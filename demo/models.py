"""Đầu vào/đầu ra model: mock phản ứng với tool results hoặc LLM OpenAI thật."""

import json
import os
from typing import Literal
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

from constraints import Constraints


def get_base_url(value: str | None = None) -> str:
    """Resolve an explicit session endpoint before server environment defaults."""
    if value is None:
        value = os.environ.get("OPENAI_BASE_URL", "")
    return value.strip().rstrip("/") or "https://api.openai.com/v1"


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["book_seat", "pay", "get_booking"]
    args: dict[str, str]


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    steps: list[Step] = Field(max_length=6)


class DemoChatModel(BaseChatModel):
    """Model giả có phản hồi theo observation, không gọi API hoặc suy luận LLM.

    Vòng lặp agent vẫn do LangChain thực thi. Model chỉ nhìn constraints,
    plan và ToolMessage; không đọc database trực tiếp.
    """
    constraints: Constraints
    plan: Plan | None = None

    @property
    def _llm_type(self) -> str:
        return "flight-demo-mock"

    def bind_tools(self, tools, **kwargs):
        return self

    @staticmethod
    def _call(name: str, **args) -> AIMessage:
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"call_{uuid4().hex}",
            "type": "tool_call",
        }])

    def _search(self) -> AIMessage:
        c = self.constraints
        return self._call("search_flights", origin=c.origin,
                          destination=c.destination, date=c.date.isoformat())

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        observations = [m for m in messages if isinstance(m, ToolMessage)]
        if not observations:
            if self.plan and self.plan.steps:
                step = self.plan.steps[0]
                answer = self._call(step.tool, **step.args)
            else:
                answer = self._search()
        else:
            last = observations[-1]
            result = json.loads(last.content)
            if result["status"] != "ok":
                if result["status"] == "unavailable" and self.plan is None:
                    answer = self._search()  # ReAct tự tìm lại sau observation lỗi.
                else:
                    answer = AIMessage(content=f"Cần xử lý tiếp: {result.get('reason', result['status'])}")
            elif last.name == "search_flights":
                valid = [f for f in result["flights"] if self.constraints.is_ok(f)]
                if valid:
                    chosen = min(valid, key=lambda f: (f["price"], f["depart"]))
                    answer = self._call("book_seat", flight=chosen["flight"])
                else:
                    answer = AIMessage(content="Không có chuyến đáp ứng tất cả ràng buộc.")
            elif last.name == "book_seat":
                answer = self._call("pay", code=result["code"])
            elif last.name == "pay":
                answer = self._call("get_booking", code=result["code"])
            else:
                answer = AIMessage(content="Đã đọc lại booking sau thanh toán.")
        return ChatResult(generations=[ChatGeneration(message=answer)])


def make_model(backend: str, model_name: str | None, constraints: Constraints,
               plan: Plan | None = None, *, base_url: str | None = None) -> BaseChatModel:
    if backend == "mock":
        return DemoChatModel(constraints=constraints, plan=plan)
    if backend != "openai":
        raise ValueError(f"Unknown backend: {backend}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Hãy đặt biến môi trường OPENAI_API_KEY để dùng LLM thật")
    if not model_name:
        raise ValueError("Hãy chọn model bằng --model hoặc biến FLIGHT_MODEL")
    from langchain_openai import ChatOpenAI
    # Không buộc temperature vì một số model không hỗ trợ tham số này.
    return ChatOpenAI(model=model_name.removeprefix("openai:"), max_retries=1,
                      timeout=45, base_url=get_base_url(base_url),
                      use_responses_api=False)


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Plan one flight booking using only the supplied facts. "
     "Meet every constraint. Return a Plan with summary and ordered steps. "
     "Tools: book_seat(flight), pay(code), get_booking(code). "
     "For the unknown booking code, use the exact string $booking_code. "
     "If a valid booking already exists, reuse its code and do not book again. "
     "If no flight fits, return an empty steps list. "
     "Use the failure/reviewer feedback when revising a plan."),
    ("human", "{payload}"),
])


def mock_plan(prompt) -> Plan:
    payload = json.loads(prompt.to_messages()[-1].content)
    constraints = Constraints.model_validate(payload["constraints"])
    existing = payload["bookings"]
    if existing:
        booking = existing[0]
        if not constraints.is_ok(booking):
            return Plan(summary="Booking hiện tại cần người kiểm tra", steps=[])
        code = booking["code"]
        steps = [] if booking["paid"] else [Step(tool="pay", args={"code": code})]
        return Plan(summary="Tiếp tục booking hiện tại", steps=[
            *steps, Step(tool="get_booking", args={"code": code})])
    flights = [f for f in payload["flights"] if constraints.is_ok(f)]
    if not flights:
        return Plan(summary="Không có chuyến đáp ứng tất cả ràng buộc", steps=[])
    flight = min(flights, key=lambda f: (f["price"], f["depart"]))
    return Plan(summary=f"Giữ chỗ, thanh toán và kiểm tra {flight['flight']}", steps=[
        Step(tool="book_seat", args={"flight": flight["flight"]}),
        Step(tool="pay", args={"code": "$booking_code"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])


def make_planner(backend: str, model_name: str | None, constraints: Constraints):
    if backend == "mock":
        model = RunnableLambda(mock_plan)
    else:
        model = make_model(backend, model_name, constraints).with_structured_output(
            Plan, method="function_calling")
    return PLANNER_PROMPT | model
