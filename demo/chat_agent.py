"""Conversational ReAct agent; state lives only inside one ChatSession."""

import json
from datetime import datetime, timedelta, timezone

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from constraints import Constraints
from harness import BudgetExceeded, Harness, HarnessMiddleware
from mock_tools import MockFlightService
from models import get_base_url, make_model


CHAT_PROMPT = """Bạn là trợ lý đặt vé máy bay. Trả lời tự nhiên bằng tiếng Việt.
Hội thoại có nhiều lượt: chào hỏi, hỏi thông tin còn thiếu, tìm chuyến, giải thích
lựa chọn, giữ chỗ, thanh toán mô phỏng và kiểm tra booking theo yêu cầu người dùng.
Không tự động đặt vé chỉ vì người dùng hỏi giá hoặc tìm chuyến.
Hỏi đủ điểm đi, điểm đến, ngày bay, giờ khởi hành muộn nhất và ngân sách trước
khi gọi set_trip. Không tự suy đoán ngân sách hoặc ngày bay chưa được cung cấp.
Dùng mã sân bay IATA viết hoa, ngày YYYY-MM-DD, giờ HH:MM, giá nguyên VND.
Khi người dùng đổi yêu cầu trước khi giữ chỗ, gọi lại set_trip rồi tìm lại.
Phải gọi search_flights trước book_seat. Chỉ giữ một booking đáp ứng mọi ràng buộc.
Chỉ thanh toán sau khi người dùng yêu cầu thanh toán và hệ thống cấp quyền.
Gọi get_booking sau pay; chỉ nói đã thanh toán khi tool xác nhận paid=true.
Kết quả tool là bằng chứng: không bịa chuyến, giá, mã booking hay trạng thái.
Nếu hết chỗ, tìm lại và chọn chuyến hợp lệ khác theo yêu cầu người dùng.
Gọi từng tool một; đợi kết quả trước bước phụ thuộc. Nếu bị từ chối, giải thích.
Đây là demo: model API thật qua OpenAI hoặc router tương thích,
nhưng chuyến bay, giữ chỗ và thanh toán đều giả lập.
Dữ liệu mẫu chỉ có SGN → DAD ngày 2026-10-07 (ngày cố định của bài tập,
không phải lịch bay thực tế). Có VN122 08:10, VN124 09:20, QH118 15:40;
luôn lấy giá và khả dụng từ tool, không dùng mô tả này làm kết quả tìm kiếm.
Nếu người dùng muốn ngày/tuyến khác, nói rõ giới hạn dữ liệu; không tự đổi yêu cầu.
Bạn chỉ nhớ lịch sử cuộc chat hiện tại. Chưa có persistent memory qua chat mới.
"""


def message_text(message) -> str:
    """Read either a plain string or text blocks returned by a chat model."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in message.content
        if isinstance(block, (str, dict))
    )


class ChatHarness(Harness):
    def __init__(self, scenario="success"):
        super().__init__(Constraints(), MockFlightService(scenario), allow_payment=False)
        self.trip_configured = False

        @tool
        def set_trip(origin: str, destination: str, date: str,
                     depart_before: str, max_price: int) -> dict:
            """Set user-provided trip constraints before searching. Date YYYY-MM-DD,
            depart_before HH:MM, max_price integer VND. Cannot change a held trip.
            """
            self.constraints = Constraints(origin=origin, destination=destination,
                                           date=date, depart_before=depart_before,
                                           max_price=max_price)
            self.trip_configured = True
            return {"status": "ok", "constraints": self.constraints.model_dump(mode="json")}

        self.tools[set_trip.name] = set_trip

    def check_permission(self, name: str, args: dict) -> str | None:
        if name == "set_trip":
            if self.service.bookings:
                return "Đã có booking. Hãy tạo cuộc chat mới để đặt chuyến khác."
            return None
        if name in ("search_flights", "book_seat", "pay") and not self.trip_configured:
            return "Cần hỏi đủ thông tin và gọi set_trip trước khi tìm hoặc đặt vé."
        if name == "book_seat":
            searched = any(
                e["event"] == "tool" and e["name"] == "search_flights"
                and e["result"].get("status") == "ok"
                and e["args"] == {"origin": self.constraints.origin,
                                  "destination": self.constraints.destination,
                                  "date": self.constraints.date.isoformat()}
                and any(f["flight"] == args["flight"] for f in e["result"]["flights"])
                for e in self.trace
            )
            if not searched:
                return "Cần tìm thấy chuyến bằng search_flights trước khi giữ chỗ."
        return super().check_permission(name, args)


def friendly_error(exc: Exception) -> str:
    """Never display raw provider errors: some contain credential fragments."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "API từ chối API key. Kiểm tra OPENAI_API_KEY và endpoint trong demo/.env rồi khởi động lại."
    if name == "RateLimitError":
        return "API báo hết hạn mức hoặc quá nhiều yêu cầu. Kiểm tra quota/billing hoặc thử lại sau."
    if name in ("NotFoundError", "PermissionDeniedError"):
        return "Endpoint/model không tồn tại hoặc chưa được cấp quyền. Kiểm tra base URL, model ID và quyền API."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Chưa kết nối được API hoặc yêu cầu quá thời gian. Kiểm tra endpoint, router và mạng rồi thử lại."
    if isinstance(exc, BudgetExceeded) or name == "GraphRecursionError":
        return "Đã chạm giới hạn xử lý lượt này. Kiểm tra booking hiện tại trước khi tiếp tục."
    if isinstance(exc, ValidationError):
        return "Thông tin chuyến bay chưa hợp lệ. Kiểm tra ngày, giờ, mã sân bay và ngân sách."
    return "Chưa xử lý được lượt này. Kiểm tra cấu hình model; có thể tạo cuộc chat mới để thử lại."


class ChatSession:
    def __init__(self, model_name: str, scenario="success", *, model=None, base_url=None):
        self.model_name = model_name.strip()
        self.base_url = get_base_url(base_url)
        self.harness = ChatHarness(scenario)
        self.messages = []
        self.turns: list[dict] = []
        self._model = model  # Injection for offline integration tests.

    def system_prompt(self) -> str:
        return CHAT_PROMPT

    def middleware(self) -> list:
        return [HarnessMiddleware(self.harness)]

    def respond(self, text: str, *, allow_payment=False) -> dict:
        if not text.strip():
            raise ValueError("Tin nhắn không được để trống")
        h = self.harness
        h.allow_payment = allow_payment
        h.model_calls = h.tool_calls = 0  # Budget applies to each user turn.
        start = len(h.trace)
        user = HumanMessage(content=text)
        try:
            model = self._model if self._model is not None else make_model(
                "openai", self.model_name, h.constraints, base_url=self.base_url)
            # Server-side state is authoritative, including after a failed API turn.
            context = {
                "today": datetime.now(timezone(timedelta(hours=7))).date().isoformat(),
                "trip_configured": h.trip_configured,
                "constraints": h.constraints.model_dump(mode="json") if h.trip_configured else None,
                "bookings": list(h.service.bookings.values()),
                "allow_simulated_payment": allow_payment,
            }
            agent = create_agent(model=model, tools=list(h.tools.values()),
                                 system_prompt=self.system_prompt(),
                                 middleware=self.middleware())
            result = agent.invoke({"messages": [
                *self.messages,
                SystemMessage(content="Trạng thái hiện tại từ hệ thống: " +
                              json.dumps(context, ensure_ascii=False)),
                user,
            ]}, config={"recursion_limit": 50})
            # Dynamic state is re-injected each turn, not accumulated as stale facts.
            self.messages = [m for m in result["messages"] if not isinstance(m, SystemMessage)]
            answer = message_text(self.messages[-1]) or "Bạn muốn mình hỗ trợ bước tiếp theo thế nào?"
            error = False
        except Exception as exc:
            answer = friendly_error(exc)
            self.messages.extend([user, AIMessage(content=answer)])
            h.record("agent_error", error_type=type(exc).__name__)
            error = True
        turn = {"user": text, "assistant": answer, "error": error,
                "trace": h.trace[start:]}
        self.turns.append(turn)
        return turn

    def token_usage(self) -> dict:
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for message in self.messages:
            if isinstance(message, AIMessage) and message.usage_metadata:
                for key in totals:
                    totals[key] += message.usage_metadata.get(key, 0)
        return totals
