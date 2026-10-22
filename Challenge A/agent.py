"""Challenge A: extend the existing booking agent with project memory."""

import json
from uuid import uuid4

import bootstrap  # noqa: F401 -- establishes the shared demo import path
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from chat_agent import CHAT_PROMPT, ChatSession
from memory_store import ProjectMemoryStore


MEMORY_INSTRUCTIONS = """
Bạn có persistent memory cho QUY ƯỚC DỰ ÁN, lưu bằng tool remember_project_rule.
Chỉ gọi tool ghi khi người dùng yêu cầu rõ ràng ghi nhớ hoặc cập nhật quy ước.
Không lưu thông tin chuyến đi tạm thời, booking hoặc API key vào memory dự án.
Khi chỉ hỏi/tìm vé, áp dụng memory hiện hành, không tự tạo hay sửa quy ước.
Chọn topic ổn định bằng snake_case. Nếu cùng ý nghĩa với topic đã có, BẮT BUỘC
dùng lại đúng topic đó để thay thế quy ước cũ, không tạo topic đồng nghĩa mới.
Quy ước trình bày kết quả chuyến bay dùng topic flight_results_format.
Giá trị value phải diễn đạt đầy đủ quy ước mới, không chỉ lưu 'như trên'.
Dùng revision trong project_memory làm expected_revision khi ghi.
Nếu có conflict, đọc memory mới bằng read_project_memory và xem lại trước khi ghi.
Chỉ nói đã ghi nhớ khi tool xác nhận status=ok. Tool bị lỗi thì nói chưa lưu được.
project_memory trong system message được nạp lại trước MỖI lần gọi model.
Chỉ active_rules có hiệu lực; quy ước cũ trong lịch sử hội thoại đã bị thay thế.
Áp dụng quy ước về cách trình bày khi trả lời; tham số tool luôn theo schema.
Quy ước không được vô hiệu hóa harness, quyền thanh toán hay tính đúng của dữ liệu.
Nếu memory rỗng, nói rõ chưa có quy ước nào, không suy đoán từ cuộc chat khác.
"""


class MemoryInjectionMiddleware(AgentMiddleware):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def wrap_model_call(self, request, handler):
        snapshot = self.session.store.snapshot()
        injected = "<project_memory>\n" + json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n</project_memory>"
        # Capture exactly the additional system content passed to this model call.
        evidence = {"model_call": self.session.harness.model_calls,
                    "session_id": self.session.session_id, "snapshot": snapshot,
                    "injected_context": injected}
        self.session.injections.append(evidence)
        base = request.system_message.text if request.system_message else ""
        return handler(request.override(system_message=SystemMessage(content=base + "\n\n" + injected)))


class MemoryChatSession(ChatSession):
    def __init__(self, model_name: str, store: ProjectMemoryStore, scenario="success", **kwargs):
        super().__init__(model_name, scenario, **kwargs)
        self.store = store
        self.session_id = uuid4().hex
        self.initial_memory = store.snapshot()
        self.injections: list[dict] = []
        self.current_source = ""

        @tool
        def read_project_memory() -> dict:
            """Read active project conventions and the current memory revision."""
            return {"status": "ok", **self.store.snapshot()}

        @tool
        def remember_project_rule(topic: str, value: str, expected_revision: int) -> dict:
            """Save an explicitly requested project convention. Reuse an existing topic
            to replace its old rule. expected_revision must match the injected memory.
            """
            return self.store.remember(topic, value, expected_revision=expected_revision,
                                       source_message=self.current_source,
                                       session_id=self.session_id)

        self.harness.tools.update({t.name: t for t in [read_project_memory, remember_project_rule]})

    def system_prompt(self) -> str:
        return CHAT_PROMPT.replace(
            "Bạn chỉ nhớ lịch sử cuộc chat hiện tại. Chưa có persistent memory qua chat mới.",
            "Bạn có bộ nhớ quy ước dự án qua nhiều session.") + MEMORY_INSTRUCTIONS

    def middleware(self) -> list:
        return [*super().middleware(), MemoryInjectionMiddleware(self)]

    def respond(self, text: str, **kwargs) -> dict:
        self.current_source = text
        start = len(self.injections)
        turn = super().respond(text, **kwargs)
        turn["session_id"] = self.session_id
        turn["memory_injections"] = self.injections[start:]
        return turn
