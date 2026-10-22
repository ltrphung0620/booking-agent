"""Bốn lớp harness dùng chung cho cả ba mẫu agent."""

import json
from typing import Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from constraints import Constraints
from mock_tools import MockFlightService


class BudgetExceeded(RuntimeError):
    pass


class Harness:
    def __init__(self, constraints: Constraints, service: MockFlightService,
                 allow_payment: bool = True, max_model_calls: int = 12,
                 max_tool_calls: int = 24):
        self.constraints = constraints
        self.service = service
        self.tools = service.make_tools()
        self.allow_payment = allow_payment
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.model_calls = 0
        self.tool_calls = 0
        self.trace: list[dict] = []

    def record(self, event: str, **data) -> None:
        self.trace.append({"event": event, **data})

    def consume_model_call(self, purpose: str) -> None:
        if self.model_calls >= self.max_model_calls:
            raise BudgetExceeded("Đã hết giới hạn gọi model")
        self.model_calls += 1
        self.record("model", purpose=purpose, number=self.model_calls)

    def check_permission(self, name: str, args: dict) -> str | None:
        if name not in self.tools:
            return "Tool không có trong danh sách cho phép"
        c = self.constraints
        if name == "search_flights":
            if args != {"origin": c.origin, "destination": c.destination,
                        "date": c.date.isoformat()}:
                return "Tham số tìm vé phải khớp ràng buộc gốc"
        elif name == "book_seat":
            flight = self.service.find_flight(args["flight"])
            if flight is None or not c.is_ok(flight):
                return "Chuyến bay không đáp ứng ràng buộc"
            if any(b["flight"] != args["flight"] for b in self.service.bookings.values()):
                return "Chỉ được giữ một booking cho yêu cầu một vé"
        elif name in ("pay", "get_booking"):
            booking = self.service.get_booking(args["code"])
            if booking["status"] != "ok":
                return "Booking không thuộc lần chạy hiện tại hoặc không tồn tại"
            if name == "pay":
                if not self.allow_payment:
                    return "Chưa được cấp quyền thanh toán"
                if not c.is_ok(booking):
                    return "Booking không còn đáp ứng ràng buộc; chặn thanh toán"
                quote = self.service.find_flight(booking["flight"])
                if quote is None or not c.is_ok(quote) or quote["price"] != booking["price"]:
                    return "Giá hoặc thông tin chuyến đã thay đổi; cần xác nhận lại"
                if not quote["available"]:
                    return "Chuyến bay không còn khả dụng; chặn thanh toán"
        return None

    def execute(self, name: str, args: dict, runner: Callable | None = None) -> dict:
        """Mọi tool call, từ ReAct hoặc executor, đều đi qua cổng này."""
        with self.service.lock:
            if self.tool_calls >= self.max_tool_calls:
                raise BudgetExceeded("Đã hết giới hạn gọi tool")
            self.tool_calls += 1
            if name not in self.tools:
                result = {"status": "denied", "reason": "Tool không được phép"}
            else:
                try:
                    schema = self.tools[name].args_schema
                    if set(args) - set(schema.model_fields):
                        raise ValueError("Có tham số không được hỗ trợ")
                    clean_args = schema.model_validate(args).model_dump()
                    reason = self.check_permission(name, clean_args)
                    if reason:
                        result = {"status": "denied", "reason": reason}
                    else:
                        raw = runner() if runner else self.tools[name].invoke(clean_args)
                        result = json.loads(raw.content) if isinstance(raw, ToolMessage) else raw
                    if not isinstance(result, dict) or "status" not in result:
                        result = {"status": "error", "reason": "Tool trả kết quả thiếu status"}
                except (ValidationError, ValueError, TypeError, KeyError) as exc:
                    result = {"status": "error", "reason": f"Dữ liệu tool không hợp lệ: {exc}"}
                except Exception as exc:
                    result = {"status": "error", "reason": f"Tool bị lỗi: {type(exc).__name__}: {exc}"}
            self.record("tool", name=name, args=args, result=result)
            return result

    def booking_code(self) -> str:
        for event in reversed(self.trace):
            if (event["event"] == "tool" and event["name"] == "book_seat"
                    and event["result"]["status"] == "ok"):
                return event["result"]["code"]
        return ""

    def is_done(self) -> bool:
        """Đọc lại dữ liệu thật: đúng MỘT booking, đã trả tiền và hợp lệ."""
        bookings = [self.service.get_booking(code) for code in self.service.bookings]
        return (len(bookings) == 1 and bookings[0].get("paid") is True
                and self.constraints.is_ok(bookings[0]))

    def handoff(self, reason: str) -> dict:
        if "timeout" in reason.lower():
            question = "Dịch vụ tìm vé lỗi. Bạn muốn thử lại sau không?"
        elif "thanh toán" in reason.lower():
            question = "Có booking chưa thanh toán. Bạn muốn kiểm tra và cấp quyền thanh toán không?"
        elif "hết chỗ" in reason.lower():
            question = "Chuyến đã hết chỗ. Bạn muốn chọn chuyến hợp lệ khác không?"
        elif "không có chuyến" in reason.lower():
            question = "Bạn muốn nới giờ khởi hành hay tăng ngân sách?"
        else:
            question = "Bạn muốn kiểm tra lịch sử và tiếp tục xử lý yêu cầu này như thế nào?"
        return {
            "reason": reason,
            "done_so_far": list(self.service.bookings.values()),
            "tried": [e for e in self.trace if e["event"] == "tool"],
            "question": question,
        }

    def finish(self, pattern: str, reason: str = "", model_answer: str = "") -> dict:
        done = self.is_done()
        self.record("verification", done=done)
        output = {"pattern": pattern, "result": "DONE" if done else "FAILED",
                  "constraints": self.constraints.model_dump(mode="json"),
                  "bookings": list(self.service.bookings.values()),
                  "model_answer": model_answer}
        if not done:
            reason = reason or "Chưa có đúng một booking đã thanh toán và hợp lệ"
            output["handoff"] = self.handoff(reason)
        output["trace"] = self.trace
        return output


class HarnessMiddleware(AgentMiddleware):
    """LangChain gọi các hook này quanh model và tool."""
    def __init__(self, harness: Harness):
        super().__init__()
        self.harness = harness

    def before_model(self, state, runtime):
        self.harness.consume_model_call("react")
        return None

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        result = self.harness.execute(call["name"], call["args"],
                                      runner=lambda: handler(request))
        return ToolMessage(content=json.dumps(result, ensure_ascii=False),
                           tool_call_id=call["id"], name=call["name"])
