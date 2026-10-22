"""Offline integration tests for multi-turn state and the chat harness."""

import unittest
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from chat_agent import ChatHarness, ChatSession, friendly_error
from constraints import Constraints
from models import get_base_url, make_model


TRIP = {"origin": "SGN", "destination": "DAD", "date": "2026-10-07",
        "depart_before": "12:00", "max_price": 2_000_000}


class ScriptedChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, identifier):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": identifier, "type": "tool_call"}])


class ChatTests(unittest.TestCase):
    def test_custom_endpoint_receives_chat_completion_and_tool_schema(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append({"path": self.path, "payload": payload,
                                 "auth": self.headers.get("Authorization")})
                response = json.dumps({
                    "id": "chatcmpl-offline", "object": "chat.completion", "created": 0,
                    "model": payload["model"], "choices": [{"index": 0,
                        "finish_reason": "tool_calls", "message": {"role": "assistant",
                            "content": None, "tool_calls": [{"id": "trip", "type": "function",
                                "function": {"name": "set_trip", "arguments": json.dumps(TRIP)}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            with patch.dict("os.environ", {
                "OPENAI_API_KEY": "offline-router-test", "OPENAI_BASE_URL": endpoint + "/"
            }):
                self.assertEqual(get_base_url(), endpoint)
                model = make_model("openai", "router/model-id", Constraints())
                result = model.bind_tools([ChatHarness().tools["set_trip"]]).invoke("Find a flight")
            self.assertEqual(received[0]["path"], "/v1/chat/completions")
            self.assertEqual(received[0]["auth"], "Bearer offline-router-test")
            self.assertEqual(received[0]["payload"]["model"], "router/model-id")
            self.assertEqual(received[0]["payload"]["tools"][0]["function"]["name"], "set_trip")
            self.assertEqual(result.tool_calls[0]["args"], TRIP)
            self.assertEqual(result.usage_metadata["total_tokens"], 15)
            with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://other.invalid/v1"}):
                session = ChatSession("router/model-id", base_url=endpoint)
                self.assertEqual(session.base_url, endpoint)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_multiple_turns_use_same_history_service_and_read_back_payment(self):
        model = ScriptedChatModel(responses=[
            AIMessage(content="Bạn muốn đi ngày nào?"),
            call("set_trip", TRIP, "trip"),
            call("search_flights", {k: TRIP[k] for k in ("origin", "destination", "date")}, "search"),
            AIMessage(content="VN122 giá 1.850.000 VND. Bạn muốn giữ chỗ không?"),
            call("book_seat", {"flight": "VN122"}, "book"),
            AIMessage(content="Đã giữ chỗ VN122-12A, chưa thanh toán."),
            call("pay", {"code": "VN122-12A"}, "pay-denied"),
            AIMessage(content="Bạn cần bật quyền thanh toán mô phỏng."),
            call("pay", {"code": "VN122-12A"}, "pay"),
            call("get_booking", {"code": "VN122-12A"}, "verify"),
            AIMessage(content="Đã kiểm tra, booking đã thanh toán."),
        ])
        session = ChatSession("offline-test", model=model)
        self.assertFalse(session.respond("Tìm vé SGN DAD, trước 12h, 2 triệu")["error"])
        self.assertFalse(session.harness.trip_configured)
        session.respond("Ngày 2026-10-07")
        self.assertEqual(session.harness.service.bookings, {})
        session.respond("Giữ chỗ VN122")
        self.assertFalse(session.harness.service.bookings["VN122-12A"]["paid"])
        session.respond("Thanh toán nhé")
        self.assertFalse(session.harness.service.bookings["VN122-12A"]["paid"])
        result = session.respond("Đã cấp quyền, thanh toán nhé", allow_payment=True)
        self.assertFalse(result["error"])
        self.assertTrue(session.harness.is_done())
        self.assertEqual(len([m for m in session.messages if isinstance(m, HumanMessage)]), 5)
        self.assertEqual(len(session.harness.service.bookings), 1)
        fresh = ChatSession("offline-test", model=model)
        self.assertFalse(fresh.messages)
        self.assertFalse(fresh.harness.trip_configured)
        self.assertFalse(fresh.harness.service.bookings)

    def test_constraints_required_and_locked_after_booking(self):
        h = ChatHarness()
        query = {k: TRIP[k] for k in ("origin", "destination", "date")}
        self.assertEqual(h.execute("search_flights", query)["status"], "denied")
        self.assertEqual(h.execute("set_trip", {**TRIP, "max_price": -1})["status"], "error")
        self.assertFalse(h.trip_configured)
        self.assertEqual(h.execute("set_trip", TRIP)["status"], "ok")
        self.assertEqual(h.execute("book_seat", {"flight": "VN122"})["status"], "denied")
        h.execute("search_flights", query)
        self.assertEqual(h.execute("book_seat", {"flight": "VN122"})["status"], "ok")
        self.assertEqual(h.execute("set_trip", {**TRIP, "max_price": 3_000_000})["status"], "denied")
        self.assertEqual(h.constraints.max_price, 2_000_000)

    def test_api_error_is_sanitized_and_next_turn_recovers(self):
        AuthenticationError = type("AuthenticationError", (Exception,), {})
        session = ChatSession("example-model")
        with patch("chat_agent.make_model", side_effect=AuthenticationError("sk-SECRET")):
            result = session.respond("Chào bạn")
        self.assertTrue(result["error"])
        self.assertNotIn("sk-SECRET", str(result))
        self.assertIn("API key", result["assistant"])
        session._model = ScriptedChatModel(responses=[AIMessage(content="Chào bạn!")])
        self.assertFalse(session.respond("Thử lại")["error"])
        self.assertEqual(len(session.turns), 2)
        self.assertEqual(friendly_error(RuntimeError("sk-SECRET")).find("sk-SECRET"), -1)

    def test_ui_configuration_and_new_chat_without_api_calls(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).with_name("chat_app.py")
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {
            "OPENAI_API_KEY": "", "FLIGHT_MODEL": ""
        }):
            app = AppTest.from_file(str(app_path)).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            self.assertTrue(app.info)

        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {
            "OPENAI_API_KEY": "offline-test-not-real", "FLIGHT_MODEL": "offline-test"
        }):
            app = AppTest.from_file(str(app_path)).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.chat_input[0].disabled)
            with patch("chat_agent.make_model", return_value=ScriptedChatModel(
                responses=[AIMessage(content="Bạn muốn đi đâu?")])):
                app.chat_input[0].set_value("Chào bạn").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["booking_chat"].turns), 1)
            app.text_input[0].set_value("http://127.0.0.1:20128/v1").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("booking_chat", app.session_state)
            self.assertFalse(app.chat_input[0].disabled)


if __name__ == "__main__":
    unittest.main()
