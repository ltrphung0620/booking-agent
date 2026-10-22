"""Offline proofs: actual model input, new processes, updates and file safety."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import bootstrap
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent import MemoryChatSession
from memory_store import MemoryFileError, ProjectMemoryStore


class RecordingModel(FakeMessagesListChatModel):
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, **kwargs)


def remember(value, revision, ident):
    return AIMessage(content="", tool_calls=[{
        "id": ident, "type": "tool_call", "name": "remember_project_rule",
        "args": {"topic": "flight_results_format", "value": value, "expected_revision": revision}
    }])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "project_memory.json"
        self.store = ProjectMemoryStore(self.path)

    def save(self, value="Bảng Markdown", revision=0, **kwargs):
        return self.store.remember("flight_results_format", value, expected_revision=revision,
                                   source_message=kwargs.get("source", "Ghi nhớ quy ước này"),
                                   session_id=kwargs.get("session", "session-one"))

    def test_two_challenge_proofs_through_agent_loop_and_actual_model_input(self):
        writer = RecordingModel(responses=[remember("Bảng Markdown", 0, "save"),
                                          AIMessage(content="Đã lưu quy ước.")])
        first = MemoryChatSession("offline", self.store, model=writer)
        source = "Ghi nhớ quy ước dự án: trình bày chuyến bay bằng bảng Markdown."
        turn = first.respond(source)
        self.assertFalse(turn["error"], turn)
        self.assertEqual(len(writer.seen), 2)
        self.assertNotIn("Bảng Markdown", first.injections[0]["injected_context"])
        self.assertIn("Bảng Markdown", writer.seen[1][0].text)
        self.assertEqual(self.store.history()["records"][0]["source_message"], source)

        reader = RecordingModel(responses=[AIMessage(content="| Chuyến | Giá |\n|---|---|\n| VN122 | 1850000 |")])
        second = MemoryChatSession("offline", ProjectMemoryStore(self.path), model=reader)
        self.assertEqual(second.messages, [])
        self.assertFalse(second.harness.service.bookings)
        result = second.respond("Tìm chuyến bay giúp tôi")
        self.assertFalse(result["error"], result)
        self.assertIn("Bảng Markdown", reader.seen[0][0].text)
        self.assertEqual(len([m for m in reader.seen[0] if isinstance(m, HumanMessage)]), 1)
        self.assertEqual(self.store.history()["revision"], 1)  # Reading never writes.

        editor = RecordingModel(responses=[remember("Danh sách gạch đầu dòng", 1, "update"),
                                          AIMessage(content="Đã cập nhật.")])
        third = MemoryChatSession("offline", ProjectMemoryStore(self.path), model=editor)
        self.assertFalse(third.respond("Từ nay dùng danh sách thay cho bảng")["error"])
        reader_new = RecordingModel(responses=[AIMessage(content="- VN122: 1.850.000 VND")])
        fourth = MemoryChatSession("offline", ProjectMemoryStore(self.path), model=reader_new)
        self.assertFalse(fourth.respond("Tìm chuyến bay giúp tôi")["error"])
        injected = fourth.injections[0]["injected_context"]
        self.assertIn("Danh sách gạch đầu dòng", reader_new.seen[0][0].text)
        self.assertNotIn("Bảng Markdown", injected)
        history = self.store.history()["records"]
        self.assertEqual([r["status"] for r in history], ["superseded", "active"])
        self.assertEqual(history[0]["superseded_by"], history[1]["id"])
        self.assertEqual(history[1]["supersedes"], history[0]["id"])

    def test_memory_survives_a_new_python_process(self):
        self.save()
        code = ("import json,sys; from memory_store import ProjectMemoryStore; "
                "print(json.dumps(ProjectMemoryStore(sys.argv[1]).snapshot(),ensure_ascii=True))")
        run = subprocess.run([sys.executable, "-c", code, str(self.path)], cwd=bootstrap.HERE,
                             capture_output=True, text=True, check=True, timeout=15)
        self.assertEqual(json.loads(run.stdout)["active_rules"][0]["value"], "Bảng Markdown")

    def test_unchanged_rule_and_stale_revision_do_not_overwrite(self):
        self.save()
        unchanged = self.save(revision=1)
        self.assertFalse(unchanged["changed"])
        stale = self.save("Danh sách", revision=0)
        self.assertEqual(stale["status"], "conflict")
        self.assertEqual(self.store.history()["revision"], 1)

    def test_parallel_sessions_cannot_lose_updates(self):
        def write(value):
            store = ProjectMemoryStore(self.path)
            return store.remember("flight_results_format", value, expected_revision=0,
                                  session_id=value, source_message="Ghi nhớ quy ước")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(write, ["Bảng", "Danh sách"]))
        self.assertCountEqual([r["status"] for r in results], ["ok", "conflict"])
        self.assertEqual(len(self.store.history()["records"]), 1)

    def test_broken_file_is_not_silently_replaced(self):
        self.path.write_text('{"records": BROKEN', encoding="utf-8")
        original = self.path.read_bytes()
        with self.assertRaises(MemoryFileError):
            self.save()
        self.assertEqual(self.path.read_bytes(), original)

    def test_failed_atomic_write_preserves_previous_revision(self):
        self.save()
        original = self.path.read_bytes()
        with patch("memory_store.os.replace", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                self.save("Danh sách", revision=1)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))

    def test_existing_session_refreshes_memory_written_by_another_session(self):
        reader = RecordingModel(responses=[AIMessage(content="Chưa có quy ước"), AIMessage(content="Dùng bảng")])
        session = MemoryChatSession("offline", self.store, model=reader)
        session.respond("Quy ước hiện tại là gì?")
        self.save()
        session.respond("Kiểm tra lại quy ước")
        self.assertEqual([i["snapshot"]["revision"] for i in session.injections], [0, 1])
        self.assertIn("Bảng Markdown", reader.seen[-1][0].text)

    def test_ui_new_chat_preserves_rules_but_clears_history_and_bookings(self):
        from streamlit.testing.v1 import AppTest

        self.save()
        with patch("dotenv.load_dotenv"), patch.dict(os.environ, {
            "OPENAI_API_KEY": "offline-not-real", "FLIGHT_MODEL": "offline",
            "CHALLENGE_A_MEMORY_PATH": str(self.path)
        }):
            app = AppTest.from_file(str(bootstrap.HERE / "app.py")).run()
            self.assertFalse(app.exception)
            initial = app.session_state["challenge_a_chat"].session_id
            model = RecordingModel(responses=[AIMessage(content="Dùng bảng Markdown.")])
            app.session_state["challenge_a_chat"]._model = model
            app.chat_input[0].set_value("Quy ước trình bày là gì?").run()
            self.assertFalse(app.exception)
            self.assertIn("Bảng Markdown", model.seen[0][0].text)
            self.assertEqual(len(app.session_state["challenge_a_chat"].turns), 1)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            fresh = app.session_state["challenge_a_chat"]
            self.assertNotEqual(initial, fresh.session_id)
            self.assertFalse(fresh.messages)
            self.assertFalse(fresh.harness.service.bookings)
            self.assertEqual(fresh.initial_memory["revision"], 1)


if __name__ == "__main__":
    unittest.main()
