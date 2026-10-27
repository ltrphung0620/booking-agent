"""Offline checks for the native read_file agent, its path policy and its meter."""

import tempfile
import unittest
from pathlib import Path

import bootstrap  # noqa: F401
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from investigation import InvestigationSession
from log_fixtures import generate_logs
from log_tools import InvestigationHarness, path_policy


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, ident):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


def reads(files):
    return [call("list_files", {"glob": "logs/*.log"}, "ls"),
            *[call("read_file", {"path": f}, f"r{i}") for i, f in enumerate(files)],
            AIMessage(content="ERR_PAY_TIMEOUT tăng mạnh ở pg-2.")]


class NativeModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.truth = generate_logs(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, responses, **kwargs):
        return InvestigationSession("fake", self.root, model=ScriptedModel(responses=responses),
                                    **kwargs)

    def tool_events(self, turn):
        return [e for e in turn["trace"] if e["event"] == "tool"]

    def test_lists_then_reads_three_files(self):
        session = self.session(reads(self.truth["files"][6:9]))
        turn = session.respond("Ngày 07–09 có lỗi gì?")
        self.assertFalse(turn["error"], turn["assistant"])
        events = self.tool_events(turn)
        self.assertEqual([e["name"] for e in events], ["list_files", *["read_file"] * 3])
        self.assertTrue(all(e["result"]["status"] == "ok" for e in events))
        self.assertEqual(events[0]["result"]["files"], self.truth["files"])
        self.assertIn("ERR_PAY_TIMEOUT", events[1]["result"]["content"])

    def test_secret_and_escaping_paths_are_denied(self):
        bad = [".env", "../demo/.env", str(self.root / ".env"), str(self.root / self.truth["files"][0]),
               "logs/../.env", "logs\\..\\.env", "C:\\Windows\\win.ini"]
        responses = [call("read_file", {"path": p}, f"b{i}") for i, p in enumerate(bad)]
        turn = self.session([*responses, AIMessage(content="Không đọc được.")]).respond("Đọc .env")
        events = self.tool_events(turn)
        self.assertEqual(len(events), len(bad))
        for event in events:
            self.assertEqual(event["result"]["status"], "denied", event["args"])
            self.assertIn("PermissionDenied", event["result"]["reason"])
        self.assertNotIn("fake-do-not-leak", str(turn["trace"]))

    def test_policy_allows_only_logs(self):
        self.assertIsNone(path_policy("logs/booking-2026-10-07.log"))
        self.assertIn(".env", path_policy(".env"))
        self.assertIsNotNone(path_policy("data/x.log"))
        self.assertIsNotNone(path_policy("logs/.hidden"))

    def test_list_files_hides_secret(self):
        events = self.tool_events(self.session([call("list_files", {"glob": "**/*"}, "a"),
                                                call("list_files", {"glob": "../*"}, "b"),
                                                AIMessage(content="ok")]).respond("Liệt kê"))
        self.assertEqual(events[0]["result"]["files"], self.truth["files"])
        self.assertEqual(events[1]["result"]["status"], "denied")

    def test_context_grows_and_usage_is_reported(self):
        session = self.session(reads(self.truth["files"][:3]))
        session.respond("Đọc 3 file đầu")
        sizes = [e["approx_tokens"] for e in session.context_log]
        self.assertEqual(len(sizes), 5)
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1] - sizes[0], 3 * 4000)  # Three raw files are now in context.
        self.assertEqual(session.context_log[0]["tools"], ["list_files", "read_file"])
        summary = session.usage_summary()
        self.assertEqual(summary["llm_calls"], 5)
        self.assertEqual(summary["source"], "estimate")  # The fake model reports no usage.
        self.assertEqual(summary["total_input"], sum(sizes))
        self.assertEqual(summary["peak_input"], sizes[-1])
        self.assertGreater(summary["total_output"], 0)

    def test_captured_input_contains_raw_log(self):
        canary_file = self.truth["canary_file"]
        session = self.session(reads([canary_file]), capture_inputs=True)
        session.respond("Đọc file có canary")
        self.assertNotIn(self.truth["canary"], session.context_log[0]["text"])
        self.assertIn(self.truth["canary"], session.context_log[-1]["text"])

    def test_twenty_reads_in_one_turn_fit_the_recursion_limit(self):
        session = self.session(reads(self.truth["files"][:20]))
        turn = session.respond("Đọc 20 file")
        self.assertFalse(turn["error"], turn["assistant"])
        self.assertEqual(sum(e["name"] == "read_file" and e["result"]["status"] == "ok"
                             for e in self.tool_events(turn)), 20)

    def test_demo_default_recursion_limit_is_too_small(self):
        session = self.session(reads(self.truth["files"][:20]))
        session.recursion_limit = 50  # ChatSession's default before the hook.
        turn = session.respond("Đọc 20 file")
        self.assertTrue(turn["error"])
        self.assertIn({"event": "agent_error", "error_type": "GraphRecursionError"}, turn["trace"])


class ExecuteInnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.truth = generate_logs(self.root, days=3)
        self.harness = InvestigationHarness(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gate_validates_denies_and_traces_without_content(self):
        h = self.harness
        ok = h.execute_inner("read_file", {"path": self.truth["files"][0]})
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(h.execute_inner("read_file", {"path": ".env"})["status"], "denied")
        self.assertEqual(h.execute_inner("run_code", {"code": "pass"})["status"], "denied")
        self.assertEqual(h.execute_inner("read_file", {"path": 1, "x": 2})["status"], "error")
        self.assertEqual(h.tool_calls, 0)  # Runtime calls have their own quota.
        events = [e for e in h.trace if e["event"] == "runtime_tool"]
        self.assertEqual([e["status"] for e in events], ["ok", "denied", "denied", "error"])
        self.assertGreater(events[0]["tokens"], 1000)
        self.assertNotIn(ok["content"][:200], str(h.trace))
        self.assertTrue(all("content" not in e for e in events))


if __name__ == "__main__":
    unittest.main()
