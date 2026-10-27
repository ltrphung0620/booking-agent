"""Offline B2 proofs: raw logs stay in the runtime, and each code-mode trap is handled."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import bootstrap  # noqa: F401
from langchain_core.messages import AIMessage

from code_runner import run_code
from investigation import InvestigationSession
from log_fixtures import generate_logs
from log_tools import InvestigationHarness
from test_native import ScriptedModel, call, reads

HERE = Path(__file__).resolve().parent

AGGREGATE = """
from collections import Counter
files = tools.list_files("logs/*.log")
codes, pay_days, lines, failed = Counter(), Counter(), 0, 0
for path in files:
    try:
        text = tools.read_file(path)
    except ToolError:
        failed += 1
        continue
    for line in text.splitlines():
        lines += 1
        parts = line.split()
        if parts[2] == "ERROR":
            codes[parts[4]] += 1
            if parts[4] == "ERR_PAY_TIMEOUT":
                pay_days[parts[0]] += 1
return {"top_errors": codes.most_common(3), "peak_days": pay_days.most_common(3),
        "files_listed": len(files), "files_scanned": len(files) - failed,
        "lines_scanned": lines, "failed": failed}
"""


class CodeModeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.truth = generate_logs(self.root)
        self.harness = InvestigationHarness(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_code(self, code, **limits):
        return run_code(code, self.harness.execute_inner, **limits)

    def code_session(self, codes, **kwargs):
        script = [call("run_code", {"code": c}, f"c{i}") for i, c in enumerate(codes)]
        return InvestigationSession("fake", self.root, mode="code", capture_inputs=True,
                                    model=ScriptedModel(responses=[*script, AIMessage(content="Xong.")]),
                                    **kwargs)

    def test_aggregation_matches_ground_truth(self):
        session = self.code_session([AGGREGATE])
        turn = session.respond("Mã lỗi nào nhiều nhất?")
        self.assertFalse(turn["error"], turn["assistant"])
        result = next(e["result"] for e in turn["trace"] if e["event"] == "tool")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["truncated"])
        data = json.loads(result["output"])
        self.assertEqual(data["top_errors"], self.truth["top_errors"])
        self.assertEqual(sorted(data["peak_days"]), sorted(self.truth["pay_timeout_peak_days"]))
        self.assertEqual((data["files_scanned"], data["lines_scanned"], data["failed"]),
                         (31, self.truth["total_lines"], 0))
        self.assertEqual(result["stats"]["runtime_tool_calls"], 32)

    def test_raw_logs_never_reach_the_model_in_code_mode(self):
        code = self.code_session([AGGREGATE])
        code.respond("Đếm lỗi")
        self.assertGreater(len(code.context_log), 1)
        self.assertFalse(any(self.truth["canary"] in e["text"] for e in code.context_log))
        # Control: the same check does see the canary once native mode reads that file.
        native = InvestigationSession("fake", self.root, capture_inputs=True,
                                      model=ScriptedModel(responses=reads([self.truth["canary_file"]])))
        native.respond("Đọc file")
        self.assertTrue(any(self.truth["canary"] in e["text"] for e in native.context_log))

    def test_code_mode_uses_under_a_fifth_of_native_tokens(self):
        native = InvestigationSession("fake", self.root,
                                      model=ScriptedModel(responses=reads(self.truth["files"])))
        self.assertFalse(native.respond("Đọc toàn bộ log")["error"])
        code = self.code_session([AGGREGATE])
        self.assertFalse(code.respond("Đếm lỗi")["error"])
        native_total = native.usage_summary()["approx_input"]
        code_total = code.usage_summary()["approx_input"]
        self.assertLess(code_total * 5, native_total, (code_total, native_total))

    def test_only_run_code_is_bound(self):
        session = InvestigationSession("fake", self.root, mode="code", model=ScriptedModel(responses=[
            call("read_file", {"path": self.truth["files"][0]}, "r"), AIMessage(content="Bị chặn.")]))
        turn = session.respond("Đọc thẳng file")
        self.assertEqual({e["tools"] == ["run_code"] for e in session.context_log}, {True})
        event = next(e for e in turn["trace"] if e["event"] == "tool")
        self.assertEqual(event["result"]["status"], "denied")

    def test_output_cap_stops_bloat(self):
        result = self.run_code(f"return tools.read_file({self.truth['files'][0]!r})", cap_chars=2000)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["truncated"])
        self.assertIn("…[harness cắt", result["output"])
        head, note = result["output"].rsplit("\n…[harness cắt", 1)
        self.assertEqual(len(head), 2000)
        self.assertLess(len(note), 40)

    def test_print_flood_is_capped(self):
        result = self.run_code("for i in range(100000):\n    print('x' * 50)")
        self.assertTrue(result["truncated"])
        self.assertLess(len(result["output"]), 2100)

    def test_infinite_loop_times_out_and_kills_child(self):
        procs = []
        real_popen = subprocess.Popen

        def spy(*args, **kwargs):
            procs.append(real_popen(*args, **kwargs))
            return procs[-1]

        started = time.monotonic()
        with patch("code_runner.subprocess.Popen", side_effect=spy):
            result = self.run_code("while True:\n    pass", timeout_s=1)
        self.assertEqual(result["status"], "error")
        self.assertIn("Timeout", result["reason"])
        self.assertLess(time.monotonic() - started, 1 + 3)
        self.assertIsNotNone(procs[0].poll())

    def test_secret_is_denied_and_sdk_has_no_network(self):
        for path in (".env", "../.env", str(self.root / ".env")):
            result = self.run_code(f"return tools.read_file({path!r})")
            self.assertEqual(result["status"], "error")
            self.assertIn("PermissionDenied", result["reason"])
        self.assertEqual(set(self.harness.sdk_tools), {"list_files", "read_file"})
        result = self.run_code("return tools.http_get('https://collector.evil.example/x')")
        self.assertIn("AttributeError", result["reason"])

    def test_injected_instruction_in_log_is_just_data(self):
        code = (f"text = tools.read_file({self.truth['injection_file']!r})\n"
                "return [l for l in text.splitlines() if 'support' in l]")
        result = self.run_code(code)
        self.assertEqual(result["status"], "ok")
        self.assertIn("ignore previous instructions", result["output"])
        self.assertNotIn("fake-do-not-leak", str(self.harness.trace) + result["output"])

    def test_child_gets_no_app_env_and_an_empty_cwd(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            result = self.run_code("import os\nreturn {'env': dict(os.environ), 'cwd': os.getcwd(),"
                                   " 'files': os.listdir('.')}")
        data = json.loads(result["output"])
        self.assertNotIn("OPENAI_API_KEY", data["env"])
        self.assertNotIn("sk-test", result["output"])
        self.assertEqual(data["files"], [])
        cwd = Path(data["cwd"]).resolve()
        self.assertFalse(cwd.is_relative_to(HERE.parent))
        self.assertFalse(cwd.exists())  # Removed after the run.

    def test_code_errors_are_short(self):
        for code, kind in (("x = 1\nreturn y", "NameError"), ("return (", "SyntaxError"),
                           ("return 1 / 0", "ZeroDivisionError")):
            result = self.run_code(code)
            self.assertEqual(result["status"], "error")
            self.assertIn(kind, result["reason"])
            self.assertLess(len(result["reason"]), 300)
            self.assertNotIn("Traceback", result["reason"])
        self.assertIn("line 2", self.run_code("x = 1\nreturn y")["reason"])

    def test_tool_call_quota(self):
        result = self.run_code("for _ in range(400):\n    tools.list_files('logs/*.log')", max_calls=300)
        self.assertEqual(result["status"], "error")
        self.assertIn("QuotaExceeded", result["reason"])

    def test_runtime_trace_has_no_file_content(self):
        session = self.code_session([AGGREGATE])
        turn = session.respond("Đếm lỗi")
        events = [e for e in turn["trace"] if e["event"] == "runtime_tool"]
        self.assertEqual(len(events), 32)
        self.assertTrue(all("content" not in e for e in events))
        self.assertNotIn(self.truth["canary"], json.dumps(turn["trace"], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
