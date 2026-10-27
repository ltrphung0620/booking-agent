"""Offline B1 proofs: compaction triggers, shrinks context and keeps what the task needs."""

import json
import tempfile
import unittest
from pathlib import Path

import bootstrap  # noqa: F401
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from compaction import Checkpoint, CompactionPolicy, ModelSummarizer
from investigation import InvestigationSession
from log_fixtures import generate_logs
from test_native import ScriptedModel, call


class RecordingModel(ScriptedModel):
    seen: list = Field(default_factory=list)

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def checkpoint(failed="Giả thuyết search timeout: loại vì đều mọi ngày, không kèm booking lỗi",
               next_action="So sánh các node thanh toán ngày 2026-10-09", **extra) -> Checkpoint:
    data = {"goal": "Tìm nguyên nhân thanh toán vé thất bại", "current_state": "Đã đọc log 07",
            "constraints": ["Chỉ kết luận từ log"], "findings": ["ERR_PAY_TIMEOUT tăng vọt ngày 07"],
            "failed_attempts": [failed], "evidence_files": ["logs/booking-2026-10-07.log"],
            "open_questions": ["Node nào lỗi?"], "next_action": next_action}
    return Checkpoint(**{**data, **extra})


class FakeSummarizer:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, previous, messages):
        self.calls.append({"previous": previous, "messages": messages})
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


class Script:
    """Builds one model response list for a multi-turn session with unique tool-call ids."""

    def __init__(self):
        self.responses, self.n = [], 0

    def turn(self, *paths, answer="Đã ghi nhận."):
        for path in paths:
            self.n += 1
            self.responses.append(call("read_file", {"path": path}, f"t{self.n}"))
        self.responses.append(AIMessage(content=answer))
        return self


class CompactionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.truth = generate_logs(self.root)
        self.files = self.truth["files"]

    def tearDown(self):
        self.tmp.cleanup()

    def day(self, n):
        return f"logs/booking-2026-10-{n:02d}.log"

    def session(self, script, summarizer=None, **policy):
        model = RecordingModel(responses=script.responses)
        compaction = CompactionPolicy(summarizer=summarizer or FakeSummarizer(checkpoint()), **policy)
        return InvestigationSession("fake", self.root, model=model, compaction=compaction,
                                    capture_inputs=True)

    def three_big_turns(self, extra_turns=2, **policy):
        """Turns 1-3 each read one ~5k-token file (turn 1 holds the canary), then small turns."""
        script = Script().turn(self.truth["canary_file"]).turn(self.day(3)).turn(self.day(4))
        for _ in range(extra_turns):
            script.turn()
        policy = {"budget_tokens": 20_000, "keep_last_turns": 1, **policy}
        session = self.session(script, **policy)
        for i in range(3 + extra_turns):
            session.respond(f"Lượt {i + 1}")
        return session

    def turn_inputs(self, session, turn):
        return [e for e in session.context_log if e["turn"] == turn]

    # 1
    def test_compacts_once_above_threshold_and_halves_context(self):
        session = self.three_big_turns()
        self.assertEqual(len(session.compactions), 1)
        entry = session.compactions[0]
        self.assertGreater(entry["tokens_before"], session.compaction.threshold)
        self.assertLess(entry["tokens_after"], 0.5 * entry["tokens_before"])
        self.assertEqual(session.turns[3]["trace"][0]["event"], "compaction")  # Before turn 4 ran.
        self.assertFalse(any(e["event"] == "compaction" for t in session.turns[:3] for e in t["trace"]))

    def test_no_compaction_below_threshold(self):
        session = self.three_big_turns(budget_tokens=200_000)
        self.assertEqual(session.compactions, [])
        self.assertEqual(session.compaction.summarizer.calls, [])

    # 2
    def test_history_stays_api_valid_after_compaction(self):
        session = self.three_big_turns()
        self.assertIsInstance(session.messages[0], HumanMessage)
        call_ids = {c["id"] for m in session.messages if isinstance(m, AIMessage) for c in m.tool_calls}
        result_ids = {m.tool_call_id for m in session.messages if isinstance(m, ToolMessage)}
        self.assertEqual(call_ids, result_ids)
        self.assertGreater(len(call_ids), 0)
        for sent in session._model.seen:  # Every request actually sent, including pruned ones.
            ids = {c["id"] for m in sent if isinstance(m, AIMessage) for c in m.tool_calls}
            self.assertEqual(ids, {m.tool_call_id for m in sent if isinstance(m, ToolMessage)})

    # 3
    def test_next_call_sees_checkpoint_not_compacted_raw_log(self):
        session = self.three_big_turns()
        text = self.turn_inputs(session, 4)[0]["text"]
        self.assertIn("<checkpoint>", text)
        self.assertIn("search timeout", text)
        self.assertIn("So sánh các node thanh toán", text)
        self.assertNotIn(self.truth["canary"], text)
        # Control: before compaction the canary file was really in context.
        self.assertIn(self.truth["canary"], self.turn_inputs(session, 1)[-1]["text"])

    def test_checkpoint_is_appended_after_static_system_prompt(self):
        session = self.three_big_turns()
        system = session._model.seen[-1][0].content
        self.assertTrue(system.startswith(session.system_prompt()))
        self.assertLess(system.index(session.system_prompt()[-40:]), system.index("<checkpoint>"))

    # 4
    def test_checkpoint_survives_later_turns(self):
        session = self.three_big_turns(extra_turns=4)
        self.assertEqual(len(session.compactions), 1)
        for turn in (5, 6, 7):
            for entry in self.turn_inputs(session, turn):
                self.assertIn("<checkpoint>", entry["text"])
                self.assertIn("search timeout", entry["text"])
        self.assertFalse(any(m.type == "system" for m in session.messages))

    # 5
    def test_second_compaction_keeps_first_failed_attempts(self):
        first = checkpoint(failed="Loại search timeout")
        second = checkpoint(failed="Loại rate limit ở gateway")  # The summarizer "forgot" the first.
        summarizer = FakeSummarizer(first, second)
        script = Script().turn(self.day(1)).turn(self.day(2)).turn(self.day(3)).turn()
        session = self.session(script, summarizer, keep_last_turns=1)
        for i in range(2):
            session.respond(f"Lượt {i + 1}")
        self.assertEqual(session.maybe_compact(force=True)["status"], "ok")
        session.respond("Lượt 3")
        self.assertEqual(session.maybe_compact(force=True)["status"], "ok")
        self.assertEqual(summarizer.calls[1]["previous"].failed_attempts, ["Loại search timeout"])
        self.assertEqual(session.checkpoint.failed_attempts, ["Loại search timeout", "Loại rate limit ở gateway"])
        session.respond("Tiếp tục")
        self.assertIn("Loại search timeout", session.context_log[-1]["text"])

    # 6
    def test_failed_summary_never_loses_messages(self):
        for bad in (RuntimeError("model down"), checkpoint(next_action=" "), checkpoint(goal="")):
            with self.subTest(bad=repr(bad)[:40]):
                script = Script().turn(self.day(1)).turn(self.day(2)).turn()
                session = self.session(script, FakeSummarizer(bad), keep_last_turns=1)
                session.respond("Lượt 1")
                session.respond("Lượt 2")
                before = list(session.messages)
                result = session.maybe_compact(force=True)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(session.messages, before)
                self.assertIsNone(session.checkpoint)
                self.assertEqual(session.harness.trace[-1]["event"], "compaction_failed")
                self.assertFalse(session.respond("Lượt 3")["error"])

    # 7
    def test_mid_turn_prune_stubs_old_outputs_and_keeps_pairs(self):
        script = Script().turn(*self.files[:10])
        session = self.session(script, budget_tokens=8_000, keep_last_tool_outputs=2)
        turn = session.respond("Đọc 10 file")
        self.assertFalse(turn["error"])
        self.assertTrue(any(e["event"] == "prune" for e in turn["trace"]))
        last = session._model.seen[-1]
        tools = [m for m in last if isinstance(m, ToolMessage)]
        self.assertEqual(len(tools), 10)
        stubs = [json.loads(m.content) for m in tools[:8]]
        self.assertTrue(all(s["status"] == "archived" for s in stubs))
        self.assertTrue(all(len(s["preview"]) <= 300 for s in stubs))
        self.assertTrue(all(json.loads(m.content)["status"] == "ok" for m in tools[8:]))
        ids = [c["id"] for m in last if isinstance(m, AIMessage) for c in m.tool_calls]
        self.assertEqual(ids, [m.tool_call_id for m in tools])
        for stub in stubs:
            raw = json.loads(session.archive[stub["archive_id"]])
            self.assertEqual(len(session.archive[stub["archive_id"]]), stub["chars"])
            self.assertEqual(raw["status"], "ok")
            self.assertIn("ERROR", raw["content"])
        # Pruning edits the request only; the stored history keeps raw outputs for layer 2.
        self.assertTrue(all(json.loads(m.content)["status"] == "ok"
                            for m in session.messages if isinstance(m, ToolMessage)))

    # 8
    def test_from_checkpoint_resumes_with_only_the_checkpoint(self):
        cp = checkpoint()
        model = RecordingModel(responses=[AIMessage(content="Bước tiếp: so sánh node.")])
        session = InvestigationSession.from_checkpoint(cp, "fake", self.root, model=model,
                                                       capture_inputs=True)
        self.assertEqual(session.messages, [])
        session.respond("Tiếp tục bước tiếp theo.")
        sent = model.seen[0]
        self.assertIn("<checkpoint>", sent[0].content)
        self.assertIn(cp.next_action, sent[0].content)
        self.assertEqual([m.type for m in sent], ["system", "human"])

    # 9
    def test_usage_summary_accumulates_across_compactions(self):
        session = self.three_big_turns(extra_turns=3)
        script_calls = 3 * 2 + 3  # Three read turns (call + answer) and three plain answers.
        summary = session.usage_summary()
        self.assertEqual(summary["llm_calls"], script_calls)
        self.assertEqual(summary["total_input"], sum(e["approx_tokens"] for e in session.context_log))
        self.assertLess(len(session.messages), 2 * 6)  # History shrank but the count did not.

    # Gain guard and manual compaction
    def test_auto_compaction_skips_when_old_part_is_tiny(self):
        script = Script().turn().turn(*self.files[:4])
        session = self.session(script, budget_tokens=8_000, keep_last_turns=1)
        session.respond("Chào")
        session.respond("Đọc 4 file")
        result = session.maybe_compact()
        self.assertEqual(result["status"], "skipped")
        self.assertIn("không đáng", result["reason"])
        self.assertEqual(session.compaction.summarizer.calls, [])

    def test_manual_compaction(self):
        script = Script().turn().turn().turn()
        session = self.session(script, keep_last_turns=2)
        session.respond("Lượt 1")
        result = session.maybe_compact(force=True)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("Chưa đủ lượt", result["reason"])
        self.assertEqual(session.harness.trace[-1]["event"], "compaction_skipped")
        self.assertEqual(session.maybe_compact()["status"], "skipped")  # Auto: far below threshold.
        result = session.maybe_compact(force=True, keep_last_turns=0)
        self.assertEqual((result["status"], result["manual"]), ("ok", True))
        self.assertEqual(session.messages, [])
        self.assertFalse(session.respond("Tiếp tục")["error"])

    def test_summarizer_gets_shortened_tool_outputs(self):
        session = self.three_big_turns()
        sent = session.compaction.summarizer.calls[0]["messages"]
        tools = [m for m in sent if isinstance(m, ToolMessage)]
        self.assertTrue(tools)
        self.assertTrue(all(len(m.content) < 1_600 and "…[cắt" in m.content for m in tools))

    def test_model_summarizer_uses_structured_output_and_keeps_usage(self):
        cp = checkpoint()

        class Structured:
            def invoke(self, prompt):
                self.prompt = prompt
                return {"raw": AIMessage(content="", usage_metadata={
                    "input_tokens": 900, "output_tokens": 80, "total_tokens": 980}),
                        "parsed": cp, "parsing_error": None}

        class Model:
            def with_structured_output(self, schema, **kwargs):
                self.args = (schema, kwargs)
                self.structured = Structured()
                return self.structured

        model = Model()
        summarizer = ModelSummarizer(model)
        self.assertIs(summarizer(None, [HumanMessage(content="Điều tra lỗi thanh toán")]), cp)
        self.assertEqual(model.args, (Checkpoint, {"method": "function_calling", "include_raw": True}))
        self.assertIn("failed_attempts", model.structured.prompt[0].content)
        self.assertEqual(summarizer.usage[0]["input_tokens"], 900)


if __name__ == "__main__":
    unittest.main()
