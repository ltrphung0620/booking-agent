"""Opt-in real-API verification of B1 (compaction) and B2 (code mode); logs live in a temp folder."""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import bootstrap
from dotenv import load_dotenv

from compaction import CompactionPolicy
from investigation import InvestigationSession
from log_fixtures import generate_logs

B2_QUESTION = ("Trong tháng 10/2026, mã lỗi nào xuất hiện nhiều nhất và bao nhiêu lần? "
               "ERR_PAY_TIMEOUT tăng cao vào những ngày nào?")
B1_TURNS = [
    "Từ đầu tháng 10 khách phản ánh thanh toán vé thất bại nhiều. Hãy điều tra nguyên nhân bằng log, "
    "bắt đầu bằng việc liệt kê file log.",
    "Đọc log ngày 2026-10-07 và cho biết lỗi nổi bật.",
    "Mình nghi do search timeout làm hỏng booking. Đọc log 2026-10-03 và 2026-10-04 để kiểm tra giả thuyết này.",
    "Ghi nhận: loại giả thuyết search timeout vì nó đều đặn mọi ngày và không đi kèm booking lỗi. "
    "Đọc tiếp log 2026-10-08.",
    "Đọc log 2026-10-09 và so sánh các node thanh toán.",
    "Tóm tắt ngắn: đang điều tra lỗi gì, đã thử và loại giả thuyết nào, bằng chứng chính là gì, "
    "bước tiếp theo là gì?",
    "Tiếp tục bước tiếp theo.",
]
SEARCH_RULED_OUT = re.compile(r"search[ _]?timeout.{0,80}(loại|bác bỏ|không phải|không liên quan)|"
                              r"(loại|bác bỏ).{0,80}search[ _]?timeout", re.IGNORECASE | re.DOTALL)
# Measured offline with a fake model; real runs add answers and retries on top.
NATIVE_ESTIMATE = {10: "~260k", 31: "~2,3M"}

ModelFactory = Callable[[str], object] | None


def mentions_day(text: str, day: str) -> bool:
    d = int(day[-2:])
    return any(re.search(p, text) for p in (day, rf"\b0?{d}/10\b", rf"\b0?{d}-10\b", rf"ngày 0?{d}\b"))


def slim(trace: list[dict]) -> list[dict]:
    """Keep reports readable: file contents become their size."""
    out = []
    for event in trace:
        result = event.get("result")
        if isinstance(result, dict) and "content" in result:
            event = {**event, "result": {**result, "content": f"<{len(result['content'])} ký tự>"}}
        out.append(event)
    return out


def tool_calls(turn: dict) -> list[dict]:
    return [{"name": e["name"], "args": e["args"], "status": e["result"].get("status")}
            for e in turn["trace"] if e["event"] == "tool"]


def run_b2(model_name: str, files: int, model_factory: ModelFactory = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="challenge-b-b2-") as directory:
        root = Path(directory)
        truth = generate_logs(root, days=files)
        runs = {}
        for mode in ("native", "code"):
            session = InvestigationSession(
                model_name, root, mode=mode, capture_inputs=True,
                model=model_factory(f"b2_{mode}") if model_factory else None,
                max_model_calls=files + 15, max_tool_calls=2 * files + 20)
            turn = session.respond(B2_QUESTION)
            usage = session.usage_summary()
            runs[mode] = {
                "answer": turn["assistant"], "error": turn["error"], "usage": usage,
                "tool_calls": sum(e["event"] == "tool" for e in turn["trace"]),
                "runtime_tool_calls": sum(e["event"] == "runtime_tool" for e in turn["trace"]),
                "tools_bound": sorted({t for e in session.context_log for t in e["tools"]}),
                "canary_in_input": any(truth["canary"] in e["text"] for e in session.context_log),
                "trace": slim(turn["trace"]),
                "per_call_input": [e["input_tokens"] if e["input_tokens"] is not None else e["approx_tokens"]
                                   for e in session.context_log]}
            print(json.dumps({"part": "b2", "mode": mode, "error": turn["error"], **usage},
                             ensure_ascii=True), flush=True)
    (top_code, top_count), peaks = truth["top_errors"][0], [d for d, _ in truth["pay_timeout_peak_days"]]

    def correct(answer: str) -> bool:
        return (top_code in answer and re.search(rf"\b{top_count}\b", answer) is not None
                and all(mentions_day(answer, d) for d in peaks))

    native, code = runs["native"], runs["code"]
    checks = {
        "native_succeeded": not native["error"],
        "code_succeeded": not code["error"],
        "native_answer_correct": correct(native["answer"]),
        "code_answer_correct": correct(code["answer"]),
        "code_binds_only_run_code": code["tools_bound"] == ["run_code"],
        "code_input_has_no_canary": not code["canary_in_input"],
        "code_total_input_smaller": code["usage"]["total_input"] < native["usage"]["total_input"],
    }
    return {"files": files, "question": B2_QUESTION,
            "truth": {k: truth[k] for k in ("top_errors", "pay_timeout_peak_days", "spike_node")},
            "native_saw_canary": native["canary_in_input"], "runs": runs, "checks": checks}


def run_b1(model_name: str, budget: int, model_factory: ModelFactory = None, summarizer=None) -> dict:
    with tempfile.TemporaryDirectory(prefix="challenge-b-b1-") as directory:
        root = Path(directory)
        generate_logs(root)
        policy = CompactionPolicy(budget_tokens=budget, keep_last_turns=1, keep_last_tool_outputs=1,
                                  summarizer=summarizer)
        session = InvestigationSession(model_name, root, compaction=policy,
                                       model=model_factory("b1") if model_factory else None)
        steps = []
        for i, prompt in enumerate(B1_TURNS, 1):
            turn = session.respond(prompt)
            events = [{k: v for k, v in e.items() if k != "checkpoint"} for e in turn["trace"]
                      if e["event"] in ("compaction", "compaction_failed", "compaction_skipped", "prune")]
            steps.append({"turn": i, "user": prompt, "assistant": turn["assistant"], "error": turn["error"],
                          "tool_calls": tool_calls(turn), "context_events": events})
            print(json.dumps({"part": "b1", "turn": i, "error": turn["error"],
                              "tools": [c["name"] for c in steps[-1]["tool_calls"]],
                              "events": [e["event"] for e in events]}, ensure_ascii=True), flush=True)
        auto = list(session.compactions)
        # Slide 22: fold everything into one checkpoint, then resume in a session that has nothing else.
        final = session.maybe_compact(force=True, keep_last_turns=0)
        checkpoint = session.checkpoint
        resumed = None
        if checkpoint is not None:
            fresh = InvestigationSession.from_checkpoint(
                checkpoint, model_name, root, model=model_factory("b1_resume") if model_factory else None)
            turn = fresh.respond(B1_TURNS[5])
            resumed = {"assistant": turn["assistant"], "error": turn["error"],
                       "tool_calls": tool_calls(turn)}
        usage = session.usage_summary()
    summary_answer = steps[5]["assistant"]
    turn7_reads = [c["args"].get("path", "") for c in steps[6]["tool_calls"] if c["name"] == "read_file"]
    cp = checkpoint.model_dump() if checkpoint else {}
    checks = {
        "all_turns_succeeded": not any(s["error"] for s in steps),
        "auto_compaction_happened": len(auto) >= 1,
        "compaction_reduced_tokens": bool(auto) and all(c["tokens_after"] < c["tokens_before"] for c in auto),
        "checkpoint_has_goal_and_next_action": bool(cp.get("goal", "").strip() and cp.get("next_action", "").strip()),
        "checkpoint_records_search_timeout_failed": any("search" in a.lower() for a in cp.get("failed_attempts", [])),
        "turn6_names_pay_timeout": "ERR_PAY_TIMEOUT" in summary_answer,
        "turn6_says_search_timeout_ruled_out": bool(SEARCH_RULED_OUT.search(summary_answer)),
        "resumed_session_answers_turn6": bool(resumed) and not resumed["error"]
            and "ERR_PAY_TIMEOUT" in resumed["assistant"]
            and bool(SEARCH_RULED_OUT.search(resumed["assistant"])),
    }
    return {"budget_tokens": budget, "policy": {"keep_last_turns": 1, "keep_last_tool_outputs": 1},
            "steps": steps, "auto_compactions": auto, "final_compaction": final, "checkpoint": cp,
            "resumed": resumed, "usage": usage, "checks": checks,
            # For the reader to judge: did "continue" repeat the ruled-out search-timeout days?
            "turn7_rereads_ruled_out_days": [p for p in turn7_reads if p.endswith(("-03.log", "-04.log"))]}


def render_report(report: dict) -> str:
    lines = ["# Kiểm chứng Challenge B bằng model thật", "",
             f"- Kết quả: **{'PASS' if report['passed'] else 'FAIL'}**",
             f"- Model: `{report['model']}`", f"- Thời điểm UTC: {report['verified_at']}",
             "- Log giả sinh từ seed cố định trong thư mục tạm.", ""]
    b2 = report.get("b2")
    if b2:
        lines += ["## B2: native và code mode", "", f"Câu hỏi ({b2['files']} file): {b2['question']}", "",
                  f"Đáp án đúng: `{b2['truth']['top_errors'][0]}`, ngày đỉnh "
                  f"{[d for d, _ in b2['truth']['pay_timeout_peak_days']]}", "",
                  "| | native | code |", "|---|---|---|"]
        rows = [("Số lần gọi LLM", "llm_calls"), ("Tổng input tokens", "total_input"),
                ("Input lớn nhất", "peak_input"), ("Output tokens", "total_output"), ("Nguồn số liệu", "source")]
        n, c = b2["runs"]["native"], b2["runs"]["code"]
        lines += [f"| {label} | {n['usage'][key]} | {c['usage'][key]} |" for label, key in rows]
        lines += [f"| Tool call (model) | {n['tool_calls']} | {c['tool_calls']} |",
                  f"| Lời gọi trong runtime | {n['runtime_tool_calls']} | {c['runtime_tool_calls']} |",
                  f"| Canary trong input | {n['canary_in_input']} | {c['canary_in_input']} |", ""]
        lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in b2["checks"].items()]
        for mode in ("native", "code"):
            lines += ["", f"### Trả lời ({mode})", "", b2["runs"][mode]["answer"]]
        lines.append("")
    b1 = report.get("b1")
    if b1:
        lines += ["## B1: compaction", "",
                  f"Budget {b1['budget_tokens']} token, giữ {b1['policy']['keep_last_turns']} lượt và "
                  f"{b1['policy']['keep_last_tool_outputs']} tool output gần nhất.", ""]
        lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in b1["checks"].items()]
        for step in b1["steps"]:
            tools = " → ".join(f"`{c['name']}` {json.dumps(c['args'], ensure_ascii=False)} ({c['status']})"
                               for c in step["tool_calls"]) or "không có"
            events = ", ".join(f"{e['event']} {e.get('tokens_before', '')}→{e.get('tokens_after', '')}"
                               for e in step["context_events"]) or "không có"
            lines += ["", f"### Lượt {step['turn']}", "", "> " + step["user"], "", step["assistant"], "",
                      f"Tool: {tools}", "", f"Sự kiện context: {events}"]
        lines += ["", "### Checkpoint cuối", "", "```json",
                  json.dumps(b1["checkpoint"], ensure_ascii=False, indent=2), "```", "",
                  "### Session mới chỉ có checkpoint, hỏi lại câu lượt 6", "",
                  (b1["resumed"] or {}).get("assistant", "(không chạy được)"), "",
                  "### Lượt 7 có lặp lại cách đã loại không?", "",
                  f"Đọc lại log ngày 03/04: {b1['turn7_rereads_ruled_out_days'] or 'không'}", ""]
    return "\n".join(lines)


def verify(parts: list[str], files: int, budget: int, report_dir: Path,
           model_name: str | None = None, model_factory: ModelFactory = None, summarizer=None) -> bool:
    if model_name is None:
        load_dotenv(bootstrap.HERE / ".env", override=False)
        load_dotenv(bootstrap.DEMO / ".env", override=False)
        model_name = os.environ.get("FLIGHT_MODEL", "").strip()
        if not os.environ.get("OPENAI_API_KEY") or not model_name:
            raise RuntimeError("Cần cấu hình API key và model trước khi chạy kiểm chứng thật.")
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "model": model_name}
    if "b2" in parts:
        report["b2"] = run_b2(model_name, files, model_factory)
    if "b1" in parts:
        report["b1"] = run_b1(model_name, budget, model_factory, summarizer)
    report["passed"] = all(ok for part in parts for ok in report[part]["checks"].values())
    report_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        single = {**{k: v for k, v in report.items() if k not in ("b1", "b2")}, part: report[part],
                  "passed": all(report[part]["checks"].values())}
        path = report_dir / f"{part}_verification.json"
        path.write_text(json.dumps(single, ensure_ascii=False, indent=2), encoding="utf-8")
        path.with_suffix(".md").write_text(render_report(single), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "reports": str(report_dir),
                      **{part: report[part]["checks"] for part in parts}}, ensure_ascii=True), flush=True)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description="Kiểm chứng B1 và B2 của Challenge B bằng API thật.")
    parser.add_argument("--live", action="store_true", help="Đồng ý gọi model thật bằng router/key đã cấu hình.")
    parser.add_argument("--part", choices=("b1", "b2", "all"), default="all")
    parser.add_argument("--files", type=int, default=10, help="Số file log cho phần B2 (9-31).")
    parser.add_argument("--b1-budget", type=int, default=8_000)
    parser.add_argument("--report-dir", type=Path, default=bootstrap.HERE / "reports")
    args = parser.parse_args()
    if not args.live:
        parser.error("Thêm --live để gọi model thật; dùng unittest cho kiểm tra offline.")
    if not 9 <= args.files <= 31:
        parser.error("--files phải trong khoảng 9-31 để có đủ ngày sự cố 07-09.")
    parts = ["b2", "b1"] if args.part == "all" else [args.part]
    if "b2" in parts:
        estimate = NATIVE_ESTIMATE.get(args.files, "tăng gần theo bình phương số file")
        print(f"Cảnh báo: lệnh này tốn token thật. B2 native với {args.files} file ước tính {estimate} "
              "input tokens (đo bằng model giả); chế độ code chỉ vài nghìn.", flush=True)
    raise SystemExit(0 if verify(parts, args.files, args.b1_budget, args.report_dir.resolve()) else 1)


if __name__ == "__main__":
    main()
