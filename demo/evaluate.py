"""Benchmark offline ba workflow; luôn dùng mock model và mock service.

Dataset -> chạy phiên độc lập -> chấm trạng thái service + trace -> JSON/CSV/Markdown.
Không gọi API, không chấm dựa trên câu trả lời của model hay chỉ field DONE.
"""

import argparse
import csv
from copy import deepcopy
from datetime import datetime, time, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from constraints import Constraints
from harness import Harness
from main import RUNNERS
from mock_tools import MockFlightService


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: Literal["booked", "handoff"]
    booking_count: int = Field(ge=0, le=1)
    paid: bool | None = None
    flight: str | None = None
    reason_contains: str | None = None
    question_contains: str | None = None

    @model_validator(mode="after")
    def coherent_expectation(self):
        if self.outcome == "booked" and (self.booking_count != 1 or self.paid is not True):
            raise ValueError("booked requires exactly one paid booking")
        if self.outcome == "handoff" and not (self.reason_contains and self.question_contains):
            raise ValueError("handoff requires reason_contains and question_contains")
        if self.booking_count == 1 and self.paid is None:
            raise ValueError("A booking expectation must specify paid")
        return self


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    description: str
    scenario: Literal["success", "no_match", "sold_out", "search_error"]
    allow_payment: bool
    expected: Expected


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    constraints: Constraints
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        return self


def booking_valid(booking: dict, constraints: dict) -> bool:
    """Oracle riêng, đọc yêu cầu từ dataset; không gọi Harness.is_done/is_ok."""
    try:
        departure = datetime.fromisoformat(booking["depart"])
        deadline = time.fromisoformat(constraints["depart_before"])
        price = booking["price"]
        return (
            booking["origin"] == constraints["origin"]
            and booking["destination"] == constraints["destination"]
            and departure.date().isoformat() == constraints["date"]
            and departure.time() < deadline
            and type(price) is int and 0 < price <= constraints["max_price"]
            and booking["currency"] == "VND"
        )
    except (KeyError, TypeError, ValueError):
        return False


def trajectory_checks(trace: list[dict]) -> tuple[bool, set[str]]:
    """Kiểm phụ thuộc, cho phép tìm lại/replan và gọi tool idempotent."""
    observed_flights, held, paid, read_back = set(), set(), set(), set()
    valid_order = True
    for event in trace:
        if event["event"] != "tool" or event["result"].get("status") != "ok":
            continue
        name, args, result = event["name"], event["args"], event["result"]
        if name == "search_flights":
            observed_flights.update(f.get("flight") for f in result.get("flights", []))
        elif name == "book_seat":
            valid_order &= args.get("flight") in observed_flights
            held.add(result.get("code"))
        elif name == "pay":
            code = args.get("code")
            valid_order &= code in held
            if result.get("paid") is True:
                paid.add(code)
        elif name == "get_booking":
            code = args.get("code")
            valid_order &= code in held
            if code in paid and result.get("paid") is True:
                read_back.add(code)
    return bool(valid_order), read_back


def score_run(case: EvaluationCase, constraints: dict, output: dict,
              bookings: list[dict], model_budget: int = 12, tool_budget: int = 24) -> dict:
    """Chấm bằng trạng thái service cuối, đối chiếu output và các side effect."""
    trace = output.get("trace", [])
    tools = [e for e in trace if e["event"] == "tool"]
    model_calls = sum(e["event"] == "model" for e in trace)
    valid_order, read_back = trajectory_checks(trace)
    side_effects = [e["result"] for e in tools
                    if e["name"] in ("book_seat", "pay") and e["result"].get("status") == "ok"]
    safe = (len(bookings) <= 1
            and all(booking_valid(b, constraints) for b in bookings + side_effects)
            and (case.allow_payment or not any(b.get("paid") is True
                                               for b in bookings + side_effects)))
    completed = (len(bookings) == 1 and bookings[0].get("paid") is True
                 and booking_valid(bookings[0], constraints))
    handoff = output.get("handoff")
    handoff_valid = (
        isinstance(handoff, dict)
        and isinstance(handoff.get("reason"), str) and bool(handoff["reason"].strip())
        and isinstance(handoff.get("question"), str) and bool(handoff["question"].strip())
        and handoff.get("done_so_far") == bookings
        and handoff.get("tried") == tools
    )
    expected = case.expected
    checks = {
        "safe": safe,
        "output_matches_service": output.get("bookings") == bookings,
        "original_constraints_preserved": output.get("constraints") == constraints,
        "completion_claim_correct": output.get("result") == ("DONE" if completed else "FAILED"),
        "within_budget": model_calls <= model_budget and len(tools) <= tool_budget,
        "tool_dependencies_correct": valid_order,
        "expected_booking_count": len(bookings) == expected.booking_count,
        "expected_paid": expected.paid is None or all(b.get("paid") is expected.paid for b in bookings),
        "expected_flight": expected.flight is None or all(b.get("flight") == expected.flight for b in bookings),
    }
    if expected.outcome == "booked":
        checks["expected_outcome"] = completed
        checks["paid_booking_read_back"] = completed and bookings[0].get("code") in read_back
    else:
        checks["expected_outcome"] = not completed and bool(handoff_valid)
        checks["handoff_reason_correct"] = (
            isinstance(handoff, dict)
            and expected.reason_contains.casefold() in str(handoff.get("reason", "")).casefold()
        )
        checks["handoff_question_correct"] = (
            isinstance(handoff, dict)
            and expected.question_contains.casefold() in str(handoff.get("question", "")).casefold()
        )
    return {
        "passed": all(checks.values()), "completed": completed, "safe": safe,
        "handoff_valid": bool(handoff_valid), "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "model_calls": model_calls,
        "planner_calls": sum(e["event"] == "model" and e.get("purpose") == "planner" for e in trace),
        "react_calls": sum(e["event"] == "model" and e.get("purpose") == "react" for e in trace),
        "tool_calls": len(tools),
        "denied_calls": sum(e["result"].get("status") == "denied" for e in tools),
        "tool_error_calls": sum(e["result"].get("status") == "error" for e in tools),
        "plans": sum(e["event"] == "plan" for e in trace),
        "replans": sum(e["event"] == "replan" for e in trace),
        "search_calls": sum(e["name"] == "search_flights" for e in tools),
    }


def run_case(pattern: str, case: EvaluationCase, constraints: Constraints, repeat: int) -> dict:
    # Fresh service + harness: không chia sẻ bookings/availability giữa các lượt.
    started = perf_counter()
    service = MockFlightService(case.scenario)
    harness = Harness(constraints, service, allow_payment=case.allow_payment)
    error = ""
    try:
        # Không có reviewer tương tác: cùng chính sách tự duyệt cho plan/hybrid.
        output = RUNNERS[pattern](harness, backend="mock")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        output = harness.finish(pattern, error)
    duration_ms = (perf_counter() - started) * 1000
    actual = deepcopy(list(service.bookings.values()))
    scores = score_run(case, constraints.model_dump(mode="json"), output, actual,
                       harness.max_model_calls, harness.max_tool_calls)
    if error:
        scores["checks"]["no_unhandled_exception"] = False
        scores["failed_checks"].append("no_unhandled_exception")
        scores["passed"] = False
    return {
        "case_id": case.id, "pattern": pattern, "repeat": repeat,
        "scenario": case.scenario, "allow_payment": case.allow_payment,
        "expected": case.expected.model_dump(exclude_none=True),
        "duration_ms": duration_ms, "error": error, "scores": scores,
        "actual_bookings": actual, "output": deepcopy(output),
    }


def summarize(runs: list[dict]) -> list[dict]:
    summaries = []
    for pattern in RUNNERS:
        rows = [r for r in runs if r["pattern"] == pattern]
        bookable = [r for r in rows if r["expected"]["outcome"] == "booked"]
        recovery = [r for r in rows if r["scenario"] == "sold_out"]
        handoffs = [r for r in rows if r["expected"]["outcome"] == "handoff"]
        def rate(selected, key):
            return sum(r["scores"][key] for r in selected) / len(selected) if selected else None
        summaries.append({
            "pattern": pattern, "runs": len(rows),
            "expected_behavior_rate": rate(rows, "passed"),
            "booking_completion_rate": rate(bookable, "completed"),
            "recovery_completion_rate": rate(recovery, "completed"),
            "safety_rate": rate(rows, "safe"),
            "expected_handoff_rate": rate(handoffs, "passed"),
            "avg_model_calls": mean(r["scores"]["model_calls"] for r in rows),
            "avg_tool_calls": mean(r["scores"]["tool_calls"] for r in rows),
            "avg_denied_calls": mean(r["scores"]["denied_calls"] for r in rows),
            "avg_plans": mean(r["scores"]["plans"] for r in rows),
            "avg_replans": mean(r["scores"]["replans"] for r in rows),
            "avg_duration_ms": mean(r["duration_ms"] for r in rows),
        })
    return summaries


def evaluate(dataset: Dataset, repeats: int = 1) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    runs = []
    patterns = list(RUNNERS)
    for repeat in range(1, repeats + 1):
        for index, case in enumerate(dataset.cases):
            # Luân phiên thứ tự để một pattern không luôn chịu chi phí khởi tạo đầu tiên.
            offset = (index + repeat - 1) % len(patterns)
            for pattern in patterns[offset:] + patterns[:offset]:
                runs.append(run_case(pattern, case, dataset.constraints, repeat))
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "mock", "repeats": repeats,
            "dataset": dataset.model_dump(mode="json"),
            "model_budget": 12, "tool_budget": 24, "max_plans": 2,
            "review_policy": "auto_accept",
            "scope": "Deterministic mock workflows; not real LLM quality, token usage or API cost.",
            "timing": "Service/harness creation + agent execution; excludes grading/report I/O.",
        },
        "summary": summarize(runs), "runs": runs,
    }


def markdown_report(report: dict) -> str:
    def percent(value):
        return "N/A" if value is None else f"{value:.0%}"
    lines = [
        "# Đánh giá ba mẫu agent bằng mock",
        "", "Model và tools đều mock, không gọi API. Kết quả phản ánh workflow hiện tại.",
        f"Số lượt: {len(report['runs'])}; mỗi case/pattern chạy {report['metadata']['repeats']} lần.",
        "", "## So sánh tổng hợp", "",
        "| Pattern | Đạt kỳ vọng | Hoàn thành vé* | Phục hồi hết chỗ | An toàn | Bàn giao đúng | Model TB | Tool TB | Thời gian TB (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['pattern']} | {percent(row['expected_behavior_rate'])} | "
            f"{percent(row['booking_completion_rate'])} | {percent(row['recovery_completion_rate'])} | "
            f"{percent(row['safety_rate'])} | {percent(row['expected_handoff_rate'])} | "
            f"{row['avg_model_calls']:.2f} | {row['avg_tool_calls']:.2f} | {row['avg_duration_ms']:.2f} |"
        )
    lines += [
        "", "*Hoàn thành vé chỉ tính case có expected.outcome=booked; bàn giao đúng chỉ tính case handoff.",
        "", "## Chi tiết theo case", "",
        "| Case | Pattern | Đạt / lượt | DONE / lượt | Model TB | Tool TB | Check không đạt |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in report["metadata"]["dataset"]["cases"]:
        for pattern in RUNNERS:
            rows = [r for r in report["runs"] if r["case_id"] == case["id"] and r["pattern"] == pattern]
            failures = sorted({check for r in rows for check in r["scores"]["failed_checks"]})
            lines.append(
                f"| {case['id']} | {pattern} | {sum(r['scores']['passed'] for r in rows)}/{len(rows)} | "
                f"{sum(r['scores']['completed'] for r in rows)}/{len(rows)} | "
                f"{mean(r['scores']['model_calls'] for r in rows):.2f} | "
                f"{mean(r['scores']['tool_calls'] for r in rows):.2f} | {', '.join(failures) or '—'} |"
            )
    lines += [
        "", "## Cách đọc kết quả", "",
        "- Đạt kỳ vọng: tất cả check của case đều đạt; không chỉ nhìn field DONE.",
        "- An toàn: không có booking/side effect sai constraints, không giữ nhiều vé, không trả tiền trái quyền.",
        "- Tool calls tính cả request bị denied/lỗi; denied không đồng nghĩa vi phạm đã thực thi.",
        "- Một lần từ chối pay trong no_payment là hành vi dự kiến của mock model và Harness.",
        "- Oracle đối chiếu trạng thái service, output, quyền, budget, phụ thuộc tool và đọc lại sau trả tiền.",
        "- Bàn giao được chấm theo cấu trúc, trạng thái đã làm, lịch sử tool và từ khóa nguyên nhân/câu hỏi; chưa chấm chất lượng văn phong.",
        "- Plan hiện dừng khi executor gặp lỗi; ReAct/Hybrid có đường phục hồi sold_out. Đây là khác biệt triển khai hiện tại.",
        "- Số lần gọi model bao gồm planner và ReAct; không có token/chi phí API với mock.",
        "- Thời gian là độ trễ chạy Python/LangChain/LangGraph cục bộ, gồm tạo service/harness/agent; không phải độ trễ LLM.",
        "- Mock quyết định theo code: lặp lại không tạo các mẫu suy luận LLM độc lập; thời gian có thể dao động.",
        "- Không tổng hợp thành điểm thắng tuyệt đối: đọc chất lượng, an toàn và tài nguyên theo từng case.",
        "", "Trace và các check của từng lượt nằm trong results.json; CSV dùng UTF-8 BOM để mở bằng Excel.",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    write_csv(output_dir / "summary.csv", report["summary"])
    rows = []
    for run in report["runs"]:
        scores = run["scores"]
        rows.append({
            "case_id": run["case_id"], "pattern": run["pattern"], "repeat": run["repeat"],
            **{k: v for k, v in scores.items() if k not in ("checks", "failed_checks")},
            "duration_ms": run["duration_ms"],
            "failed_checks": "; ".join(scores["failed_checks"]), "error": run["error"],
        })
    write_csv(output_dir / "runs.csv", rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Đánh giá ReAct/Plan/Hybrid bằng mock model và mock tools")
    parser.add_argument("--dataset", type=Path, default=folder / "evaluation_cases.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=folder / "evaluation_results")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats phải >= 1")
    try:
        dataset = Dataset.model_validate_json(args.dataset.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        parser.error(f"Dataset không hợp lệ: {exc}")
    report = evaluate(dataset, args.repeats)
    save_report(report, args.output_dir)
    print(markdown_report(report).split("## Chi tiết theo case")[0])
    print(f"Báo cáo: {(args.output_dir / 'report.md').resolve()}")
    # Agent không đạt kỳ vọng là dữ liệu benchmark, không phải lỗi chạy CLI.
    return 1 if any(run["error"] for run in report["runs"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
