"""Opt-in real-API verification. All test memory is isolated in a temporary folder."""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import bootstrap
from dotenv import load_dotenv

from agent import MemoryChatSession
from memory_store import ProjectMemoryStore


SEARCH = "Tìm vé SGN đi DAD ngày 2026-10-07, khởi hành trước 12 giờ, ngân sách tối đa 2 triệu đồng. Chỉ tìm vé."


def render_report(report: dict) -> str:
    lines = ["# Kiểm chứng Challenge A bằng model thật", "",
             f"- Kết quả: **{'PASS' if report['passed'] else 'FAIL'}**",
             f"- Model: `{report['model']}`", f"- Thời điểm UTC: {report['verified_at']}",
             "- Bốn session độc lập; memory kiểm thử nằm trong thư mục tạm riêng.", "",
             "## Các kiểm tra", ""]
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in report["checks"].items())
    for step in report["steps"]:
        turn = step["turn"]
        lines.extend(["", f"## {step['label']}", "", f"Session: `{turn['session_id']}`", "",
                      "Yêu cầu:", "", "> " + turn["user"], "", "Trả lời thực tế:", "", turn["assistant"], "",
                      "Tool đã gọi: " + " → ".join(
                          f"`{e['name']}` ({e['result']['status']})" for e in turn["trace"] if e["event"] == "tool"),
                      "", "Memory thực sự đã inject (mỗi revision một lần):", ""])
        seen = set()
        for evidence in turn["memory_injections"]:
            revision = evidence["snapshot"]["revision"]
            if revision not in seen:
                lines.extend(["```text", evidence["injected_context"], "```", ""])
                seen.add(revision)
    lines.extend(["## Lịch sử quy ước", "", "```json",
                  json.dumps(report["history"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def verify(report_path: Path):
    load_dotenv(bootstrap.HERE / ".env", override=False)
    load_dotenv(bootstrap.DEMO / ".env", override=False)
    model_name = os.environ.get("FLIGHT_MODEL", "").strip()
    if not os.environ.get("OPENAI_API_KEY") or not model_name:
        raise RuntimeError("Cần cấu hình API key và model trước khi chạy kiểm chứng thật.")
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "model": model_name,
              "memory_isolated": True, "steps": [], "checks": {}, "passed": False}
    with tempfile.TemporaryDirectory(prefix="challenge-a-live-") as directory:
        path = Path(directory) / "project_memory.json"

        def turn(label, prompt):
            session = MemoryChatSession(model_name, ProjectMemoryStore(path))
            fresh = not session.messages and not session.harness.service.bookings
            result = session.respond(prompt)
            report["steps"].append({"label": label, "fresh_session": fresh,
                                     "turn": result, "usage": session.token_usage()})
            print(json.dumps({"step": label, "error": result["error"],
                              "tools": [e["name"] for e in result["trace"] if e["event"] == "tool"],
                              "tokens": session.token_usage()["total_tokens"]}, ensure_ascii=True), flush=True)
            return result

        store = ProjectMemoryStore(path)
        first = turn("1_save", "Ghi nhớ quy ước dự án: trình bày kết quả chuyến bay bằng bảng Markdown.")
        saved = store.snapshot()
        second = turn("2_new_session_uses_table", SEARCH)
        third = turn("3_update", "Cập nhật quy ước dự án: từ nay trình bày kết quả chuyến bay bằng danh sách gạch đầu dòng, thay cho bảng.")
        fourth = turn("4_new_session_uses_list", SEARCH)
        current = store.snapshot()
        history = store.history()
        report["history"] = history
        report["checks"] = {
            "all_turns_succeeded": not any(t["turn"]["error"] for t in report["steps"]),
            "independent_sessions": len({s["turn"]["session_id"] for s in report["steps"]}) == 4
                                    and all(s["fresh_session"] for s in report["steps"]),
            "first_rule_saved_by_agent_tool": saved["revision"] == 1 and any(
                e["event"] == "tool" and e["name"] == "remember_project_rule" and e["result"]["status"] == "ok"
                for e in first["trace"]),
            "proof_1_new_session_uses_table": bool(re.search(r"(?m)^\s*\|.+\|", second["assistant"]))
                and bool(second["memory_injections"])
                and second["memory_injections"][0]["snapshot"] == saved,
            "proof_2_new_session_uses_updated_list": bool(re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", fourth["assistant"]))
                and not bool(re.search(r"(?m)^\s*\|.+\|", fourth["assistant"]))
                and bool(fourth["memory_injections"])
                and fourth["memory_injections"][0]["snapshot"] == current,
            "old_rule_superseded": len(history["records"]) == 2
                and [r["status"] for r in history["records"]] == ["superseded", "active"]
                and len(current["active_rules"]) == 1,
            "same_topic_updated": len(history["records"]) == 2
                and history["records"][0]["topic"] == history["records"][1]["topic"],
            "search_tools_used": all(any(e["event"] == "tool" and e["name"] == "search_flights"
                and e["result"]["status"] == "ok" for e in result["trace"]) for result in [second, fourth]),
            "sources_and_times_preserved": all(r["source_message"] and r["created_at"] and r["session_id"]
                                                for r in history["records"]),
        }
        report["passed"] = all(report["checks"].values())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "report": str(report_path)}, ensure_ascii=True), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify both Challenge A requirements using real API calls.")
    parser.add_argument("--live", action="store_true", help="Opt in to real model calls using the configured router/key.")
    parser.add_argument("--report", type=Path, default=bootstrap.HERE / "reports" / "live_verification.json")
    args = parser.parse_args()
    if not args.live:
        parser.error("Thêm --live để gọi model thật; dùng unittest cho kiểm tra offline.")
    raise SystemExit(0 if verify(args.report.resolve()) else 1)
