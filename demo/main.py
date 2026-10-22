"""CLI: chọn mẫu agent, dữ liệu mock và backend model."""

import argparse
import json
import os
import sys

from constraints import Constraints
from harness import Harness
from hybrid_agent import run_hybrid
from mock_tools import MockFlightService, SCENARIOS
from plan_agent import run_plan
from react_agent import run_react
from pydantic import ValidationError

RUNNERS = {"react": run_react, "plan": run_plan, "hybrid": run_hybrid}


def human_review(plan) -> bool:
    print("\nKế hoạch đề xuất:")
    print(plan.summary)
    for number, step in enumerate(plan.steps, 1):
        print(f"  {number}. {step.tool}({step.args})")
    while True:
        answer = input("Duyệt kế hoạch? (y/n): ").strip().lower()
        if answer in ("y", "n"):
            return answer == "y"


def run_demo(pattern: str, scenario: str = "success", backend: str = "mock",
             model_name: str | None = None, reviewer=None, allow_payment: bool = True,
             constraints: Constraints | None = None) -> dict:
    service = MockFlightService(scenario)
    harness = Harness(constraints or Constraints(), service, allow_payment=allow_payment)
    if pattern == "react":
        return run_react(harness, backend, model_name)
    return RUNNERS[pattern](harness, backend, model_name, reviewer)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demo agent đặt vé: ReAct, Plan, Hybrid")
    parser.add_argument("--pattern", choices=RUNNERS, default="react")
    parser.add_argument("--scenario", choices=SCENARIOS, default="success")
    parser.add_argument("--backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--model", default=os.environ.get("FLIGHT_MODEL"))
    parser.add_argument("--origin", default="SGN")
    parser.add_argument("--destination", default="DAD")
    parser.add_argument("--date", default="2026-10-07")
    parser.add_argument("--depart-before", default="12:00")
    parser.add_argument("--max-price", type=int, default=2_000_000)
    parser.add_argument("--review", action="store_true", help="Duyệt plan bằng y/n")
    parser.add_argument("--no-payment", action="store_true", help="Không cấp quyền thanh toán")
    parser.add_argument("--json", action="store_true", help="In kết quả và toàn bộ trace dạng JSON")
    args = parser.parse_args()
    if args.review and args.pattern == "react":
        parser.error("--review dùng cho mẫu plan hoặc hybrid")
    if args.review and args.json:
        parser.error("Không kết hợp --review với --json vì có lời nhắc tương tác")
    if args.backend == "openai":
        if not os.environ.get("OPENAI_API_KEY") or not args.model:
            parser.error("Backend openai cần OPENAI_API_KEY và --model (hoặc FLIGHT_MODEL)")
    try:
        constraints = Constraints(origin=args.origin, destination=args.destination,
                                  date=args.date, depart_before=args.depart_before,
                                  max_price=args.max_price)
    except ValidationError as exc:
        parser.error(f"Ràng buộc không hợp lệ: {exc}")
    result = run_demo(args.pattern, args.scenario, args.backend, args.model,
                      human_review if args.review else None, not args.no_payment,
                      constraints)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"=== {args.pattern} | {args.scenario} | backend={args.backend} ===")
        for number, event in enumerate(result["trace"], 1):
            print(f"[{number:02}] {json.dumps(event, ensure_ascii=False)}")
        print(f"\nKết quả do code xác nhận: {result['result']}")
        print("Bookings:", json.dumps(result["bookings"], ensure_ascii=False))
        if "handoff" in result:
            print("Bàn giao:", json.dumps(result["handoff"], ensure_ascii=False, indent=2))
    # FAILED là kết quả nghiệp vụ có bàn giao, không phải lỗi CLI.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
