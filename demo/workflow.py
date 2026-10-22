"""Các node và state dùng chung cho hai graph có planner."""

import json
from typing import Callable, TypedDict

from harness import Harness
from models import Plan, make_planner


class WorkflowState(TypedDict, total=False):
    flights: list[dict]
    plan: Plan | None
    attempts: int
    approved: bool
    error: str
    feedback: str
    answer: str
    last_status: str
    done: bool
    output: dict


class WorkflowNodes:
    def __init__(self, harness: Harness, pattern: str, backend: str = "mock",
                 model_name: str | None = None,
                 reviewer: Callable[[Plan], bool] | None = None, max_plans: int = 2):
        self.harness = harness
        self.pattern = pattern
        self.backend = backend
        self.model_name = model_name
        self.reviewer = reviewer
        self.max_plans = max_plans

    def search(self, state: WorkflowState) -> dict:
        c = self.harness.constraints
        try:
            result = self.harness.execute("search_flights", {
                "origin": c.origin, "destination": c.destination, "date": c.date.isoformat()})
            return {"flights": result.get("flights", []),
                    "error": "" if result["status"] == "ok" else result.get("reason", result["status"]),
                    "feedback": state.get("error", "") or state.get("feedback", ""),
                    "last_status": result["status"]}
        except Exception as exc:
            return {"flights": [], "error": f"{type(exc).__name__}: {exc}",
                    "last_status": "error"}

    def plan(self, state: WorkflowState) -> dict:
        attempt = state.get("attempts", 0) + 1
        try:
            self.harness.consume_model_call("planner")
            planner = make_planner(self.backend, self.model_name, self.harness.constraints)
            payload = {
                "goal": self.harness.constraints.to_prompt(),
                "constraints": self.harness.constraints.model_dump(mode="json"),
                "flights": state["flights"],
                "bookings": list(self.harness.service.bookings.values()),
                "feedback": state.get("error", "") or state.get("feedback", ""),
            }
            draft = planner.invoke({"payload": json.dumps(payload, ensure_ascii=False)})
            # Kiểm schema cả khi dùng một planner thay thế trong bài học.
            draft = Plan.model_validate(draft)
            self.harness.record("plan", attempt=attempt, plan=draft.model_dump())
            error = "" if draft.steps else draft.summary
            return {"plan": draft, "attempts": attempt, "error": error}
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.harness.record("planner_error", reason=reason)
            return {"plan": None, "attempts": attempt, "error": reason}

    def review(self, state: WorkflowState) -> dict:
        try:
            approved = self.reviewer(state["plan"]) if self.reviewer else True
            self.harness.record("review", approved=approved,
                                mode="human" if self.reviewer else "auto")
            return {"approved": approved,
                    "error": "" if approved else "Người duyệt từ chối kế hoạch",
                    "feedback": "" if approved else "Người duyệt từ chối kế hoạch"}
        except Exception as exc:
            return {"approved": False, "error": f"Reviewer error: {exc}"}

    def execute_plan(self, state: WorkflowState) -> dict:
        """Plain Python executor: không hỏi lại model sau mỗi bước."""
        try:
            for index, step in enumerate(state["plan"].steps, 1):
                args = {k: self.harness.booking_code() if v == "$booking_code" else v
                        for k, v in step.args.items()}
                result = self.harness.execute(step.tool, args)
                self.harness.record("step", number=index, status=result["status"])
                if result["status"] != "ok":
                    return {"error": result.get("reason", result["status"]),
                            "last_status": result["status"]}
            return {"error": "", "last_status": "ok"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "last_status": "error"}

    def verify(self, state: WorkflowState) -> dict:
        done = self.harness.is_done()
        self.harness.record("graph_check", done=done)
        return {"done": done}

    def finalize(self, state: WorkflowState) -> dict:
        return {"output": self.harness.finish(self.pattern, state.get("error", ""),
                                               state.get("answer", ""))}

    @staticmethod
    def after_search(state: WorkflowState) -> str:
        return "finalize" if state.get("error") else "plan"

    @staticmethod
    def after_plan(state: WorkflowState) -> str:
        plan = state.get("plan")
        return "review" if plan and plan.steps else "finalize"

    def after_review(self, state: WorkflowState) -> str:
        if state["approved"]:
            return "execute"
        return "plan" if state["attempts"] < self.max_plans else "finalize"
