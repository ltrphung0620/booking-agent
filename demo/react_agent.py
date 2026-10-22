"""ReAct: model chọn hành động kế tiếp sau mỗi kết quả tool."""

import json

from langchain.agents import create_agent

from harness import Harness, HarnessMiddleware
from models import Plan, make_model

SYSTEM_PROMPT = (
    "You are a flight booking agent. The original constraints are authoritative. "
    "Search before booking; hold exactly one valid booking, then pay and read it back. "
    "Call one tool at a time and wait for its result before any dependent action. "
    "A tool result is evidence; never claim success without an actual paid booking. "
    "If a flight becomes unavailable, search again and consider another valid flight. "
    "If a tool is denied or the service fails, stop and explain the failure. "
    "All tools here operate on a mock booking system."
)


def invoke_react(harness: Harness, backend: str = "mock", model_name: str | None = None,
                 plan: Plan | None = None) -> dict:
    """Có plan: dùng ReAct làm executor bên trong graph lai."""
    start = len(harness.trace)
    prompt = SYSTEM_PROMPT
    if plan is not None:
        prompt += (" Use this plan as guidance, observing each actual tool result. "
                   "If this plan cannot be completed, stop so the outer planner can revise it. "
                   f"Plan: {plan.model_dump_json()}. "
                   f"Existing bookings: {json.dumps(list(harness.service.bookings.values()))}.")
    try:
        agent = create_agent(
            model=make_model(backend, model_name, harness.constraints, plan),
            tools=list(harness.tools.values()), system_prompt=prompt,
            middleware=[HarnessMiddleware(harness)],
        )
        result = agent.invoke( #hàm có sẵn của langchain
            {"messages": [{"role": "user", "content": harness.constraints.to_prompt()}]},
            config={"recursion_limit": 50},
        )
        answer = str(result["messages"][-1].content)
        tools = [e for e in harness.trace[start:] if e["event"] == "tool"]
        last = tools[-1]["result"] if tools else {}
        reason = last.get("reason", "") if last.get("status") != "ok" else ""
        if (last.get("status") == "ok" and "flights" in last
                and not any(harness.constraints.is_ok(f) for f in last["flights"])):
            reason = "Không có chuyến đáp ứng tất cả ràng buộc"
        return {"answer": answer, "error": reason, "last_status": last.get("status")}
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        harness.record("agent_error", reason=reason)
        return {"answer": "", "error": reason, "last_status": "error"}


def run_react(harness: Harness, backend: str = "mock", model_name: str | None = None) -> dict:
    result = invoke_react(harness, backend, model_name)
    return harness.finish("react", result["error"], result["answer"])
