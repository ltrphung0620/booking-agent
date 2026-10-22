"""BTVN#3 - Flight booking agent, PLAN-THEN-EXECUTE pattern (LangChain).

    Goal -> Model writes the WHOLE plan (1 call) -> Human reviews the plan
         -> approved: execute Step 1, Step 2, Step 3 -> Result
         -> rejected: the model writes a new plan

Compared with flight_agent.py (ReAct):
    ReAct              the model decides ONE step at a time, after each result.
    Plan-then-execute  the model is called ONCE for the whole plan. Then plain
                       code runs the steps in order, WITHOUT asking the model again.

    + The plan is visible BEFORE anything runs: a human can approve or reject it.
    - The plan is written before the steps run. If a step fails, the plan
      cannot adapt: we stop and hand off.

The same harness as flight_agent.py is kept:
    1. Constraints are DATA     -> the Constraints class
    2. Permission check         -> check_permission(), runs BEFORE each step
    3. Done is checked by CODE  -> is_done(), reads the booking back
    4. Handoff to a human       -> handoff(), when the agent fails

Run:  python flight_agent_plan_then_execute.py   then choose a case:
    1. success   a valid flight exists -> plan approved -> booked and paid -> DONE
    2. fail      no valid flight -> the plan is rejected or blocked -> FAILED + handoff
"""
import json
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel


# =====================================================================
# 1. CONSTRAINTS ARE DATA
# =====================================================================
@dataclass
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"      # morning flight
    max_price: int = 2_000_000        # VND

    def to_prompt(self) -> str:
        return (f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
                f"departing before {self.depart_before}, price at most {self.max_price:,} VND.")

    def is_ok(self, flight: dict) -> bool:
        """Does this flight satisfy ALL constraints?"""
        return (flight["depart"].startswith(self.date)
                and flight["depart"][11:16] < self.depart_before
                and flight["price"] <= self.max_price)


CONSTRAINTS = Constraints()


# =====================================================================
# MOCKUP TOOLS - same as flight_agent.py, wrapped as LangChain tools
# =====================================================================
FLIGHTS = {
    "success": [
        {"flight": "VN122", "depart": "2026-10-07T08:10", "price": 1_850_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
    "fail": [   # no flight is BOTH in the morning AND under 2 million
        {"flight": "VJ604", "depart": "2026-10-07T08:10", "price": 2_480_000},
        {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},
    ],
}
CASE = "success"      # which data set the tools use (chosen by the user in main)
BOOKINGS = {}         # our fake booking database


@tool
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    return {"status": "ok", "flights": FLIGHTS[CASE]}


@tool
def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
    info = next(f for f in FLIGHTS[CASE] if f["flight"] == flight)
    code = f"{flight}-12A"
    BOOKINGS[code] = {"code": code, "paid": False, **info}
    return {"status": "ok", **BOOKINGS[code]}


@tool
def pay(code: str) -> dict:
    """Pay for a held booking. This spends money and cannot be undone."""
    if code not in BOOKINGS:
        return {"status": "not_found", "code": code}
    BOOKINGS[code]["paid"] = True
    return {"status": "ok", **BOOKINGS[code]}


@tool
def get_booking(code: str) -> dict:
    """Read a booking back from the system."""
    return {"status": "ok", **BOOKINGS[code]} if code in BOOKINGS else {"status": "not_found"}


TOOLS = {t.name: t for t in [search_flights, book_seat, pay, get_booking]}


# =====================================================================
# THE PLAN - the model must answer with this structure (LangChain structured output)
# =====================================================================
class Step(BaseModel):
    tool: str          # one of: book_seat, pay, get_booking
    args: dict         # e.g. {"flight": "VN122"} or {"code": "$booking_code"}


class Plan(BaseModel):
    steps: list[Step]  # an EMPTY list means: "no flight meets the constraints"


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You plan a flight booking. Write the WHOLE plan at once, as a list of steps. "
     "Tools: book_seat(flight), pay(code), get_booking(code). "
     "The booking code is not known yet: write \"$booking_code\" and it will be filled in. "
     "If no flight meets ALL constraints, return an empty plan."),
    ("human", "Goal: {goal}\nAvailable flights: {flights}"),
])

MAX_PLANS = 2       # budget: the model may write at most 2 plans


# =====================================================================
# THE HARNESS - same ideas as flight_agent.py
# =====================================================================
LOG = []    # every tool call: (tool name, args, result)


# ---- 2. PERMISSION CHECK: runs BEFORE each step, using the constraint DATA
def check_permission(tool_name: str, args: dict) -> str | None:
    """Return None if allowed, or a reason if the step must be blocked."""
    if tool_name == "book_seat":
        flight = next((f for f in FLIGHTS[CASE] if f["flight"] == args["flight"]), None)
        if flight is None or not CONSTRAINTS.is_ok(flight):
            return f"{args['flight']} breaks the constraints: {CONSTRAINTS.to_prompt()}"
    return None


# ---- 3. DONE IS CHECKED BY CODE
def is_done() -> bool:
    """Done = a booking exists, is paid, and satisfies the constraints."""
    for code in BOOKINGS:
        b = TOOLS["get_booking"].invoke({"code": code})   # read back from the system
        if b["paid"] and CONSTRAINTS.is_ok(b):
            return True
    return False


# ---- 4. HANDOFF: when the agent fails, give a human 3 things
def handoff() -> dict:
    return {
        "done_so_far": [f"{c}: paid={b['paid']}" for c, b in BOOKINGS.items()] or ["Nothing booked, nothing paid"],
        "tried": [f"{tool}({args}) -> {result['status']}" for tool, args, result in LOG],
        "question": "No flight meets all constraints. Which one can we relax: "
                    "departure time or maximum price?",
    }


def fill_placeholders(args: dict) -> dict:
    """Replace "$booking_code" with the code returned by book_seat."""
    code = next((r["code"] for t, _, r in reversed(LOG) if t == "book_seat" and r["status"] == "ok"), "")
    return {k: (code if v == "$booking_code" else v) for k, v in args.items()}


# =====================================================================
# A SCRIPTED FAKE MODEL - so the demo runs without an API key.
# It returns the plans a real LLM would write, one plan per call. (Optional reading.)
# =====================================================================
def fake_model(case: str):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.runnables import RunnableLambda

    book_pay_check = lambda flight: Plan(steps=[
        Step(tool="book_seat", args={"flight": flight}),
        Step(tool="pay", args={"code": "$booking_code"}),
        Step(tool="get_booking", args={"code": "$booking_code"}),
    ])
    plans = {
        "success": [book_pay_check("VN122"), book_pay_check("VN122")],
        # 1st plan: the model "forgets" the price limit and picks the morning flight VJ604.
        # 2nd plan (after the human rejects it): the model admits no flight fits.
        "fail": [book_pay_check("VJ604"), Plan(steps=[])],
    }[case]
    plans = iter(plans)

    class ScriptedModel(GenericFakeChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: next(plans))

    return ScriptedModel(messages=iter([]))


# =====================================================================
if __name__ == "__main__":
    # ---- Let the user choose which case to demo
    print("Choose a case:")
    print("  1. success  - a valid flight exists")
    print("  2. fail     - no valid flight, the harness must hand off")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Your choice (1/2): ").strip()
    CASE = "success" if choice == "1" else "fail"
    model = fake_model(CASE)

    print(f"\n=== case: {CASE} ===")
    print("User's request:", CONSTRAINTS.to_prompt())

    # ---- GOAL: the request + the flight list (a read-only search, arguments from DATA)
    c = CONSTRAINTS
    found = TOOLS["search_flights"].invoke({"origin": c.origin, "destination": c.destination, "date": c.date})
    LOG.append(("search_flights", {"origin": c.origin, "destination": c.destination, "date": c.date}, found))

    # ---- PLAN + HUMAN REVIEW: the model writes the whole plan, a human approves or rejects it
    planner = PLANNER_PROMPT | model.with_structured_output(Plan)     # a LangChain chain
    plan = None
    for attempt in range(1, MAX_PLANS + 1):
        draft = planner.invoke({"goal": c.to_prompt(), "flights": json.dumps(found["flights"])})

        print(f"\n--- plan #{attempt} (written by the model in ONE call) ---")
        if not draft.steps:
            print("(empty plan: the model found no flight that meets the constraints)")
            break
        for i, step in enumerate(draft.steps, 1):
            print(f"  Step {i}: {step.tool}({step.args})")

        answer = ""
        while answer not in ("y", "n"):
            answer = input("Human reviewer - approve this plan? (y/n): ").strip().lower()
        if answer == "y":
            plan = draft
            break
        print("Rejected -> ask the model for a new plan.")

    # ---- EXECUTE: plain code runs the steps in order. NO model call from here on.
    if plan:
        print("\n--- execute ---")
        for i, step in enumerate(plan.steps, 1):
            args = fill_placeholders(step.args)
            reason = check_permission(step.tool, args)                # harness, BEFORE the step
            result = {"status": "denied", "reason": reason} if reason else TOOLS[step.tool].invoke(args)
            LOG.append((step.tool, args, result))
            print(f"  Step {i}: {step.tool}({args}) -> {result['status']} {result.get('reason', '')}")
            if result["status"] != "ok":
                print("  A step failed. Plan-then-execute cannot change the plan -> stop.")
                break

    # ---- RESULT: the harness decides, not the model
    if is_done():
        output = {"result": "DONE", "booking": list(BOOKINGS.values())}
    else:
        output = {"result": "FAILED", "handoff": handoff()}
    print("\n--- result ---")
    print(json.dumps(output, indent=2))
