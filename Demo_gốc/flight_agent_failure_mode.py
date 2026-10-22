"""BTVN#3 - Four classic agent failure modes, and the harness fix for each.

    Agent = Model + Harness

Same flight booking agent as flight_agent.py. This time the model makes
mistakes on purpose, so we can see what goes wrong WITHOUT a harness and how
the harness catches it.

    1. Infinite loop        the agent repeats the same tool call forever
    2. Tool hallucination   the final answer contains data no tool returned
    3. Goal drift           the agent forgets a constraint of the user
    4. State corruption     an empty tool result is read as "nothing exists"

Run:  python flight_agent_failure_mode.py
      then choose a failure mode (1-4), and turn the harness fix OFF or ON.
"""
import json
import re
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage


# =====================================================================
# CONSTRAINTS ARE DATA
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
# MOCKUP TOOLS - fake data, no network
# =====================================================================
FLIGHTS = [
    {"flight": "VN122", "depart": "2026-10-07T08:10", "price": 1_850_000},   # valid
    {"flight": "QH118", "depart": "2026-10-07T15:40", "price": 1_640_000},   # cheaper, but afternoon
]
MODE = "loop"       # which failure mode we are demoing (chosen in main)
BOOKINGS = {}       # our fake booking database


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    if MODE == "loop":
        # A badly designed tool: the error says nothing useful, so the model
        # has no idea what to change and simply tries again.
        return {"status": "error", "error": "not found"}
    if MODE == "state":
        # The flight service timed out, and the tool silently returned nothing.
        return {}
    return {"status": "ok", "flights": FLIGHTS}


def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
    info = next((f for f in FLIGHTS if f["flight"] == flight), None)
    if info is None:
        return {"status": "not_found", "flight": flight}
    code = f"{flight}-12A"
    BOOKINGS[code] = {"code": code, "paid": False, **info}
    return {"status": "ok", **BOOKINGS[code]}


def pay(code: str) -> dict:
    """Pay for a held booking. This spends money and cannot be undone."""
    if code not in BOOKINGS:
        return {"status": "not_found", "code": code}
    BOOKINGS[code]["paid"] = True
    return {"status": "ok", **BOOKINGS[code]}


def get_booking(code: str) -> dict:
    """Read a booking back from the system."""
    return {"status": "ok", **BOOKINGS[code]} if code in BOOKINGS else {"status": "not_found"}


TOOLS = [search_flights, book_seat, pay, get_booking]


# =====================================================================
# THE HARNESS
# Each fix below only runs when FIX_ON is True, so we can compare.
# =====================================================================
FIX_ON = False      # harness fixes ON or OFF (chosen in main)
LOG = []            # every tool call: (tool name, args, result)


class StopAgent(Exception):
    """Raised by the harness to stop the agent immediately, with a reason."""


# ---- FIX 1 (infinite loop): the same (tool, args) 3 times = a loop
def is_looping(tool: str, args: dict) -> bool:
    same_calls = [1 for t, a, _ in LOG if t == tool and a == args]
    return len(same_calls) >= 3


# ---- FIX 3 (goal drift): check every booking against the constraint DATA
def check_permission(tool: str, args: dict) -> str | None:
    """Return None if allowed, or a reason if the call must be blocked."""
    if tool == "book_seat":
        flight = next((f for f in FLIGHTS if f["flight"] == args["flight"]), None)
        if flight is None or not CONSTRAINTS.is_ok(flight):
            return f"{args['flight']} breaks the constraints: {CONSTRAINTS.to_prompt()}"
    return None


# ---- FIX 4 (state corruption): an empty result is an ERROR, not "nothing found"
def is_empty(result: dict) -> bool:
    return not result or "status" not in result


# ---- FIX 2 (tool hallucination): every fact in the answer must come from a tool
def ungrounded_facts(answer: str) -> list:
    """Flight codes and prices in the answer that no tool result contains."""
    tool_text = json.dumps([result for _, _, result in LOG])
    facts = re.findall(r"\b[A-Z]{2}\d{3}\b", answer)                       # VN122
    facts += [p.replace(",", "") for p in re.findall(r"\d{1,3}(?:,\d{3})+", answer)]  # 1,850,000
    return [f for f in facts if f not in tool_text]


# ---- Done is checked by CODE
def is_done() -> bool:
    return any(b["paid"] and CONSTRAINTS.is_ok(b) for b in BOOKINGS.values())


# ---- Handoff: what was done, what was tried, one concrete question
def handoff(question: str) -> dict:
    return {
        "done_so_far": [f"{c}: paid={b['paid']}" for c, b in BOOKINGS.items()] or ["Nothing booked, nothing paid"],
        "tried": [f"{tool}({args}) -> {result}" for tool, args, result in LOG],
        "question": question,
    }


# ---- Every tool call of the agent goes through the harness
@wrap_tool_call
def harness(request, handler):
    call = request.tool_call
    name, args = call["name"], call["args"]

    if FIX_ON and is_looping(name, args):                                   # FIX 1
        raise StopAgent(f"LOOP: {name}({args}) was already called 3 times")

    reason = check_permission(name, args) if FIX_ON else None               # FIX 3
    if reason:
        result = {"status": "denied", "reason": reason}
    else:
        result = globals()[name](**args)

    LOG.append((name, args, result))

    if FIX_ON and is_empty(result):                                         # FIX 4
        raise StopAgent(f"EMPTY RESULT: {name} returned {result}. "
                        "This is a tool failure, NOT 'no flights'.")

    return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])


SYSTEM_PROMPT = ("You are a flight booking agent. Use the tools: search_flights, "
                 "book_seat, pay. Only book a flight that meets ALL constraints.")


# =====================================================================
# A SCRIPTED FAKE MODEL - it makes the classic mistakes on purpose.
# =====================================================================
def fake_model(mode: str):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    ids = iter(range(100))    # every tool call needs a unique id

    def call(name, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(ids)}"}])

    search = call("search_flights", origin="SGN", destination="DAD", date="2026-10-07")
    script = {
        # 1. The tool says only "not found", so the model retries the SAME call again and again.
        "loop": [call("search_flights", origin="SGN", destination="DAD", date="07/10") for _ in range(12)]
                + [AIMessage(content="Sorry, I could not find any flight.")],

        # 2. The model never books anything, but "reports" a booking it made up.
        "hallucination": [search,
                          AIMessage(content="Done! Booked VN999, seat 5C, for 1,200,000 VND.")],

        # 3. After a few steps the model chases "cheapest" and forgets "morning".
        "drift": [search,
                  call("book_seat", flight="QH118"),
                  call("pay", code="QH118-12A"),
                  AIMessage(content="Booked the cheapest flight QH118 for 1,640,000 VND.")],

        # 4. The search returns {} and the model reads it as "there are no flights".
        "state": [search,
                  AIMessage(content="There are no flights from SGN to DAD on 2026-10-07.")],
    }

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return ScriptedModel(messages=iter(script[mode]))


# =====================================================================
if __name__ == "__main__":
    # ---- 1. Choose the failure mode and whether the harness fix is ON
    modes = {"1": "loop", "2": "hallucination", "3": "drift", "4": "state"}
    print("Choose a failure mode:")
    print("  1. Infinite loop")
    print("  2. Tool hallucination")
    print("  3. Goal drift")
    print("  4. State corruption")
    choice = ""
    while choice not in modes:
        choice = input("Your choice (1-4): ").strip()
    MODE = modes[choice]

    fix = ""
    while fix not in ("y", "n"):
        fix = input("Turn the harness fix ON? (y/n): ").strip().lower()
    FIX_ON = fix == "y"

    print(f"\n=== failure mode: {MODE} - harness fix: {'ON' if FIX_ON else 'OFF'} ===")

    # ---- 2. Build and run the agent. A budget of 10 model calls is always on.
    agent = create_agent(
        model=fake_model(MODE),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[harness, ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")],
    )
    stop_reason = None
    try:
        out = agent.invoke({"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]})
        answer = out["messages"][-1].content
    except StopAgent as e:
        stop_reason, answer = str(e), ""

    print("\n--- trace ---")
    for i, (tool, args, result) in enumerate(LOG, 1):
        print(f"[{i}] {tool}({args}) -> {result}")

    # ---- 3a. Harness OFF: we simply believe whatever the model says
    if not FIX_ON:
        print("\n--- result (no harness: we trust the model) ---")
        print("Model says:", answer)
        print("Bookings in the system:", list(BOOKINGS.values()) or "none")

    # ---- 3b. Harness ON: code decides the result, and hands off on failure
    else:
        questions = {
            "loop": "search_flights keeps answering 'not found' for date='07/10'. "
                    "Should the date be YYYY-MM-DD (2026-10-07)?",
            "hallucination": "The answer mentions a booking that no tool made. "
                             "Should I search and book again?",
            "drift": "The agent tried a flight that breaks the constraints. Valid options: "
                     + ", ".join(f["flight"] for f in FLIGHTS if CONSTRAINTS.is_ok(f))
                     + ". Book one of these?",
            "state": "The flight search returned nothing (service error). Retry later?",
        }
        fake_facts = ungrounded_facts(answer)                              # FIX 2
        if fake_facts:
            stop_reason = f"HALLUCINATION: {fake_facts} do not come from any tool result"

        print("\n--- result (harness ON: code decides) ---")
        if is_done() and not stop_reason:
            output = {"result": "DONE", "booking": list(BOOKINGS.values())}
        else:
            output = {"result": "FAILED", "stop_reason": stop_reason or "GOAL NOT REACHED: no paid booking meets the constraints",
                      "model_said": answer, "handoff": handoff(questions[MODE])}
        print(json.dumps(output, indent=2))
