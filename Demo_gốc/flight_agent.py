"""BTVN#3 - Flight booking agent with a simple harness (LangChain).

    Agent = Model + Harness

The MODEL decides which tool to call next.
The HARNESS (plain Python code) decides whether the call is allowed,
and whether the job is really done.

This demo shows 4 harness ideas:
    1. Constraints are DATA     -> the Constraints class
    2. Permission check         -> check_permission(), runs BEFORE a tool
    3. Done is checked by CODE  -> is_done(), reads the booking back
    4. Handoff to a human       -> handoff(), when the agent fails

Run:  python flight_agent.py   then choose a case:
    1. success   a valid flight exists -> booked and paid -> DONE
    2. fail      no valid flight -> harness blocks the booking -> FAILED + handoff
"""
import json
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage


# =====================================================================
# 1. CONSTRAINTS ARE DATA
#    The user's request is stored as data. The prompt is built FROM it,
#    and the harness CHECKS it in code. The model cannot "forget" it.
# =====================================================================
@dataclass
class Constraints: 
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"      # morning flight
    max_price: int = 2_000_000        # VND

    def to_prompt(self) -> str: #chuyển dữ liệu có cấu trúc thành ngôn ngữ tự nhiên để gửi cho llm
        return (f"Book one ticket {self.origin} -> {self.destination} on {self.date}, "
                f"departing before {self.depart_before}, price at most {self.max_price:,} VND.")

    def is_ok(self, flight: dict) -> bool: #nhận dữ liệu một chuyến bay và xem nó có đáp ứng được các điều kiện trong constraint hay không
        """Does this flight satisfy ALL constraints?"""
        return (flight["depart"].startswith(self.date)
                and flight["depart"][11:16] < self.depart_before
                and flight["price"] <= self.max_price)


CONSTRAINTS = Constraints()


# =====================================================================
# MOCKUP TOOLS - fake data, no network
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


def search_flights(origin: str, destination: str, date: str) -> dict:
    """Search flights by route and date (YYYY-MM-DD)."""
    return {"status": "ok", "flights": FLIGHTS[CASE]}


def book_seat(flight: str) -> dict:
    """Hold a seat on a flight. Returns a booking code. No money is charged yet."""
    info = next(f for f in FLIGHTS[CASE] if f["flight"] == flight)
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
# =====================================================================
LOG = []    # every tool call: (tool name, args, result)


# ---- 2. PERMISSION CHECK: runs BEFORE the tool, using the constraint DATA
def check_permission(tool: str, args: dict) -> str | None:
    """Return None if allowed, or a reason if the call must be blocked."""
    if tool == "book_seat":
        flight = next((f for f in FLIGHTS[CASE] if f["flight"] == args["flight"]), None)
        if flight is None or not CONSTRAINTS.is_ok(flight):
            return f"{args['flight']} breaks the constraints: {CONSTRAINTS.to_prompt()}"
    return None


# ---- 3. DONE IS CHECKED BY CODE: never trust the model saying "I'm done"
def is_done() -> bool:
    """Done = a booking exists, is paid, and satisfies the constraints."""
    for code in BOOKINGS:
        b = get_booking(code)                  # read back from the system
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


# ---- Every tool call of the agent goes through the harness first
@wrap_tool_call
def harness(request, handler):
    call = request.tool_call
    reason = check_permission(call["name"], call["args"])
    if reason:                                         # blocked: the tool does NOT run
        result = {"status": "denied", "reason": reason}
    else:                                              # allowed: run the real tool
        result = globals()[call["name"]](**call["args"])
    LOG.append((call["name"], call["args"], result))
    return ToolMessage(content=json.dumps(result), tool_call_id=call["id"])


# =====================================================================
# THE AGENT (ReAct loop, built by LangChain)
# =====================================================================
SYSTEM_PROMPT = ("You are a flight booking agent. Use the tools: search_flights, "
                 "book_seat, pay. Only book a flight that meets ALL constraints. "
                 "If no flight meets them, stop and say so.")


# =====================================================================
# A SCRIPTED FAKE MODEL - so the demo runs without an API key.
# It replays the tool calls a real LLM would make. (Optional reading.)
# =====================================================================
def fake_model(case: str):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    def call(name, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])

    script = {
        "success": [call("search_flights", origin="SGN", destination="DAD", date="2026-10-07"),
                    call("book_seat", flight="VN122"),
                    call("pay", code="VN122-12A"),
                    AIMessage(content="Booked VN122, seat 12A, paid 1,850,000 VND.")],
        # The model "forgets" the price limit and tries the morning flight.
        "fail": [call("search_flights", origin="SGN", destination="DAD", date="2026-10-07"),
                 call("book_seat", flight="VJ604"),
                 AIMessage(content="I could not book a flight that meets the constraints.")],
    }

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    return ScriptedModel(messages=iter(script[case]))


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

    print(f"=== case: {CASE} ===")
    print("Constraints (data):", CONSTRAINTS)

    # Build the agent: model + tools + harness. LangChain runs the ReAct loop.
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[harness,                                   # our harness
                    ModelCallLimitMiddleware(run_limit=10)],   # budget: max 10 model calls
    )

    # User's request
    print("User's request:", CONSTRAINTS.to_prompt())
    agent.invoke({"messages": [{"role": "user", "content": CONSTRAINTS.to_prompt()}]})

    print("\n--- trace ---")
    for i, (tool, args, result) in enumerate(LOG, 1):
        print(f"[{i}] {tool}({args}) -> {result['status']} {result.get('reason', '')}")

    # The harness decides the result, not the model.
    if is_done():
        output = {"result": "DONE", "booking": list(BOOKINGS.values())}
    else:
        output = {"result": "FAILED", "handoff": handoff()}
    print("\n--- result ---")
    print(json.dumps(output, indent=2))
