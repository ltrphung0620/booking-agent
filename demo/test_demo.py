"""Kiểm tra chức năng và guardrails; chưa đánh giá hiệu quả các mẫu agent."""

import unittest
from unittest.mock import patch

from constraints import Constraints
from harness import BudgetExceeded, Harness
from main import RUNNERS, run_demo
from mock_tools import MockFlightService
from models import DemoChatModel, make_model, make_planner
from react_agent import run_react


def make_harness(scenario="success", **options):
    return Harness(Constraints(), MockFlightService(scenario), **options)


class AgentFlowTests(unittest.TestCase):
    def test_all_patterns_success(self):
        for pattern in RUNNERS:
            with self.subTest(pattern=pattern):
                result = run_demo(pattern)
                self.assertEqual(result["result"], "DONE")
                self.assertEqual(len(result["bookings"]), 1)
                self.assertTrue(result["bookings"][0]["paid"])

    def test_all_patterns_no_match_handoff_without_side_effects(self):
        for pattern in RUNNERS:
            with self.subTest(pattern=pattern):
                result = run_demo(pattern, "no_match")
                self.assertEqual(result["result"], "FAILED")
                self.assertEqual(result["bookings"], [])
                self.assertIn("ngân sách", result["handoff"]["question"])

    def test_search_errors_are_not_reported_as_no_flights(self):
        for pattern in RUNNERS:
            with self.subTest(pattern=pattern):
                result = run_demo(pattern, "search_error")
                self.assertEqual(result["result"], "FAILED")
                self.assertIn("timeout", result["handoff"]["reason"])
                self.assertIn("thử lại", result["handoff"]["question"])
                self.assertEqual(result["bookings"], [])

    def test_payment_requires_permission_in_all_patterns(self):
        for pattern in RUNNERS:
            with self.subTest(pattern=pattern):
                result = run_demo(pattern, allow_payment=False)
                self.assertEqual(result["result"], "FAILED")
                self.assertEqual(len(result["bookings"]), 1)
                self.assertFalse(result["bookings"][0]["paid"])
                self.assertIn("quyền thanh toán", result["handoff"]["reason"])

    def test_plan_stops_on_execution_failure(self):
        result = run_demo("plan", "sold_out")
        self.assertEqual(result["result"], "FAILED")
        self.assertEqual(result["bookings"], [])
        self.assertFalse(any(e["event"] == "tool" and e["name"] == "pay"
                             for e in result["trace"]))

    def test_react_uses_new_observation_after_sold_out(self):
        result = run_demo("react", "sold_out")
        self.assertEqual(result["result"], "DONE")
        self.assertEqual(result["bookings"][0]["flight"], "VN124")
        searches = [e for e in result["trace"] if e["event"] == "tool"
                    and e["name"] == "search_flights"]
        self.assertEqual(len(searches), 2)

    def test_hybrid_replans_using_updated_flights(self):
        result = run_demo("hybrid", "sold_out")
        self.assertEqual(result["result"], "DONE")
        self.assertEqual(result["bookings"][0]["flight"], "VN124")
        plans = [e for e in result["trace"] if e["event"] == "plan"]
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["plan"]["steps"][0]["args"]["flight"], "VN122")
        self.assertEqual(plans[1]["plan"]["steps"][0]["args"]["flight"], "VN124")
        self.assertTrue(any(e["event"] == "replan" for e in result["trace"]))

    def test_rejected_plans_do_not_execute(self):
        for pattern in ("plan", "hybrid"):
            with self.subTest(pattern=pattern):
                result = run_demo(pattern, reviewer=lambda _: False)
                self.assertEqual(result["result"], "FAILED")
                self.assertEqual(result["bookings"], [])
                self.assertEqual(len([e for e in result["trace"] if e["event"] == "plan"]), 2)

    def test_final_answer_is_not_the_completion_criterion(self):
        class LyingModel(DemoChatModel):
            def _generate(self, messages, **kwargs):
                from langchain_core.messages import AIMessage
                from langchain_core.outputs import ChatGeneration, ChatResult
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="DONE!"))])

        harness = make_harness()
        with patch("react_agent.make_model", return_value=LyingModel(constraints=Constraints())):
            result = run_react(harness)
        self.assertEqual(result["result"], "FAILED")
        self.assertEqual(result["model_answer"], "DONE!")

    def test_loop_stops_at_model_budget(self):
        class LoopingModel(DemoChatModel):
            def _generate(self, messages, **kwargs):
                from langchain_core.outputs import ChatGeneration, ChatResult
                return ChatResult(generations=[ChatGeneration(message=self._search())])

        harness = make_harness(max_model_calls=3)
        with patch("react_agent.make_model", return_value=LoopingModel(constraints=Constraints())):
            result = run_react(harness)
        self.assertEqual(result["result"], "FAILED")
        self.assertEqual(harness.model_calls, 3)
        self.assertIn("BudgetExceeded", result["handoff"]["reason"])

    def test_invalid_model_plan_is_rejected_before_execution(self):
        from langchain_core.runnables import RunnableLambda
        invalid_plan = {"summary": "Bad tool", "steps": [{"tool": "delete_everything", "args": {}}]}
        with patch("workflow.make_planner", return_value=RunnableLambda(lambda _: invalid_plan)):
            result = run_demo("plan")
        self.assertEqual(result["result"], "FAILED")
        self.assertEqual(result["bookings"], [])
        self.assertIn("ValidationError", result["handoff"]["reason"])


class HarnessTests(unittest.TestCase):
    def test_wrong_route_and_invalid_flight_data_fail(self):
        harness = make_harness()
        flight = dict(harness.service.flights[0], destination="HAN")
        self.assertFalse(harness.constraints.is_ok(flight))
        self.assertFalse(harness.constraints.is_ok({}))
        self.assertFalse(harness.constraints.is_ok(dict(flight, price="cheap")))

    def test_invalid_flight_booking_is_blocked(self):
        harness = make_harness("no_match")
        result = harness.execute("book_seat", {"flight": "VJ604"})
        self.assertEqual(result["status"], "denied")
        self.assertEqual(harness.service.bookings, {})

    def test_unknown_tool_missing_extra_and_invalid_args(self):
        harness = make_harness()
        cases = [("unknown", {}, "denied"), ("book_seat", {}, "error"),
                 ("book_seat", {"flight": "VN122", "extra": "oops"}, "error"),
                 ("pay", {"code": "fake-code"}, "denied")]
        for name, args, expected in cases:
            with self.subTest(name=name, args=args):
                self.assertEqual(harness.execute(name, args)["status"], expected)

    def test_cannot_book_two_tickets(self):
        harness = make_harness()
        self.assertEqual(harness.execute("book_seat", {"flight": "VN122"})["status"], "ok")
        self.assertEqual(harness.execute("book_seat", {"flight": "VN124"})["status"], "denied")
        self.assertEqual(len(harness.service.bookings), 1)

    def test_price_change_before_payment_is_blocked(self):
        harness = make_harness()
        held = harness.execute("book_seat", {"flight": "VN122"})
        harness.service.flights[0]["price"] = 2_500_000
        result = harness.execute("pay", {"code": held["code"]})
        self.assertEqual(result["status"], "denied")
        self.assertFalse(harness.service.bookings[held["code"]]["paid"])

    def test_payment_is_idempotent_and_verification_reads_service(self):
        harness = make_harness()
        held = harness.execute("book_seat", {"flight": "VN122"})
        for _ in range(2):
            self.assertEqual(harness.execute("pay", {"code": held["code"]})["status"], "ok")
        self.assertTrue(harness.is_done())
        harness.service.bookings[held["code"]]["paid"] = False
        self.assertFalse(harness.is_done())

    def test_empty_tool_response_and_exception_are_logged(self):
        harness = make_harness()
        args = {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"}
        self.assertEqual(harness.execute("search_flights", args, runner=lambda: {})["status"], "error")
        def broken_tool():
            raise RuntimeError("service broke")
        result = harness.execute("search_flights", args, runner=broken_tool)
        self.assertEqual(result["status"], "error")
        self.assertIn("service broke", harness.trace[-1]["result"]["reason"])

    def test_tool_budget_prevents_additional_side_effect(self):
        harness = make_harness(max_tool_calls=1)
        harness.execute("search_flights", {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"})
        with self.assertRaises(BudgetExceeded):
            harness.execute("book_seat", {"flight": "VN122"})
        self.assertEqual(harness.service.bookings, {})

    def test_search_must_use_original_constraints(self):
        harness = make_harness()
        result = harness.execute("search_flights", {"origin": "SGN", "destination": "HAN", "date": "2026-10-07"})
        self.assertEqual(result["status"], "denied")


class ModelSetupTests(unittest.TestCase):
    def test_real_backend_requires_credentials_and_model(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                make_model("openai", "example-model", Constraints())
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-not-a-real-key"}):
            with self.assertRaisesRegex(ValueError, "model"):
                make_model("openai", None, Constraints())

    def test_openai_client_and_structured_planner_construct_without_api_calls(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-not-a-real-key"}):
            model = make_model("openai", "example-model", Constraints())
            planner = make_planner("openai", "example-model", Constraints())
        self.assertEqual(model.model_name, "example-model")
        self.assertTrue(callable(planner.invoke))


if __name__ == "__main__":
    unittest.main()
