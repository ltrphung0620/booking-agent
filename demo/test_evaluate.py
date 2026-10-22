"""Kiểm oracle, benchmark độc lập giữa các lượt và định dạng báo cáo."""

from contextlib import redirect_stderr
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from evaluate import Dataset, evaluate, main, run_case, save_report, score_run


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.model_validate_json(
            Path(__file__).with_name("evaluation_cases.json").read_text(encoding="utf-8"))
        cls.report = evaluate(cls.dataset)

    def row(self, case_id, pattern="react"):
        return deepcopy(next(r for r in self.report["runs"]
                             if r["case_id"] == case_id and r["pattern"] == pattern))

    def grade(self, row):
        case = next(c for c in self.dataset.cases if c.id == row["case_id"])
        return score_run(case, self.dataset.constraints.model_dump(mode="json"),
                         row["output"], row["actual_bookings"])

    def test_matrix_measures_real_architecture_difference(self):
        self.assertEqual(len(self.report["runs"]), 15)
        for run in self.report["runs"]:
            should_pass = not (run["pattern"] == "plan" and run["case_id"] == "sold_out")
            self.assertEqual(run["scores"]["passed"], should_pass,
                             (run["case_id"], run["pattern"], run["scores"]["failed_checks"]))
            self.assertTrue(run["scores"]["safe"])
        summaries = {r["pattern"]: r for r in self.report["summary"]}
        self.assertEqual(summaries["plan"]["expected_behavior_rate"], 0.8)
        self.assertEqual(summaries["plan"]["booking_completion_rate"], 0.5)
        self.assertEqual(summaries["plan"]["recovery_completion_rate"], 0)
        self.assertEqual(summaries["react"]["recovery_completion_rate"], 1)

    def test_expected_handoff_is_pass_despite_failed_business_result(self):
        for case_id in ("no_match", "search_error", "no_payment"):
            run = self.row(case_id)
            self.assertEqual(run["output"]["result"], "FAILED")
            self.assertTrue(run["scores"]["passed"])
            self.assertFalse(run["scores"]["completed"])
        denied = self.row("no_payment")["scores"]
        self.assertEqual(denied["denied_calls"], 1)
        self.assertTrue(denied["safe"])

    def test_false_done_without_booking_is_not_accepted(self):
        run = self.row("success")
        run["actual_bookings"] = []
        run["output"]["bookings"] = []
        run["output"]["model_answer"] = "Đã đặt vé thành công!"
        scores = self.grade(run)
        self.assertFalse(scores["passed"])
        self.assertFalse(scores["completed"])
        self.assertIn("completion_claim_correct", scores["failed_checks"])

    def test_independent_oracle_rejects_invalid_booking_fields(self):
        variants = ({"destination": "HAN"}, {"price": 2_500_000},
                    {"depart": "2026-10-07T12:00:00+07:00"},
                    {"depart": "2026-10-08T08:10:00+07:00"},
                    {"currency": "USD"}, {"price": "1850000"})
        # Chấm oracle không phụ thuộc bộ kiểm Constraints.is_ok của agent.
        with patch("constraints.Constraints.is_ok", return_value=True):
            for changes in variants:
                with self.subTest(changes=changes):
                    run = self.row("success")
                    run["actual_bookings"][0].update(changes)
                    run["output"]["bookings"] = deepcopy(run["actual_bookings"])
                    self.assertFalse(self.grade(run)["safe"])

    def test_snapshot_mismatch_and_multiple_bookings_fail(self):
        run = self.row("success")
        run["output"]["bookings"] = []
        self.assertIn("output_matches_service", self.grade(run)["failed_checks"])
        run["actual_bookings"].append(deepcopy(run["actual_bookings"][0]))
        run["output"]["bookings"] = deepcopy(run["actual_bookings"])
        scores = self.grade(run)
        self.assertFalse(scores["safe"])
        self.assertFalse(scores["passed"])

    def test_successful_unauthorized_payment_is_unsafe(self):
        run = self.row("no_payment")
        run["actual_bookings"][0]["paid"] = True
        run["output"]["bookings"] = deepcopy(run["actual_bookings"])
        run["output"]["result"] = "DONE"
        self.assertFalse(self.grade(run)["safe"])

    def test_successful_bad_side_effect_is_detected_even_if_final_booking_is_valid(self):
        run = self.row("success")
        event = next(e for e in run["output"]["trace"]
                     if e["event"] == "tool" and e["name"] == "book_seat")
        event["result"]["price"] = 2_500_000
        self.assertFalse(self.grade(run)["safe"])

    def test_missing_search_and_missing_paid_readback_fail(self):
        for name, check in (("search_flights", "tool_dependencies_correct"),
                            ("get_booking", "paid_booking_read_back")):
            with self.subTest(tool=name):
                run = self.row("success")
                run["output"]["trace"] = [e for e in run["output"]["trace"]
                                          if not (e["event"] == "tool" and e["name"] == name)]
                self.assertIn(check, self.grade(run)["failed_checks"])

    def test_wrong_handoff_reason_or_progress_is_rejected(self):
        run = self.row("search_error")
        run["output"]["handoff"]["reason"] = "Không có chuyến bay"
        self.assertIn("handoff_reason_correct", self.grade(run)["failed_checks"])
        run = self.row("no_payment")
        run["output"]["handoff"]["done_so_far"] = []
        self.assertFalse(self.grade(run)["handoff_valid"])

    def test_changed_constraints_and_budget_overflow_are_rejected(self):
        run = self.row("success")
        run["output"]["constraints"]["max_price"] = 3_000_000
        self.assertIn("original_constraints_preserved", self.grade(run)["failed_checks"])
        run["output"]["trace"].extend({"event": "model", "purpose": "react"} for _ in range(13))
        self.assertIn("within_budget", self.grade(run)["failed_checks"])

    def test_repeated_sold_out_runs_have_isolated_service_state(self):
        data = self.dataset.model_dump(mode="json")
        data["cases"] = [c for c in data["cases"] if c["id"] == "sold_out"]
        report = evaluate(Dataset.model_validate(data), repeats=2)
        self.assertEqual(len(report["runs"]), 6)
        for run in report["runs"]:
            if run["pattern"] == "plan":
                self.assertEqual(run["actual_bookings"], [])
            else:
                self.assertEqual(run["actual_bookings"][0]["flight"], "VN124")
                self.assertEqual(run["scores"]["search_calls"], 2)
        self.assertIsNone(report["summary"][0]["expected_handoff_rate"])

    def test_runner_exception_is_recorded_and_other_runs_can_continue(self):
        with patch.dict("evaluate.RUNNERS", {"react": lambda *a, **k: 1 / 0}):
            run = run_case("react", self.dataset.cases[0], self.dataset.constraints, 1)
        self.assertIn("ZeroDivisionError", run["error"])
        self.assertFalse(run["scores"]["passed"])
        self.assertIn("no_unhandled_exception", run["scores"]["failed_checks"])

    def test_dataset_validation_and_nonpositive_repeats(self):
        data = self.dataset.model_dump(mode="json")
        data["cases"].append(deepcopy(data["cases"][0]))
        with self.assertRaises(ValidationError):
            Dataset.model_validate(data)
        with self.assertRaises(ValueError):
            evaluate(self.dataset, repeats=0)
        with patch("sys.argv", ["evaluate.py", "--repeats", "0"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    def test_reports_preserve_traces_and_csv_row_counts(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_report(self.report, folder)
            report = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["runs"]), 15)
            self.assertTrue(report["runs"][0]["output"]["trace"])
            for filename, count in (("runs.csv", 15), ("summary.csv", 3)):
                with (folder / filename).open(encoding="utf-8-sig", newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            markdown = (folder / "report.md").read_text(encoding="utf-8")
            self.assertIn("80%", markdown)
            self.assertIn("Plan hiện dừng", markdown)


if __name__ == "__main__":
    unittest.main()
