"""
Unit tests for evaluation metrics and judge calibration.
"""

import unittest

from src.eval.metrics import calculate_rouge_l, evaluate_predictions
from src.eval.judge import LLMSupportJudge
from src.eval.calibration import calibrate_judge_vs_human


class TestEvaluationMetrics(unittest.TestCase):

    def test_rouge_l_identical(self):
        text = "Check http://reportaproblem.apple.com for refunds."
        self.assertAlmostEqual(calculate_rouge_l(text, text), 1.0)

    def test_evaluate_predictions_shape(self):
        metrics = evaluate_predictions(
            y_true_intent=["billing_subscription", "software_update"],
            y_pred_intent=["billing_subscription", "device_troubleshooting"],
            y_true_escalate=[False, True],
            y_pred_escalate=[False, True],
            reference_replies=["Visit reportaproblem.", "Please DM us."],
            predicted_replies=["Visit reportaproblem.", "Please DM us."],
        )
        self.assertIn("intent_accuracy", metrics)
        self.assertIn("escalation_f1", metrics)
        self.assertEqual(metrics["intent_accuracy"], 0.5)

    def test_judge_returns_bounded_score(self):
        judge = LLMSupportJudge()
        result = judge.evaluate_reply(
            customer_text="@AppleSupport refund please",
            predicted_intent="billing_subscription",
            drafted_reply="Visit http://reportaproblem.apple.com for refund help.",
            reference_reply="You can request a refund at http://reportaproblem.apple.com.",
            should_escalate=False,
        )
        self.assertGreaterEqual(result["overall_score"], 1.0)
        self.assertLessEqual(result["overall_score"], 5.0)

    def test_calibration_perfect_agreement(self):
        scores = [4.0, 5.0, 3.0]
        cal = calibrate_judge_vs_human(scores, scores)
        self.assertEqual(cal["pearson_correlation"], 1.0)
        self.assertEqual(cal["mean_absolute_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
