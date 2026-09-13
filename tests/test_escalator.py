"""
Unit tests for EscalationRouter.
"""

import unittest
from src.agent.escalator import EscalationRouter

class TestEscalationRouter(unittest.TestCase):

    def setUp(self):
        self.router = EscalationRouter()

    def test_thermal_hazard_escalation(self):
        res = self.router.evaluate("My iPhone battery started smoking and got burning hot!", "device_troubleshooting")
        self.assertTrue(res["should_escalate"])
        self.assertEqual(res["trigger_rule"], "safety_thermal_hazard")

    def test_fraud_escalation(self):
        res = self.router.evaluate("Someone stole my credit card and spent $500 on App Store fraud!", "billing_subscription")
        self.assertTrue(res["should_escalate"])
        self.assertEqual(res["trigger_rule"], "financial_fraud_security")

    def test_routine_auto_handle(self):
        res = self.router.evaluate("How do I cancel my subscription on iPhone?", "billing_subscription")
        self.assertFalse(res["should_escalate"])

if __name__ == "__main__":
    unittest.main()
