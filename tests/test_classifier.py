"""
Unit tests for IntentClassifier.
"""

import unittest
from src.agent.classifier import IntentClassifier
from src.agent.taxonomy import Intent

class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = IntentClassifier()

    def test_billing_classification(self):
        res = self.clf.classify("I was double charged for Apple Music subscription.")
        self.assertEqual(res["predicted_intent"], Intent.BILLING_SUBSCRIPTION.value)

    def test_device_troubleshooting(self):
        res = self.clf.classify("My iPhone 14 battery is draining fast and screen is frozen.")
        self.assertEqual(res["predicted_intent"], Intent.DEVICE_TROUBLESHOOTING.value)

    def test_account_apple_id(self):
        res = self.clf.classify("Locked out of my Apple ID and password reset link isn't working.")
        self.assertEqual(res["predicted_intent"], Intent.ACCOUNT_APPLE_ID.value)

    def test_out_of_scope(self):
        res = self.clf.classify("What is the recipe for baking chocolate cookies?")
        self.assertEqual(res["predicted_intent"], Intent.UNKNOWN_OTHER.value)

if __name__ == "__main__":
    unittest.main()
