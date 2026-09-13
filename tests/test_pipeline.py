"""
Unit tests for SupportAgentPipeline.
"""

import unittest
from src.agent.pipeline import SupportAgentPipeline

class TestSupportAgentPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = SupportAgentPipeline()

    def test_full_pipeline_execution(self):
        res = self.pipeline.process("@AppleSupport My iPhone battery dies fast.")
        self.assertIn("predicted_intent", res)
        self.assertIn("drafted_reply", res)
        self.assertIn("should_escalate", res)
        self.assertIn("retrieved_context", res)
        self.assertLessEqual(len(res["drafted_reply"]), 280)

if __name__ == "__main__":
    unittest.main()
