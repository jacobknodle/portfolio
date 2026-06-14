"""Tests for the framework-agnostic API layer."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import handle_differentiate, health  # noqa: E402


class HandleDifferentiateTest(unittest.TestCase):
    def test_basic_success(self):
        status, body = handle_differentiate({"expression": "x^2"})
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["derivative"], "2*x")
        self.assertTrue(body["steps"])

    def test_custom_variable(self):
        status, body = handle_differentiate({"expression": "t^3", "variable": "t"})
        self.assertEqual(status, 200)
        self.assertEqual(body["derivative"], "3*t^2")

    def test_missing_expression(self):
        status, body = handle_differentiate({})
        self.assertEqual(status, 400)
        self.assertFalse(body["ok"])

    def test_blank_expression(self):
        status, body = handle_differentiate({"expression": "   "})
        self.assertEqual(status, 400)

    def test_parse_error_is_400(self):
        status, body = handle_differentiate({"expression": "sin(x"})
        self.assertEqual(status, 400)
        self.assertIn("parse", body["error"].lower())

    def test_bad_variable(self):
        status, body = handle_differentiate({"expression": "x^2", "variable": "2x"})
        self.assertEqual(status, 400)

    def test_evaluation_at_point(self):
        status, body = handle_differentiate({"expression": "x^2", "at": 3})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["evaluation"]["value"], 9.0)
        self.assertAlmostEqual(body["evaluation"]["derivative_value"], 6.0)

    def test_non_object_body(self):
        status, body = handle_differentiate([1, 2, 3])
        self.assertEqual(status, 400)


class HealthTest(unittest.TestCase):
    def test_health_lists_functions(self):
        info = health()
        self.assertTrue(info["ok"])
        self.assertIn("sin", info["functions"])
        self.assertIn("ln", info["functions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
