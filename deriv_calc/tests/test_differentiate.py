"""Correctness tests for the differentiation engine.

The central idea: rather than hard-coding expected derivative strings (which are
sensitive to formatting), we verify each symbolic derivative *numerically*. For
a function f, the symbolic derivative f' must agree with the central
finite-difference approximation (f(x+h) - f(x-h)) / (2h) at several sample
points. This catches genuine mathematical errors regardless of how the answer
is formatted or simplified.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculus import differentiate, evaluate, parse, simplify  # noqa: E402
from calculus.parser import ParseError  # noqa: E402


def numeric_derivative(expr_tree, x, h=1e-6):
    plus = evaluate(expr_tree, {"x": x + h})
    minus = evaluate(expr_tree, {"x": x - h})
    return (plus - minus) / (2 * h)


class NumericAgreementTest(unittest.TestCase):
    """Symbolic derivative must match a finite-difference approximation."""

    # Each entry: (expression, sample points to test at).
    CASES = [
        # polynomials and power rule
        ("x", [0.3, 1.5, -2.1]),
        ("x^2", [0.5, 1.7, -1.2]),
        ("x^5", [0.4, 1.1, -0.9]),
        ("3*x^4 - 2*x^2 + 7*x - 1", [0.6, -1.3, 2.0]),
        ("x^(1/2)", [0.5, 2.0, 4.0]),
        ("x^(-3)", [0.7, 1.4, -2.0]),
        ("1/x", [0.8, -1.5, 3.0]),
        ("1/x^2", [0.9, -1.1, 2.2]),
        # products and quotients
        ("x^2 * sin(x)", [0.5, 1.2, -0.7]),
        ("(x+1)*(x-2)", [0.3, 1.9, -1.4]),
        ("(x^2+1)/(x-3)", [0.5, 1.7, -0.6]),
        ("x/(x^2+1)", [0.4, 2.1, -1.8]),
        # chain rule + trig
        ("sin(x)", [0.3, 1.2, -0.8]),
        ("cos(x)", [0.5, 1.1, -1.3]),
        ("tan(x)", [0.3, 0.8, -0.6]),
        ("sin(x^2)", [0.6, 1.0, -0.7]),
        ("sin(cos(x))", [0.4, 1.3, -0.9]),
        ("sec(x)", [0.3, 0.7, -0.5]),
        ("csc(x)", [0.4, 1.0, 2.0]),
        ("cot(x)", [0.5, 1.1, 2.2]),
        # exponentials and logs
        ("exp(x)", [0.2, 1.0, -0.5]),
        ("e^x", [0.2, 1.0, -0.5]),
        ("e^(x^2)", [0.3, 0.9, -0.6]),
        ("2^x", [0.5, 1.5, -1.0]),
        ("ln(x)", [0.5, 2.0, 5.0]),
        ("ln(x^2+1)", [0.4, 1.2, -0.8]),
        ("log10(x)", [0.5, 2.0, 5.0]),
        ("log2(x)", [0.5, 2.0, 8.0]),
        # roots
        ("sqrt(x)", [0.5, 2.0, 4.0]),
        ("sqrt(x^2+1)", [0.4, 1.3, -0.9]),
        ("cbrt(x)", [0.5, 2.0, 4.0]),
        # inverse trig
        ("asin(x)", [-0.5, 0.2, 0.6]),
        ("acos(x)", [-0.4, 0.3, 0.7]),
        ("atan(x)", [-1.0, 0.5, 2.0]),
        # hyperbolic
        ("sinh(x)", [-0.5, 0.4, 1.0]),
        ("cosh(x)", [-0.6, 0.3, 1.1]),
        ("tanh(x)", [-0.7, 0.2, 1.2]),
        # the genuinely general case: variable base and exponent
        ("x^x", [0.5, 1.5, 2.0]),
        ("x^sin(x)", [0.5, 1.2, 2.0]),
        # nested composition
        ("ln(sin(x^2)+2)", [0.4, 1.1, -0.7]),
        ("(3*x^2+1)^4", [0.3, 1.0, -0.8]),
    ]

    def test_cases_match_finite_difference(self):
        for expr, points in self.CASES:
            with self.subTest(expr=expr):
                result = differentiate(expr)
                deriv_tree = parse(result["derivative"])
                original = parse(expr)
                for x in points:
                    approx = numeric_derivative(original, x)
                    exact = evaluate(deriv_tree, {"x": x})
                    self.assertAlmostEqual(
                        exact, approx, delta=1e-3,
                        msg=f"{expr}: symbolic {exact} vs numeric {approx} at x={x}",
                    )


class StepsTest(unittest.TestCase):
    def test_steps_are_recorded(self):
        result = differentiate("x^2 * sin(x)")
        rules = [s["rule"] for s in result["steps"]]
        self.assertIn("Product Rule", rules)
        self.assertIn("Power Rule", rules)
        self.assertIn("Chain Rule", rules)
        self.assertTrue(all(s["before"] and s["after"] for s in result["steps"]))

    def test_quotient_rule_named(self):
        result = differentiate("(x^2+1)/(x-3)")
        self.assertIn("Quotient Rule", [s["rule"] for s in result["steps"]])

    def test_constant_derivative_is_zero(self):
        result = differentiate("5")
        self.assertEqual(result["derivative"], "0")


class SimplifyTest(unittest.TestCase):
    def test_known_simple_derivatives(self):
        self.assertEqual(differentiate("x^2")["derivative"], "2*x")
        self.assertEqual(differentiate("x")["derivative"], "1")
        self.assertEqual(differentiate("3*x")["derivative"], "3")
        self.assertEqual(differentiate("sin(x)")["derivative"], "cos(x)")

    def test_simplify_preserves_value(self):
        tree = parse("x^2 + 0*x + 1*x - x")
        simplified = simplify(tree)
        for x in (0.3, 1.5, -2.0):
            self.assertAlmostEqual(
                evaluate(tree, {"x": x}), evaluate(simplified, {"x": x})
            )


class ParseErrorTest(unittest.TestCase):
    def test_empty_input(self):
        with self.assertRaises(ParseError):
            parse("")

    def test_unbalanced_parens(self):
        with self.assertRaises(ParseError):
            parse("sin(x")

    def test_dangling_operator(self):
        with self.assertRaises(ParseError):
            parse("x +")


if __name__ == "__main__":
    unittest.main(verbosity=2)
