"""Symbolic differentiation with a step-by-step explanation.

The :class:`Differentiator` walks an expression tree and, for every rule it
applies, records a :class:`Step` describing what happened. Steps are emitted
top-down: the outermost rule first, with the not-yet-evaluated inner
derivatives shown as ``d/dx[...]`` placeholders, then the steps that resolve
those placeholders. This mirrors how the rules are presented in a textbook.

Supported rules
---------------
* constant rule, constant-multiple rule
* sum and difference rules
* product rule, quotient rule
* power rule (constant exponent)
* exponential rule (constant base, including base ``e``)
* generalised power rule / logarithmic differentiation (``u^v``)
* chain rule, applied to every supported function
* trigonometric, inverse-trigonometric, hyperbolic, inverse-hyperbolic,
  exponential, logarithmic, root and absolute-value functions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from .nodes import (
    Add, Const, Div, Func, Mul, Neg, Node, Num, Pow, Sub, Var,
    ONE, TWO, ZERO,
)
from .parser import parse
from .simplify import simplify


@dataclass
class Step:
    """A single rule application in the worked solution."""

    rule: str            # short name, e.g. "Product Rule"
    before: str          # what we are differentiating, plain text
    after: str           # the result of applying the rule, plain text
    explanation: str     # prose describing the rule
    before_latex: str = ""
    after_latex: str = ""

    def as_dict(self) -> dict:
        return {
            "rule": self.rule,
            "before": self.before,
            "after": self.after,
            "explanation": self.explanation,
            "before_latex": self.before_latex,
            "after_latex": self.after_latex,
        }


class DifferentiationError(ValueError):
    """Raised when an expression cannot be differentiated."""


def depends_on(node: Node, var: str) -> bool:
    """True if ``node`` contains the differentiation variable ``var``."""
    if isinstance(node, Var):
        return node.name == var
    if isinstance(node, (Num, Const)):
        return False
    return any(depends_on(c, var) for c in node.children())


# Outer-derivative templates for each supported function: given the inner
# argument ``u`` they return f'(u) (without the chain-rule factor u'). Defined
# as a function so the description and the node stay in sync.
def _func_derivative(name: str, u: Node) -> Node:
    sqrt = lambda n: Func("sqrt", n)
    p = lambda b, e: Pow(b, Num(e))

    table = {
        "sin": lambda: Func("cos", u),
        "cos": lambda: Neg(Func("sin", u)),
        "tan": lambda: p(Func("sec", u), 2),
        "cot": lambda: Neg(p(Func("csc", u), 2)),
        "sec": lambda: Mul(Func("sec", u), Func("tan", u)),
        "csc": lambda: Neg(Mul(Func("csc", u), Func("cot", u))),
        "asin": lambda: Div(ONE, sqrt(Sub(ONE, p(u, 2)))),
        "acos": lambda: Neg(Div(ONE, sqrt(Sub(ONE, p(u, 2))))),
        "atan": lambda: Div(ONE, Add(ONE, p(u, 2))),
        "acot": lambda: Neg(Div(ONE, Add(ONE, p(u, 2)))),
        "asec": lambda: Div(ONE, Mul(Func("abs", u), sqrt(Sub(p(u, 2), ONE)))),
        "acsc": lambda: Neg(Div(ONE, Mul(Func("abs", u), sqrt(Sub(p(u, 2), ONE))))),
        "sinh": lambda: Func("cosh", u),
        "cosh": lambda: Func("sinh", u),
        "tanh": lambda: p(Func("sech", u), 2),
        "coth": lambda: Neg(p(Func("csch", u), 2)),
        "sech": lambda: Neg(Mul(Func("sech", u), Func("tanh", u))),
        "csch": lambda: Neg(Mul(Func("csch", u), Func("coth", u))),
        "asinh": lambda: Div(ONE, sqrt(Add(p(u, 2), ONE))),
        "acosh": lambda: Div(ONE, sqrt(Sub(p(u, 2), ONE))),
        "atanh": lambda: Div(ONE, Sub(ONE, p(u, 2))),
        "exp": lambda: Func("exp", u),
        "ln": lambda: Div(ONE, u),
        "log": lambda: Div(ONE, Mul(u, Func("ln", Num(10)))),
        "log10": lambda: Div(ONE, Mul(u, Func("ln", Num(10)))),
        "log2": lambda: Div(ONE, Mul(u, Func("ln", Num(2)))),
        "sqrt": lambda: Div(ONE, Mul(TWO, sqrt(u))),
        "cbrt": lambda: Div(ONE, Mul(Num(3), p(Func("cbrt", u), 2))),
        "abs": lambda: Div(u, Func("abs", u)),
    }
    if name not in table:
        raise DifferentiationError(f"No differentiation rule for function {name!r}")
    return table[name]()


# Human-readable description of each function's derivative, for the steps.
_FUNC_RULE_TEXT = {
    "sin": "the derivative of sin is cos",
    "cos": "the derivative of cos is -sin",
    "tan": "the derivative of tan is sec^2",
    "cot": "the derivative of cot is -csc^2",
    "sec": "the derivative of sec is sec*tan",
    "csc": "the derivative of csc is -csc*cot",
    "asin": "the derivative of arcsin is 1/sqrt(1-u^2)",
    "acos": "the derivative of arccos is -1/sqrt(1-u^2)",
    "atan": "the derivative of arctan is 1/(1+u^2)",
    "acot": "the derivative of arccot is -1/(1+u^2)",
    "asec": "the derivative of arcsec is 1/(|u|*sqrt(u^2-1))",
    "acsc": "the derivative of arccsc is -1/(|u|*sqrt(u^2-1))",
    "sinh": "the derivative of sinh is cosh",
    "cosh": "the derivative of cosh is sinh",
    "tanh": "the derivative of tanh is sech^2",
    "coth": "the derivative of coth is -csch^2",
    "sech": "the derivative of sech is -sech*tanh",
    "csch": "the derivative of csch is -csch*coth",
    "asinh": "the derivative of arcsinh is 1/sqrt(u^2+1)",
    "acosh": "the derivative of arccosh is 1/sqrt(u^2-1)",
    "atanh": "the derivative of arctanh is 1/(1-u^2)",
    "exp": "the derivative of e^u is e^u",
    "ln": "the derivative of ln(u) is 1/u",
    "log": "log is treated as base-10; d/dx log_10(u) = 1/(u*ln 10)",
    "log10": "d/dx log_10(u) = 1/(u*ln 10)",
    "log2": "d/dx log_2(u) = 1/(u*ln 2)",
    "sqrt": "the derivative of sqrt(u) is 1/(2*sqrt(u))",
    "cbrt": "the derivative of cbrt(u) is 1/(3*cbrt(u)^2)",
    "abs": "the derivative of |u| is u/|u| (the sign of u)",
}


class Differentiator:
    def __init__(self, var: str = "x"):
        self.var = var
        self.steps: List[Step] = []

    # -- step recording helpers -------------------------------------------
    def _ddx(self, node: Node) -> str:
        return f"d/d{self.var}[{node.to_string()}]"

    def _ddx_latex(self, node: Node) -> str:
        return f"\\frac{{d}}{{d{self.var}}}\\left[{node.to_latex()}\\right]"

    def _record(self, rule: str, node: Node, after: str, explanation: str,
                after_latex: str) -> None:
        self.steps.append(Step(
            rule=rule,
            before=self._ddx(node),
            after=after,
            explanation=explanation,
            before_latex=self._ddx_latex(node),
            after_latex=after_latex,
        ))

    # -- main dispatch -----------------------------------------------------
    def diff(self, node: Node) -> Node:
        # Anything free of the variable is a constant.
        if not depends_on(node, self.var):
            self._record(
                "Constant Rule", node, "0",
                "This expression does not contain the variable, so its "
                "derivative is 0.", "0",
            )
            return ZERO

        if isinstance(node, Var):
            self._record(
                "Variable Rule", node, "1",
                f"The derivative of {self.var} with respect to {self.var} is 1.",
                "1",
            )
            return ONE

        if isinstance(node, Neg):
            return self._diff_neg(node)
        if isinstance(node, Add):
            return self._diff_add(node)
        if isinstance(node, Sub):
            return self._diff_sub(node)
        if isinstance(node, Mul):
            return self._diff_mul(node)
        if isinstance(node, Div):
            return self._diff_div(node)
        if isinstance(node, Pow):
            return self._diff_pow(node)
        if isinstance(node, Func):
            return self._diff_func(node)

        raise DifferentiationError(
            f"Cannot differentiate node of type {type(node).__name__}"
        )

    # -- individual rules --------------------------------------------------
    def _diff_neg(self, node: Neg) -> Node:
        u = node.operand
        self._record(
            "Constant-Multiple Rule", node, f"-{self._ddx(u)}",
            "The derivative of a negation is the negation of the derivative.",
            f"-{self._ddx_latex(u)}",
        )
        return Neg(self.diff(u))

    def _diff_add(self, node: Add) -> Node:
        u, v = node.left, node.right
        self._record(
            "Sum Rule", node, f"{self._ddx(u)} + {self._ddx(v)}",
            "The derivative of a sum is the sum of the derivatives.",
            f"{self._ddx_latex(u)} + {self._ddx_latex(v)}",
        )
        return Add(self.diff(u), self.diff(v))

    def _diff_sub(self, node: Sub) -> Node:
        u, v = node.left, node.right
        self._record(
            "Difference Rule", node, f"{self._ddx(u)} - {self._ddx(v)}",
            "The derivative of a difference is the difference of the derivatives.",
            f"{self._ddx_latex(u)} - {self._ddx_latex(v)}",
        )
        return Sub(self.diff(u), self.diff(v))

    def _diff_mul(self, node: Mul) -> Node:
        u, v = node.left, node.right
        u_const = not depends_on(u, self.var)
        v_const = not depends_on(v, self.var)

        if u_const:
            self._record(
                "Constant-Multiple Rule", node,
                f"{u.to_string()}*{self._ddx(v)}",
                "A constant factor can be pulled out of the derivative.",
                f"{u.to_latex()} \\cdot {self._ddx_latex(v)}",
            )
            return Mul(u, self.diff(v))
        if v_const:
            self._record(
                "Constant-Multiple Rule", node,
                f"{v.to_string()}*{self._ddx(u)}",
                "A constant factor can be pulled out of the derivative.",
                f"{v.to_latex()} \\cdot {self._ddx_latex(u)}",
            )
            return Mul(v, self.diff(u))

        self._record(
            "Product Rule", node,
            f"{self._ddx(u)}*{v.to_string()} + {u.to_string()}*{self._ddx(v)}",
            "Product rule: (u*v)' = u'*v + u*v'.",
            f"{self._ddx_latex(u)} \\cdot {v.to_latex()} + "
            f"{u.to_latex()} \\cdot {self._ddx_latex(v)}",
        )
        return Add(Mul(self.diff(u), v), Mul(u, self.diff(v)))

    def _diff_div(self, node: Div) -> Node:
        u, v = node.left, node.right
        if not depends_on(v, self.var):
            self._record(
                "Constant-Multiple Rule", node,
                f"(1/{v.to_string()})*{self._ddx(u)}",
                "A constant denominator can be pulled out of the derivative.",
                f"\\frac{{1}}{{{v.to_latex()}}} \\cdot {self._ddx_latex(u)}",
            )
            return Div(self.diff(u), v)

        self._record(
            "Quotient Rule", node,
            f"({self._ddx(u)}*{v.to_string()} - {u.to_string()}*{self._ddx(v)})"
            f"/{v.to_string()}^2",
            "Quotient rule: (u/v)' = (u'*v - u*v') / v^2.",
            f"\\frac{{{self._ddx_latex(u)} \\cdot {v.to_latex()} - "
            f"{u.to_latex()} \\cdot {self._ddx_latex(v)}}}{{{v.to_latex()}^2}}",
        )
        numerator = Sub(Mul(self.diff(u), v), Mul(u, self.diff(v)))
        return Div(numerator, Pow(v, TWO))

    def _diff_pow(self, node: Pow) -> Node:
        base, exp = node.base, node.exp
        base_has = depends_on(base, self.var)
        exp_has = depends_on(exp, self.var)

        # Case 1: constant exponent -> power rule.
        if base_has and not exp_has:
            new_exp = simplify(Sub(exp, ONE))
            after = f"{exp.to_string()}*{base.to_string()}^({new_exp.to_string()})*{self._ddx(base)}"
            self._record(
                "Power Rule", node, after,
                "Power rule (constant exponent): d/dx[u^n] = n*u^(n-1)*u', "
                "where the last factor comes from the chain rule.",
                f"{exp.to_latex()} \\cdot {base.to_latex()}^{{{new_exp.to_latex()}}}"
                f" \\cdot {self._ddx_latex(base)}",
            )
            return Mul(Mul(exp, Pow(base, new_exp)), self.diff(base))

        # Case 2: constant base -> exponential rule.
        if exp_has and not base_has:
            if base == Const("e"):
                after = f"e^({exp.to_string()})*{self._ddx(exp)}"
                self._record(
                    "Exponential Rule", node, after,
                    "Exponential rule (base e): d/dx[e^u] = e^u * u'.",
                    f"e^{{{exp.to_latex()}}} \\cdot {self._ddx_latex(exp)}",
                )
                return Mul(Pow(base, exp), self.diff(exp))
            after = (f"{base.to_string()}^({exp.to_string()})*"
                     f"ln({base.to_string()})*{self._ddx(exp)}")
            self._record(
                "Exponential Rule", node, after,
                "Exponential rule (constant base a): d/dx[a^u] = a^u * ln(a) * u'.",
                f"{base.to_latex()}^{{{exp.to_latex()}}} \\cdot "
                f"\\ln\\left({base.to_latex()}\\right) \\cdot {self._ddx_latex(exp)}",
            )
            return Mul(Mul(Pow(base, exp), Func("ln", base)), self.diff(exp))

        # Case 3: both depend on the variable -> logarithmic differentiation.
        after = (f"{node.to_string()}*({self._ddx(exp)}*ln({base.to_string()})"
                 f" + {exp.to_string()}*{self._ddx(base)}/{base.to_string()})")
        self._record(
            "Generalised Power Rule", node, after,
            "When both base and exponent contain the variable, use logarithmic "
            "differentiation: d/dx[u^v] = u^v * (v'*ln(u) + v*u'/u).",
            f"{node.to_latex()} \\left({self._ddx_latex(exp)} \\cdot "
            f"\\ln\\left({base.to_latex()}\\right) + {exp.to_latex()} \\cdot "
            f"\\frac{{{self._ddx_latex(base)}}}{{{base.to_latex()}}}\\right)",
        )
        du = self.diff(base)
        dv = self.diff(exp)
        bracket = Add(Mul(dv, Func("ln", base)), Div(Mul(exp, du), base))
        return Mul(Pow(base, exp), bracket)

    def _diff_func(self, node: Func) -> Node:
        u = node.arg
        outer = _func_derivative(node.name, u)
        rule_text = _FUNC_RULE_TEXT.get(node.name, "")
        after = f"{outer.to_string()}*{self._ddx(u)}"
        self._record(
            "Chain Rule", node, after,
            f"Chain rule: d/dx[f(u)] = f'(u) * u'. Here {rule_text}.",
            f"{outer.to_latex()} \\cdot {self._ddx_latex(u)}",
        )
        return Mul(outer, self.diff(u))


def differentiate(expression: str, var: str = "x") -> dict:
    """Differentiate ``expression`` and return a structured result.

    Returns a dict with the parsed input, the raw and simplified derivatives in
    both plain-text and LaTeX, and the list of explanation steps.
    """
    tree = parse(expression, variable=var)
    engine = Differentiator(var=var)
    raw = engine.diff(tree)
    final = simplify(raw)

    # Append a closing simplification step when it actually changed something.
    raw_str = raw.to_string()
    final_str = final.to_string()
    if raw_str != final_str:
        engine.steps.append(Step(
            rule="Simplify",
            before=raw_str,
            after=final_str,
            explanation="Combine like terms and constants to tidy the result.",
            before_latex=raw.to_latex(),
            after_latex=final.to_latex(),
        ))

    return {
        "input": tree.to_string(),
        "input_latex": tree.to_latex(),
        "variable": var,
        "derivative": final_str,
        "derivative_latex": final.to_latex(),
        "derivative_unsimplified": raw_str,
        "steps": [s.as_dict() for s in engine.steps],
    }
