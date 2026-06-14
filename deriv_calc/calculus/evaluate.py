"""Numeric evaluation of expression trees.

This module exists mainly so the test-suite can verify correctness: we compare
the symbolic derivative against a finite-difference approximation of the
original function at several points. It is also exposed through the API so a
front-end can plot or sanity-check results.
"""

from __future__ import annotations

import math
from typing import Dict

from .nodes import (
    Add, Const, Div, Func, Mul, Neg, Node, Num, Pow, Sub, Var,
)

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "cot": lambda x: 1.0 / math.tan(x),
    "sec": lambda x: 1.0 / math.cos(x),
    "csc": lambda x: 1.0 / math.sin(x),
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "acot": lambda x: math.atan(1.0 / x) if x != 0 else math.pi / 2,
    "asec": lambda x: math.acos(1.0 / x),
    "acsc": lambda x: math.asin(1.0 / x),
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "coth": lambda x: math.cosh(x) / math.sinh(x),
    "sech": lambda x: 1.0 / math.cosh(x),
    "csch": lambda x: 1.0 / math.sinh(x),
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "exp": math.exp, "ln": math.log,
    "log": math.log10, "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "abs": abs,
}


class EvaluationError(ValueError):
    """Raised when an expression cannot be evaluated at a given point."""


def evaluate(node: Node, env: Dict[str, float]) -> float:
    """Evaluate ``node`` numerically given variable bindings in ``env``."""
    if isinstance(node, Num):
        return float(node.value)
    if isinstance(node, Const):
        return _CONSTANTS[node.name]
    if isinstance(node, Var):
        if node.name not in env:
            raise EvaluationError(f"No value supplied for variable {node.name!r}")
        return float(env[node.name])
    if isinstance(node, Neg):
        return -evaluate(node.operand, env)
    if isinstance(node, Add):
        return evaluate(node.left, env) + evaluate(node.right, env)
    if isinstance(node, Sub):
        return evaluate(node.left, env) - evaluate(node.right, env)
    if isinstance(node, Mul):
        return evaluate(node.left, env) * evaluate(node.right, env)
    if isinstance(node, Div):
        return evaluate(node.left, env) / evaluate(node.right, env)
    if isinstance(node, Pow):
        return evaluate(node.base, env) ** evaluate(node.exp, env)
    if isinstance(node, Func):
        fn = _FUNCS.get(node.name)
        if fn is None:
            raise EvaluationError(f"Unknown function {node.name!r}")
        return fn(evaluate(node.arg, env))
    raise EvaluationError(f"Cannot evaluate node of type {type(node).__name__}")
