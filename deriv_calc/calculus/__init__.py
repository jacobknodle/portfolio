"""Symbolic differentiation engine for the derivative calculator.

Public API
----------
``differentiate(expression, var="x")``  -> structured result with steps
``parse(expression, var="x")``          -> expression tree
``simplify(node)``                       -> simplified expression tree
``evaluate(node, env)``                  -> numeric value
"""

from .differentiate import (
    DifferentiationError, Differentiator, Step, differentiate,
)
from .evaluate import EvaluationError, evaluate
from .parser import ParseError, parse
from .simplify import simplify
from .tokenizer import TokenizeError

__all__ = [
    "differentiate",
    "Differentiator",
    "Step",
    "DifferentiationError",
    "parse",
    "ParseError",
    "TokenizeError",
    "simplify",
    "evaluate",
    "EvaluationError",
]
