"""Expression tree (AST) for the derivative calculator.

Every mathematical expression is represented as an immutable tree of ``Node``
objects. The nodes carry no calculus logic themselves; differentiation,
simplification and evaluation are implemented as functions that walk the tree
in the sibling modules. Keeping the data and the algorithms separate makes the
step-by-step differentiator easy to reason about and test.

Node types
----------
Num     numeric literal (stored as an exact ``Fraction`` when possible)
Const   named mathematical constant such as ``pi`` or ``e``
Var     a variable, e.g. ``x``
Neg     unary negation, ``-u``
Add     binary sum,        ``u + v``
Sub     binary difference, ``u - v``
Mul     binary product,    ``u * v``
Div     binary quotient,   ``u / v``
Pow     power,             ``u ^ v``
Func    unary function,    ``sin(u)``, ``ln(u)`` ...

The tree is intentionally binary (rather than n-ary) because the textbook
differentiation rules - product rule, quotient rule, chain rule - map cleanly
onto binary nodes, which keeps the generated explanation close to how the rules
are taught.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, Union

Number = Union[int, Fraction]

# Operator precedence used when rendering, so we only emit parentheses we need.
_PREC = {
    "Add": 1,
    "Sub": 1,
    "Mul": 2,
    "Div": 2,
    "Neg": 3,
    "Pow": 4,
    "atom": 5,
}

# Functions the calculator understands. The value is used only for display of
# the canonical name; differentiation rules live in ``differentiate.py``.
KNOWN_FUNCTIONS = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh",
    "exp", "ln", "log", "log10", "log2", "sqrt", "cbrt", "abs",
}

# Named constants understood by the parser/evaluator.
KNOWN_CONSTANTS = {"pi", "e", "tau"}


class Node:
    """Base class for every expression-tree node."""

    precedence: int = _PREC["atom"]

    # -- structural helpers ------------------------------------------------
    def children(self) -> Tuple["Node", ...]:
        return ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def key(self):  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    # -- rendering ---------------------------------------------------------
    def to_string(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_latex(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_string()

    def _wrap(self, child: "Node", *, strict: bool = False) -> str:
        """Render ``child`` adding parentheses only when precedence requires it."""
        text = child.to_string()
        needs = child.precedence < self.precedence
        if strict and child.precedence == self.precedence:
            needs = True
        return f"({text})" if needs else text

    def _wrap_latex(self, child: "Node", *, strict: bool = False) -> str:
        text = child.to_latex()
        needs = child.precedence < self.precedence
        if strict and child.precedence == self.precedence:
            needs = True
        return f"\\left({text}\\right)" if needs else text


@dataclass(frozen=True, eq=False)
class Num(Node):
    value: Number

    def __post_init__(self):
        # Normalise to Fraction so arithmetic stays exact.
        if not isinstance(self.value, Fraction):
            object.__setattr__(self, "value", Fraction(self.value))

    precedence = _PREC["atom"]

    def key(self):
        return ("Num", self.value)

    def is_integer(self) -> bool:
        return self.value.denominator == 1

    def to_string(self) -> str:
        if self.is_integer():
            return str(self.value.numerator)
        return f"{self.value.numerator}/{self.value.denominator}"

    def to_latex(self) -> str:
        if self.is_integer():
            return str(self.value.numerator)
        return f"\\frac{{{self.value.numerator}}}{{{self.value.denominator}}}"


@dataclass(frozen=True, eq=False)
class Const(Node):
    name: str
    precedence = _PREC["atom"]

    def key(self):
        return ("Const", self.name)

    def to_string(self) -> str:
        return self.name

    def to_latex(self) -> str:
        return {"pi": "\\pi", "tau": "\\tau", "e": "e"}.get(self.name, self.name)


@dataclass(frozen=True, eq=False)
class Var(Node):
    name: str
    precedence = _PREC["atom"]

    def key(self):
        return ("Var", self.name)

    def to_string(self) -> str:
        return self.name

    def to_latex(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class Neg(Node):
    operand: Node
    precedence = _PREC["Neg"]

    def children(self):
        return (self.operand,)

    def key(self):
        return ("Neg", self.operand.key())

    def to_string(self) -> str:
        return f"-{self._wrap(self.operand)}"

    def to_latex(self) -> str:
        return f"-{self._wrap_latex(self.operand)}"


@dataclass(frozen=True, eq=False)
class Add(Node):
    left: Node
    right: Node
    precedence = _PREC["Add"]

    def children(self):
        return (self.left, self.right)

    def key(self):
        return ("Add", self.left.key(), self.right.key())

    def to_string(self) -> str:
        # Render "a + (-b)" as "a - b" for readability.
        if isinstance(self.right, Neg):
            return f"{self._wrap(self.left)} - {self._wrap(self.right.operand)}"
        return f"{self._wrap(self.left)} + {self._wrap(self.right)}"

    def to_latex(self) -> str:
        if isinstance(self.right, Neg):
            return f"{self._wrap_latex(self.left)} - {self._wrap_latex(self.right.operand)}"
        return f"{self._wrap_latex(self.left)} + {self._wrap_latex(self.right)}"


@dataclass(frozen=True, eq=False)
class Sub(Node):
    left: Node
    right: Node
    precedence = _PREC["Sub"]

    def children(self):
        return (self.left, self.right)

    def key(self):
        return ("Sub", self.left.key(), self.right.key())

    def to_string(self) -> str:
        return f"{self._wrap(self.left)} - {self._wrap(self.right, strict=True)}"

    def to_latex(self) -> str:
        return f"{self._wrap_latex(self.left)} - {self._wrap_latex(self.right, strict=True)}"


@dataclass(frozen=True, eq=False)
class Mul(Node):
    left: Node
    right: Node
    precedence = _PREC["Mul"]

    def children(self):
        return (self.left, self.right)

    def key(self):
        return ("Mul", self.left.key(), self.right.key())

    def to_string(self) -> str:
        return f"{self._wrap(self.left)}*{self._wrap(self.right)}"

    def to_latex(self) -> str:
        return f"{self._wrap_latex(self.left)} \\cdot {self._wrap_latex(self.right)}"


@dataclass(frozen=True, eq=False)
class Div(Node):
    left: Node
    right: Node
    precedence = _PREC["Div"]

    def children(self):
        return (self.left, self.right)

    def key(self):
        return ("Div", self.left.key(), self.right.key())

    def to_string(self) -> str:
        return f"{self._wrap(self.left)}/{self._wrap(self.right, strict=True)}"

    def to_latex(self) -> str:
        return f"\\frac{{{self.left.to_latex()}}}{{{self.right.to_latex()}}}"


@dataclass(frozen=True, eq=False)
class Pow(Node):
    base: Node
    exp: Node
    precedence = _PREC["Pow"]

    def children(self):
        return (self.base, self.exp)

    def key(self):
        return ("Pow", self.base.key(), self.exp.key())

    def to_string(self) -> str:
        # Exponentiation is right-associative, so the base needs parentheses at
        # equal precedence. The exponent must be parenthesised unless it is a
        # plain atom, otherwise "x^-1/2" would re-parse as "(x^-1)/2".
        exp = self.exp
        simple_exp = (
            (isinstance(exp, Num) and exp.is_integer() and exp.value >= 0)
            or isinstance(exp, (Var, Const))
        )
        exp_str = exp.to_string() if simple_exp else f"({exp.to_string()})"
        return f"{self._wrap(self.base, strict=True)}^{exp_str}"

    def to_latex(self) -> str:
        return f"{self._wrap_latex(self.base, strict=True)}^{{{self.exp.to_latex()}}}"


@dataclass(frozen=True, eq=False)
class Func(Node):
    name: str
    arg: Node
    precedence = _PREC["atom"]

    def children(self):
        return (self.arg,)

    def key(self):
        return ("Func", self.name, self.arg.key())

    def to_string(self) -> str:
        return f"{self.name}({self.arg.to_string()})"

    def to_latex(self) -> str:
        arg = self.arg.to_latex()
        if self.name == "sqrt":
            return f"\\sqrt{{{arg}}}"
        if self.name == "cbrt":
            return f"\\sqrt[3]{{{arg}}}"
        if self.name == "abs":
            return f"\\left|{arg}\\right|"
        if self.name == "ln":
            return f"\\ln\\left({arg}\\right)"
        if self.name == "exp":
            return f"e^{{{arg}}}"
        trig = {
            "sin", "cos", "tan", "cot", "sec", "csc",
            "sinh", "cosh", "tanh", "log",
        }
        if self.name in trig:
            return f"\\{self.name}\\left({arg}\\right)"
        return f"\\operatorname{{{self.name}}}\\left({arg}\\right)"


# -- convenience constructors ---------------------------------------------
ZERO = Num(0)
ONE = Num(1)
NEG_ONE = Num(-1)
TWO = Num(2)


def num(value: Number) -> Num:
    return Num(value)


def is_num(node: Node, value: Number | None = None) -> bool:
    if not isinstance(node, Num):
        return False
    return value is None or node.value == Fraction(value)
