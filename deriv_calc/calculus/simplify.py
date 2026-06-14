"""Algebraic simplification of expression trees.

The differentiator produces correct but verbose results - things like
``1*x^(2-1) + 0`` - so we run a simplifier afterwards to present a tidy answer.
Simplification is purely cosmetic: it never changes the mathematical value of an
expression, which the test-suite checks numerically.

The two interesting passes are:

* :func:`_simplify_sum`    - flatten nested additions/subtractions, fold
  numeric constants and combine like terms (``2x + 3x -> 5x``).
* :func:`_simplify_product`- flatten nested products, fold numeric factors and
  combine repeated factors into powers (``x*x -> x^2``).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from .nodes import (
    Add, Const, Div, Func, Mul, Neg, Node, Num, Pow, Sub, Var, ZERO, ONE,
)


def simplify(node: Node) -> Node:
    """Return a simplified, value-equivalent copy of ``node``."""
    prev = None
    cur = node
    # A few simplifications expose further opportunities (e.g. folding a power
    # reveals a constant that a surrounding product can absorb), so iterate to
    # a fixed point. The tree shrinks each pass, so this always terminates.
    for _ in range(50):
        cur = _simplify_once(cur)
        if cur == prev:
            break
        prev = cur
    return cur


def _simplify_once(node: Node) -> Node:
    if isinstance(node, (Num, Var, Const)):
        return node

    if isinstance(node, Neg):
        return _simplify_neg(node)
    if isinstance(node, (Add, Sub)):
        return _simplify_sum(node)
    if isinstance(node, Mul):
        return _simplify_product(node)
    if isinstance(node, Div):
        return _simplify_div(node)
    if isinstance(node, Pow):
        return _simplify_pow(node)
    if isinstance(node, Func):
        return Func(node.name, _simplify_once(node.arg))
    return node


# -- negation --------------------------------------------------------------
def _simplify_neg(node: Neg) -> Node:
    operand = _simplify_once(node.operand)
    if isinstance(operand, Num):
        return Num(-operand.value)
    if isinstance(operand, Neg):
        return operand.operand
    return Neg(operand)


# -- sums ------------------------------------------------------------------
def _flatten_sum(node: Node, sign: int, terms: List[Tuple[int, Node]]) -> None:
    if isinstance(node, Add):
        _flatten_sum(node.left, sign, terms)
        _flatten_sum(node.right, sign, terms)
    elif isinstance(node, Sub):
        _flatten_sum(node.left, sign, terms)
        _flatten_sum(node.right, -sign, terms)
    elif isinstance(node, Neg):
        _flatten_sum(node.operand, -sign, terms)
    else:
        terms.append((sign, node))


def _split_coefficient(node: Node) -> Tuple[Fraction, Node | None]:
    """Split ``node`` into (numeric coefficient, remaining factor tree)."""
    if isinstance(node, Num):
        return node.value, None
    if isinstance(node, Mul):
        c_left, rest_left = _split_coefficient(node.left)
        c_right, rest_right = _split_coefficient(node.right)
        coeff = c_left * c_right
        if rest_left is None:
            rest = rest_right
        elif rest_right is None:
            rest = rest_left
        else:
            rest = Mul(rest_left, rest_right)
        return coeff, rest
    return Fraction(1), node


def _simplify_sum(node: Node) -> Node:
    raw: List[Tuple[int, Node]] = []
    _flatten_sum(node, 1, raw)

    constant = Fraction(0)
    # Map a term's structural key -> [coefficient, representative node].
    grouped: Dict[object, list] = {}
    order: List[object] = []

    for sign, term in raw:
        term = _simplify_once(term)
        coeff, rest = _split_coefficient(term)
        coeff *= sign
        if rest is None:
            constant += coeff
            continue
        k = rest.key()
        if k not in grouped:
            grouped[k] = [Fraction(0), rest]
            order.append(k)
        grouped[k][0] += coeff

    pieces: List[Node] = []
    for k in order:
        coeff, rest = grouped[k]
        if coeff == 0:
            continue
        pieces.append(_coeff_times(coeff, rest))

    if constant != 0:
        pieces.append(Num(constant))

    if not pieces:
        return ZERO
    return _rebuild_sum(pieces)


def _coeff_times(coeff: Fraction, rest: Node) -> Node:
    if coeff == 1:
        return rest
    if coeff == -1:
        return Neg(rest)
    return Mul(Num(coeff), rest)


def _rebuild_sum(pieces: List[Node]) -> Node:
    result = pieces[0]
    for piece in pieces[1:]:
        if isinstance(piece, Neg):
            result = Sub(result, piece.operand)
        elif isinstance(piece, Num) and piece.value < 0:
            result = Sub(result, Num(-piece.value))
        elif isinstance(piece, Mul) and isinstance(piece.left, Num) and piece.left.value < 0:
            result = Sub(result, Mul(Num(-piece.left.value), piece.right))
        else:
            result = Add(result, piece)
    return result


# -- products --------------------------------------------------------------
def _flatten_product(node: Node, factors: List[Node], sign: list) -> None:
    if isinstance(node, Mul):
        _flatten_product(node.left, factors, sign)
        _flatten_product(node.right, factors, sign)
    elif isinstance(node, Neg):
        sign[0] *= -1
        _flatten_product(node.operand, factors, sign)
    else:
        factors.append(node)


def _simplify_product(node: Node) -> Node:
    factors: List[Node] = []
    sign = [1]
    _flatten_product(node, factors, sign)

    coeff = Fraction(sign[0])
    # Map a base's key -> [exponent-sum, base node].
    powers: Dict[object, list] = {}
    order: List[object] = []

    for f in factors:
        f = _simplify_once(f)
        if isinstance(f, Num):
            coeff *= f.value
            continue
        base, exp = _as_power(f)
        k = base.key()
        if k not in powers:
            powers[k] = [Fraction(0), base, False]
            order.append(k)
        if exp is None:
            powers[k][2] = True  # non-numeric exponent, keep separate
            powers[k][0] += 1
        else:
            powers[k][0] += exp

    if coeff == 0:
        return ZERO

    built: List[Node] = []
    for k in order:
        exp_sum, base, _ = powers[k]
        if exp_sum == 0:
            continue
        if exp_sum == 1:
            built.append(base)
        else:
            built.append(Pow(base, Num(exp_sum)))

    if not built:
        return Num(coeff)

    result = built[0]
    for b in built[1:]:
        result = Mul(result, b)

    if coeff == 1:
        return result
    if coeff == -1:
        return Neg(result)
    return Mul(Num(coeff), result)


def _as_power(node: Node) -> Tuple[Node, Fraction | None]:
    """Return (base, exponent) where exponent is a Fraction if numeric."""
    if isinstance(node, Pow) and isinstance(node.exp, Num):
        return node.base, node.exp.value
    return node, None


# -- division --------------------------------------------------------------
def _simplify_div(node: Div) -> Node:
    left = _simplify_once(node.left)
    right = _simplify_once(node.right)

    if isinstance(right, Num) and right.value == 1:
        return left
    if isinstance(left, Num) and left.value == 0:
        return ZERO
    if isinstance(left, Num) and isinstance(right, Num) and right.value != 0:
        return Num(left.value / right.value)
    if left == right:
        return ONE
    return Div(left, right)


# -- powers ----------------------------------------------------------------
def _simplify_pow(node: Pow) -> Node:
    base = _simplify_once(node.base)
    exp = _simplify_once(node.exp)

    if isinstance(exp, Num):
        if exp.value == 0:
            return ONE
        if exp.value == 1:
            return base
    if isinstance(base, Num):
        if base.value == 1:
            return ONE
        if base.value == 0 and isinstance(exp, Num) and exp.value > 0:
            return ZERO
        if isinstance(exp, Num) and exp.value.denominator == 1:
            # Fold integer powers of numeric bases, e.g. 2^3 -> 8.
            p = exp.value.numerator
            if p >= 0:
                return Num(base.value ** p)
            return Num(Fraction(1) / (base.value ** (-p)))
    # (a^b)^c -> a^(b*c) when it is safe (numeric exponents).
    if isinstance(base, Pow) and isinstance(base.exp, Num) and isinstance(exp, Num):
        return _simplify_pow(Pow(base.base, Num(base.exp.value * exp.value)))
    return Pow(base, exp)
