"""Recursive-descent parser building an expression tree from tokens.

Grammar (lowest to highest precedence)::

    expr     := term (("+" | "-") term)*
    term     := factor (("*" | "/") factor)*           # also implicit mult.
    factor   := ("+" | "-") factor | power
    power    := atom ("^" factor)?                     # right associative
    atom     := NUMBER
              | NAME                                    # variable or constant
              | NAME "(" expr ("," expr)* ")"           # function call
              | "(" expr ")"

Implicit multiplication is supported wherever it is unambiguous, e.g. ``2x``,
``3sin(x)``, ``(x+1)(x-2)`` and ``2pi`` all parse as products.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional

from .nodes import (
    Add, Const, Div, Func, KNOWN_CONSTANTS, KNOWN_FUNCTIONS, Mul, Neg, Node,
    Num, Pow, Sub, Var,
)
from .tokenizer import Token, tokenize


class ParseError(ValueError):
    """Raised when the token stream does not form a valid expression."""


# Aliases that let users type natural variants which map to a canonical name.
_FUNC_ALIASES = {
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "arccot": "acot", "arcsec": "asec", "arccsc": "acsc",
    "arcsinh": "asinh", "arccosh": "acosh", "arctanh": "atanh",
    "loge": "ln",
}


class Parser:
    def __init__(self, tokens: List[Token], variable: str = "x"):
        self.tokens = tokens
        self.pos = 0
        self.variable = variable

    # -- token stream helpers ---------------------------------------------
    def _peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            got = "end of input" if tok is None else f"{tok.value!r}"
            raise ParseError(f"Expected {kind} but found {got}")
        return self._advance()

    # -- grammar rules ----------------------------------------------------
    def parse(self) -> Node:
        node = self._expr()
        if self._peek() is not None:
            tok = self._peek()
            raise ParseError(f"Unexpected token {tok.value!r} at position {tok.pos}")
        return node

    def _expr(self) -> Node:
        node = self._term()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in "+-":
                self._advance()
                rhs = self._term()
                node = Add(node, rhs) if tok.value == "+" else Sub(node, rhs)
            else:
                return node

    def _term(self) -> Node:
        node = self._factor()
        while True:
            tok = self._peek()
            if tok and tok.kind == "OP" and tok.value in "*/":
                self._advance()
                rhs = self._factor()
                node = Mul(node, rhs) if tok.value == "*" else Div(node, rhs)
            elif self._starts_implicit_factor(tok):
                # Implicit multiplication: 2x, 3sin(x), (x+1)(x-2), 2pi.
                rhs = self._factor()
                node = Mul(node, rhs)
            else:
                return node

    def _starts_implicit_factor(self, tok: Optional[Token]) -> bool:
        if tok is None:
            return False
        return tok.kind in ("NUMBER", "NAME", "LPAREN")

    def _factor(self) -> Node:
        tok = self._peek()
        if tok and tok.kind == "OP" and tok.value in "+-":
            self._advance()
            operand = self._factor()
            return Neg(operand) if tok.value == "-" else operand
        return self._power()

    def _power(self) -> Node:
        base = self._atom()
        tok = self._peek()
        if tok and tok.kind == "OP" and tok.value == "^":
            self._advance()
            # Right-associative, and the exponent may itself be signed: x^-2.
            exp = self._factor()
            return Pow(base, exp)
        return base

    def _atom(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of input")

        if tok.kind == "NUMBER":
            self._advance()
            return Num(_parse_number(tok.value))

        if tok.kind == "LPAREN":
            self._advance()
            node = self._expr()
            self._expect("RPAREN")
            return node

        if tok.kind == "NAME":
            return self._name()

        raise ParseError(f"Unexpected token {tok.value!r} at position {tok.pos}")

    def _name(self) -> Node:
        tok = self._advance()
        name = tok.value
        canonical = _FUNC_ALIASES.get(name, name)

        # Function application: name immediately followed by '('.
        nxt = self._peek()
        if canonical in KNOWN_FUNCTIONS and nxt and nxt.kind == "LPAREN":
            self._advance()  # consume '('
            arg = self._expr()
            # Accept a 2-argument log(base, x) -> log_base, otherwise single arg.
            if self._peek() and self._peek().kind == "COMMA":
                raise ParseError(
                    "Multi-argument functions are not supported; "
                    "use ln(x), log10(x) or log2(x)."
                )
            self._expect("RPAREN")
            return Func(canonical, arg)

        if name in KNOWN_CONSTANTS:
            return Const(name)

        if canonical in KNOWN_FUNCTIONS:
            raise ParseError(f"Function {name!r} must be called with parentheses")

        # Treat any other identifier as a (single-letter style) variable. Each
        # character lets users write multi-variable-looking input, but we keep
        # it simple: the whole identifier is one variable name.
        return Var(name)


def _parse_number(text: str) -> Fraction:
    if "." in text:
        # Keep decimals exact, e.g. "0.25" -> 1/4.
        return Fraction(text)
    return Fraction(int(text))


def parse(text: str, variable: str = "x") -> Node:
    """Parse ``text`` into an expression tree.

    Raises :class:`ParseError` (or :class:`TokenizeError`) on malformed input.
    """
    if not text or not text.strip():
        raise ParseError("Empty expression")
    tokens = tokenize(text)
    return Parser(tokens, variable=variable).parse()
