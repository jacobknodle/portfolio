"""Turn a raw expression string into a flat list of tokens.

The tokenizer is deliberately small: it recognises numbers, identifiers
(variables, function names and named constants), operators and parentheses.
Everything else - operator precedence, implicit multiplication, function
application - is handled by the parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


class TokenizeError(ValueError):
    """Raised when the input contains a character we cannot tokenize."""


@dataclass(frozen=True)
class Token:
    kind: str   # NUMBER, NAME, OP, LPAREN, RPAREN, COMMA
    value: str
    pos: int    # index in the source string, for error messages


_OPERATORS = set("+-*/^")


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch.isspace():
            i += 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            start = i
            seen_dot = False
            while i < n and (text[i].isdigit() or text[i] == "."):
                if text[i] == ".":
                    if seen_dot:
                        raise TokenizeError(f"Malformed number at position {start}")
                    seen_dot = True
                i += 1
            tokens.append(Token("NUMBER", text[start:i], start))
            continue

        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(Token("NAME", text[start:i], start))
            continue

        if ch in _OPERATORS:
            tokens.append(Token("OP", ch, i))
            i += 1
            continue

        if ch == "(":
            tokens.append(Token("LPAREN", ch, i))
            i += 1
            continue

        if ch == ")":
            tokens.append(Token("RPAREN", ch, i))
            i += 1
            continue

        if ch == ",":
            tokens.append(Token("COMMA", ch, i))
            i += 1
            continue

        raise TokenizeError(f"Unexpected character {ch!r} at position {i}")

    return tokens
