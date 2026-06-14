"""Framework-agnostic request handling for the derivative calculator.

The actual web servers (``app.py`` for Flask, ``server.py`` for the dependency
-free standard-library server) are thin wrappers around the functions here.
Keeping the logic framework-agnostic means it is easy to test and easy to host
behind whatever HTTP layer you prefer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from calculus import (
    DifferentiationError, EvaluationError, ParseError, TokenizeError,
    differentiate, evaluate, parse,
)

# A reasonable cap so a pathological request cannot tie up the server.
MAX_EXPRESSION_LENGTH = 2000


def _bad_request(message: str) -> Tuple[int, Dict[str, Any]]:
    return 400, {"ok": False, "error": message}


def handle_differentiate(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """Compute a derivative with steps.

    Expected JSON payload::

        {"expression": "x^2 * sin(x)", "variable": "x"}

    Returns ``(status_code, body)``.
    """
    if not isinstance(payload, dict):
        return _bad_request("Request body must be a JSON object.")

    expression = payload.get("expression")
    variable = payload.get("variable", "x")

    if not isinstance(expression, str) or not expression.strip():
        return _bad_request("Field 'expression' is required and must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return _bad_request(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters.")
    if not isinstance(variable, str) or not variable.isidentifier():
        return _bad_request("Field 'variable' must be a valid identifier (e.g. 'x').")

    try:
        result = differentiate(expression, var=variable)
    except (ParseError, TokenizeError) as exc:
        return _bad_request(f"Could not parse expression: {exc}")
    except DifferentiationError as exc:
        return _bad_request(f"Could not differentiate expression: {exc}")
    except RecursionError:
        return _bad_request("Expression is too deeply nested.")

    body: Dict[str, Any] = {"ok": True, **result}

    # Optional: evaluate the original and the derivative at a point.
    if "at" in payload and payload["at"] is not None:
        body["evaluation"] = _evaluate_at(expression, result["derivative"],
                                          variable, payload["at"])
    return 200, body


def _evaluate_at(expression: str, derivative: str, variable: str,
                 at: Any) -> Dict[str, Any]:
    try:
        point = float(at)
    except (TypeError, ValueError):
        return {"ok": False, "error": "'at' must be a number."}
    env = {variable: point}
    out: Dict[str, Any] = {"point": point}
    try:
        out["value"] = evaluate(parse(expression, variable), env)
    except (EvaluationError, ZeroDivisionError, ValueError) as exc:
        out["value_error"] = str(exc)
    try:
        out["derivative_value"] = evaluate(parse(derivative, variable), env)
    except (EvaluationError, ZeroDivisionError, ValueError) as exc:
        out["derivative_value_error"] = str(exc)
    # Replace any non-finite floats so the JSON stays valid.
    for k, v in list(out.items()):
        if isinstance(v, float) and not math.isfinite(v):
            out.pop(k)
            out[f"{k}_error"] = "result is undefined at this point"
    return out


def supported_functions() -> List[str]:
    from calculus.nodes import KNOWN_FUNCTIONS
    return sorted(KNOWN_FUNCTIONS)


def health() -> Dict[str, Any]:
    return {"ok": True, "service": "deriv_calc", "functions": supported_functions()}
