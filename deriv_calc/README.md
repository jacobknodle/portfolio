# Derivative Calculator — Backend

A symbolic differentiation engine that computes the derivative of an expression
and explains every step, exposed as a small JSON HTTP API. It is the backend
for a derivative-calculator webpage: send it an expression, get back the
simplified derivative plus a worked, rule-by-rule solution.

The engine is written from scratch (its own tokenizer, parser, differentiator
and simplifier) and has **no third-party dependencies** — it runs on the Python
standard library alone. Flask is optional and only used by the production-style
server in `app.py`.

## What it can differentiate

* Polynomials and the power rule, including negative and fractional exponents
  (`x^5`, `x^(-3)`, `x^(1/2)`)
* Sums, differences, products (product rule) and quotients (quotient rule)
* The chain rule, to arbitrary nesting depth
* Trigonometric: `sin cos tan cot sec csc`
* Inverse trigonometric: `asin acos atan acot asec acsc` (also `arcsin`, …)
* Hyperbolic and inverse hyperbolic: `sinh cosh tanh coth sech csch asinh acosh atanh`
* Exponentials: `exp(x)`, `e^x`, and constant bases such as `2^x`
* Logarithms: `ln`, `log` (base 10), `log10`, `log2`
* Roots and `abs`: `sqrt`, `cbrt`, `abs`
* The fully general case where both base and exponent contain the variable
  (`x^x`, `x^sin(x)`), handled by logarithmic differentiation
* Constants `pi`, `e`, `tau`, and differentiation with respect to any variable

Input is forgiving: implicit multiplication (`2x`, `3sin(x)`, `(x+1)(x-2)`),
`^` for powers, and common function aliases (`arctan`, `loge`, …) all work.

## Running it

No dependencies required:

```bash
cd deriv_calc
python server.py          # serves http://127.0.0.1:5000
```

Or with Flask (production-style):

```bash
pip install -r requirements.txt
python app.py
```

From the command line:

```bash
python cli.py "x^2 * sin(x)"
python cli.py "x^x" --var x
```

## HTTP API

### `GET /api/health`

Returns service info and the list of supported functions.

### `POST /api/derivative`

Request body:

```json
{ "expression": "x^2 * sin(x)", "variable": "x", "at": 1.0 }
```

`variable` (default `"x"`) and `at` (evaluate f and f′ at a point) are optional.

Response:

```json
{
  "ok": true,
  "input": "x^2*sin(x)",
  "input_latex": "x^2 \\cdot \\sin\\left(x\\right)",
  "variable": "x",
  "derivative": "2*x*sin(x) + x^2*cos(x)",
  "derivative_latex": "2 \\cdot x \\cdot \\sin\\left(x\\right) + ...",
  "derivative_unsimplified": "2*x^(1)*1*sin(x) + x^2*cos(x)*1",
  "steps": [
    {
      "rule": "Product Rule",
      "before": "d/dx[x^2*sin(x)]",
      "after": "d/dx[x^2]*sin(x) + x^2*d/dx[sin(x)]",
      "explanation": "Product rule: (u*v)' = u'*v + u*v'.",
      "before_latex": "...",
      "after_latex": "..."
    }
  ]
}
```

Each step carries a short `rule` name, the sub-expression being differentiated
(`before`), the result of applying the rule (`after`) with not-yet-evaluated
parts shown as `d/dx[...]`, a prose `explanation`, and LaTeX renderings for a
front-end to typeset. Errors return HTTP 400 with `{"ok": false, "error": ...}`.

Example with `curl`:

```bash
curl -s http://127.0.0.1:5000/api/derivative \
  -H 'Content-Type: application/json' \
  -d '{"expression": "(x^2+1)/(x-3)"}'
```

## How it works

```
expression string
      │  tokenizer.py     -> list of tokens
      │  parser.py        -> expression tree (nodes.py)
      │  differentiate.py -> derivative tree + ordered list of Steps
      │  simplify.py      -> tidy, value-equivalent result
      ▼
JSON result (api.py)  ──>  app.py (Flask)  or  server.py (stdlib)
```

The expression tree uses binary nodes because the textbook differentiation
rules map cleanly onto them, which keeps the generated explanation faithful to
how the rules are taught.

## Tests

```bash
cd deriv_calc
python -m unittest discover -s tests
```

Correctness is verified *numerically*: every symbolic derivative is checked
against a central finite-difference approximation of the original function at
several points, so a genuine mathematical error fails the suite regardless of
formatting. The API layer and parser error handling are tested separately.
