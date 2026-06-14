"""Command-line interface for the derivative calculator.

Examples::

    python cli.py "x^2 * sin(x)"
    python cli.py "x^x" --var x
    python cli.py "(x^2+1)/(x-3)" --no-steps
"""

from __future__ import annotations

import argparse
import sys

from calculus import DifferentiationError, ParseError, TokenizeError, differentiate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Differentiate an expression, with steps.")
    parser.add_argument("expression", help="the expression to differentiate, e.g. 'x^2*sin(x)'")
    parser.add_argument("--var", default="x", help="the variable to differentiate with respect to")
    parser.add_argument("--no-steps", action="store_true", help="print only the final derivative")
    args = parser.parse_args(argv)

    try:
        result = differentiate(args.expression, var=args.var)
    except (ParseError, TokenizeError, DifferentiationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"f({args.var})  = {result['input']}")
    print(f"f'({args.var}) = {result['derivative']}")

    if not args.no_steps:
        print("\nSteps:")
        for i, step in enumerate(result["steps"], 1):
            print(f"  {i}. [{step['rule']}] {step['before']} = {step['after']}")
            print(f"     {step['explanation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
