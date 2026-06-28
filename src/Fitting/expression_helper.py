"""Compile a string expression to a Python callable for the scipy fallback.

The Rust backend has its own parser (`bmri_fit.check_expression`). This
module mirrors the same syntax so curve_fit can be used when the Rust
wheel is not available. Keep both parsers in lock-step.
"""

from __future__ import annotations

import ast
from typing import Callable

import numpy as np

_ALLOWED_FUNCS = {
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pow": np.power,
    "min": np.minimum,
    "max": np.maximum,
}

_ALIASES = {
    "S0": 0,
    "T": 1,
    "T1": 1,
    "T2": 1,
    "T2s": 1,
    "T2star": 1,
    "T1rho": 1,
    "D": 1,
    "ADC": 1,
    "K": 2,
    "D_star": 2,
    "Dstar": 2,
    "T_long": 2,
    "Tl": 2,
}


def _alias_index(name: str, n_params: int) -> int | None:
    if name.startswith("p") and name[1:].isdigit():
        idx = int(name[1:])
        return idx if idx < n_params else None
    if name in _ALIASES:
        idx = _ALIASES[name]
        return idx if idx < n_params else None
    if name in ("offset", "C", "f", "alpha"):
        if n_params >= 4:
            return 3
        if n_params == 3:
            return 2
    return None


def _to_lambda(tree: ast.AST, n_params: int) -> Callable:
    """Walk a parsed AST and translate it into a numeric callable."""

    def walk(node: ast.AST):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            return lambda x, p, _v=value: _v
        if isinstance(node, ast.Name):
            if node.id == "x":
                return lambda x, p: x
            idx = _alias_index(node.id, n_params)
            if idx is None:
                raise ValueError(f"unknown identifier '{node.id}'")
            return lambda x, p, _i=idx: p[_i]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = walk(node.operand)
            return lambda x, p, _f=inner: -_f(x, p)
        if isinstance(node, ast.BinOp):
            left = walk(node.left)
            right = walk(node.right)
            if isinstance(node.op, ast.Add):
                return lambda x, p, l=left, r=right: l(x, p) + r(x, p)
            if isinstance(node.op, ast.Sub):
                return lambda x, p, l=left, r=right: l(x, p) - r(x, p)
            if isinstance(node.op, ast.Mult):
                return lambda x, p, l=left, r=right: l(x, p) * r(x, p)
            if isinstance(node.op, ast.Div):
                return lambda x, p, l=left, r=right: l(x, p) / r(x, p)
            if isinstance(node.op, ast.Pow):
                return lambda x, p, l=left, r=right: l(x, p) ** r(x, p)
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in _ALLOWED_FUNCS:
                raise ValueError(f"unsupported function '{name}'")
            args = [walk(a) for a in node.args]
            fn = _ALLOWED_FUNCS[name]
            return lambda x, p, _fn=fn, _args=args: _fn(*[a(x, p) for a in _args])
        raise ValueError(f"unsupported syntax node {type(node).__name__}")

    return walk(tree)


def compile_callable(expression: str, n_params: int) -> Callable:
    """Return a callable f(x, p0, p1, ..., pN-1) -> ndarray for the expression.

    scipy.curve_fit needs to introspect the signature to determine the
    parameter count, so we generate a function with explicit positional
    arguments rather than *args.
    """
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    walker = _to_lambda(tree, n_params)

    arg_names = ", ".join(f"p{i}" for i in range(n_params))
    tuple_args = ", ".join(f"p{i}" for i in range(n_params))
    src = f"def _generated(x, {arg_names}):\n    return _walker(x, ({tuple_args},))\n"
    ns: dict = {"_walker": walker}
    exec(src, ns)
    return ns["_generated"]
