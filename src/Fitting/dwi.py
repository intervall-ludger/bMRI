"""Diffusion-weighted imaging fitters: ADC, Kurtosis, IVIM.

All classes pass the same Rust backend used by T2/T2*; switch between
backends with `method="rust" | "curvefit"`.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numba import njit

from Fitting.AbstractFitting import AbstractFitting, cpu_count


@njit
def _adc_model(x: np.ndarray, S0: float, D: float, offset: float) -> np.ndarray:
    return S0 * np.exp(-x * D) + offset


@njit
def _kurtosis_model(x: np.ndarray, S0: float, D: float, K: float) -> np.ndarray:
    return S0 * np.exp(-x * D + (x * D) ** 2 * K / 6.0)


def _ivim_model(x: np.ndarray, S0: float, D: float, D_star: float, f: float) -> np.ndarray:
    return S0 * (f * np.exp(-x * D_star) + (1.0 - f) * np.exp(-x * D))


def _t2star_biexp_model(
    x: np.ndarray, S0: float, T_short: float, T_long: float, f: float
) -> np.ndarray:
    return S0 * (f * np.exp(-x / T_short) + (1.0 - f) * np.exp(-x / T_long))


class DWIMonoExp(AbstractFitting):
    """ADC mapping with S(b) = S0 * exp(-b*D) + offset.

    Default bounds: S0 in [0, 10], D in [0, 5e-3] mm^2/s, offset in [-1, 1].
    """

    def __init__(
        self,
        dim: int = 3,
        boundary: Optional[Tuple] = None,
        normalize: bool = True,
    ):
        if boundary is None:
            boundary = ([0.0, 0.0, -1.0], [10.0, 5e-3, 1.0])
        super().__init__(
            _adc_model,
            boundary=boundary,
            normalize=normalize,
            rust_model="dwi_mono_exp",
        )
        self.dim = dim


class DWIKurtosis(AbstractFitting):
    """Kurtosis: S(b) = S0 * exp(-b*D + (b*D)^2 * K/6).

    Default bounds: S0 in [0, 10], D in [0, 5e-3] mm^2/s, K in [0, 3].
    """

    def __init__(
        self,
        dim: int = 3,
        boundary: Optional[Tuple] = None,
        normalize: bool = True,
    ):
        if boundary is None:
            boundary = ([0.0, 0.0, 0.0], [10.0, 5e-3, 3.0])
        super().__init__(
            _kurtosis_model,
            boundary=boundary,
            normalize=normalize,
            rust_model="dwi_kurtosis",
        )
        self.dim = dim


class DWIIvim(AbstractFitting):
    """IVIM bi-exponential: S(b) = S0 * (f*exp(-b*D*) + (1-f)*exp(-b*D)).

    4 parameters: S0, D, D*, f. Default bounds suitable for abdominal IVIM:
    D in [0, 5e-3], D* in [5e-3, 1e-1] mm^2/s, f in [0, 1].
    """

    def __init__(
        self,
        dim: int = 3,
        boundary: Optional[Tuple] = None,
        normalize: bool = True,
    ):
        if boundary is None:
            boundary = (
                [0.0, 0.0, 5e-3, 0.0],
                [10.0, 5e-3, 1e-1, 1.0],
            )
        super().__init__(
            _ivim_model,
            boundary=boundary,
            normalize=normalize,
            rust_model="dwi_ivim",
        )
        self.dim = dim


class T2StarBiExp(AbstractFitting):
    """Two-compartment T2*: S(TE) = S0 * (f*exp(-TE/Ts) + (1-f)*exp(-TE/Tl)).

    4 parameters: S0, T_short, T_long, f. Useful for myelin water imaging
    or other tissues with short + long T2* components.
    """

    def __init__(
        self,
        dim: int = 3,
        boundary: Optional[Tuple] = None,
        normalize: bool = True,
    ):
        if boundary is None:
            boundary = (
                [0.0, 1.0, 20.0, 0.0],
                [10.0, 30.0, 300.0, 1.0],
            )
        super().__init__(
            _t2star_biexp_model,
            boundary=boundary,
            normalize=normalize,
            rust_model="t2star_biexp",
        )
        self.dim = dim


class StretchedExp(AbstractFitting):
    """Stretched exponential: S(x) = S0 * exp(-(x/T)^alpha).

    3 parameters: S0, T, alpha. alpha = 1 recovers a normal mono-exp.
    """

    def __init__(
        self,
        dim: int = 3,
        boundary: Optional[Tuple] = None,
        normalize: bool = True,
    ):
        if boundary is None:
            boundary = ([0.0, 1.0, 0.1], [10.0, 1e4, 2.0])
        super().__init__(
            lambda x, S0, T, a: S0 * np.exp(-((x / T) ** a)),
            boundary=boundary,
            normalize=normalize,
            rust_model="stretched_exp",
        )
        self.dim = dim


class CustomExpression(AbstractFitting):
    """Fit an arbitrary model defined as a string expression in `x` and `pN`.

    Identifiers `x`, `p0`..`p7` and the aliases `S0`, `T`, `T1`, `T2`, `T2s`,
    `T1rho`, `D`, `D_star`, `K`, `f`, `alpha`, `offset`, `C` are recognised.
    Functions: exp, log, sin, cos, tan, sqrt, abs, pow, min, max.

    Example:
        m = CustomExpression(
            expression="S0 * exp(-x/T) + C",
            boundary=([0, 0, -1], [10, 200, 1]),
        )
        fit_maps, r2 = m.fit(data, mask, te, method="rust")
    """

    def __init__(
        self,
        expression: str,
        boundary: Tuple,
        dim: int = 3,
        normalize: bool = True,
    ):
        n_params = len(boundary[0])

        # Python fallback: parse the expression to a callable that takes
        # (x, *params). We use a small eval-based shim because scipy's
        # curve_fit also needs to call it; we keep this simple by routing
        # through the same Rust string parser via a lambda that delegates
        # to the compiled program.
        from .expression_helper import compile_callable

        py_fn = compile_callable(expression, n_params)

        super().__init__(
            py_fn,
            boundary=boundary,
            normalize=normalize,
            rust_model="expression",
        )
        self.rust_expression = expression
        self.dim = dim
