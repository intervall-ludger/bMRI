"""Unit tests for AbstractFitting. The fit infrastructure assumes a
3-parameter model (S0, T, offset); older 2-parameter linear tests were
removed when the Rust backend landed."""

import numpy as np
import pytest
from numba import njit

from src.Fitting import AbstractFitting


def test_init():
    def fit_function(x, a, b, c):
        return a * x + b + c

    boundary = (0, 1)
    fitting = AbstractFitting(fit_function, boundary)
    assert fitting.fit_function == fit_function
    assert fitting.bounds == boundary


def test_set_fit_config():
    def fit_function(x, a, b, c):
        return a * x + b + c

    boundary = (0, 1)
    fitting = AbstractFitting(fit_function, boundary)
    fit_config = {"maxfev": 1000}
    fitting.set_fit_config(fit_config)
    assert fitting.fit_config == fit_config


@pytest.mark.parametrize("pools", [0, 1])
def test_fit_three_params(pools):
    @njit
    def fit_function(x, a, b, c):
        return a * x + b + c

    fitting = AbstractFitting(fit_function)
    a, b, c = 1.0, 2.0, 0.0
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = fit_function(x, a, b, c)
    dicom = np.array([np.ones((2, 2)) * y[i] for i in range(len(x))])
    dicom = dicom.reshape((6, 2, 2, 1))
    mask = np.ones((2, 2, 1))

    fit_maps, _ = fitting.fit(dicom, mask, x, pools=pools)
    assert len(fit_maps) == 3
    assert abs(fit_maps[0][0, 0, 0] - a) < 0.01


if __name__ == "__main__":
    pytest.main()
