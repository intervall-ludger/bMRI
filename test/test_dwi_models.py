"""Validate DWI, IVIM, Kurtosis, T2* bi-exp and CustomExpression fitters."""

from __future__ import annotations

import numpy as np
import pytest

from bmri.fitting import (
    CustomExpression,
    DWIIvim,
    DWIKurtosis,
    DWIMonoExp,
    StretchedExp,
    T2StarBiExp,
)

try:
    import bmri_fit  # noqa: F401

    HAVE_RUST = True
except ImportError:
    HAVE_RUST = False


def _adc_signal(b, S0=1.0, D=1e-3):
    return S0 * np.exp(-b * D)


def _ivim_signal(b, S0=1.0, D=1e-3, D_star=2e-2, f=0.2):
    return S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))


def _kurt_signal(b, S0=1.0, D=8e-4, K=0.7):
    return S0 * np.exp(-b * D + (b * D) ** 2 * K / 6.0)


def _t2s_biexp_signal(te, S0=1.0, Ts=8.0, Tl=60.0, f=0.3):
    return S0 * (f * np.exp(-te / Ts) + (1 - f) * np.exp(-te / Tl))


def _make_volume(signal_fn, x, shape=(4, 4, 2)):
    data = np.empty((len(x), *shape), dtype=np.float64)
    for i, xi in enumerate(x):
        data[i] = signal_fn(xi)
    return data, np.ones(shape, dtype=np.int16)


@pytest.mark.parametrize("method", ["curvefit", "rust"])
def test_dwi_mono_exp(method):
    if method == "rust" and not HAVE_RUST:
        pytest.skip("rust not built")
    b = np.array([0, 200, 400, 800, 1500], dtype=np.float64)
    data, mask = _make_volume(_adc_signal, b)
    fit_maps, r2 = DWIMonoExp().fit(data, mask, b, method=method)
    assert np.allclose(fit_maps[1][mask > 0], 1e-3, rtol=0.05)
    assert (r2[mask > 0] > 0.99).all()


@pytest.mark.parametrize("method", ["curvefit", "rust"])
def test_dwi_kurtosis(method):
    if method == "rust" and not HAVE_RUST:
        pytest.skip("rust not built")
    b = np.array([0, 200, 400, 600, 1000, 1500, 2000], dtype=np.float64)
    data, mask = _make_volume(_kurt_signal, b)
    fit_maps, r2 = DWIKurtosis().fit(data, mask, b, method=method)
    assert np.allclose(np.nanmedian(fit_maps[1][mask > 0]), 8e-4, rtol=0.05)
    assert np.allclose(np.nanmedian(fit_maps[2][mask > 0]), 0.7, atol=0.1)


@pytest.mark.skipif(not HAVE_RUST, reason="IVIM 4-param needs rust backend")
def test_dwi_ivim_rust():
    b = np.array([0, 10, 30, 60, 100, 200, 400, 800, 1500], dtype=np.float64)
    data, mask = _make_volume(_ivim_signal, b)
    fit_maps, r2 = DWIIvim().fit(data, mask, b, method="rust")
    in_mask = mask > 0
    assert np.allclose(np.nanmedian(fit_maps[1][in_mask]), 1e-3, rtol=0.1)
    assert np.allclose(np.nanmedian(fit_maps[2][in_mask]), 2e-2, rtol=0.2)
    assert np.allclose(np.nanmedian(fit_maps[3][in_mask]), 0.2, atol=0.05)


@pytest.mark.skipif(not HAVE_RUST, reason="bi-exp T2* 4-param needs rust backend")
def test_t2star_biexp_rust():
    te = np.array([2, 5, 10, 15, 20, 30, 50, 80, 120], dtype=np.float64)
    data, mask = _make_volume(_t2s_biexp_signal, te)
    fit_maps, r2 = T2StarBiExp().fit(data, mask, te, method="rust")
    in_mask = mask > 0
    assert np.allclose(np.nanmedian(fit_maps[1][in_mask]), 8.0, rtol=0.1)
    assert np.allclose(np.nanmedian(fit_maps[2][in_mask]), 60.0, rtol=0.1)
    assert np.allclose(np.nanmedian(fit_maps[3][in_mask]), 0.3, atol=0.05)


@pytest.mark.skipif(not HAVE_RUST, reason="stretched exp needs rust for now")
def test_stretched_exp_rust():
    x = np.array([1, 5, 10, 20, 40, 80, 160], dtype=np.float64)
    S0, T, a = 1.0, 40.0, 0.8
    sig = S0 * np.exp(-((x / T) ** a))
    data = np.broadcast_to(sig[:, None, None, None], (len(x), 3, 3, 2)).copy()
    mask = np.ones(data.shape[1:], dtype=np.int16)
    fit_maps, r2 = StretchedExp().fit(data, mask, x, method="rust")
    in_mask = mask > 0
    assert abs(np.nanmedian(fit_maps[1][in_mask]) - T) / T < 0.1
    assert abs(np.nanmedian(fit_maps[2][in_mask]) - a) < 0.1


@pytest.mark.parametrize("method", ["curvefit", "rust"])
def test_custom_expression_mono_exp(method):
    """User defines mono-exp by string, both backends recover the same values."""
    if method == "rust" and not HAVE_RUST:
        pytest.skip("rust not built")
    x = np.array([5, 10, 20, 40, 80, 120], dtype=np.float64)
    S0, T = 1.0, 45.0
    sig = S0 * np.exp(-x / T)
    data = np.broadcast_to(sig[:, None, None, None], (len(x), 3, 3, 2)).copy()
    mask = np.ones(data.shape[1:], dtype=np.int16)
    model = CustomExpression(
        expression="S0 * exp(-x/T) + C",
        boundary=([0.5, 1.0, -0.5], [2.0, 200.0, 0.5]),
    )
    # curvefit with dynamic function: use single-threaded path (pools=0)
    fit_maps, r2 = model.fit(data, mask, x, method=method, pools=0)
    in_mask = mask > 0
    assert abs(np.nanmedian(fit_maps[1][in_mask]) - T) / T < 0.02


@pytest.mark.skipif(not HAVE_RUST, reason="rust needed")
def test_custom_expression_pn_names():
    """Positional p0..pN-1 names work even without aliases."""
    x = np.array([5, 10, 20, 40, 80], dtype=np.float64)
    sig = 2.0 * np.exp(-x * 0.05)
    data = np.broadcast_to(sig[:, None, None, None], (len(x), 2, 2, 1)).copy()
    mask = np.ones(data.shape[1:], dtype=np.int16)
    model = CustomExpression(
        expression="p0 * exp(-x * p1)",
        boundary=([0.0, 0.0], [10.0, 1.0]),
    )
    fit_maps, _ = model.fit(data, mask, x, method="rust")
    assert abs(np.nanmedian(fit_maps[0]) - 2.0) < 0.05
    assert abs(np.nanmedian(fit_maps[1]) - 0.05) < 0.005
