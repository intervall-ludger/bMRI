import time

import numpy as np
import pytest
from scipy.optimize import curve_fit

bmri_fit = pytest.importorskip("bmri_fit")


def mono_exp(x, s0, t, offset):
    return s0 * np.exp(-x / t) + offset


def aronen_t1rho(x, s0, t, offset, TR, T1, alpha, TE, T2star):
    tau = TR - x
    num = s0 * np.exp(-x / t) * (1 - np.exp(-tau / T1)) * np.sin(alpha) * np.exp(-TE / T2star)
    den = 1 - np.cos(alpha) * np.exp(-tau / T1) * np.exp(-x / t)
    return num / den + offset


def _bounds(n, lo, hi):
    lower = np.tile(np.array(lo, dtype=np.float64), (n, 1))
    upper = np.tile(np.array(hi, dtype=np.float64), (n, 1))
    return lower, upper


def test_mono_exp_matches_scipy():
    rng = np.random.default_rng(0)
    x = np.array([0.0, 10.0, 20.0, 40.0, 70.0])
    n = 200
    true = np.column_stack(
        [
            rng.uniform(800, 1200, n),  # S0
            rng.uniform(20, 80, n),  # T
            rng.uniform(-20, 20, n),  # offset
        ]
    )
    signals = np.array([mono_exp(x, *p) for p in true])
    signals += rng.normal(0, 2.0, signals.shape)

    lower, upper = _bounds(n, [1.0, 1.0, -200.0], [5000.0, 500.0, 200.0])
    params, r2 = bmri_fit.fit_volume(
        signals.astype(np.float64),
        x.astype(np.float64),
        lower,
        upper,
        "mono_exp",
        normalize=False,
        max_iter=200,
    )

    # scipy reference
    sp = np.zeros_like(params)
    for i in range(n):
        try:
            sp[i], _ = curve_fit(
                mono_exp,
                x,
                signals[i],
                p0=[1000, 40, 0],
                bounds=(lower[i], upper[i]),
                maxfev=5000,
            )
        except RuntimeError:
            sp[i] = np.nan

    # Compare T (the parameter of interest) where both converged
    rust_t = params[:, 1]
    scipy_t = sp[:, 1]
    ok = np.isfinite(scipy_t)
    rel = np.abs(rust_t[ok] - scipy_t[ok]) / np.abs(scipy_t[ok])
    assert np.median(rel) < 0.02, f"median rel T error {np.median(rel):.4f}"
    assert (r2 > 0.9).mean() > 0.95


def test_aronen_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.array([0.0, 10.0, 40.0, 70.0])
    seq = dict(TR=5000.0, T1=1200.0, alpha=np.deg2rad(90.0), TE=4.0, T2star=30.0)
    n = 100
    true = np.column_stack(
        [
            rng.uniform(800, 1200, n),
            rng.uniform(30, 90, n),
            np.zeros(n),
        ]
    )
    signals = np.array([aronen_t1rho(x, p[0], p[1], p[2], **seq) for p in true])
    signals += rng.normal(0, 1.0, signals.shape)

    lower, upper = _bounds(n, [1.0, 1.0, -500.0], [10000.0, 500.0, 500.0])
    params, r2 = bmri_fit.fit_volume(
        signals.astype(np.float64),
        x.astype(np.float64),
        lower,
        upper,
        "aronen_t1rho",
        tr=seq["TR"],
        t1=seq["T1"],
        alpha=seq["alpha"],
        te=seq["TE"],
        t2star=seq["T2star"],
        normalize=False,
        max_iter=300,
    )
    rel = np.abs(params[:, 1] - true[:, 1]) / true[:, 1]
    assert np.median(rel) < 0.05, f"median rel T error {np.median(rel):.4f}"


def test_region_bounds_clamp():
    # A pixel whose true T is out of its tight region bound must be clamped to it.
    x = np.array([0.0, 10.0, 20.0, 40.0, 70.0])
    signal = mono_exp(x, 1000.0, 200.0, 0.0)[None, :]  # true T=200
    lower = np.array([[1.0, 1.0, -10.0]])
    upper = np.array([[5000.0, 50.0, 10.0]])  # cap T at 50
    params, _ = bmri_fit.fit_volume(signal, x, lower, upper, "mono_exp", max_iter=200)
    assert params[0, 1] <= 50.0 + 1e-6


def test_speed_smoke():
    rng = np.random.default_rng(2)
    x = np.array([0.0, 10.0, 20.0, 40.0, 70.0])
    n = 20000
    signals = rng.uniform(100, 1000, (n, len(x)))
    lower, upper = _bounds(n, [1.0, 1.0, -200.0], [5000.0, 500.0, 200.0])
    t0 = time.perf_counter()
    bmri_fit.fit_volume(signals, x, lower, upper, "mono_exp", max_iter=100)
    dt = time.perf_counter() - t0
    # 20k pixels should be well under a second on any modern machine
    assert dt < 5.0, f"too slow: {dt:.3f}s"
    print(f"\n20k pixels fitted in {dt * 1000:.0f} ms")
