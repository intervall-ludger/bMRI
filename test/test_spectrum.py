"""NNLS T2 spectrum recovery on synthetic two-peak phantoms."""

from __future__ import annotations

import numpy as np
import pytest

from bmri.fitting import T2Spectrum

try:
    import bmri_fit  # noqa: F401

    HAVE_RUST = True
except ImportError:
    HAVE_RUST = False

pytestmark = pytest.mark.skipif(not HAVE_RUST, reason="NNLS needs the rust backend")


def _two_peak_signal(te, t_grid, amps_at):
    truth = np.zeros(len(t_grid))
    for t_target, a in amps_at.items():
        idx = int(np.argmin(np.abs(t_grid - t_target)))
        truth[idx] = a
    A = np.exp(-te[:, None] / t_grid[None, :])
    return A @ truth, truth


def test_t2_spectrum_recovers_two_peaks():
    te = np.array([5, 10, 20, 40, 80, 120, 160, 200], dtype=np.float64)
    t_grid = T2Spectrum.log_grid(5, 300, 40)
    sig, truth = _two_peak_signal(te, t_grid, {20.0: 0.3, 80.0: 0.7})
    data = sig[:, None, None, None].repeat(2, axis=1).repeat(2, axis=2).repeat(1, axis=3)
    mask = np.ones(data.shape[1:], dtype=np.int16)

    sp = T2Spectrum(t_grid=t_grid, lambda_reg=0.0)
    spec, err = sp.fit(data, mask, te)

    # Per voxel, peaks should land within one grid bin of the truth.
    voxel = spec[:, 0, 0, 0]
    peak_idx = np.argsort(voxel)[::-1][:2]
    recovered_t2s = sorted(t_grid[peak_idx])
    assert abs(recovered_t2s[0] - 20.0) < t_grid[1] - t_grid[0] + 5
    assert abs(recovered_t2s[1] - 80.0) < 10
    assert (err < 1e-3).all()


def test_integrate_band():
    te = np.array([5, 10, 20, 40, 80, 120, 160], dtype=np.float64)
    t_grid = T2Spectrum.log_grid(5, 300, 40)
    sig, _ = _two_peak_signal(te, t_grid, {15.0: 0.4, 90.0: 0.6})
    data = sig[:, None, None, None]
    mask = np.ones((1, 1, 1), dtype=np.int16)

    sp = T2Spectrum(t_grid=t_grid)
    spec, _ = sp.fit(data, mask, te)
    short = sp.integrate(spec, 5, 40)  # short-T2 fraction (myelin-like)
    long_ = sp.integrate(spec, 40, 300)  # long-T2 fraction
    assert abs(short[0, 0, 0] - 0.4) < 0.05
    assert abs(long_[0, 0, 0] - 0.6) < 0.05


def test_peak_t2():
    te = np.array([5, 10, 20, 40, 80, 120], dtype=np.float64)
    t_grid = T2Spectrum.log_grid(5, 300, 50)
    sig, _ = _two_peak_signal(te, t_grid, {50.0: 1.0})
    data = sig[:, None, None, None]
    mask = np.ones((1, 1, 1), dtype=np.int16)
    sp = T2Spectrum(t_grid=t_grid)
    spec, _ = sp.fit(data, mask, te)
    peak = sp.peak_t2(spec)
    assert abs(peak[0, 0, 0] - 50.0) < (t_grid[1] - t_grid[0]) + 5
