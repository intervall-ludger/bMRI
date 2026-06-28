"""Rust and scipy backends must produce identical results on the same data."""

from __future__ import annotations

import numpy as np
import pytest

from bmri.fitting import T2T2star

from .phantoms import t2_phantom, two_region_phantom

try:
    import bmri_fit  # noqa: F401

    HAVE_RUST = True
except ImportError:
    HAVE_RUST = False

pytestmark = pytest.mark.skipif(not HAVE_RUST, reason="Rust backend not built")


def test_rust_vs_scipy_uniform():
    data, te, truth = t2_phantom(t2_value=45.0, noise_sigma=0.01)
    mask = np.ones(data.shape[1:], dtype=np.int16)
    bounds = ([0.5, 1.0, -0.5], [2.0, 200.0, 0.5])

    fitter_py = T2T2star(dim=3, boundary=bounds, normalize=True)
    py_maps, py_r2 = fitter_py.fit(data, mask, te)

    fitter_rs = T2T2star(dim=3, boundary=bounds, normalize=True)
    rs_maps, rs_r2 = fitter_rs.fit(data, mask, te, method="rust")

    py_t2 = np.nanmedian(py_maps[1])
    rs_t2 = np.nanmedian(rs_maps[1])
    assert abs(py_t2 - rs_t2) < 0.5  # within 0.5 ms


def test_rust_full_volume_matches_masked():
    """fit_region='full' must give the same results inside the mask as fit_region='mask'."""
    data, te, _ = t2_phantom(t2_value=45.0, noise_sigma=0.01)
    mask = np.zeros(data.shape[1:], dtype=np.int16)
    mask[2:6, 2:6, :] = 1
    bounds = ([0.5, 1.0, -0.5], [2.0, 200.0, 0.5])

    fitter = T2T2star(dim=3, boundary=bounds, normalize=True)
    masked_maps, _ = fitter.fit(data, mask, te, method="rust", fit_region="mask")
    full_maps, _ = fitter.fit(data, mask, te, method="rust", fit_region="full")

    masked_t2 = np.nanmedian(masked_maps[1][mask > 0])
    full_t2 = np.nanmedian(full_maps[1][mask > 0])
    assert abs(masked_t2 - full_t2) < 0.5


def test_region_bounds_separate_regions():
    """Each label uses its own bounds; mixing regions does not bleed values."""
    data, te, truth, mask = two_region_phantom(t2_left=45.0, t2_right=150.0)
    cart_bounds = ([0.5, 1.0, -0.5], [2.0, 100.0, 0.5])
    hoffa_bounds = ([0.5, 1.0, -0.5], [2.0, 500.0, 0.5])

    fitter = T2T2star(dim=3, boundary=cart_bounds, normalize=True)
    fit_maps, _ = fitter.fit(
        data,
        mask,
        te,
        method="rust",
        fit_region="full",
        region_bounds={1: cart_bounds, 6: hoffa_bounds},
    )

    left = np.nanmedian(fit_maps[1][mask == 1])
    right = np.nanmedian(fit_maps[1][mask == 6])
    assert abs(left - 45.0) < 1.0
    assert abs(right - 150.0) < 2.0


def test_rust_r2_filter():
    """min_r2 must drop voxels with poor fits (set to NaN)."""
    data, te, _ = t2_phantom(t2_value=45.0, noise_sigma=0.05)
    mask = np.ones(data.shape[1:], dtype=np.int16)
    bounds = ([0.5, 1.0, -0.5], [2.0, 200.0, 0.5])

    fitter = T2T2star(dim=3, boundary=bounds, normalize=True)
    fit_lo, _ = fitter.fit(data, mask, te, method="rust", min_r2=-np.inf)
    fit_hi, _ = fitter.fit(data, mask, te, method="rust", min_r2=0.99)

    n_lo = (~np.isnan(fit_lo[1])).sum()
    n_hi = (~np.isnan(fit_hi[1])).sum()
    assert n_hi <= n_lo
