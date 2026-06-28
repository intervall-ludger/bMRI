"""Recover ground-truth relaxation times from synthetic phantoms."""

from __future__ import annotations

import numpy as np
import pytest

from bmri.fitting import T2T2star

from .phantoms import t2_gradient_phantom, t2_phantom, two_region_phantom


@pytest.mark.parametrize("t2_value", [25.0, 45.0, 80.0])
def test_recover_uniform_t2_no_noise(t2_value):
    """Without noise, the fit must recover T2 within 0.5 %."""
    data, te, truth = t2_phantom(t2_value=t2_value, noise_sigma=0.0)
    mask = np.ones(data.shape[1:], dtype=np.int16)
    fitter = T2T2star(dim=3, boundary=([0.5, 1.0, -0.5], [2.0, 200.0, 0.5]), normalize=True)
    fit_maps, r2 = fitter.fit(data, mask, te)

    recovered_t2 = fit_maps[1]
    assert np.allclose(recovered_t2, truth[1], rtol=0.005)
    assert (r2 > 0.999).all()


@pytest.mark.parametrize("noise_sigma", [0.01, 0.02])
def test_recover_uniform_t2_with_noise(noise_sigma):
    """With moderate noise, median recovery must be within 5 %."""
    data, te, truth = t2_phantom(t2_value=50.0, noise_sigma=noise_sigma)
    mask = np.ones(data.shape[1:], dtype=np.int16)
    fitter = T2T2star(dim=3, boundary=([0.5, 1.0, -0.5], [2.0, 200.0, 0.5]), normalize=True)
    fit_maps, _ = fitter.fit(data, mask, te)

    recovered = fit_maps[1][~np.isnan(fit_maps[1])]
    assert abs(np.median(recovered) - 50.0) / 50.0 < 0.05


def test_recover_gradient():
    """Each X-slab should recover its own T2 (gradient phantom)."""
    data, te, truth = t2_gradient_phantom(shape=(8, 8, 2), t2_min=30, t2_max=80)
    mask = np.ones(data.shape[1:], dtype=np.int16)
    fitter = T2T2star(dim=3, boundary=([0.5, 1.0, -0.5], [2.0, 200.0, 0.5]), normalize=True)
    fit_maps, _ = fitter.fit(data, mask, te)

    for ix in range(data.shape[1]):
        recovered = np.nanmedian(fit_maps[1][ix])
        expected = float(np.median(truth[1][ix]))
        assert abs(recovered - expected) / expected < 0.01


def test_two_region_no_bounds_interference():
    """Cartilage-like and Hoffa-like region recovered jointly with one bound set."""
    data, te, truth, mask = two_region_phantom(t2_left=45.0, t2_right=150.0)
    fitter = T2T2star(dim=3, boundary=([0.5, 1.0, -0.5], [2.0, 500.0, 0.5]), normalize=True)
    fit_maps, _ = fitter.fit(data, mask, te)

    left = fit_maps[1][mask == 1]
    right = fit_maps[1][mask == 6]
    assert abs(np.median(left) - 45.0) / 45.0 < 0.02
    assert abs(np.median(right) - 150.0) / 150.0 < 0.02
