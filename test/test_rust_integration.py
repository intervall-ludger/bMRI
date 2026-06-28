import numpy as np
import pytest

pytest.importorskip("bmri_fit")

from src.Fitting.T2_T2star import T2_T2star, mono_exp


def _make_volume(shape=(6, 6, 2), te=(0.0, 10.0, 20.0, 40.0, 70.0), t_true=35.0):
    x = np.array(te)
    nx, ny, nz = shape
    dicom = np.zeros((len(x), nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dicom[:, i, j, k] = mono_exp(x, 1000.0, t_true, 0.0)
    return dicom, x


def test_rust_mask_region_matches_truth():
    dicom, x = _make_volume(t_true=35.0)
    mask = np.zeros(dicom.shape[1:])
    mask[2:4, 2:4, :] = 1  # small ROI

    fitter = T2_T2star(dim=3, boundary=((1.0, 1.0, -200.0), (5000.0, 500.0, 200.0)))
    fit_maps, r2 = fitter.fit(dicom, mask, x, method="rust")

    t_map = fit_maps[1]
    inside = t_map[mask > 0]
    assert np.allclose(inside, 35.0, rtol=0.05)
    # outside the mask stays NaN in mask mode
    assert np.isnan(t_map[mask == 0]).all()


def test_rust_full_image_then_apply_mask():
    dicom, x = _make_volume(t_true=35.0)
    mask = np.zeros(dicom.shape[1:])
    mask[0:2, 0:2, :] = 1

    fitter = T2_T2star(dim=3, boundary=((1.0, 1.0, -200.0), (5000.0, 500.0, 200.0)))
    fit_maps, r2 = fitter.fit(dicom, mask, x, method="rust", fit_region="full")

    t_map = fit_maps[1]
    # whole volume is fitted, not just the mask
    assert np.isfinite(t_map).all()
    assert np.allclose(t_map, 35.0, rtol=0.05)


def test_rust_region_bounds_clamp_per_label():
    dicom, x = _make_volume(t_true=200.0)  # true T well above the tight cap
    mask = np.zeros(dicom.shape[1:])
    mask[1:3, 1:3, :] = 1  # label 1
    mask[4:6, 4:6, :] = 2  # label 2

    fitter = T2_T2star(dim=3, boundary=((1.0, 1.0, -200.0), (5000.0, 500.0, 200.0)))
    region_bounds = {
        1: ((1.0, 1.0, -10.0), (5000.0, 50.0, 10.0)),  # cap T at 50
    }
    fit_maps, _ = fitter.fit(
        dicom, mask, x, method="rust", fit_region="full", region_bounds=region_bounds
    )
    t_map = fit_maps[1]

    # label 1 is clamped to its region cap
    assert (t_map[mask == 1] <= 50.0 + 1e-6).all()
    # label 2 uses the default wide bounds and recovers the true high T
    assert np.allclose(t_map[mask == 2], 200.0, rtol=0.1)
