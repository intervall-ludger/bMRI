"""Synthetic phantoms with known ground-truth relaxation times.

Used to validate fitting accuracy and backend equivalence. All phantoms
return a (n_echoes, X, Y, Z) data array and the matching x-array (TE, TSL, TI).
"""

from __future__ import annotations

import numpy as np


def mono_exp(x: np.ndarray, S0: float, T: float, offset: float = 0.0) -> np.ndarray:
    """Three-parameter mono-exponential decay with offset."""
    return S0 * np.exp(-x / T) + offset


def t2_phantom(
    shape: tuple[int, int, int] = (8, 8, 4),
    te: np.ndarray | None = None,
    t2_value: float = 45.0,
    s0_value: float = 1.0,
    offset_value: float = 0.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform T2 phantom (every voxel has the same T2).

    Returns:
        data:  (n_te, X, Y, Z) magnitude signals
        te:    echo times
        truth: (3, X, Y, Z) ground-truth (S0, T2, offset) per voxel
    """
    if te is None:
        te = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    if rng is None:
        rng = np.random.default_rng(0)

    n_te = len(te)
    data = np.empty((n_te, *shape), dtype=np.float64)
    for i, t in enumerate(te):
        data[i] = mono_exp(np.array([t]), s0_value, t2_value, offset_value)[0]

    if noise_sigma > 0:
        data = data + rng.normal(0.0, noise_sigma, size=data.shape)

    truth = np.stack(
        [
            np.full(shape, s0_value, dtype=np.float64),
            np.full(shape, t2_value, dtype=np.float64),
            np.full(shape, offset_value, dtype=np.float64),
        ]
    )
    return data, np.asarray(te, dtype=np.float64), truth


def t2_gradient_phantom(
    shape: tuple[int, int, int] = (16, 16, 4),
    te: np.ndarray | None = None,
    t2_min: float = 20.0,
    t2_max: float = 80.0,
    s0_value: float = 1.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T2 phantom with a linear gradient along the X axis.

    Each X-slab has a different T2 between t2_min and t2_max. Useful to test
    that region_bounds work and that the fit recovers spatially varying values.
    """
    if te is None:
        te = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    if rng is None:
        rng = np.random.default_rng(0)

    nx, ny, nz = shape
    t2_per_x = np.linspace(t2_min, t2_max, nx)
    truth_t2 = np.broadcast_to(t2_per_x[:, None, None], shape).astype(np.float64)
    truth_s0 = np.full(shape, s0_value, dtype=np.float64)
    truth_offset = np.zeros(shape, dtype=np.float64)

    data = np.empty((len(te), *shape), dtype=np.float64)
    for i, t in enumerate(te):
        data[i] = s0_value * np.exp(-t / truth_t2)

    if noise_sigma > 0:
        data = data + rng.normal(0.0, noise_sigma, size=data.shape)

    truth = np.stack([truth_s0, truth_t2, truth_offset])
    return data, np.asarray(te, dtype=np.float64), truth


def t2star_phantom(
    shape: tuple[int, int, int] = (8, 8, 4),
    te: np.ndarray | None = None,
    t2star_value: float = 25.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform T2* phantom. Same model as T2, just shorter typical times."""
    if te is None:
        te = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
    return t2_phantom(shape=shape, te=te, t2_value=t2star_value, **kwargs)


def two_region_phantom(
    shape: tuple[int, int, int] = (16, 8, 4),
    te: np.ndarray | None = None,
    t2_left: float = 45.0,
    t2_right: float = 150.0,
    s0_value: float = 1.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Phantom with two distinct T2 regions (e.g. cartilage vs fat pad).

    Left half has T2=t2_left (cartilage-like ~45ms), right half t2_right (fat-like
    ~150ms). Returns a label mask: 1 = left region, 6 = right region (matches
    LenaMinko cart/Hoffa convention).
    """
    if te is None:
        te = np.array([10.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    if rng is None:
        rng = np.random.default_rng(0)

    nx, ny, nz = shape
    mid = nx // 2
    truth_t2 = np.empty(shape, dtype=np.float64)
    truth_t2[:mid] = t2_left
    truth_t2[mid:] = t2_right
    truth_s0 = np.full(shape, s0_value)
    truth_offset = np.zeros(shape)

    data = np.empty((len(te), *shape), dtype=np.float64)
    for i, t in enumerate(te):
        data[i] = s0_value * np.exp(-t / truth_t2)

    if noise_sigma > 0:
        data = data + rng.normal(0.0, noise_sigma, size=data.shape)

    mask = np.empty(shape, dtype=np.int16)
    mask[:mid] = 1
    mask[mid:] = 6

    truth = np.stack([truth_s0, truth_t2, truth_offset])
    return data, np.asarray(te, dtype=np.float64), truth, mask
