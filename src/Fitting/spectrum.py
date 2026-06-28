"""Non-negative least squares spectrum analysis (T2 spectrum, IVIM diffusion
spectrum, generally any non-negative basis decomposition).

Instead of fitting a single relaxation time per voxel, NNLS recovers a
distribution of relaxation times. Useful for multi-compartment tissues like
myelin water (short T2), free water (long T2), or perfusion vs tissue
diffusion in DWI.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


class T2Spectrum:
    """Multi-component T2 spectrum via NNLS with Tikhonov regularisation.

    Fits, per voxel, a non-negative amplitude vector `a` such that
        S(TE) ≈ sum_j a_j * exp(-TE / t_grid[j])
    subject to a >= 0, optionally with a smoothness penalty
        + lambda^2 * ||L a||^2  (L = second-difference matrix).

    Use cases:
      - Myelin water imaging: integrate amplitude in [10, 40] ms
      - Total water content: sum(a)
      - Peak T2: t_grid[argmax(a)]

    Args:
        t_grid: array of candidate T2 values in ms (log-spaced typical)
        lambda_reg: Tikhonov smoothness penalty (0 = pure NNLS, 0.005 typical)
        max_iter: NNLS iterations per voxel (500 default)
    """

    def __init__(
        self,
        t_grid: np.ndarray,
        lambda_reg: float = 0.0,
        max_iter: int = 500,
    ):
        self.t_grid = np.asarray(t_grid, dtype=np.float64)
        if (self.t_grid <= 0).any():
            raise ValueError("t_grid must be positive")
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)

    @staticmethod
    def log_grid(t_min: float, t_max: float, n: int = 40) -> np.ndarray:
        """Convenience: log-spaced grid from t_min to t_max with n bins."""
        return np.logspace(np.log10(t_min), np.log10(t_max), n)

    def fit(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        te: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the spectrum on every mask voxel.

        Returns:
            spectrum: (n_bins, X, Y, Z) amplitudes per voxel
            recon_err: (X, Y, Z) relative reconstruction error
        """
        try:
            import bmri_fit
        except ImportError as e:
            raise ImportError(
                "T2Spectrum needs the bmri_fit Rust wheel. "
                "Build it with `cd rust && uvx maturin build --release && "
                "uv pip install target/wheels/bmri_fit-*.whl`"
            ) from e

        te = np.asarray(te, dtype=np.float64)
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[..., None]
            dicom = dicom[..., None] if dicom.ndim == 3 else dicom
        sel = mask > 0
        coords = np.nonzero(sel)
        if len(coords[0]) == 0:
            n_bins = len(self.t_grid)
            return (
                np.zeros((n_bins, *mask.shape)),
                np.zeros(mask.shape),
            )

        signals = np.ascontiguousarray(
            dicom[:, coords[0], coords[1], coords[2]].T, dtype=np.float64
        )
        spec = bmri_fit.fit_spectrum(
            signals,
            te,
            self.t_grid,
            kernel="t2",
            lambda_reg=self.lambda_reg,
            max_iter=self.max_iter,
        )
        # Compute relative reconstruction error per voxel
        A = np.exp(-te[:, None] / self.t_grid[None, :])
        recon = spec @ A.T  # (n_pix, n_echoes)
        err = np.linalg.norm(recon - signals, axis=1) / (np.linalg.norm(signals, axis=1) + 1e-12)

        n_bins = len(self.t_grid)
        spec_out = np.zeros((n_bins, *mask.shape))
        err_out = np.zeros(mask.shape)
        for j in range(n_bins):
            spec_out[j][coords] = spec[:, j]
        err_out[coords] = err
        return spec_out, err_out

    def integrate(
        self,
        spectrum: np.ndarray,
        t_low: float,
        t_high: float,
    ) -> np.ndarray:
        """Sum amplitudes in a T2 range. Returns a 3D map.

        Example: myelin water fraction = integrate(spec, 10, 40) / spec.sum(axis=0)
        """
        in_range = (self.t_grid >= t_low) & (self.t_grid <= t_high)
        return spectrum[in_range].sum(axis=0)

    def peak_t2(self, spectrum: np.ndarray) -> np.ndarray:
        """Argmax T2 per voxel. NaN where the spectrum is empty."""
        out = np.full(spectrum.shape[1:], np.nan)
        amax = spectrum.argmax(axis=0)
        has_signal = spectrum.sum(axis=0) > 1e-12
        out[has_signal] = self.t_grid[amax[has_signal]]
        return out

    def geometric_mean_t2(self, spectrum: np.ndarray) -> np.ndarray:
        """Amplitude-weighted geometric mean T2 per voxel."""
        out = np.full(spectrum.shape[1:], np.nan)
        total = spectrum.sum(axis=0)
        sel = total > 1e-12
        log_t = np.log(self.t_grid)
        weighted = (spectrum * log_t[:, None, None, None]).sum(axis=0)
        out[sel] = np.exp(weighted[sel] / total[sel])
        return out


class DiffusionSpectrum(T2Spectrum):
    """Same NNLS engine, but the kernel is exp(-b * D) for diffusion fitting.

    `t_grid` here is actually a grid of D values (in mm²/s), e.g.
    `np.logspace(np.log10(1e-4), np.log10(1e-1), 40)` for combined tissue +
    pseudo-diffusion analysis.
    """

    def fit(self, dicom, mask, b_values):
        try:
            import bmri_fit
        except ImportError as e:
            raise ImportError("DiffusionSpectrum needs the bmri_fit Rust wheel") from e

        b_values = np.asarray(b_values, dtype=np.float64)
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[..., None]
            dicom = dicom[..., None] if dicom.ndim == 3 else dicom
        sel = mask > 0
        coords = np.nonzero(sel)
        n_bins = len(self.t_grid)
        if len(coords[0]) == 0:
            return np.zeros((n_bins, *mask.shape)), np.zeros(mask.shape)

        signals = np.ascontiguousarray(
            dicom[:, coords[0], coords[1], coords[2]].T, dtype=np.float64
        )
        spec = bmri_fit.fit_spectrum(
            signals,
            b_values,
            self.t_grid,
            kernel="dwi",
            lambda_reg=self.lambda_reg,
            max_iter=self.max_iter,
        )
        A = np.exp(-b_values[:, None] * self.t_grid[None, :])
        recon = spec @ A.T
        err = np.linalg.norm(recon - signals, axis=1) / (np.linalg.norm(signals, axis=1) + 1e-12)

        spec_out = np.zeros((n_bins, *mask.shape))
        err_out = np.zeros(mask.shape)
        for j in range(n_bins):
            spec_out[j][coords] = spec[:, j]
        err_out[coords] = err
        return spec_out, err_out
