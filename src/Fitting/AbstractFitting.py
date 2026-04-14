from abc import ABC
from functools import partial
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Tuple, Optional, Union

import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from src.Utilitis.utils import get_function_parameter


class AbstractFitting(ABC):
    def __init__(
        self,
        fit_function: Callable,
        boundary: Tuple[float, float] = None,
        fit_config: Optional[dict] = None,
        normalize: bool = False,
    ) -> None:
        """
        Initializes the AbstractFitting object.

        Parameters:
        - fit_function (Callable): The function used for fitting
        - boundary (Tuple[float, float], optional): Boundary for the parameters during fitting
        - fit_config (dict, optional): Additional configuration for the fit function
        - normalize (bool, optional): Normalize the data before fitting (default is False)
        """
        self.fit_function = fit_function
        self.bounds = boundary
        self.fit_config = fit_config
        self.normalize = normalize

    def set_fit_config(self, fit_config: dict) -> None:
        """
        Set the fit configuration.

        Parameters:
        - fit_config (dict): Configuration for the fit function
        """
        self.fit_config = fit_config

    def fit(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        pools: int = cpu_count(),
        min_r2: float = -np.inf,
        method: str = "curvefit",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the given data using parallel processing.

        Parameters:
        - dicom (np.ndarray): The input data
        - mask (np.ndarray): The mask array
        - x (np.ndarray): Independent variable data
        - pools (int, optional): Number of parallel processes to use (default is the number of CPUs)
        - min_r2 (float, optional): Minimum R squared value to consider (default is negative infinity)
        - method (str): "curvefit" (default, 3-param with offset) or "loglinear" (fast 2-param, no offset)

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Arrays of fit parameters and R squared values
        """
        dicom = dicom.astype("float64")
        assert len(mask.shape) == len(dicom.shape) - 1
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
            dicom = np.expand_dims(dicom, axis=3)
        assert dicom.shape[0] == len(x)

        if mask.shape != dicom.shape[1:]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match DICOM data shape {dicom.shape[1:]}. "
                "Ensure the mask was created from the same image data."
            )

        if method == "loglinear":
            return self._fit_loglinear(dicom, mask, x, min_r2)
        return self._fit_curvefit(dicom, mask, x, pools, min_r2)

    def _fit_loglinear(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        min_r2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized log-linear fitting: y = S0 * exp(-x/T). No offset parameter."""
        # 3 output params: S0, T, offset (offset=0 for compatibility)
        num_params = 3
        fit_maps = np.full((num_params, *mask.shape), np.nan)
        r2_map = np.zeros(mask.shape)

        coords = np.nonzero(mask)
        pixel_data = dicom[:, coords[0], coords[1], coords[2]]  # (n_echoes, n_pixels)

        if self.normalize:
            maxvals = pixel_data.max(axis=0)
            maxvals[maxvals == 0] = 1
            pixel_data = pixel_data / maxvals

        # Clamp for log
        pixel_data = np.clip(pixel_data, 1e-10, None)
        log_data = np.log(pixel_data)  # (n_echoes, n_pixels)

        # Solve log(y) = log(S0) - x/T for all pixels at once
        # Design matrix: [1, -x] → params: [log(S0), 1/T]
        A = np.column_stack([np.ones(len(x)), -x])  # (n_echoes, 2)
        params, residuals, _, _ = np.linalg.lstsq(A, log_data, rcond=None)
        # params shape: (2, n_pixels) → [log(S0), 1/T]

        s0_vals = np.exp(params[0])
        inv_t = params[1]
        # Avoid division by zero / negative
        valid = inv_t > 1e-10
        t_vals = np.full(inv_t.shape, np.nan)
        t_vals[valid] = 1.0 / inv_t[valid]

        # Apply bounds if set
        if self.bounds is not None:
            t_lower, t_upper = self.bounds[0][1], self.bounds[1][1]
            out_of_bounds = (t_vals < t_lower) | (t_vals > t_upper)
            t_vals[out_of_bounds] = np.nan

        # Calculate R² for all pixels vectorized
        fitted = s0_vals[np.newaxis, :] * np.exp(-x[:, np.newaxis] * inv_t[np.newaxis, :])
        if self.normalize:
            ss_res = np.sum((pixel_data - fitted) ** 2, axis=0)
            ss_tot = np.sum((pixel_data - pixel_data.mean(axis=0)) ** 2, axis=0)
        else:
            ss_res = np.sum((pixel_data - fitted) ** 2, axis=0)
            ss_tot = np.sum((pixel_data - pixel_data.mean(axis=0)) ** 2, axis=0)
        r2_vals = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

        # Store results
        good = np.isfinite(t_vals) & (r2_vals > min_r2)
        idx = (coords[0][good], coords[1][good], coords[2][good])
        fit_maps[0][idx] = s0_vals[good] if not self.normalize else s0_vals[good] * maxvals[good]
        fit_maps[1][idx] = t_vals[good]
        fit_maps[2][idx] = 0.0  # offset = 0

        r2_idx = (coords[0], coords[1], coords[2])
        r2_map[r2_idx] = r2_vals

        return np.array(fit_maps), r2_map

    def _fit_curvefit(
        self,
        dicom: np.ndarray,
        mask: np.ndarray,
        x: np.ndarray,
        pools: int,
        min_r2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel curve_fit with log-linear initial guess and chunked multiprocessing."""
        num_params = len(curve_fit(self.fit_function, x, dicom[:, 0, 0, 0])[0])
        fit_maps = np.full((num_params, *mask.shape), np.nan)
        r2_map = np.zeros(mask.shape)

        calc_p0 = len(get_function_parameter(self.fit_function)) == 3

        coords = list(zip(*np.nonzero(mask)))
        pixel_data = [dicom[:, i, j, k].copy() for i, j, k in coords]

        # Vectorized log-linear pre-fit for initial guesses
        p0_list = _loglinear_initial_guess(pixel_data, x, self.normalize, self.bounds)

        chunk_size = max(100, len(pixel_data) // (pools if pools > 0 else cpu_count()))
        chunks = []
        p0_chunks = []
        for start in range(0, len(pixel_data), chunk_size):
            chunks.append(pixel_data[start : start + chunk_size])
            p0_chunks.append(p0_list[start : start + chunk_size])

        fit_chunk_fixed = partial(
            _fit_chunk,
            x=x,
            fit_function=self.fit_function,
            bounds=self.bounds,
            config=self.fit_config,
            normalize=self.normalize,
            calc_p0=calc_p0,
        )

        if pools != 0:
            with Pool(pools) as pool:
                chunk_results = pool.starmap(
                    fit_chunk_fixed,
                    [(chunk, p0) for chunk, p0 in zip(chunks, p0_chunks)],
                )
        else:
            chunk_results = [fit_chunk_fixed(c, p0) for c, p0 in zip(chunks, p0_chunks)]

        all_results = [r for chunk in chunk_results for r in chunk]
        for (i, j, k), (param, r2) in zip(coords, all_results):
            if param is None:
                continue
            r2_map[i, j, k] = r2
            if r2 > min_r2:
                for p_num, p in enumerate(param):
                    fit_maps[p_num][i, j, k] = p

        return np.array(fit_maps), r2_map

    def save_times(self, times: np.ndarray, file_path: Path) -> None:
        """
        Save times for further evaluations.

        Args:
            times: np.ndarray with acquisition times.
            file_path: file path
        """
        np.savetxt(file_path, times)

    def load_times(self, file_path: Path) -> np.ndarray:
        """
        Load acquisition times.

        Args:
            file_path: file path to txt

        Returns:
            acquisition times (np.ndarray)
        """
        return np.loadtxt(file_path)


def _loglinear_initial_guess(
    pixel_data_list: list,
    x: np.ndarray,
    normalize: bool,
    bounds: Optional[Tuple] = None,
) -> list:
    """Compute log-linear initial guesses for all pixels vectorized."""
    arr = np.array(pixel_data_list, dtype=np.float64)  # (n_pixels, n_echoes)
    if normalize:
        maxvals = arr.max(axis=1, keepdims=True)
        maxvals[maxvals == 0] = 1
        arr = arr / maxvals

    arr_clipped = np.clip(arr, 1e-10, None)
    log_data = np.log(arr_clipped)  # (n_pixels, n_echoes)

    A = np.column_stack([np.ones(len(x)), -x])  # (n_echoes, 2)
    params, _, _, _ = np.linalg.lstsq(A, log_data.T, rcond=None)
    # params: (2, n_pixels) → [log(S0), 1/T]

    s0_init = np.exp(params[0])
    inv_t = params[1]
    t_init = np.where(inv_t > 1e-10, 1.0 / inv_t, 30.0)
    offset_init = arr[:, -1] * 0.1

    # Clamp to bounds
    if bounds is not None:
        s0_init = np.clip(s0_init, bounds[0][0], bounds[1][0])
        t_init = np.clip(t_init, bounds[0][1], bounds[1][1])
        offset_init = np.clip(offset_init, bounds[0][2], bounds[1][2])

    return [[float(s0_init[i]), float(t_init[i]), float(offset_init[i])] for i in range(len(pixel_data_list))]


def _fit_chunk(
    pixel_data_list: list,
    p0_list: list = None,
    x: np.ndarray = None,
    fit_function: Callable = None,
    bounds: Optional[Tuple[float, float]] = None,
    config: Optional[dict] = None,
    normalize: bool = False,
    calc_p0: bool = True,
) -> list:
    """Fit a chunk of pixels and return (params, r2) tuples."""
    results = []
    for idx, y in enumerate(pixel_data_list):
        p0 = p0_list[idx] if p0_list is not None else None
        param = fit_pixel(y, x, fit_function, bounds, config, normalize, calc_p0, p0_override=p0)
        if param is None:
            results.append((None, 0.0))
        else:
            y_eval = y / (np.max(y) if normalize and np.max(y) > 0 else 1)
            residuals = y_eval - fit_function(x, *param)
            r2 = float(get_r2(residuals, y_eval))
            results.append((param, r2))
    return results


def fit_pixel(
    y: np.ndarray,
    x: np.ndarray,
    fit_function: Callable,
    bounds: Optional[Tuple[float, float]] = None,
    config: Optional[dict] = None,
    normalize: bool = False,
    calc_p0: bool = True,
    p0_override: Optional[list] = None,
) -> Union[np.ndarray, None]:
    if normalize:
        y = y.copy()
        y /= np.max(y) if np.max(y) > 0 else 1

    if p0_override is not None:
        p0 = p0_override
        kwargs = {"xtol": 1e-6, "p0": p0}
    elif calc_p0:
        S0_init = np.max(y)
        offset_init = np.min(y)
        slope, _ = np.polyfit(x, np.log(y - offset_init + 0.0001), 1)
        t2_t2star_init = -1.0 / slope

        if t2_t2star_init > 5:
            t2_t2star_init = 5

        if bounds is None:
            p0 = [S0_init, np.exp(t2_t2star_init), offset_init]
        else:
            p0 = [
                max(S0_init, bounds[0][0]),
                max(np.exp(t2_t2star_init), bounds[0][1]),
                max(offset_init, bounds[0][2]),
            ]
            p0 = [
                min(p0[0], bounds[1][0]),
                min(p0[1], bounds[1][1]),
                min(p0[2], bounds[1][2]),
            ]

        kwargs = {"xtol": 1e-6, "p0": p0}
    else:
        kwargs = {"xtol": 1e-6}

    if bounds is not None:
        kwargs["bounds"] = bounds
    if config is not None:
        kwargs.update(config)

    try:
        param, _ = curve_fit(fit_function, x, y, **kwargs)
    except (RuntimeError, ValueError):
        param = None
    return param


def calculate_r2(
    y: np.ndarray,
    fit_function: Callable,
    param: np.ndarray,
    x: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Calculate the R squared value of the fitted curve.

    Parameters:
    - y (np.ndarray): 1D array of dependent variable data
    - fit_function (Callable): function used for fitting the curve
    - param (np.ndarray): array of curve fit parameters
    - x (np.ndarray): 1D array of independent variable data
    - normalize (bool, optional): normalize the data before calculation (default is False)

    Returns:
    - float: R squared value
    """
    if normalize:
        y /= y.max()
    residuals = y - fit_function(x, *param)
    return get_r2(residuals, y)


@njit
def get_r2(residuals: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the R squared value from the residuals and dependent variable data.

    Parameters:
    - residuals (np.ndarray): Residuals of the fit
    - y (np.ndarray): 1D array of dependent variable data

    Returns:
    - float: R squared value
    """
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Handle edge case: all y values are equal (constant signal)
    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)
