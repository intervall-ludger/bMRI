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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the given data using parallel processing.

        Parameters:
        - dicom (np.ndarray): The input data
        - mask (np.ndarray): The mask array
        - x (np.ndarray): Independent variable data
        - pools (int, optional): Number of parallel processes to use (default is the number of CPUs)
        - min_r2 (float, optional): Minimum R squared value to consider (default is negative infinity)

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Arrays of fit parameters and R squared values
        """
        dicom = dicom.astype("float64")
        assert len(mask.shape) == len(dicom.shape) - 1
        if len(mask.shape) == 2:
            # Change size of dicom and mask to 3D
            mask = np.expand_dims(mask, axis=2)
            dicom = np.expand_dims(dicom, axis=3)
        assert dicom.shape[0] == len(x)
        # Get the number of parameters in the fit function
        num_params = len(curve_fit(self.fit_function, x, dicom[:, 0, 0, 0])[0])

        # Initialize fit_maps list
        fit_maps = np.full((num_params, *mask.shape), np.nan)
        r2_map = np.zeros(mask.shape)

        # Create a partial function with the fixed arguments for fit_pixel
        fit_pixel_fixed = partial(
            fit_pixel,
            fit_function=self.fit_function,
            bounds=self.bounds,
            config=self.fit_config,
            normalize=self.normalize,
            calc_p0=True
            if len(get_function_parameter(self.fit_function)) == 3
            else False,
        )

        # Create an iterator of arguments to pass to fit_pixel
        pixel_args = zip(
            (dicom[:, i, j, k] for i, j, k in zip(*np.nonzero(mask))),
            repeat(x),
        )

        # Use a Pool to fit the pixels in parallel
        if pools != 0:
            with Pool(pools) as pool:
                # Call fit_pixel for each pixel and store the result in a list
                pixel_results = pool.starmap(fit_pixel_fixed, pixel_args)
        else:
            pixel_results = [fit_pixel_fixed(*p) for p in pixel_args]

        # Store fitting parameters and r2 values in appropriate arrays
        for i, j, k, param in zip(*np.nonzero(mask), pixel_results):
            if param is None:
                continue
            r2_map[i, j, k] = calculate_r2(
                dicom[:, i, j, k], self.fit_function, param, x, self.normalize
            )
            if r2_map[i, j, k] > min_r2:
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


def fit_pixel(
    y: np.ndarray,
    x: np.ndarray,
    fit_function: Callable,
    bounds: Optional[Tuple[float, float]] = None,
    config: Optional[dict] = None,
    normalize: bool = False,
    calc_p0: bool = True,
) -> Union[np.ndarray, None]:
    """
    Fits a curve to the given data using the provided fit function.

    Parameters:
    - y (np.ndarray): 1D array of dependent variable data
    - x (np.ndarray): 1D array of independent variable data
    - fit_function (Callable): function to use for fitting the curve
    - bounds (Tuple[float, float], optional): optional bounds for the curve fit parameters
    - config (dict, optional): optional config dictionary for curve_fit function
    - normalize (bool, optional): normalize the data before fitting (default is False)
    - calc_p0 (bool, optional): Calc initial parameter as start values

    Returns:
    - np.ndarray, None: array of curve fit parameters or None if fitting fails
    """

    # Normalize data if requested
    if normalize:
        y /= np.max(y) if np.max(y) > 0 else 1

    # Set initial parameter values based on data
    if calc_p0:
        S0_init = np.max(y)
        offset_init = np.min(y)
        slope, _ = np.polyfit(x, np.log(y - offset_init + 0.0001), 1)
        t2_t2star_init = -1.0 / slope

        if t2_t2star_init > 5:
            t2_t2star_init = 5  # np.exp(10) == 22026 ms

        # Check bounds and set initial parameter values accordingly
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

    # Set bounds and configuration if provided
    if bounds is not None:
        kwargs["bounds"] = bounds
    if config is not None:
        kwargs.update(config)

    try:
        # Perform curve fitting
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
    return 1 - (ss_res / ss_tot)
