"""Type definitions and aliases for bMRI.

This module centralizes all type hints and provides type aliases for better
code readability and maintainability.
"""

from pathlib import Path
from typing import Callable, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

# Array types
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int32]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

# 2D arrays
FloatArray2D: TypeAlias = npt.NDArray[np.float64]  # Shape: (height, width)
IntArray2D: TypeAlias = npt.NDArray[np.int32]

# 3D arrays
FloatArray3D: TypeAlias = npt.NDArray[np.float64]  # Shape: (height, width, depth)
IntArray3D: TypeAlias = npt.NDArray[np.int32]

# 4D arrays (e.g., multi-echo DICOM)
FloatArray4D: TypeAlias = npt.NDArray[np.float64]  # Shape: (echoes, height, width, depth)

# Fitting-related types
TimePoints: TypeAlias = FloatArray  # Acquisition times (e.g., echo times, TSL)
Intensities: TypeAlias = FloatArray  # Signal intensities
FitParams: TypeAlias = tuple[float, ...]  # Fitted parameters (S0, T2, offset, etc.)
BoundaryTuple: TypeAlias = tuple[tuple[float, ...], tuple[float, ...]]  # ((lower...), (upper...))

# File paths
DICOMPath: TypeAlias = Path
NIfTIPath: TypeAlias = Path
CSVPath: TypeAlias = Path


class FitFunction(Protocol):
    """Protocol for curve fitting functions.

    Defines the signature that all fitting functions must follow.
    """

    def __call__(self, x: TimePoints, *params: float) -> FloatArray:
        """Evaluate the fitting function.

        Args:
            x: Independent variable (e.g., echo times)
            *params: Model parameters (e.g., S0, T2, offset)

        Returns:
            Predicted signal intensities
        """
        ...


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.

    Allows different progress reporting implementations (Rich, tqdm, etc.)
    """

    def __call__(self, n: int = 1) -> None:
        """Update progress by n steps.

        Args:
            n: Number of steps to advance
        """
        ...
