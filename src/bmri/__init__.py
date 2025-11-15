"""bMRI - Bio-sensitive MRI Analysis Framework.

A Python framework for analyzing MRI relaxation times (T1, T2, T2*, T1rho)
from DICOM data with professional CLI tools and rich terminal output.
"""

# Add src directory to path for legacy module imports
import sys
from pathlib import Path

_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

__version__ = "0.2.0"
__author__ = "Ludger Radke"

# Re-export main classes for convenience
from bmri.config import T1Config, T1rhoConfig, T2Config
from bmri.exceptions import BMRIError, DICOMError, FittingError, ValidationError

__all__ = [
    "BMRIError",
    "DICOMError",
    "FittingError",
    "ValidationError",
    "T2Config",
    "T1Config",
    "T1rhoConfig",
    "__version__",
]
