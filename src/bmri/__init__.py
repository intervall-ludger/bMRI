"""bMRI - Quantitative MRI Analysis Framework.

Per-voxel fitting of T1, T2, T2* and T1rho relaxation times from DICOM data,
with a native Rust backend for fast volumes, ROI statistics, GLCM textures,
and a built-in web viewer.

Quick start:
    from bmri.fitting import T2T2star
    from bmri.io import load_nii

    fitter = T2T2star(dim=3, boundary=([0.9, 0, -1], [3, 100, 1]), normalize=True)
    data, te = fitter.read_data(dicom_folder)
    mask = load_nii(mask_path).array
    fit_maps, r2 = fitter.fit(data, mask, te, method="rust")
"""

import sys
from pathlib import Path

# Legacy modules (Fitting/, Utilitis/, Visualization/) sit one level up from
# this package. Add that to sys.path so they remain importable while we
# transition downstream code to the bmri.* namespace.
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

__version__ = "0.4.3"
__author__ = "Ludger Radke"

from bmri.config import T1Config, T1rhoConfig, T2Config
from bmri.exceptions import BMRIError, DICOMError, FittingError, ValidationError

__all__ = [
    "__version__",
    "BMRIError",
    "DICOMError",
    "FittingError",
    "ValidationError",
    "T1Config",
    "T1rhoConfig",
    "T2Config",
]
