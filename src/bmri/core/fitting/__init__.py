"""Per-voxel relaxometric fitting.

Public API for T2, T2*, T1 and T1rho mapping. All classes accept a method
argument ("curvefit", "loglinear" or "rust") and an optional region_bounds
dict for label-specific parameter ranges.
"""

from Fitting.AbstractFitting import AbstractFitting
from Fitting.T1 import InversionRecoveryT1
from Fitting.T1rho_T2prep import T1rho_T2prep
from Fitting.T2_T2star import T2_T2star

# Cleaner aliases that match Python naming conventions.
T2T2star = T2_T2star
T1RhoT2prep = T1rho_T2prep
T1 = InversionRecoveryT1

__all__ = [
    "AbstractFitting",
    "InversionRecoveryT1",
    "T1",
    "T1rho_T2prep",
    "T1RhoT2prep",
    "T2_T2star",
    "T2T2star",
]
