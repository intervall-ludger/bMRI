"""Per-voxel relaxometric fitting.

Public API for T2, T2*, T1, T1rho mapping plus DWI (ADC, Kurtosis, IVIM),
multi-component T2* and user-defined string expressions. All classes accept
a method argument ("curvefit", "loglinear" or "rust") and an optional
region_bounds dict for label-specific parameter ranges.
"""

from Fitting.AbstractFitting import AbstractFitting
from Fitting.dwi import (
    CustomExpression,
    DWIIvim,
    DWIKurtosis,
    DWIMonoExp,
    StretchedExp,
    T2StarBiExp,
)
from Fitting.T1 import InversionRecoveryT1
from Fitting.T1rho_T2prep import T1rho_T2prep
from Fitting.T2_T2star import T2_T2star

T2T2star = T2_T2star
T1RhoT2prep = T1rho_T2prep
T1 = InversionRecoveryT1

__all__ = [
    "AbstractFitting",
    "CustomExpression",
    "DWIIvim",
    "DWIKurtosis",
    "DWIMonoExp",
    "InversionRecoveryT1",
    "StretchedExp",
    "T1",
    "T1rho_T2prep",
    "T1RhoT2prep",
    "T2_T2star",
    "T2T2star",
    "T2StarBiExp",
]
