"""Legacy fitting module. Prefer `from bmri.fitting import ...`."""
from Fitting.AbstractFitting import AbstractFitting
from Fitting.FittingMap import FittedMap
from Fitting.T1 import InversionRecoveryT1
from Fitting.T1rho_T2prep import T1rho_T2prep
from Fitting.T2_T2star import T2_T2star

__all__ = [
    "AbstractFitting",
    "FittedMap",
    "InversionRecoveryT1",
    "T1rho_T2prep",
    "T2_T2star",
]
