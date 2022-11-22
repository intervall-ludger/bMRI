from pathlib import Path

import numpy as np

from Utilitis.read import get_dcm_list, get_dcm_array, split_dcm_list
from Fitting.AbstractFitting import AbstractFitting


# Assessment of T1, T1ρ, and T2 values of the ulnocarpal disc in healthy subjects at 3 tesla
# DOI: 10.1016/j.mri.2014.05.010
# Eq. 4

def fit_T1rho_wrapper_raush(TR, T1, alpha):
    def fit(x, S0, t1rho, offset):
        counter = (1-np.exp(-(TR - x) / T1))*np.exp(-x/t1rho)
        denominator = 1-np.cos(alpha)*np.exp(-x/t1rho)*np.exp(-(TR - x) / T1)
        return S0 * np.sin(alpha) * counter / denominator + offset
    return fit

# 3D SPIN-LOCK IMAGING OF HUMAN GLIOMAS
# https://doi.org/10.1016/S0730-725X(99)00041-7
# Appendix
def fit_T1rho_wrapper_aronen(TR, T1, alpha, TE, T2star):
    def fit(x, S0, t1rho, offset):
        tau = TR - x
        counter = S0 * np.exp(-x / t1rho) * (1 - np.exp(-tau / T1)) * np.sin(alpha) * np.exp(-TE / T2star)
        denominator = 1 - np.cos(alpha) * np.exp(tau / T1) * np.exp(x / t1rho)
        return counter / denominator + offset
    return fit


class T1rho(AbstractFitting):

    def __init__(self, dim, config, boundary=None):
        #fit = fit_T1rho_wrapper_raush(config["TR"], config["T1"], config["alpha"])
        fit = fit_T1rho_wrapper_aronen(config["TR"], config["T1"], config["alpha"], config["TE"], config["T2star"])
        super(T1rho, self).__init__(fit, boundary=boundary)
        self.dim = dim

    def set_fit_config(self):
        pass

    def read_data(self, folder: str | Path):

        folder = Path(folder)
        if self.dim == 2:
            dcm_files = get_dcm_list(folder)
            dcm_files = [[dcm] for dcm in dcm_files]
        elif self.dim == 3:
            dcm_files = get_dcm_list(folder)
            if len(dcm_files) == 0:
                echos = folder.glob('*/')
                dcm_files = [get_dcm_list(echo) for echo in echos]
                dcm_files = [item for sublist in dcm_files for item in sublist]
            dcm_files = split_dcm_list(dcm_files)
        else:
            raise NotImplementedError
        # echos, z, x, y --> echos, x, y, z
        dicom = np.array([get_dcm_array(dcm) for dcm in dcm_files]).transpose(0, 3, 2, 1)
        return dicom, None
