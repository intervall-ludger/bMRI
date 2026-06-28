"""Legacy utilities. Prefer `from bmri.io import ...`."""
from Utilitis.read import (
    get_dcm_array,
    get_dcm_list,
    load_nii,
    split_dcm_list,
)
from Utilitis.results_writer import save_nii, save_results

__all__ = [
    "get_dcm_array",
    "get_dcm_list",
    "load_nii",
    "save_nii",
    "save_results",
    "split_dcm_list",
]
