from Utilitis.read import get_dcm_array, load_nii
from pathlib import Path
import numpy as np


class FittedMap:

    def __init__(self,
                 low_percentile: int = 5,
                 up_percentile: int = 95):

        assert low_percentile < up_percentile
        assert low_percentile >= 0 and low_percentile <= 100
        assert up_percentile >= 0 and up_percentile <= 100

        self.low_percentile = low_percentile
        self.up_percentile = up_percentile

    def __call__(self,
                 dcm_folder: str | Path,
                 mask_file: str | Path):
        dcm_folder = Path(dcm_folder)
        mask_file = Path(mask_file)
        fitted_map = get_dcm_array([_ for _ in dcm_folder.glob('*.dcm')]).transpose((2, 1, 0))
        mask = load_nii(mask_file)

        for i in range(1, int(mask.array.max()) + 1):
            fit_map = fitted_map.copy().astype('int16')
            fit_map[mask != i] = 0
            low = np.percentile(fit_map[mask == i], self.low_percentile)
            up = np.percentile(fit_map[mask == i], self.up_percentile)
            fit_map[fit_map < low] = 0
            fit_map[fit_map > up] = 0
            mask = np.where(fit_map != 0.0, fit_map, np.nan)

        return fitted_map, mask