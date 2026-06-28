# Quick start

## Python

```python
from bmri.fitting import T2T2star
from bmri.io import load_nii

fitter = T2T2star(
    dim=3,
    boundary=([0.9, 0, -1.0], [3.0, 100.0, 1.0]),
    normalize=True,
)
data, te = fitter.read_data("path/to/T2_folder")
mask = load_nii("path/to/mask.nii.gz").array

fit_maps, r2 = fitter.fit(data, mask, te, method="rust")
# fit_maps[0] = S0, fit_maps[1] = T2 in ms, fit_maps[2] = offset
```

## Region-specific bounds

When one volume contains tissues with very different relaxation times
(cartilage vs fat pad), pass per-label bounds in a single fit instead of
fitting twice and merging:

```python
fit_maps, r2 = fitter.fit(
    data, mask, te,
    method="rust",
    fit_region="full",
    region_bounds={
        1: ([0.9, 0, -1], [3, 100, 1]),   # cartilage
        6: ([0.9, 0, -1], [3, 500, 1]),   # fat pad
    },
    min_r2=0.8,
)
```
