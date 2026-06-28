# Validation

bMRI is validated against synthetic phantoms with known ground-truth
relaxation times. The phantom generator lives in `test/phantoms.py`.

## What we test

- Without noise, the fit recovers the true T2 within 0.5 %.
- With moderate noise (σ = 0.01-0.02 on normalised signal), median
  recovery is within 5 %.
- A T2 gradient phantom is recovered slab by slab.
- Two disjoint regions (cartilage-like 45 ms vs fat-like 150 ms) are
  recovered without bleeding when fit jointly with `region_bounds`.
- Rust and scipy backends agree within 0.5 ms on the same data.
- `fit_region="full"` and `fit_region="mask"` give identical results
  inside the mask.
- `min_r2` correctly drops voxels whose fit quality falls below the
  threshold.

Run the validation suite locally:

```bash
uv run pytest test/test_fitting_accuracy.py test/test_backend_equivalence.py
```
