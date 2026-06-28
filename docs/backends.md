# Fitting backends

| Backend | When to use |
|---|---|
| `method="curvefit"` (default) | scipy-based, accepts any 3-param model. Reference implementation. |
| `method="loglinear"` | Fast linearised solution. No offset support. |
| `method="rust"` | 20-40× faster than scipy on whole volumes. Optional, see install below. |

## Building the Rust backend

```bash
cd rust
uvx maturin build --release
uv pip install target/wheels/bmri_fit-*.whl
```

After install, set `method="rust"` on any fit call. The Rust backend also
supports:

- `fit_region="full"`: fit every voxel above `signal_threshold * max(signal)`
- `region_bounds={label: bounds}`: per-label parameter ranges
- `min_r2=0.8`: drop voxels whose R² is below the threshold (set to NaN)

## Performance

On a 256×256×96 T2* volume with 6 echoes, the Rust backend fits the full
volume in about 4 seconds on a M2 Mac. The scipy backend takes about
150 seconds for the same volume.
