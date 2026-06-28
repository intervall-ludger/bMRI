# bMRI

Quantitative MRI analysis library. Per-voxel fitting of T1, T2, T2* and T1ρ
relaxation times from DICOM data, with a native Rust backend for fast
whole-volume fits, ROI statistics, GLCM textures and a built-in web viewer.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why bMRI

Most relaxometry pipelines stitch together scipy `curve_fit`, custom DICOM
loaders, and ad-hoc ROI scripts. bMRI does all of that in one place with a
consistent API, sane defaults, region-specific parameter bounds, and an
optional Rust backend that fits a 256³ volume in seconds instead of minutes.

## Installation

```bash
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI
uv sync
```

Optional extras:

```bash
uv sync --extra viewer      # web viewer (FastAPI + image overlay)
uv sync --extra qt          # PyQt5 + pyvista (heavy 3D visualisation)
uv sync --extra mcp         # MCP server
uv sync --extra dev         # tests, ruff, mypy
```

### Optional: native Rust backend

A native Levenberg-Marquardt backend can fit whole volumes in a fraction
of the scipy time. It is optional. Without it, bMRI uses scipy.

```bash
cd rust
uvx maturin build --release
uv pip install target/wheels/bmri_fit-*.whl
```

After install, pass `method="rust"` to `fit()`.

## What you can fit

| Family | Class | Params |
|---|---|---|
| T2, T2*, T1ρ mono-exp | `T2T2star`, `T1rho_T2prep` | 3 |
| T1 inversion recovery | `InversionRecoveryT1` | 3 |
| DWI ADC | `DWIMonoExp` | 3 |
| DWI Kurtosis | `DWIKurtosis` | 3 |
| DWI IVIM bi-exp | `DWIIvim` | 4 |
| T2\* bi-exponential | `T2StarBiExp` | 4 |
| Stretched exponential | `StretchedExp` | 3 |
| **Your own model as string** | `CustomExpression` | up to 4 |

All of these work with `method="rust"` for a 20-40\* speed-up.

## Quick start

### From Python

```python
from bmri.fitting import T2T2star
from bmri.io import load_nii

fitter = T2T2star(
    dim=3,
    boundary=([0.9, 0, -1.0], [3.0, 100.0, 1.0]),
    normalize=True,
)
data, te = fitter.read_data("path/to/T2_dicom_folder")
mask = load_nii("path/to/mask.nii.gz").array

fit_maps, r2 = fitter.fit(data, mask, te, method="rust")
# fit_maps[0] = S0,  fit_maps[1] = T2 in ms,  fit_maps[2] = offset
```

### Region-specific bounds

When one volume contains tissues with very different relaxation times (for
example cartilage and fat pad), you can pass per-label bounds in a single
fit instead of fitting twice and merging:

```python
fit_maps, r2 = fitter.fit(
    data,
    mask,
    te,
    method="rust",
    fit_region="full",
    region_bounds={
        1: ([0.9, 0, -1], [3, 100, 1]),    # cartilage
        6: ([0.9, 0, -1], [3, 500, 1]),    # fat pad
    },
    min_r2=0.8,
)
```

### From the CLI

```bash
bmri fit t2 --dicom path/to/T2_folder --mask mask.nii.gz --out output/
bmri fit t2star --dicom path/to/T2star_folder --mask mask.nii.gz --backend rust
```

### Your own model in one line

The Rust backend can fit any user-defined expression:

```python
from bmri.fitting import CustomExpression

model = CustomExpression(
    expression="S0 * exp(-x/T) + C",
    boundary=([0, 0, -1], [10, 200, 1]),
)
fit_maps, r2 = model.fit(data, mask, te, method="rust")
```

The expression is parsed and compiled to a stack-based program by Rust,
so per-voxel evaluation has zero Python-callback overhead. Same speed as
built-in models. Identifiers `x`, `p0..p7`, `S0`, `T`, `D`, `K`, `f`, `C`
are recognised; functions `exp`, `log`, `sin`, `cos`, `sqrt`, `pow` and
basic operators work as expected. See [docs/models.md](docs/models.md).

## API surface

| Module | What it gives you |
|---|---|
| `bmri.fitting` | `T2T2star`, `T1rho_T2prep`, `InversionRecoveryT1`, `DWIMonoExp`, `DWIKurtosis`, `DWIIvim`, `T2StarBiExp`, `StretchedExp`, `CustomExpression`, `AbstractFitting` |
| `bmri.io` | `load_nii`, `save_nii`, `get_dcm_list`, `get_dcm_array`, `split_dcm_list` |
| `bmri.postprocessing.texture` | GLCM texture features (Haralick) |
| `bmri.viewer` | Web-based DICOM + map viewer (needs `viewer` extra) |
| `bmri.cli` | `bmri ...` command-line entry point |

## Fitting backends

| Backend | When to use |
|---|---|
| `method="curvefit"` (default) | Reference, scipy-based, accepts any model |
| `method="loglinear"` | Fast linearised fit when no offset is needed |
| `method="rust"` | 20-40× faster than scipy on whole volumes, requires the optional wheel |

The Rust backend supports `fit_region="full"` to fit every voxel above a
`signal_threshold`, with `region_bounds` for per-label parameter ranges and
`min_r2` to drop poorly fitted voxels.

## Validation

Tests live in `test/`. Synthetic phantoms with known ground-truth
relaxation times (`test/phantoms.py`) validate that:

- the fit recovers known T2 within 0.5 % without noise
- both backends agree within 0.5 ms on the same data
- `fit_region="full"` and `fit_region="mask"` give identical results inside the mask
- per-label `region_bounds` separate disjoint tissue types

```bash
uv run pytest                       # full suite
uv run pytest test/test_fitting_accuracy.py test/test_backend_equivalence.py
```

## Project layout

```
src/
  bmri/                # public package (use this in your code)
    fitting/             # T1, T2, T2*, T1ρ
    io/                  # DICOM + NIfTI loaders
    postprocessing/      # GLCM and ROI stats
    viewer/              # web viewer
    cli/                 # typer entry points
  Fitting/, Utilitis/    # legacy modules, kept for backwards compatibility
test/
  phantoms.py            # synthetic phantoms
  test_fitting_accuracy.py
  test_backend_equivalence.py
  ...
rust/
  src/lib.rs             # native LM solver
```

Project-specific pipelines (study code, paper-specific scripts) live in
their own repositories and import bMRI as a dependency.

## License

MIT. See [LICENSE](LICENSE).
