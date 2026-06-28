# bMRI

Quantitative MRI analysis library. Per-voxel fitting of T1, T2, T2* and T1ρ
relaxation times from DICOM data, with a native Rust backend for fast
whole-volume fits.

## Why

Most relaxometry pipelines stitch together scipy `curve_fit`, custom DICOM
loaders, and ad-hoc ROI scripts. bMRI does all of that in one place with a
consistent API, sane defaults, region-specific parameter bounds, and an
optional Rust backend that fits a 256³ volume in seconds instead of minutes.

## Install

```bash
git clone https://github.com/ludgerradke/bMRI.git
cd bMRI && uv sync
```

See [Quick start](quickstart.md) for your first fit.
