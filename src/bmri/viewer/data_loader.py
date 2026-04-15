import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


@dataclass
class ViewerData:
    results_dir: Path
    manifest: dict = field(default_factory=dict)
    dicom: np.ndarray | None = None
    parameter_maps: dict[str, np.ndarray] = field(default_factory=dict)
    params_raw: np.ndarray | None = None  # All params including rejected fits
    r2: np.ndarray | None = None
    mask: np.ndarray | None = None
    times: np.ndarray | None = None
    num_slices: int = 0
    shape: tuple = ()

    @property
    def modality(self) -> str:
        return self.manifest.get("modality", "unknown")

    @property
    def parameters(self) -> list[str]:
        return list(self.parameter_maps.keys())


def load_nifti(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return nib.load(path).get_fdata()


def _detect_modality(results_dir: Path) -> str:
    for f in results_dir.glob("*_map.nii.gz"):
        name = f.stem.replace("_map.nii", "").replace(".gz", "")
        if "t1rho" in name.lower():
            return "t1rho"
        if "t2_t2star" in name.lower():
            # Check parent or sibling files for clues
            if any("t2star" in p.name.lower() for p in results_dir.parent.iterdir()):
                return "t2star"
            return "t2star"
    return "unknown"


def _build_fallback_manifest(results_dir: Path) -> dict:
    """Build manifest by scanning directory for known file patterns."""
    manifest = {"modality": _detect_modality(results_dir), "files": {}}

    # Find parameter maps
    param_maps = {}
    for f in sorted(results_dir.glob("*_map.nii.gz")):
        if f.name.startswith("._"):
            continue
        name = f.stem.replace("_map.nii", "")
        param_maps[name] = f.name
    manifest["files"]["parameter_maps"] = param_maps
    manifest["parameters"] = list(param_maps.keys())

    # Standard files
    for key, pattern in [("r2", "r2.nii.gz"), ("params", "params.nii.gz"), ("dicom", "dicom.nii.gz")]:
        p = results_dir / pattern
        if p.exists():
            manifest["files"][key] = pattern

    # Mask: check parent directory
    for mask_pattern in ["mask*.nii", "mask*.nii.gz"]:
        masks = list(results_dir.parent.glob(mask_pattern))
        if masks:
            manifest["files"]["mask"] = str(masks[0])
            break

    # Times
    times_file = results_dir / "acquisition_times.txt"
    if times_file.exists():
        manifest["times"] = np.loadtxt(times_file).tolist()

    return manifest


def load_results(results_dir: Path) -> ViewerData:
    """Load all fitting results from a directory."""
    results_dir = Path(results_dir)
    data = ViewerData(results_dir=results_dir)

    # Try manifest first, fallback to auto-detection
    manifest_path = results_dir / "bmri_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            data.manifest = json.load(f)
    else:
        data.manifest = _build_fallback_manifest(results_dir)

    files = data.manifest.get("files", {})

    # Load parameter maps
    for name, filename in files.get("parameter_maps", {}).items():
        arr = load_nifti(results_dir / filename)
        if arr is not None:
            data.parameter_maps[name] = arr

    # Load R²
    if "r2" in files:
        data.r2 = load_nifti(results_dir / files["r2"])

    # Load raw params (all fits, including rejected)
    if "params" in files:
        data.params_raw = load_nifti(results_dir / files["params"])
    else:
        params_path = results_dir / "params.nii.gz"
        if params_path.exists():
            data.params_raw = load_nifti(params_path)

    # Load DICOM
    if "dicom" in files:
        data.dicom = load_nifti(results_dir / files["dicom"])

    # Load mask
    if "mask" in files:
        mask_path = Path(files["mask"])
        if not mask_path.is_absolute():
            mask_path = results_dir / mask_path
        data.mask = load_nifti(mask_path)
        if data.mask is not None:
            data.mask = data.mask.round().astype(np.int16)

    # Load times
    times = data.manifest.get("times")
    if times:
        data.times = np.array(times)
    else:
        times_file = results_dir / "acquisition_times.txt"
        if times_file.exists():
            data.times = np.loadtxt(times_file)

    # Determine shape from first available volume
    for arr in [data.r2, data.mask, *data.parameter_maps.values()]:
        if arr is not None:
            data.shape = arr.shape
            data.num_slices = arr.shape[-1]
            break

    return data


def render_slice_png(
    arr: np.ndarray,
    slice_idx: int,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> bytes:
    """Render a 2D slice as PNG bytes."""
    s = np.clip(slice_idx, 0, arr.shape[-1] - 1)
    if arr.ndim == 3:
        sl = np.rot90(arr[:, :, s], 3)
    elif arr.ndim == 4:
        sl = np.rot90(arr[0, :, :, s], 3)
    else:
        sl = arr

    sl = sl.astype(np.float64)
    sl[~np.isfinite(sl)] = np.nan

    buf = io.BytesIO()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.axis("off")
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_overlay_png(
    data: ViewerData,
    slice_idx: int,
    param_name: str,
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "hot",
    show_mask: bool = True,
) -> bytes:
    """Render DICOM + parameter map overlay as PNG. Output is exactly shape[0] x shape[1] pixels."""
    from PIL import Image

    s = np.clip(slice_idx, 0, data.num_slices - 1)

    # Start with DICOM grayscale background
    if data.dicom is not None:
        dcm_sl = data.dicom[:, :, s] if data.dicom.ndim == 3 else data.dicom[0, :, :, s]
        dcm_rot = np.rot90(dcm_sl, 3).astype(np.float64)
        dcm_norm = np.clip((dcm_rot - dcm_rot.min()) / (dcm_rot.max() - dcm_rot.min() + 1e-10), 0, 1)
        canvas = np.stack([dcm_norm] * 3, axis=-1)  # (H_rot, W_rot, 3)
    else:
        # rot90(k=3) swaps dimensions: (h, w) -> (w, h)
        canvas_h, canvas_w = data.shape[1], data.shape[0]
        canvas = np.zeros((canvas_h, canvas_w, 3))

    # Parameter map overlay (opaque where valid)
    if param_name in data.parameter_maps:
        pmap = data.parameter_maps[param_name][:, :, s].astype(np.float64)
        pmap_rot = np.rot90(pmap, 3)
        valid = np.isfinite(pmap_rot) & (pmap_rot > 0)

        if vmin is None:
            v = pmap_rot[valid]
            vmin = float(np.percentile(v, 2)) if len(v) > 0 else 0
        if vmax is None:
            v = pmap_rot[valid]
            vmax = float(np.percentile(v, 98)) if len(v) > 0 else 1

        colormap = plt.get_cmap(cmap)
        pmap_normed = np.clip((pmap_rot - vmin) / (vmax - vmin + 1e-10), 0, 1)
        pmap_colored = colormap(pmap_normed)[:, :, :3]  # (H, W, 3)
        canvas[valid] = pmap_colored[valid]

    # Mask contours as colored outlines
    if show_mask and data.mask is not None and s < data.mask.shape[-1]:
        mask_sl = np.rot90(data.mask[:, :, s], 3)
        tab10 = plt.get_cmap("tab10")
        for roi in np.unique(mask_sl):
            if roi == 0:
                continue
            roi_mask = (mask_sl == roi).astype(np.uint8)
            # Simple edge detection: dilate - original
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(roi_mask, iterations=1)
            edge = dilated & ~roi_mask.astype(bool)
            c = tab10(int(roi) % 10)[:3]
            canvas[edge] = c

    # Convert to PIL and save as PNG
    canvas_uint8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(canvas_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def compute_stats(data: ViewerData) -> dict:
    """Compute per-ROI statistics for each parameter map."""
    if data.mask is None:
        return {}

    stats = {}
    for name, pmap in data.parameter_maps.items():
        roi_stats = []
        for roi in sorted(np.unique(data.mask)):
            if roi == 0:
                continue
            roi_mask = (data.mask == roi) & np.isfinite(pmap) & (pmap > 0)
            total = int(np.sum(data.mask == roi))
            valid = int(np.sum(roi_mask))
            if valid == 0:
                roi_stats.append({"roi": int(roi), "pixels": 0, "total": total, "mean": None, "std": None})
                continue
            vals = pmap[roi_mask]
            roi_stats.append({
                "roi": int(roi),
                "pixels": valid,
                "total": total,
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals)), 2),
            })
        stats[name] = roi_stats

    # R² stats
    if data.r2 is not None:
        r2_stats = []
        for roi in sorted(np.unique(data.mask)):
            if roi == 0:
                continue
            roi_mask = (data.mask == roi) & (data.r2 > 0)
            if np.any(roi_mask):
                r2_stats.append({
                    "roi": int(roi),
                    "mean_r2": round(float(np.mean(data.r2[roi_mask])), 3),
                })
        stats["r2"] = r2_stats

    return stats


def get_pixel_fit(data: ViewerData, img_x: int, img_y: int, slice_idx: int) -> dict | None:
    """Get signal decay + fit curve for a pixel. img_x=col, img_y=row in rotated PNG."""
    s = np.clip(slice_idx, 0, data.num_slices - 1)
    N = data.shape[0]

    # Invert rot90(k=3): rotated[row, col] came from arr[N-1-col, row]
    arr_x = N - 1 - img_x
    arr_y = img_y

    if arr_x < 0 or arr_x >= data.shape[0] or arr_y < 0 or arr_y >= data.shape[1]:
        return None

    result = {
        "x": int(arr_x),
        "y": int(arr_y),
        "z": int(s),
        "roi": int(data.mask[arr_x, arr_y, s]) if data.mask is not None else 0,
    }

    # Parameter values from filtered maps (good R² only)
    param_values = {}
    for name, pmap in data.parameter_maps.items():
        v = float(pmap[arr_x, arr_y, s])
        param_values[name] = v if np.isfinite(v) and v > 0 else None
    result["params"] = param_values
    result["rejected"] = all(v is None for v in param_values.values()) and result["roi"] > 0

    # Try raw params (includes rejected fits) for fit curve
    # params.nii.gz order matches the fit function: S0, t2_t2star, offset (NOT alphabetical)
    raw_params = {}
    if data.params_raw is not None and data.params_raw.ndim == 4:
        # Determine correct order from manifest or guess from common patterns
        param_order = data.manifest.get("parameters")
        if not param_order:
            # Default fitting order: S0 first, then relaxation param, then offset
            names = list(data.parameter_maps.keys())
            relax = [n for n in names if "t2" in n.lower() or "t1rho" in n.lower()]
            s0 = [n for n in names if n.lower() == "s0"]
            offset = [n for n in names if n.lower() == "offset"]
            param_order = s0 + relax + offset
        for i, name in enumerate(param_order):
            if i < data.params_raw.shape[0]:
                v = float(data.params_raw[i, arr_x, arr_y, s])
                if np.isfinite(v):
                    raw_params[name] = v
    # Use raw params as fallback when filtered maps are empty
    display_params = {k: param_values.get(k) or raw_params.get(k) for k in param_values}
    result["params"] = display_params

    # R² at this pixel
    if data.r2 is not None:
        r2_val = float(data.r2[arr_x, arr_y, s])
        result["r2"] = r2_val if r2_val > 0 else None
    else:
        result["r2"] = None

    # Signal decay curve from DICOM
    if data.dicom is not None and data.times is not None:
        times = data.times.tolist()
        if data.dicom.ndim == 4:
            signal = [float(data.dicom[e, arr_x, arr_y, s]) for e in range(data.dicom.shape[0])]
        elif data.dicom.ndim == 3:
            signal = [float(data.dicom[arr_x, arr_y, s])]
        else:
            signal = []

        result["times"] = times
        result["signal"] = signal

        # Reconstruct fit curve — use raw params, scale to raw signal
        fit_params = {k: display_params.get(k) or raw_params.get(k) for k in display_params}
        main_param = None
        for name in data.parameter_maps:
            if "t2" in name.lower() or "t1rho" in name.lower():
                main_param = name
                break

        s0 = fit_params.get("S0")
        t_relax = fit_params.get(main_param) if main_param else None

        # If no valid params (rejected pixel IN mask), do a quick log-linear fit
        if (not s0 or s0 < 0 or not t_relax or t_relax <= 0) and len(signal) >= 2 and result["roi"] > 0:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
            from src.Fitting.AbstractFitting import loglinear_fit
            s0_arr, t_arr = loglinear_fit(np.array(times), np.array(signal))
            s0 = float(s0_arr)
            t_relax = float(t_arr)
            if t_relax <= 0:
                t_relax = None
            else:
                fit_params["S0"] = s0
                fit_params[main_param] = t_relax
                fit_params["offset"] = 0
                result["params"] = fit_params
                result["params_source"] = "loglinear"

        if s0 and s0 > 0 and t_relax and t_relax > 0:
            offset = fit_params.get("offset", 0) or 0
            t_fit = np.linspace(min(times), max(times), 50)
            y_fit = s0 * np.exp(-t_fit / t_relax) + offset

            # If params are normalized (S0 ~ 1.0), scale fit to raw signal range
            max_signal = max(signal) if signal else 1
            if s0 < 10 and max_signal > 10:
                scale = max_signal / s0
                y_fit = y_fit * scale

            result["fit_times"] = t_fit.tolist()
            result["fit_signal"] = y_fit.tolist()

    return result
