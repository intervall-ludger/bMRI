import asyncio
import json
import shlex
import subprocess
import threading
import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bmri.viewer.data_loader import ViewerData, compute_stats, get_pixel_fit, load_results, render_overlay_png, render_slice_png

STATIC_DIR = Path(__file__).parent / "static"

VALID_CMAPS = {
    "hot", "viridis", "plasma", "inferno", "magma", "RdBu_r", "jet",
    "gray", "RdYlGn", "tab10", "coolwarm", "bone", "turbo",
}

app = FastAPI(title="bMRI Viewer")
state: ViewerData | None = None
patients: list[dict] = []
patient_idx: int = 0
fit_state: dict = {"running": False, "progress": 0, "message": "", "done": True, "error": None}
_fit_lock = threading.Lock()
_mask_cache: dict[int, dict] = {}


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    return JSONResponse({"error": "Internal server error"}, status_code=500)


def _is_valid_results_dir(d: Path) -> bool:
    return (d / "bmri_manifest.json").exists() or any(d.glob("*_map.nii.gz"))


def _detect_patients(results_dir: Path) -> list[dict]:
    subfolder_name = results_dir.name
    grandparent = results_dir.parent.parent

    # Pattern: grandparent/*/subfolder_name (e.g. Knie_01/T2star, Knie_02/T2star, ...)
    found = []
    for d in sorted(grandparent.iterdir()):
        if not d.is_dir():
            continue
        candidate = d / subfolder_name
        if candidate.is_dir() and _is_valid_results_dir(candidate):
            found.append({"id": d.name, "path": str(candidate)})

    if not found:
        for d in sorted(results_dir.parent.iterdir()):
            if d.is_dir() and _is_valid_results_dir(d):
                found.append({"id": d.name, "path": str(d)})

    return found


def _reload_state(results_path: Path):
    global state, _mask_cache
    state = load_results(results_path)
    _mask_cache = {}


def _init_app(results_dir: Path) -> FastAPI:
    global state, patients, patient_idx
    results_dir = Path(results_dir)

    # Parent dir takes priority over results_dir for config (parent is the study root)
    config_path = results_dir.parent / "bmri_config.json"
    if not config_path.exists():
        config_path = results_dir / "bmri_config.json"

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        patients = cfg.get("patients", [])

    if not patients:
        patients = _detect_patients(results_dir)

    patient_idx = next(
        (i for i, p in enumerate(patients) if Path(p["path"]) == results_dir),
        None,
    )
    if patient_idx is None:
        # results_dir not found in detected list — add it as sole entry
        patients = [{"id": results_dir.name, "path": str(results_dir)}]
        patient_idx = 0

    _reload_state(results_dir)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/manifest")
def get_manifest():
    return JSONResponse(state.manifest)


@app.get("/api/info")
def get_info():
    return JSONResponse({
        "num_slices": state.num_slices,
        "shape": list(state.shape),
        "parameters": state.parameters,
        "modality": state.modality,
        "has_dicom": state.dicom is not None,
        "has_mask": state.mask is not None,
        "has_r2": state.r2 is not None,
        "times": state.manifest.get("times"),
        "times_label": state.manifest.get("times_label", "Time"),
        "boundary": state.manifest.get("boundary"),
        "min_r2": state.manifest.get("min_r2"),
        "has_fit_cmd": bool(state.manifest.get("fit_cmd")),
    })


@app.get("/api/patients")
def get_patients():
    return JSONResponse({"patients": patients, "current": patient_idx})


@app.post("/api/patient/{idx}")
def switch_patient(idx: int):
    global patient_idx
    if idx < 0 or idx >= len(patients):
        return Response(status_code=404)
    with _fit_lock:
        if fit_state["running"]:
            return JSONResponse({"error": "Cannot switch while fitting"}, status_code=409)
    patient_idx = idx
    _reload_state(Path(patients[idx]["path"]))
    return JSONResponse({"ok": True, "id": patients[idx]["id"]})


@app.get("/api/slice/{layer}/{slice_idx}")
def get_slice(
    layer: str,
    slice_idx: int,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
):
    if cmap not in VALID_CMAPS:
        cmap = "gray"

    if layer == "dicom" and state.dicom is not None:
        arr = state.dicom
    elif layer == "r2" and state.r2 is not None:
        arr = state.r2
        cmap = cmap if cmap != "gray" else "RdYlGn"
    elif layer == "mask" and state.mask is not None:
        arr = state.mask.astype(float)
        cmap = "tab10"
        vmin, vmax = 0, 10
    elif layer in state.parameter_maps:
        arr = state.parameter_maps[layer]
        cmap = cmap if cmap != "gray" else "hot"
    else:
        return Response(status_code=404)

    png = render_slice_png(arr, slice_idx, cmap=cmap, vmin=vmin, vmax=vmax)
    return Response(content=png, media_type="image/png")


@app.get("/api/overlay/{slice_idx}")
def get_overlay(
    slice_idx: int,
    param: str = "",
    alpha: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "hot",
    mask: bool = True,
):
    if cmap not in VALID_CMAPS:
        cmap = "hot"
    if not param and state.parameters:
        param = state.parameters[0]
    png = render_overlay_png(state, slice_idx, param, alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap, show_mask=mask)
    return Response(content=png, media_type="image/png")


@app.get("/api/stats")
def get_stats():
    return JSONResponse(compute_stats(state))


@app.get("/api/pixel/{img_x}/{img_y}/{slice_idx}")
def get_pixel(img_x: int, img_y: int, slice_idx: int):
    result = get_pixel_fit(state, img_x, img_y, slice_idx)
    if result is None:
        return Response(status_code=404)
    return JSONResponse(result)


@app.get("/api/mask_data/{slice_idx}")
def get_mask_data(slice_idx: int):
    if state.mask is None:
        return JSONResponse({"data": [], "shape": [0, 0]})
    if slice_idx in _mask_cache:
        return JSONResponse(_mask_cache[slice_idx])
    s = int(np.clip(slice_idx, 0, state.num_slices - 1))
    mask_sl = np.rot90(state.mask[:, :, s], 3).astype(int)
    result = {"data": mask_sl.tolist(), "shape": list(mask_sl.shape)}
    _mask_cache[slice_idx] = result
    return JSONResponse(result)


@app.post("/api/fit")
async def start_fit():
    global fit_state
    with _fit_lock:
        if fit_state["running"]:
            return JSONResponse({"error": "already running"}, status_code=409)
        fit_cmd = state.manifest.get("fit_cmd")
        if not fit_cmd:
            return JSONResponse({"error": "No fit_cmd configured in manifest"}, status_code=400)
        # Capture results_dir before thread starts to avoid race with switch_patient
        results_dir = state.results_dir
        fit_state = {"running": True, "progress": 0, "message": "Starting...", "done": False, "error": None}

    def _run():
        try:
            cmd = fit_cmd if isinstance(fit_cmd, list) else shlex.split(fit_cmd)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(results_dir),
            )
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                with _fit_lock:
                    fit_state["message"] = line
                    if "%" in line:
                        try:
                            pct = float(line.split("%")[0].split()[-1])
                            fit_state["progress"] = int(pct)
                        except (ValueError, IndexError):
                            pass
            proc.wait()
            if proc.returncode != 0:
                with _fit_lock:
                    fit_state["error"] = f"Process exited with code {proc.returncode}"
            else:
                _reload_state(results_dir)
                with _fit_lock:
                    fit_state["message"] = "Done"
        except Exception:
            with _fit_lock:
                fit_state["error"] = "Fit process failed"
        finally:
            with _fit_lock:
                fit_state["running"] = False
                fit_state["done"] = True
                fit_state["progress"] = 100

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"started": True})


@app.get("/api/fit/progress")
async def fit_progress_sse():
    async def generate():
        while True:
            with _fit_lock:
                snapshot = dict(fit_state)
            data = json.dumps({
                "progress": snapshot["progress"],
                "message": snapshot["message"],
                "done": snapshot["done"],
                "error": snapshot["error"],
            })
            yield f"data: {data}\n\n"
            if snapshot["done"]:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def launch_viewer(results_dir: Path, port: int = 8050, open_browser: bool = True):
    _init_app(Path(results_dir))

    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=[f"http://localhost:{port}"]).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
