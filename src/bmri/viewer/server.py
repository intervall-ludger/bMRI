import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from bmri.viewer.data_loader import ViewerData, compute_stats, get_pixel_fit, load_results, render_overlay_png, render_slice_png

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="bMRI Viewer")
state: ViewerData | None = None


def _init_app(results_dir: Path) -> FastAPI:
    global state
    state = load_results(results_dir)
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
    })


@app.get("/api/slice/{layer}/{slice_idx}")
def get_slice(
    layer: str,
    slice_idx: int,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
):
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
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "hot",
    mask: bool = True,
):
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
    """Return mask as flat array for cursor detection in frontend."""
    if state.mask is None:
        return JSONResponse({"data": [], "shape": [0, 0]})
    import numpy as np
    s = np.clip(slice_idx, 0, state.num_slices - 1)
    mask_sl = np.rot90(state.mask[:, :, s], 3).astype(int)
    return JSONResponse({
        "data": mask_sl.tolist(),
        "shape": list(mask_sl.shape),
    })


def launch_viewer(results_dir: Path, port: int = 8050, open_browser: bool = True):
    _init_app(Path(results_dir))

    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=[f"http://localhost:{port}"]).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
