let info = {};
let stats = {};
let currentSlice = 0;
let currentParam = "";
let zoomLevel = 1.0;
let maskData = null;
let maskShape = [0, 0];

const img = document.getElementById("main-img");
const sliceSlider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");
const vminInput = document.getElementById("vmin");
const vmaxInput = document.getElementById("vmax");
const maskToggle = document.getElementById("mask-toggle");
const statsBody = document.getElementById("stats-body");
const metaBody = document.getElementById("meta-body");
const fitChart = document.getElementById("fit-chart");
const fitInfo = document.getElementById("fit-overlay-info");
const zoomSlider = document.getElementById("zoom-slider");
const zoomLabel = document.getElementById("zoom-label");
const container = document.getElementById("main-container");
const markerCanvas = document.getElementById("marker-canvas");
const paramTabs = document.getElementById("param-tabs");

// --- Init ---

async function init() {
    info = await (await fetch("/api/info")).json();
    stats = await (await fetch("/api/stats")).json();

    document.getElementById("header-info").textContent =
        `${info.shape[0]}×${info.shape[1]}×${info.shape[2]}  •  ${info.modality.toUpperCase()}`;

    sliceSlider.max = info.num_slices - 1;
    currentSlice = Math.floor(info.num_slices / 2);
    sliceSlider.value = currentSlice;

    // Build parameter tabs
    paramTabs.innerHTML = "";
    for (const p of info.parameters) {
        const btn = document.createElement("button");
        btn.className = "param-tab";
        btn.textContent = p;
        btn.dataset.param = p;
        btn.addEventListener("click", () => selectParam(p));
        paramTabs.appendChild(btn);
    }
    if (info.parameters.length > 0) {
        const mainParam = info.parameters.find(p => p.includes("t2") || p.includes("t1rho")) || info.parameters[0];
        selectParam(mainParam);
    }

    renderMeta();
    updateSlice();

    // Load center pixel fit chart
    const centerX = Math.floor(info.shape[0] / 2);
    const centerY = Math.floor(info.shape[1] / 2);
    try {
        const resp = await fetch(`/api/pixel/${centerX}/${centerY}/${currentSlice}`);
        if (resp.ok) {
            const px = await resp.json();
            drawFitChart(px);
        }
    } catch (e) {
        fitInfo.textContent = "Error loading fit data";
    }
}

function selectParam(name) {
    currentParam = name;
    document.querySelectorAll(".param-tab").forEach(t =>
        t.classList.toggle("active", t.dataset.param === name));
    updateSlice();
    updateStats();
}

function cycleParam(dir) {
    const idx = info.parameters.indexOf(currentParam);
    const next = (idx + dir + info.parameters.length) % info.parameters.length;
    selectParam(info.parameters[next]);
}

// --- Rendering ---

function buildOverlayUrl() {
    const params = new URLSearchParams({
        param: currentParam,
        alpha: 1.0,
        mask: maskToggle.checked,
    });
    if (vminInput.value !== "") params.set("vmin", vminInput.value);
    if (vmaxInput.value !== "") params.set("vmax", vmaxInput.value);
    return `/api/overlay/${currentSlice}?${params}`;
}

async function updateSlice() {
    sliceLabel.textContent = `${parseInt(currentSlice) + 1}/${info.num_slices}`;
    img.src = buildOverlayUrl();
    if (info.has_mask) {
        const resp = await fetch(`/api/mask_data/${currentSlice}`);
        const d = await resp.json();
        maskData = d.data;
        maskShape = d.shape;
    }
}

function updateStats() {
    const paramStats = stats[currentParam] || [];
    const r2Map = {};
    (stats["r2"] || []).forEach(s => r2Map[s.roi] = s.mean_r2);

    statsBody.innerHTML = "";
    for (const s of paramStats) {
        const r2 = r2Map[s.roi];
        const r2Class = r2 >= 0.95 ? "r2-good" : r2 >= 0.85 ? "r2-ok" : "r2-bad";
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><b>${s.roi}</b></td>
            <td>${s.mean !== null ? s.mean.toFixed(1) : "–"}</td>
            <td>${s.std !== null ? s.std.toFixed(1) : "–"}</td>
            <td>${s.pixels}</td>
            <td>${r2 ? `<span class="r2-badge ${r2Class}">${r2.toFixed(2)}</span>` : "–"}</td>
        `;
        statsBody.appendChild(tr);
    }
}

function renderMeta() {
    let html = "";
    const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
    if (info.times) {
        const label = info.modality === "t1rho" ? "TSL" : "TE";
        html += row(label, info.times.map(t => t.toFixed(1)).join(", ") + " ms");
    }
    if (info.boundary) html += row("Bounds", `[${info.boundary[0].join(",")}] [${info.boundary[1].join(",")}]`);
    if (info.min_r2 != null) html += row("Min R²", info.min_r2);
    html += row("DICOM", info.has_dicom ? "✓" : "✗");
    html += row("Mask", info.has_mask ? "✓" : "✗");
    metaBody.innerHTML = html;
}

// --- Zoom + Pan ---

function applyZoom() {
    if (img.naturalWidth > 0) {
        img.style.width = (img.naturalWidth * zoomLevel) + "px";
        img.style.height = (img.naturalHeight * zoomLevel) + "px";
    }
    zoomLabel.textContent = zoomLevel.toFixed(1) + "x";
    zoomSlider.value = Math.round(zoomLevel * 100);
}

zoomSlider.addEventListener("input", () => { zoomLevel = zoomSlider.value / 100; applyZoom(); });

container.addEventListener("wheel", (e) => {
    e.preventDefault();
    const oldZoom = zoomLevel;
    zoomLevel = Math.max(0.5, Math.min(5, zoomLevel + (e.deltaY < 0 ? 0.15 : -0.15)));
    if (zoomLevel !== oldZoom) {
        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left + container.scrollLeft;
        const my = e.clientY - rect.top + container.scrollTop;
        const ratio = zoomLevel / oldZoom;
        applyZoom();
        container.scrollLeft = mx * ratio - (e.clientX - rect.left);
        container.scrollTop = my * ratio - (e.clientY - rect.top);
    }
}, { passive: false });

let isPanning = false;
let didPan = false;
let panStartX, panStartY, scrollStartX, scrollStartY;

container.addEventListener("mousedown", (e) => {
    if (e.button === 1 || (e.button === 0 && zoomLevel > 1)) {
        isPanning = true;
        didPan = false;
        panStartX = e.clientX;
        panStartY = e.clientY;
        scrollStartX = container.scrollLeft;
        scrollStartY = container.scrollTop;
        container.style.cursor = "grabbing";
        e.preventDefault();
    }
});

document.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    const dx = e.clientX - panStartX;
    const dy = e.clientY - panStartY;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) didPan = true;
    container.scrollLeft = scrollStartX - dx;
    container.scrollTop = scrollStartY - dy;
});

document.addEventListener("mouseup", () => {
    if (isPanning) {
        isPanning = false;
        container.style.cursor = zoomLevel > 1 ? "grab" : "crosshair";
    }
});

// --- Hover cursor ---

function isInMask(imgX, imgY) {
    // maskData is rotated [row][col], imgY=row, imgX=col
    if (!maskData || imgY < 0 || imgY >= maskShape[0] || imgX < 0 || imgX >= maskShape[1]) return false;
    return maskData[imgY][imgX] > 0;
}

img.addEventListener("mousemove", (e) => {
    if (isPanning) return;
    const rect = img.getBoundingClientRect();
    const ax = Math.round((e.clientX - rect.left) / rect.width * info.shape[0]);
    const ay = Math.round((e.clientY - rect.top) / rect.height * info.shape[1]);
    if (zoomLevel > 1) {
        container.style.cursor = "grab";
    } else {
        container.style.cursor = isInMask(ax, ay) ? "pointer" : "crosshair";
    }
});

// --- Pixel click ---

img.addEventListener("click", async (e) => {
    if (didPan) { didPan = false; return; }

    const rect = img.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;
    // PNG is exactly shape[0] x shape[1] pixels, so direct mapping works
    const imgX = Math.round(relX * info.shape[0]);
    const imgY = Math.round(relY * info.shape[1]);

    if (imgX < 0 || imgX >= info.shape[0] || imgY < 0 || imgY >= info.shape[1]) return;

    const resp = await fetch(`/api/pixel/${imgX}/${imgY}/${currentSlice}`);
    if (!resp.ok) { fitInfo.textContent = "No data"; return; }
    const px = await resp.json();
    drawPixelMarker(relX, relY, px.roi > 0);
    drawFitChart(px);
});

function drawPixelMarker(relX, relY, inRoi) {
    const w = markerCanvas.width = img.clientWidth;
    const h = markerCanvas.height = img.clientHeight;
    const ctx = markerCanvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    const cx = relX * w, cy = relY * h;
    const size = Math.max(6, w * 0.015);
    ctx.strokeStyle = inRoi ? "#7cd5ff" : "#ff6b6b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - size, cy); ctx.lineTo(cx + size, cy);
    ctx.moveTo(cx, cy - size); ctx.lineTo(cx, cy + size);
    ctx.stroke();
}

// --- Fit chart ---

function drawFitChart(px) {
    // Fixed canvas size to avoid clientWidth=0 issues
    const cw = 256, ch = 180;
    fitChart.width = cw * 2;
    fitChart.height = ch * 2;
    fitChart.style.width = cw + "px";
    fitChart.style.height = ch + "px";
    const ctx = fitChart.getContext("2d");
    ctx.scale(2, 2);

    const pad = { l: 42, r: 8, t: 28, b: 28 };
    const pw = cw - pad.l - pad.r;
    const ph = ch - pad.t - pad.b;

    ctx.fillStyle = "#0b0c10";
    ctx.fillRect(0, 0, cw, ch);

    // Info line
    let infoText = `(${px.x}, ${px.y}, ${px.z})`;
    if (px.roi > 0) infoText += `  ROI ${px.roi}`;
    if (px.r2 != null) infoText += `  R²=${px.r2.toFixed(3)}`;
    if (px.rejected) infoText += `  ⚠ rejected`;
    fitInfo.textContent = infoText;
    fitInfo.style.color = px.rejected ? "#f44336" : "#aaa";

    // Param values
    const paramText = Object.entries(px.params)
        .filter(([, v]) => v != null && v > 0)
        .map(([k, v]) => `${k}=${v.toFixed(1)}`)
        .join("  ");
    ctx.fillStyle = "#888";
    ctx.font = "9px monospace";
    ctx.fillText(paramText, pad.l, pad.t - 8);

    if (!px.signal || px.signal.length === 0) {
        ctx.fillStyle = "#555";
        ctx.font = "11px monospace";
        ctx.textAlign = "center";
        ctx.fillText("No DICOM data", cw / 2, ch / 2);
        ctx.textAlign = "start";
        return;
    }

    const times = px.times, signal = px.signal;
    const allY = [...signal, ...(px.fit_signal || [])];
    const xMin = Math.min(...times), xMax = Math.max(...times);
    const yMin = Math.min(0, Math.min(...allY)), yMax = Math.max(...allY) * 1.1;
    if (yMax <= yMin) return;

    const toX = (t) => pad.l + (t - xMin) / (xMax - xMin) * pw;
    const toY = (v) => pad.t + ph - (v - yMin) / (yMax - yMin) * ph;

    // Grid
    ctx.strokeStyle = "#1a1c23";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ph * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = "#555";
    ctx.font = "8px monospace";
    ctx.textAlign = "center";
    for (const t of times) ctx.fillText(t.toFixed(0), toX(t), pad.t + ph + 14);
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
        const v = yMin + (yMax - yMin) * (4 - i) / 4;
        ctx.fillText(v.toFixed(0), pad.l - 4, pad.t + ph * i / 4 + 3);
    }
    ctx.textAlign = "center";
    ctx.fillStyle = "#444";
    ctx.fillText("time (ms)", pad.l + pw / 2, ch - 2);

    // Fit curve
    if (px.fit_times && px.fit_signal && px.fit_signal.length > 1) {
        ctx.strokeStyle = px.rejected ? "#ff9800" : "#ff6b6b";
        ctx.lineWidth = 2;
        if (px.rejected) ctx.setLineDash([5, 3]);
        ctx.beginPath();
        for (let i = 0; i < px.fit_times.length; i++) {
            const x = toX(px.fit_times[i]), y = toY(px.fit_signal[i]);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Data points
    for (let i = 0; i < times.length; i++) {
        const x = toX(times[i]), y = toY(signal[i]);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#7cd5ff";
        ctx.fill();
        ctx.strokeStyle = "#0b0c10";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// --- Events ---

sliceSlider.addEventListener("input", () => { currentSlice = sliceSlider.value; updateSlice(); });
vminInput.addEventListener("change", updateSlice);
vmaxInput.addEventListener("change", updateSlice);
maskToggle.addEventListener("change", updateSlice);
img.addEventListener("load", () => applyZoom());

// Keyboard
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        currentSlice = Math.max(0, parseInt(currentSlice) - 1);
        sliceSlider.value = currentSlice;
        updateSlice();
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        currentSlice = Math.min(info.num_slices - 1, parseInt(currentSlice) + 1);
        sliceSlider.value = currentSlice;
        updateSlice();
    } else if (e.key === "+" || e.key === "=") {
        zoomLevel = Math.min(5, zoomLevel + 0.25);
        applyZoom();
    } else if (e.key === "-") {
        zoomLevel = Math.max(0.5, zoomLevel - 0.25);
        applyZoom();
    } else if (e.key === "0") {
        zoomLevel = 1.0;
        applyZoom();
        container.scrollLeft = 0;
        container.scrollTop = 0;
    } else if (e.key === "Tab") {
        e.preventDefault();
        cycleParam(e.shiftKey ? -1 : 1);
    }
});

init();
