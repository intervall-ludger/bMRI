let info = {};
let stats = {};
let currentSlice = 0;
let currentParam = "";
let zoomLevel = 1.0;
let maskData = null;
let maskShape = [0, 0];
let maskCache = {};
let patients = [];
let patientIdx = 0;
let currentCmap = "hot";
let currentAlpha = 1.0;

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
const cmapSelect = document.getElementById("cmap-select");
const alphaSlider = document.getElementById("alpha-slider");
const alphaLabel = document.getElementById("alpha-label");
const patientNav = document.getElementById("patient-nav");
const patientSelect = document.getElementById("patient-select");
const prevPatientBtn = document.getElementById("prev-patient");
const nextPatientBtn = document.getElementById("next-patient");
const fitBtnGroup = document.getElementById("fit-btn-group");
const fitBtn = document.getElementById("fit-btn");
const fitProgressWrap = document.getElementById("fit-progress-wrap");
const fitProgressFill = document.getElementById("fit-progress-fill");
const fitStatus = document.getElementById("fit-status");

// --- Init ---

async function init() {
    setupEventListeners();
    await initPatients();
    await loadData();
}

async function loadData() {
    info = await (await fetch("/api/info")).json();
    stats = await (await fetch("/api/stats")).json();
    maskCache = {};
    maskData = null;

    document.getElementById("header-info").textContent =
        `${info.shape[0]}×${info.shape[1]}×${info.shape[2]}  •  ${info.modality.toUpperCase()}`;

    sliceSlider.max = info.num_slices - 1;
    currentSlice = Math.floor(info.num_slices / 2);
    sliceSlider.value = currentSlice;

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
    } else {
        updateSlice();
    }

    fitBtnGroup.style.display = info.has_fit_cmd ? "" : "none";
    renderMeta();
    updateStats();
}

// --- Patient navigation ---

async function initPatients() {
    const resp = await fetch("/api/patients");
    const d = await resp.json();
    patients = d.patients;
    patientIdx = d.current;

    if (patients.length <= 1) {
        patientNav.style.display = "none";
        return;
    }

    patientNav.style.display = "";
    patientSelect.innerHTML = "";
    for (let i = 0; i < patients.length; i++) {
        const opt = document.createElement("option");
        opt.value = i;
        opt.textContent = patients[i].id;
        if (i === patientIdx) opt.selected = true;
        patientSelect.appendChild(opt);
    }
}

async function switchPatient(idx) {
    if (idx < 0 || idx >= patients.length) return;
    const resp = await fetch(`/api/patient/${idx}`, { method: "POST" });
    if (!resp.ok) return;
    patientIdx = idx;
    patientSelect.value = idx;
    await loadData();
}

// --- Rendering ---

function buildOverlayUrl() {
    const params = new URLSearchParams({
        param: currentParam,
        alpha: currentAlpha,
        mask: maskToggle.checked,
        cmap: currentCmap,
    });
    if (vminInput.value !== "") params.set("vmin", vminInput.value);
    if (vmaxInput.value !== "") params.set("vmax", vmaxInput.value);
    return `/api/overlay/${currentSlice}?${params}`;
}

async function updateSlice() {
    sliceLabel.textContent = `${parseInt(currentSlice) + 1}/${info.num_slices}`;
    img.src = buildOverlayUrl();
    if (info.has_mask) {
        const d = await getMaskData(currentSlice);
        maskData = d.data;
        maskShape = d.shape;
    }
}

async function getMaskData(sliceIdx) {
    if (maskCache[sliceIdx]) return maskCache[sliceIdx];
    const resp = await fetch(`/api/mask_data/${sliceIdx}`);
    const d = await resp.json();
    maskCache[sliceIdx] = d;
    return d;
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
            <td>${r2 != null ? `<span class="r2-badge ${r2Class}">${r2.toFixed(2)}</span>` : "–"}</td>
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

let isPanning = false;
let didPan = false;
let panStartX, panStartY, scrollStartX, scrollStartY;

// --- Hover cursor ---

function isInMask(imgX, imgY) {
    if (!maskData || imgY < 0 || imgY >= maskShape[0] || imgX < 0 || imgX >= maskShape[1]) return false;
    return maskData[imgY][imgX] > 0;
}

img.addEventListener("mousemove", (e) => {
    if (isPanning) return;
    const rect = img.getBoundingClientRect();
    const ax = Math.round((e.clientX - rect.left) / rect.width * info.shape[0]);
    const ay = Math.round((e.clientY - rect.top) / rect.height * info.shape[1]);
    container.style.cursor = zoomLevel > 1 ? "grab" : (isInMask(ax, ay) ? "pointer" : "crosshair");
});

// --- Pixel click ---

img.addEventListener("click", async (e) => {
    if (didPan) { didPan = false; return; }

    const rect = img.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;
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

    let infoText = `(${px.x}, ${px.y}, ${px.z})`;
    if (px.roi > 0) infoText += `  ROI ${px.roi}`;
    if (px.r2 != null) infoText += `  R²=${px.r2.toFixed(3)}`;
    if (px.rejected) infoText += `  ⚠ rejected`;
    fitInfo.textContent = infoText;
    fitInfo.style.color = px.rejected ? "#f44336" : "#aaa";

    const paramText = Object.entries(px.params || {})
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

    const toX = t => pad.l + (t - xMin) / (xMax - xMin) * pw;
    const toY = v => pad.t + ph - (v - yMin) / (yMax - yMin) * ph;

    ctx.strokeStyle = "#1a1c23";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + ph * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    }

    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ph); ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.stroke();

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

// --- Fit from viewer ---

fitBtn.addEventListener("click", async () => {
    fitBtn.disabled = true;
    fitProgressWrap.style.display = "";
    fitProgressFill.style.width = "0%";
    fitStatus.textContent = "Starting...";

    await fetch("/api/fit", { method: "POST" });

    const evtSource = new EventSource("/api/fit/progress");
    evtSource.onmessage = (e) => {
        const d = JSON.parse(e.data);
        fitProgressFill.style.width = d.progress + "%";
        fitStatus.textContent = d.message || "";
        if (d.done) {
            evtSource.close();
            fitBtn.disabled = false;
            if (d.error) {
                fitStatus.textContent = "Error: " + d.error;
                fitStatus.style.color = "#f44336";
            } else {
                fitStatus.textContent = "Done";
                fitStatus.style.color = "#4caf50";
                setTimeout(() => loadData(), 500);
            }
        }
    };
    evtSource.onerror = () => {
        evtSource.close();
        fitBtn.disabled = false;
        fitStatus.textContent = "Connection error";
        fitStatus.style.color = "#f44336";
    };
});

// --- Event listeners setup ---

function setupEventListeners() {
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

    sliceSlider.addEventListener("input", () => { currentSlice = sliceSlider.value; updateSlice(); });
    vminInput.addEventListener("change", updateSlice);
    vmaxInput.addEventListener("change", updateSlice);
    maskToggle.addEventListener("change", updateSlice);
    img.addEventListener("load", () => applyZoom());

    cmapSelect.addEventListener("change", () => { currentCmap = cmapSelect.value; updateSlice(); });

    alphaSlider.addEventListener("input", () => {
        currentAlpha = alphaSlider.value / 100;
        alphaLabel.textContent = currentAlpha.toFixed(1);
        updateSlice();
    });

    patientSelect.addEventListener("change", () => switchPatient(parseInt(patientSelect.value)));
    prevPatientBtn.addEventListener("click", () => switchPatient(patientIdx - 1));
    nextPatientBtn.addEventListener("click", () => switchPatient(patientIdx + 1));

    document.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        // Alt+Arrow: patient navigation (must come before plain Arrow check)
        if (e.altKey && e.key === "ArrowLeft") {
            switchPatient(patientIdx - 1);
        } else if (e.altKey && e.key === "ArrowRight") {
            switchPatient(patientIdx + 1);
        } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
            currentSlice = Math.max(0, parseInt(currentSlice) - 1);
            sliceSlider.value = currentSlice;
            updateSlice();
        } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
            currentSlice = Math.min(info.num_slices - 1, parseInt(currentSlice) + 1);
            sliceSlider.value = currentSlice;
            updateSlice();
        } else if (e.key === "+" || e.key === "=") {
            zoomLevel = Math.min(5, zoomLevel + 0.25); applyZoom();
        } else if (e.key === "-") {
            zoomLevel = Math.max(0.5, zoomLevel - 0.25); applyZoom();
        } else if (e.key === "0") {
            zoomLevel = 1.0; applyZoom();
            container.scrollLeft = 0; container.scrollTop = 0;
        } else if (e.key === "Tab") {
            e.preventDefault();
            cycleParam(e.shiftKey ? -1 : 1);
        }
    });
}

init();
