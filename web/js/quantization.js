document.addEventListener("DOMContentLoaded", () => {
  const canvas = new window.DigitCanvas("digit-canvas");
  const W = window.ModelWeights;

  // ── Activation grid ──────────────────────────────────────────────
  const actContainer = document.getElementById("hidden-activations");
  const nodeTelemetry = document.getElementById("node-telemetry");
  actContainer.innerHTML = "";
  const cells = [];

  for (let i = 0; i < 128; i++) {
    const cell = document.createElement("div");
    cell.className = "dense-cell";
    cell.id = `act-${i}`;
    cell.dataset.qVal = "0";
    cell.addEventListener("mouseenter", () => {
      const q = parseInt(cell.dataset.qVal || "0", 10);
      const hex = (q & 0xFF).toString(16).toUpperCase().padStart(2, "0");
      const fp  = (q * parseFloat(document.getElementById("param-scale").value)).toFixed(4);
      nodeTelemetry.textContent =
        `a${String(i).padStart(3,"0")}: q=${String(q).padStart(4," ")}  hex=0x${hex}  deq≈${fp}`;
    });
    cell.addEventListener("mouseleave", () => {
      nodeTelemetry.textContent = "HOVER NODE FOR REG DUMP";
    });
    actContainer.appendChild(cell);
    cells.push(cell);
  }

  // ── Prediction rows ────────────────────────────────────────────
  const probContainer = document.getElementById("prob-container");
  probContainer.innerHTML = "";
  const predRows = [];
  const DIGIT_LABELS = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"];
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "pred-row";
    row.innerHTML = `
      <span class="pred-idx">${i}</span>
      <div class="pred-bar-bg"><div class="pred-bar-fill"></div></div>
      <span class="pred-pct">0.0%</span>`;
    probContainer.appendChild(row);
    predRows.push({
      row,
      fill: row.querySelector(".pred-bar-fill"),
      pct:  row.querySelector(".pred-pct")
    });
  }

  // ── Chart (line style — easier to compare FP32 vs INT8) ────────
  const mono = { family: "'JetBrains Mono', monospace", size: 10 };
  const logitChart = new Chart(
    document.getElementById("logit-chart").getContext("2d"), {
      type: "bar",
      data: {
        labels: ["0","1","2","3","4","5","6","7","8","9"],
        datasets: [
          {
            label: "FP32 REF",
            data: new Array(10).fill(0),
            backgroundColor: "rgba(56,189,248,0.55)",
            borderColor:     "rgba(56,189,248,0.9)",
            borderWidth: 1,
            borderRadius: 3,
            borderSkipped: false,
          },
          {
            label: "INT8 Q31",
            data: new Array(10).fill(0),
            backgroundColor: "rgba(74,222,128,0.45)",
            borderColor:     "rgba(74,222,128,0.9)",
            borderWidth: 1,
            borderRadius: 3,
            borderSkipped: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250, easing: "easeOutQuart" },
        scales: {
          y: {
            grid:  { color: "rgba(56,189,248,0.07)" },
            ticks: { color: "#475569", font: mono }
          },
          x: {
            grid:  { display: false },
            ticks: { color: "#475569", font: { ...mono, size: 11 } }
          }
        },
        plugins: {
          legend: {
            labels: {
              color: "#94a3b8", font: mono,
              boxWidth: 10, boxHeight: 10,
              usePointStyle: true, pointStyle: "rectRounded"
            }
          },
          tooltip: {
            backgroundColor: "rgba(5,10,20,0.92)",
            borderColor:     "rgba(56,189,248,0.3)",
            borderWidth: 1,
            titleColor:  "#e2e8f0",
            bodyColor:   "#94a3b8",
            titleFont:   mono,
            bodyFont:    mono,
            padding: 10,
            callbacks: {
              label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}`
            }
          }
        }
      }
    }
  );

  // ── Controls ───────────────────────────────────────────────────
  const scaleSlider = document.getElementById("param-scale");
  const zpSlider    = document.getElementById("param-zp");
  const scaleDisp   = document.getElementById("scale-val");
  const zpDisp      = document.getElementById("zp-val");
  const topLabel    = document.getElementById("top-pred-label");

  let currentScale = parseFloat(scaleSlider.value);
  let currentZP    = parseInt(zpSlider.value, 10);

  scaleSlider.addEventListener("input", (e) => {
    currentScale = parseFloat(e.target.value);
    scaleDisp.textContent = currentScale.toFixed(5);
    schedule();
  });

  zpSlider.addEventListener("input", (e) => {
    currentZP = parseInt(e.target.value, 10);
    zpDisp.textContent = currentZP;
    schedule();
  });

  // ── Quantization helpers ───────────────────────────────────────
  const clamp     = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const qVal      = (v, s, z)  => clamp(Math.round(v / s) + z, -128, 127);
  const deqVal    = (q, s, z)  => (q - z) * s;

  function softmax(arr) {
    const mx = Math.max(...arr);
    const ex = arr.map(v => Math.exp(v - mx));
    const sm = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / sm);
  }

  // ── Weight quantization cache ──────────────────────────────────
  const REF_SCALE = 0.00784;
  const maxAbs = (arr) => { let m = 0; for (let v of arr) m = Math.max(m, Math.abs(v)); return m; };
  const BASE_SW1 = maxAbs(W.W1) / 127;
  const BASE_SW2 = maxAbs(W.W2) / 127;
  const wCache = {};

  function getQW(key, src, scale) {
    const hit = wCache[key];
    if (hit && hit.scale === scale) return hit.q;
    const q = new Int8Array(src.length);
    for (let i = 0; i < src.length; i++) q[i] = qVal(src[i], scale, 0);
    wCache[key] = { scale, q };
    return q;
  }

  // ── Animated numeric readouts ──────────────────────────────────
  function tween(el, to, fmt) {
    const from = el._v ?? to;
    el._v = to;
    cancelAnimationFrame(el._raf);
    if (from === to || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.textContent = fmt(to); return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / 280);
      const k = 1 - Math.pow(1 - t, 3);
      el.textContent = fmt(from + (to - from) * k);
      if (t < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  // ── Heatmap color (dark → blue → cyan → green → amber → white) ─
  function heatColor(norm) {
    // norm: 0..1
    if (norm <= 0.05) return `rgba(5,10,18,0.9)`;  // dead node
    const stops = [
      [0.00, [8,  20, 50,  0.6]],
      [0.25, [56, 130, 220, 0.8]],
      [0.50, [30, 180, 210, 0.9]],
      [0.75, [74, 222, 128, 1.0]],
      [1.00, [251,191, 36,  1.0]],
    ];
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (norm >= stops[i][0] && norm <= stops[i+1][0]) {
        lo = stops[i]; hi = stops[i+1]; break;
      }
    }
    const t = (norm - lo[0]) / (hi[0] - lo[0] + 1e-9);
    const c = lo[1].map((v, i) => lo[1][i] + (hi[1][i] - lo[1][i]) * t);
    return `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${c[3].toFixed(2)})`;
  }

  // ── Hex inspector ──────────────────────────────────────────────
  let prevBytes = [];
  function updateHex(pixels) {
    const el = document.getElementById("hex-inspector");
    let html = "";
    const bytes = [];
    for (let line = 0; line < 4; line++) {
      const addr = (0x80001000 + (10 + line) * 28 + 10).toString(16).toUpperCase();
      let bs = "";
      for (let b = 0; b < 8; b++) {
        const idx = line * 8 + b;
        const v   = qVal(pixels[(10 + line) * 28 + 10 + b], currentScale, currentZP);
        const hex = (v & 0xFF).toString(16).toUpperCase().padStart(2, "0");
        bytes[idx] = hex;
        const chg = prevBytes.length && prevBytes[idx] !== hex;
        bs += `<b${chg ? ' class="chg"' : ""}>${hex}</b> `;
      }
      html += `<div class="hex-line"><span class="hex-addr">0x${addr}:</span><span class="hex-bytes">${bs}</span></div>`;
    }
    prevBytes = bytes;
    el.innerHTML = html;
  }

  // ── Main inference ─────────────────────────────────────────────
  function runInference() {
    const input = canvas.getDownsampledPixels();
    updateHex(input);

    // FP32 reference pass
    const h1_fp = new Float32Array(128);
    for (let c = 0; c < 128; c++) {
      let s = W.b1[c];
      for (let r = 0; r < 784; r++) s += input[r] * W.W1[r * 128 + c];
      h1_fp[c] = Math.max(0, s);
    }
    const out_fp = new Float32Array(10);
    for (let c = 0; c < 10; c++) {
      let s = W.b2[c];
      for (let r = 0; r < 128; r++) s += h1_fp[r] * W.W2[r * 10 + c];
      out_fp[c] = s;
    }

    // INT8 fixed-point pass
    const ratio = currentScale / REF_SCALE;
    const sW1   = BASE_SW1 * ratio;
    const sW2   = BASE_SW2 * ratio;
    const qW1   = getQW("W1", W.W1, sW1);
    const xs    = new Int32Array(784);
    for (let r = 0; r < 784; r++) xs[r] = qVal(input[r], currentScale, currentZP) - currentZP;

    let maxH = 1e-6;
    for (let c = 0; c < 128; c++) if (h1_fp[c] > maxH) maxH = h1_fp[c];
    let maxO = 1e-6;
    for (let c = 0; c < 10; c++) maxO = Math.max(maxO, Math.abs(out_fp[c]));
    const sA = (maxH / 127) * ratio;
    const sO = (maxO / 127) * ratio;

    const h1_q8 = new Float32Array(128);
    const qA    = new Int32Array(128);
    const r1    = currentScale * sW1;

    for (let c = 0; c < 128; c++) {
      let acc = 0;
      for (let r = 0; r < 784; r++) acc += xs[r] * qW1[r * 128 + c];
      const rv  = Math.max(0, acc * r1 + W.b1[c]);
      const qo  = qVal(rv, sA, currentZP);
      qA[c]     = qo - currentZP;
      h1_q8[c]  = deqVal(qo, sA, currentZP);

      // Animate changed cells
      const cell = cells[c];
      if (cell.dataset.qVal !== String(qo)) {
        cell.dataset.qVal = qo;
        cell.animate(
          [{ boxShadow: "0 0 0 2px rgba(56,189,248,0.9)" },
           { boxShadow: "0 0 0 0 transparent" }],
          { duration: 400, easing: "ease-out" }
        );
      }
    }

    const out_q8 = new Float32Array(10);
    const qW2   = getQW("W2", W.W2, sW2);
    for (let c = 0; c < 10; c++) {
      let acc = 0;
      for (let r = 0; r < 128; r++) acc += qA[r] * qW2[r * 10 + c];
      out_q8[c] = deqVal(qVal(acc * sA * sW2 + W.b2[c], sO, currentZP), sO, currentZP);
    }

    // ── Telemetry ────────────────────────────────────────────────
    let sigP = 0, noisP = 0, maxErr = 0;
    for (let i = 0; i < 10; i++) {
      const e = Math.abs(out_fp[i] - out_q8[i]);
      if (e > maxErr) maxErr = e;
      sigP  += out_fp[i] * out_fp[i];
      noisP += e * e;
    }
    const sqnr = noisP < 1e-9 ? 60 : Math.min(60, 10 * Math.log10(sigP / noisP));
    const rmse = Math.sqrt(noisP / 10);

    // SQNR color: green=good, amber=warn, red=poor
    const sqEl = document.getElementById("metric-sqnr");
    const sqColor = sqnr > 35 ? "var(--neon-green)" : sqnr > 20 ? "var(--neon-amber)" : "var(--neon-red)";
    sqEl.style.color = sqColor;
    sqEl.style.textShadow = `0 0 8px currentColor`;
    tween(sqEl, sqnr, v => `${v.toFixed(1)} dB`);
    tween(document.getElementById("metric-rmse"),  rmse,   v => v.toFixed(4));
    tween(document.getElementById("metric-linf"),  maxErr, v => v.toFixed(4));

    // Estimated cycles (simple heuristic for demo)
    const cyc = Math.round((784*128 + 128*10) * (1 + currentScale / 0.030 * 0.3));
    tween(document.getElementById("metric-cycles"), cyc, v => v.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ","));

    // ── Activation grid heatmap ───────────────────────────────────
    let maxAct = 0.001;
    for (let i = 0; i < 128; i++) if (h1_q8[i] > maxAct) maxAct = h1_q8[i];
    for (let i = 0; i < 128; i++) {
      const norm = Math.min(1.0, h1_q8[i] / maxAct);
      cells[i].style.backgroundColor = heatColor(norm);
      cells[i].style.boxShadow = norm > 0.7
        ? `0 0 ${(norm * 8)|0}px rgba(74,222,128,${(norm * 0.6).toFixed(2)})`
        : "none";
    }

    // ── Chart update ─────────────────────────────────────────────
    logitChart.data.datasets[0].data = Array.from(out_fp);
    logitChart.data.datasets[1].data = Array.from(out_q8);
    logitChart.update("none"); // skip animation for live updates

    // ── Prediction bars ───────────────────────────────────────────
    const probs = softmax(Array.from(out_q8));
    let topIdx = 0, topP = -1;
    probs.forEach((p, i) => { if (p > topP) { topP = p; topIdx = i; } });

    if (topLabel) {
      topLabel.textContent = topP > 0.3
        ? `PRED: ${topIdx} (${DIGIT_LABELS[topIdx]}) @ ${(topP*100).toFixed(1)}%`
        : "UNCERTAIN";
    }

    probs.forEach((p, i) => {
      const row = predRows[i];
      const isTop = i === topIdx;
      row.row.classList.toggle("active", isTop);
      row.fill.style.width = `${(p * 100).toFixed(2)}%`;
      row.pct.textContent  = `${(p * 100).toFixed(1)}%`;
    });
  }

  // ── Debounce to one run per animation frame ────────────────────
  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; runInference(); });
  }

  window.triggerInference = schedule;
  runInference();
});