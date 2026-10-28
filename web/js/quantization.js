document.addEventListener("DOMContentLoaded", () => {
  const canvas = new window.DigitCanvas("digit-canvas");
  const W = window.ModelWeights;

  // ---- Activation grid ----
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
      const q = cell.dataset.qVal || "0";
      const hex = (parseInt(q, 10) & 0xFF).toString(16).toUpperCase().padStart(2, "0");
      nodeTelemetry.textContent = `a${i.toString().padStart(3, "0")}: INT8=${q.padStart(4, " ")} | HEX=0x${hex}`;
    });
    actContainer.appendChild(cell);
    cells.push(cell);
  }

  // ---- Prediction rows (built once so bar widths can transition) ----
  const probContainer = document.getElementById("prob-container");
  probContainer.innerHTML = "";
  const predRows = [];
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
      pct: row.querySelector(".pred-pct")
    });
  }

  // ---- Chart ----
  const mono = { family: "monospace", size: 10 };
  const logitChart = new Chart(document.getElementById("logit-chart").getContext("2d"), {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [
        { label: "FP32 REF", data: new Array(10).fill(0), backgroundColor: "#58a6ff", borderRadius: 2 },
        { label: "INT8 Q31", data: new Array(10).fill(0), backgroundColor: "#3fb950", borderRadius: 2 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300, easing: "easeOutQuart" },
      scales: {
        y: { grid: { color: "#21262d" }, ticks: { color: "#8b949e", font: mono } },
        x: { grid: { display: false }, ticks: { color: "#8b949e", font: { ...mono, size: 11 } } }
      },
      plugins: {
        legend: { labels: { color: "#8b949e", font: mono, boxWidth: 10, boxHeight: 10 } },
        tooltip: {
          backgroundColor: "#161b22",
          borderColor: "#30363d",
          borderWidth: 1,
          titleColor: "#c9d1d9",
          bodyColor: "#8b949e",
          titleFont: mono,
          bodyFont: mono,
          callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(4)}` }
        }
      }
    }
  });

  // ---- Controls ----
  const scaleSlider = document.getElementById("param-scale");
  const zpSlider = document.getElementById("param-zp");
  const scaleValDisplay = document.getElementById("scale-val");
  const zpValDisplay = document.getElementById("zp-val");

  let currentScale = parseFloat(scaleSlider.value);
  let currentZP = parseInt(zpSlider.value, 10);

  scaleSlider.addEventListener("input", (e) => {
    currentScale = parseFloat(e.target.value);
    scaleValDisplay.textContent = currentScale.toFixed(5);
    schedule();
  });

  zpSlider.addEventListener("input", (e) => {
    currentZP = parseInt(e.target.value, 10);
    zpValDisplay.textContent = currentZP;
    schedule();
  });

  // ---- Quantization helpers ----
  const quantizeVal = (val, scale, zp) => Math.max(-128, Math.min(127, Math.round(val / scale) + zp));
  const dequantizeVal = (q, scale, zp) => (q - zp) * scale;

  function softmax(arr) {
    let max = -Infinity;
    for (const v of arr) if (v > max) max = v;
    const exps = arr.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // Quantized W1 only depends on scale (zero-point 0), so cache it.
  let qW1 = null;
  let qW1Scale = NaN;
  function getQuantizedW1(scale) {
    if (qW1Scale !== scale) {
      qW1 = new Int8Array(W.W1.length);
      for (let i = 0; i < W.W1.length; i++) qW1[i] = quantizeVal(W.W1[i], scale, 0);
      qW1Scale = scale;
    }
    return qW1;
  }

  // ---- Animated numeric readouts ----
  function tween(el, to, format) {
    const from = el._v ?? to;
    el._v = to;
    cancelAnimationFrame(el._raf);
    if (from === to || matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.textContent = format(to);
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / 250);
      const k = 1 - Math.pow(1 - t, 3);
      el.textContent = format(from + (to - from) * k);
      if (t < 1) el._raf = requestAnimationFrame(step);
    };
    el._raf = requestAnimationFrame(step);
  }

  // ---- Hex inspector ----
  let prevBytes = [];
  function updateHexInspector(pixels) {
    const hexContainer = document.getElementById("hex-inspector");
    let html = "";
    const bytes = [];
    for (let line = 0; line < 4; line++) {
      const addr = (0x80001000 + line * 8).toString(16).toUpperCase();
      let bytesStr = "";
      for (let b = 0; b < 8; b++) {
        const idx = line * 8 + b;
        const val = quantizeVal(pixels[idx], currentScale, currentZP);
        const hex = (val & 0xFF).toString(16).toUpperCase().padStart(2, "0");
        bytes[idx] = hex;
        const changed = prevBytes.length && prevBytes[idx] !== hex;
        bytesStr += `<b${changed ? ' class="chg"' : ""}>${hex}</b> `;
      }
      html += `<div class="hex-line"><span class="hex-addr">0x${addr}:</span> <span class="hex-bytes">${bytesStr}</span></div>`;
    }
    prevBytes = bytes;
    hexContainer.innerHTML = html;
  }

  // ---- Inference ----
  function runInference() {
    const input = canvas.getDownsampledPixels();
    updateHexInspector(input);

    // FP32 reference
    const h1_fp32 = new Float32Array(128);
    for (let c = 0; c < 128; c++) {
      let sum = W.b1[c];
      for (let r = 0; r < 784; r++) sum += input[r] * W.W1[r * 128 + c];
      h1_fp32[c] = Math.max(0, sum);
    }

    const out_fp32 = new Float32Array(10);
    for (let c = 0; c < 10; c++) {
      let sum = W.b2[c];
      for (let r = 0; r < 128; r++) sum += h1_fp32[r] * W.W2[r * 10 + c];
      out_fp32[c] = sum;
    }

    // INT8 path: quantize inputs and weights once, outside the MAC loop
    const qW = getQuantizedW1(currentScale);
    const xs = new Int32Array(784);
    for (let r = 0; r < 784; r++) xs[r] = quantizeVal(input[r], currentScale, currentZP) - currentZP;

    const h1_int8 = new Float32Array(128);
    const outScale = currentScale * currentScale;
    for (let c = 0; c < 128; c++) {
      let acc = 0;
      for (let r = 0; r < 784; r++) acc += xs[r] * qW[r * 128 + c];
      const relu_val = Math.max(0, acc * outScale + W.b1[c]);
      const q_out = quantizeVal(relu_val, currentScale, currentZP);
      h1_int8[c] = dequantizeVal(q_out, currentScale, currentZP);

      const cell = cells[c];
      const prevQ = cell.dataset.qVal;
      cell.dataset.qVal = q_out;
      if (prevQ !== String(q_out)) {
        cell.animate(
          [{ boxShadow: "0 0 0 1px #58a6ff" }, { boxShadow: "0 0 0 0 transparent" }],
          { duration: 350, easing: "ease-out" }
        );
      }
    }

    const out_int8 = new Float32Array(10);
    const logitScale = currentScale * 2.5;
    for (let c = 0; c < 10; c++) {
      let sum = W.b2[c];
      for (let r = 0; r < 128; r++) sum += h1_int8[r] * W.W2[r * 10 + c];
      out_int8[c] = dequantizeVal(quantizeVal(sum, logitScale, currentZP), logitScale, currentZP);
    }

    // Telemetry
    let signalPower = 0, noisePower = 0, maxAbsErr = 0;
    for (let i = 0; i < 10; i++) {
      const err = Math.abs(out_fp32[i] - out_int8[i]);
      if (err > maxAbsErr) maxAbsErr = err;
      signalPower += out_fp32[i] * out_fp32[i];
      noisePower += err * err;
    }
    const sqnr = noisePower < 1e-9 ? 60.0 : Math.min(60.0, 10 * Math.log10(signalPower / noisePower));
    const rmse = Math.sqrt(noisePower / 10.0);

    tween(document.getElementById("metric-sqnr"), sqnr, (v) => `${v.toFixed(1)} dB`);
    tween(document.getElementById("metric-rmse"), rmse, (v) => v.toFixed(4));
    tween(document.getElementById("metric-linf"), maxAbsErr, (v) => v.toFixed(4));

    // Activation grid colors
    let maxAct = 0.001;
    for (let i = 0; i < 128; i++) if (h1_int8[i] > maxAct) maxAct = h1_int8[i];
    for (let i = 0; i < 128; i++) {
      const norm = Math.min(1.0, h1_int8[i] / maxAct);
      cells[i].style.backgroundColor = norm > 0.05
        ? `rgba(63, 185, 80, ${Math.max(0.15, norm)})`
        : "#0d1117";
    }

    // Chart
    logitChart.data.datasets[0].data = Array.from(out_fp32);
    logitChart.data.datasets[1].data = Array.from(out_int8);
    logitChart.update();

    // Predictions (updated in place so bars animate)
    const probs = softmax(Array.from(out_int8));
    let topDigit = 0, topProb = -1;
    probs.forEach((p, idx) => {
      if (p > topProb) { topProb = p; topDigit = idx; }
    });
    probs.forEach((p, idx) => {
      const r = predRows[idx];
      r.row.classList.toggle("active", idx === topDigit);
      r.fill.style.width = `${(p * 100).toFixed(1)}%`;
      r.pct.textContent = `${(p * 100).toFixed(1)}%`;
    });
  }

  // Coalesce rapid triggers (drawing, slider drags) into one run per frame
  let queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      runInference();
    });
  }

  window.triggerInference = schedule;
  runInference();
});