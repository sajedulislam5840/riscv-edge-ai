document.addEventListener("DOMContentLoaded", () => {
  const canvas = new window.DigitCanvas("digit-canvas");

  // Dense 128-node grid
  const actContainer = document.getElementById("hidden-activations");
  const nodeTelemetry = document.getElementById("node-telemetry");
  actContainer.innerHTML = "";

  for (let i = 0; i < 128; i++) {
    const cell = document.createElement("div");
    cell.className = "dense-cell";
    cell.id = `act-${i}`;
    cell.addEventListener("mouseenter", () => {
      const q = cell.dataset.qVal || "0";
      const hex = (parseInt(q, 10) & 0xFF).toString(16).toUpperCase().padStart(2, '0');
      nodeTelemetry.textContent = `a${i.toString().padStart(3, '0')}: INT8=${q.padStart(4, ' ')} | HEX=0x${hex}`;
    });
    actContainer.appendChild(cell);
  }

  // Instrument Chart
  const chartCtx = document.getElementById("logit-chart").getContext("2d");
  const logitChart = new Chart(chartCtx, {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [
        {
          label: "FP32 REF",
          data: new Array(10).fill(0),
          backgroundColor: "#58a6ff",
          borderRadius: 2
        },
        {
          label: "INT8 Q31",
          data: new Array(10).fill(0),
          backgroundColor: "#3fb950",
          borderRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 50 },
      scales: {
        y: {
          grid: { color: "#21262d" },
          ticks: { color: "#8b949e", font: { family: "monospace", size: 10 } }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#8b949e", font: { family: "monospace", size: 11 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#8b949e", font: { family: "monospace", size: 10 } } }
      }
    }
  });

  const scaleSlider = document.getElementById("param-scale");
  const zpSlider = document.getElementById("param-zp");
  const scaleValDisplay = document.getElementById("scale-val");
  const zpValDisplay = document.getElementById("zp-val");

  let currentScale = parseFloat(scaleSlider.value);
  let currentZP = parseInt(zpSlider.value, 10);

  scaleSlider.addEventListener("input", (e) => {
    currentScale = parseFloat(e.target.value);
    scaleValDisplay.textContent = currentScale.toFixed(5);
    runInference();
  });

  zpSlider.addEventListener("input", (e) => {
    currentZP = parseInt(e.target.value, 10);
    zpValDisplay.textContent = currentZP;
    runInference();
  });

  function quantizeVal(val, scale, zp) {
    const q = Math.round(val / scale) + zp;
    return Math.max(-128, Math.min(127, q));
  }

  function dequantizeVal(q, scale, zp) {
    return (q - zp) * scale;
  }

  function softmax(arr) {
    let max = -Infinity;
    for (let v of arr) if (v > max) max = v;
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  function updateHexInspector(pixels) {
    const hexContainer = document.getElementById("hex-inspector");
    let html = "";
    for (let line = 0; line < 4; line++) {
      const addr = (0x80001000 + line * 8).toString(16).toUpperCase();
      let bytesStr = "";
      for (let b = 0; b < 8; b++) {
        const val = quantizeVal(pixels[line * 8 + b], currentScale, currentZP);
        const hex = (val & 0xFF).toString(16).toUpperCase().padStart(2, '0');
        bytesStr += hex + " ";
      }
      html += `<div class="hex-line"><span class="hex-addr">0x${addr}:</span> <span class="hex-bytes">${bytesStr}</span></div>`;
    }
    hexContainer.innerHTML = html;
  }

  function runInference() {
    const input = canvas.getDownsampledPixels();
    updateHexInspector(input);

    const W = window.ModelWeights;

    // FP32 Execution
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

    // INT8 Execution
    const h1_int8 = new Float32Array(128);
    for (let c = 0; c < 128; c++) {
      let acc = 0;
      for (let r = 0; r < 784; r++) {
        const qX = quantizeVal(input[r], currentScale, currentZP);
        const qW = quantizeVal(W.W1[r * 128 + c], currentScale, 0);
        acc += (qX - currentZP) * qW;
      }
      const real_val = acc * (currentScale * currentScale) + W.b1[c];
      const relu_val = Math.max(0, real_val);
      const q_out = quantizeVal(relu_val, currentScale, currentZP);
      h1_int8[c] = dequantizeVal(q_out, currentScale, currentZP);

      const cell = document.getElementById(`act-${c}`);
      cell.dataset.qVal = q_out;
    }

    const out_int8 = new Float32Array(10);
    for (let c = 0; c < 10; c++) {
      let sum = W.b2[c];
      for (let r = 0; r < 128; r++) sum += h1_int8[r] * W.W2[r * 10 + c];
      const q_logit = quantizeVal(sum, currentScale * 2.5, currentZP);
      out_int8[c] = dequantizeVal(q_logit, currentScale * 2.5, currentZP);
    }

    // Telemetry Calculation
    let signalPower = 0.0, noisePower = 0.0, maxAbsErr = 0.0;
    for (let i = 0; i < 10; i++) {
      const err = Math.abs(out_fp32[i] - out_int8[i]);
      if (err > maxAbsErr) maxAbsErr = err;
      signalPower += out_fp32[i] * out_fp32[i];
      noisePower += err * err;
    }

    const sqnr = noisePower < 1e-9 ? 60.0 : Math.min(60.0, 10 * Math.log10(signalPower / noisePower));
    const rmse = Math.sqrt(noisePower / 10.0);

    document.getElementById("metric-sqnr").textContent = `${sqnr.toFixed(1)} dB`;
    document.getElementById("metric-rmse").textContent = rmse.toFixed(4);
    document.getElementById("metric-linf").textContent = maxAbsErr.toFixed(4);

    // Update Activation Grid
    let maxAct = 0.001;
    for (let i = 0; i < 128; i++) if (h1_int8[i] > maxAct) maxAct = h1_int8[i];
    for (let i = 0; i < 128; i++) {
      const cell = document.getElementById(`act-${i}`);
      const norm = Math.min(1.0, h1_int8[i] / maxAct);
      cell.style.backgroundColor = norm > 0.05 
        ? `rgba(63, 185, 80, ${Math.max(0.15, norm)})` 
        : "#0d1117";
    }

    // Update Chart
    logitChart.data.datasets[0].data = Array.from(out_fp32);
    logitChart.data.datasets[1].data = Array.from(out_int8);
    logitChart.update();

    // Update Prediction List
    const probs = softmax(Array.from(out_int8));
    let topDigit = 0, topProb = -1;
    probs.forEach((p, idx) => {
      if (p > topProb) {
        topProb = p;
        topDigit = idx;
      }
    });

    const probContainer = document.getElementById("prob-container");
    probContainer.innerHTML = "";
    probs.forEach((p, idx) => {
      const row = document.createElement("div");
      row.className = `pred-row ${idx === topDigit ? "active" : ""}`;
      row.innerHTML = `
        <span class="pred-idx">${idx}</span>
        <div class="pred-bar-bg">
          <div class="pred-bar-fill" style="width: ${(p * 100).toFixed(1)}%;"></div>
        </div>
        <span class="pred-pct">${(p * 100).toFixed(1)}%</span>
      `;
      probContainer.appendChild(row);
    });
  }

  window.triggerInference = runInference;
  runInference();
});
