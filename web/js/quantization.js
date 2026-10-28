document.addEventListener("DOMContentLoaded", () => {
  const canvas = new window.DigitCanvas("digit-canvas");

  // Build 128-node visual grid
  const actContainer = document.getElementById("hidden-activations");
  actContainer.innerHTML = "";
  for (let i = 0; i < 128; i++) {
    const cell = document.createElement("div");
    cell.className = "act-cell";
    cell.id = `act-${i}`;
    actContainer.appendChild(cell);
  }

  // Setup Chart.js for FP32 vs INT8 Logits
  const chartCtx = document.getElementById("logit-chart").getContext("2d");
  const logitChart = new Chart(chartCtx, {
    type: "bar",
    data: {
      labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      datasets: [
        {
          label: "FP32 Baseline",
          data: new Array(10).fill(0),
          backgroundColor: "rgba(56, 189, 248, 0.7)",
          borderColor: "#38bdf8",
          borderWidth: 1
        },
        {
          label: "INT8 Fixed-Point",
          data: new Array(10).fill(0),
          backgroundColor: "rgba(129, 140, 248, 0.7)",
          borderColor: "#818cf8",
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      animation: { duration: 150 },
      scales: {
        y: {
          grid: { color: "#334155" },
          ticks: { color: "#94a3b8" }
        },
        x: {
          ticks: { color: "#f8fafc" },
          grid: { display: false }
        }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });

  // Quantization Sliders
  const scaleSlider = document.getElementById("param-scale");
  const zpSlider = document.getElementById("param-zp");
  const scaleValDisplay = document.getElementById("scale-val");
  const zpValDisplay = document.getElementById("zp-val");

  let currentScale = parseFloat(scaleSlider.value);
  let currentZP = parseInt(zpSlider.value, 10);

  scaleSlider.addEventListener("input", (e) => {
    currentScale = parseFloat(e.target.value);
    scaleValDisplay.textContent = currentScale.toFixed(4);
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

  function runInference() {
    const input = canvas.getDownsampledPixels();
    const W = window.ModelWeights;

    // --- Path A: FP32 Reference ---
    const h1_fp32 = new Float32Array(128);
    for (let c = 0; c < 128; c++) {
      let sum = W.b1[c];
      for (let r = 0; r < 784; r++) {
        sum += input[r] * W.W1[r * 128 + c];
      }
      h1_fp32[c] = Math.max(0, sum); // ReLU
    }

    const out_fp32 = new Float32Array(10);
    for (let c = 0; c < 10; c++) {
      let sum = W.b2[c];
      for (let r = 0; r < 128; r++) {
        sum += h1_fp32[r] * W.W2[r * 10 + c];
      }
      out_fp32[c] = sum;
    }

    // --- Path B: INT8 Fixed-Point Engine ---
    const h1_int8 = new Float32Array(128);
    for (let c = 0; c < 128; c++) {
      let acc = 0;
      for (let r = 0; r < 784; r++) {
        const qX = quantizeVal(input[r], currentScale, currentZP);
        const qW = quantizeVal(W.W1[r * 128 + c], currentScale, 0);
        acc += (qX - currentZP) * qW;
      }
      // Fixed-point scaling simulated requantization
      const real_val = acc * (currentScale * currentScale) + W.b1[c];
      const relu_val = Math.max(0, real_val);
      const q_out = quantizeVal(relu_val, currentScale, currentZP);
      h1_int8[c] = dequantizeVal(q_out, currentScale, currentZP);
    }

    const out_int8 = new Float32Array(10);
    for (let c = 0; c < 10; c++) {
      let sum = W.b2[c];
      for (let r = 0; r < 128; r++) {
        sum += h1_int8[r] * W.W2[r * 10 + c];
      }
      const q_logit = quantizeVal(sum, currentScale * 2.0, currentZP);
      out_int8[c] = dequantizeVal(q_logit, currentScale * 2.0, currentZP);
    }

    // Update Hidden Activation Heatmap
    let maxAct = 0.0001;
    for (let i = 0; i < 128; i++) if (h1_int8[i] > maxAct) maxAct = h1_int8[i];
    for (let i = 0; i < 128; i++) {
      const cell = document.getElementById(`act-${i}`);
      const norm = Math.min(1.0, h1_int8[i] / maxAct);
      cell.style.backgroundColor = `rgb(${Math.floor(norm * 56)}, ${Math.floor(norm * 189)}, ${Math.floor(norm * 248)})`;
    }

    // Update Chart
    logitChart.data.datasets[0].data = Array.from(out_fp32);
    logitChart.data.datasets[1].data = Array.from(out_int8);
    logitChart.update();

    // Update Prediction List
    const probs = softmax(Array.from(out_int8));
    let bestDigit = 0;
    let bestProb = -1;
    probs.forEach((p, idx) => {
      if (p > bestProb) {
        bestProb = p;
        bestDigit = idx;
      }
    });

    const probContainer = document.getElementById("prob-container");
    probContainer.innerHTML = "";
    probs.forEach((p, idx) => {
      const row = document.createElement("div");
      row.className = `prob-row ${idx === bestDigit ? "top-prediction" : ""}`;
      row.innerHTML = `
        <span style="width: 18px; font-weight: bold;">${idx}</span>
        <div class="prob-bar-container">
          <div class="prob-bar" style="width: ${(p * 100).toFixed(1)}%;"></div>
        </div>
        <span style="width: 48px; text-align: right; color: var(--text-muted);">${(p * 100).toFixed(1)}%</span>
      `;
      probContainer.appendChild(row);
    });
  }

  window.triggerInference = runInference;
  runInference();
});
