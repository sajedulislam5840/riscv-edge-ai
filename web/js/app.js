// Page logic: drawing pad, preprocessing, and the live views of each step.
(function () {
  "use strict";

  const engine = Int8Engine.createEngine(window.MODEL_INT8_QBIN_B64, window.QUANT_PARAMS);
  const fp32 = Int8Engine.createFp32(window.TRAINED_WEIGHTS_B64);
  const $ = (id) => document.getElementById(id);

  // Published numbers come from the generated files, so the page never drifts from the code.
  $("r-fp32").textContent = window.FP32_TEST_ACCURACY.toFixed(2) + "%";
  $("r-int8").textContent = window.QUANT_PARAMS.int8TestAccuracy.toFixed(2) + "%";
  $("t-int8").textContent = window.QUANT_PARAMS.int8TestAccuracy.toFixed(2) + "%";
  $("t-fp32").textContent = window.FP32_TEST_ACCURACY.toFixed(2) + "%";
  $("r-agree").textContent = window.QUANT_PARAMS.agreementWithFp32.toFixed(2) + "%";

  // ---------------------------------------------------------------- drawing pad
  const pad = $("pad");
  const ctx = pad.getContext("2d", { willReadFrequently: true });
  let pixels = new Uint8Array(784); // current 28x28 input, 0..255
  let hasInk = false;
  let drawing = false;
  let last = null;

  function clearPad() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, pad.width, pad.height);
    pixels = new Uint8Array(784);
    hasInk = false;
    update();
  }

  function point(e) {
    const r = pad.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (pad.width / r.width), y: (e.clientY - r.top) * (pad.height / r.height) };
  }

  pad.addEventListener("pointerdown", (e) => {
    drawing = true;
    pad.setPointerCapture(e.pointerId);
    last = point(e);
    stroke(last, last);
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = point(e);
    stroke(last, p);
    last = p;
  });
  const stop = () => { drawing = false; last = null; };
  pad.addEventListener("pointerup", stop);
  pad.addEventListener("pointercancel", stop);

  function stroke(a, b) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    hasInk = true;
    schedule(() => { pixels = preprocess(); });
  }

  // MNIST-style preprocessing: fit the digit into a 20x20 box, centre its mass in 28x28.
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true });

  function preprocess() {
    const w = pad.width, h = pad.height;
    const src = ctx.getImageData(0, 0, w, h).data;
    let minX = w, minY = h, maxX = -1, maxY = -1, mass = 0, cx = 0, cy = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = src[(y * w + x) * 4];
        if (v > 25) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          mass += v; cx += x * v; cy += y * v;
        }
      }
    }
    const out = new Uint8Array(784);
    if (!mass) return out;
    cx /= mass; cy /= mass;
    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    sctx.fillStyle = "#000";
    sctx.fillRect(0, 0, 28, 28);
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(pad, minX, minY, bw, bh,
      Math.round(14 + (minX - cx) * scale), Math.round(14 + (minY - cy) * scale),
      Math.max(1, Math.round(bw * scale)), Math.max(1, Math.round(bh * scale)));
    const d = sctx.getImageData(0, 0, 28, 28).data;
    for (let i = 0; i < 784; i++) out[i] = d[i * 4];
    return out;
  }

  // Test images are shown as-is: they are already 28x28 and centred.
  let exampleIdx = 0;
  $("btn-example").addEventListener("click", () => {
    const ex = window.TEST_EXAMPLES[exampleIdx++ % window.TEST_EXAMPLES.length];
    const s = atob(ex.pixels);
    pixels = new Uint8Array(784);
    for (let i = 0; i < 784; i++) pixels[i] = s.charCodeAt(i);
    const img = sctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) img.data.set([pixels[i], pixels[i], pixels[i], 255], i * 4);
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, pad.width, pad.height);
    hasInk = true;
    $("pad-hint").textContent = `MNIST test image #${ex.index}, labelled ${ex.label}. Draw over it or clear it.`;
    update();
  });
  $("btn-clear").addEventListener("click", () => {
    $("pad-hint").textContent = "Draw one digit, large, in the middle of the box.";
    clearPad();
  });

  let pending = null;
  function schedule(fn) {
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = null; fn(); update(); });
  }

  // ---------------------------------------------------------------- static views
  const bars = $("bars");
  const barRows = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="bar-digit">${d}</span>
      <span class="bar-track"><span class="bar bar-fp32"></span><span class="bar bar-int8"></span></span>
      <span class="bar-pct"></span>`;
    bars.appendChild(row);
    barRows.push({ row, fp: row.querySelector(".bar-fp32"), q: row.querySelector(".bar-int8"), pct: row.querySelector(".bar-pct") });
  }

  const neuronBox = $("viz-neurons");
  const neuronCells = [];
  for (let i = 0; i < 128; i++) {
    const c = document.createElement("span");
    c.className = "neuron";
    c.tabIndex = 0;
    c.dataset.i = i;
    neuronBox.appendChild(c);
    neuronCells.push(c);
  }
  let lastResult = null;
  function showNeuron(e) {
    const i = Number(e.target.dataset.i);
    if (Number.isNaN(i) || !lastResult) return;
    const q = lastResult.hidden[i];
    const real = (q - engine.params.hidden.zeroPoint) * engine.params.hidden.scale;
    $("neuron-readout").textContent =
      `Neuron ${i}: 32-bit sum ${lastResult.hiddenAcc[i].toLocaleString("en")}, stored as the byte ${q} (≈ ${real.toFixed(3)})`;
  }
  neuronBox.addEventListener("pointerover", showNeuron);
  neuronBox.addEventListener("focusin", showNeuron);

  const scoreBox = $("viz-scores");
  const scoreCells = [];
  for (let d = 0; d < 10; d++) {
    const c = document.createElement("div");
    c.className = "score";
    c.innerHTML = `<span class="score-digit">${d}</span><span class="score-val">0</span>`;
    scoreBox.appendChild(c);
    scoreCells.push(c);
  }

  const vizPixels = $("viz-pixels").getContext("2d");
  const pixelImg = vizPixels.createImageData(28, 28);

  // Precision chart from the precomputed test-set numbers.
  const bitChart = $("bit-chart");
  const bitAcc = window.WEIGHT_BITS_ACCURACY;
  [8, 7, 6, 5, 4, 3, 2].forEach((b) => {
    const row = document.createElement("div");
    row.className = "bit-row";
    row.dataset.bits = b;
    row.innerHTML = `<span class="bit-label">${b} bits</span>
      <span class="bit-track"><span class="bit-fill" style="width:${bitAcc[b]}%"></span></span>
      <span class="bit-val">${bitAcc[b].toFixed(1)}%</span>`;
    bitChart.appendChild(row);
  });

  const bitsInput = $("bits");
  bitsInput.addEventListener("input", () => update());

  const article = (d) => (d === 8 ? "an " : "a ") + d;
  const pct = (p) => (p >= 0.9995 ? "over 99.9%" : (p * 100).toFixed(1) + "%");

  // ---------------------------------------------------------------- update
  function update() {
    const blank = !hasInk || pixels.every((v) => v === 0);
    const q = engine.infer(pixels);
    const f = fp32.infer(pixels);
    lastResult = q;
    const pq = Int8Engine.softmax(q.logits);
    const pf = Int8Engine.softmax(f.logits);

    $("answer-digit").textContent = blank ? "?" : String(q.prediction);
    $("answer-text").textContent = blank
      ? "Waiting for a drawing."
      : q.prediction === f.prediction
        ? `Both versions read ${article(q.prediction)}. Confidence: ${pct(pq[q.prediction])} with integers, ${pct(pf[f.prediction])} with floating point.`
        : `They disagree: integers say ${q.prediction}, floating point says ${f.prediction}. This happens on about 0.2% of test images.`;

    for (let d = 0; d < 10; d++) {
      const r = barRows[d];
      r.fp.style.width = blank ? "0%" : (pf[d] * 100).toFixed(1) + "%";
      r.q.style.width = blank ? "0%" : (pq[d] * 100).toFixed(1) + "%";
      r.pct.textContent = blank ? "" : (pq[d] * 100).toFixed(0) + "%";
      r.row.classList.toggle("is-top", !blank && d === q.prediction);
    }

    // Step 1: the 28x28 input
    for (let i = 0; i < 784; i++) pixelImg.data.set([pixels[i], pixels[i], pixels[i], 255], i * 4);
    vizPixels.putImageData(pixelImg, 0, 0);

    // Step 2: middle row as signed bytes
    let html = "";
    for (let x = 0; x < 28; x++) {
      const v = q.input[14 * 28 + x];
      html += `<span class="${v > -128 ? "ink" : ""}">${v}</span> `;
    }
    $("viz-bytes").innerHTML = html;

    // Step 3: hidden neurons
    const zh = engine.params.hidden.zeroPoint;
    let max = 1;
    for (let i = 0; i < 128; i++) max = Math.max(max, q.hidden[i] - zh);
    for (let i = 0; i < 128; i++) {
      const t = blank ? 0 : (q.hidden[i] - zh) / max;
      neuronCells[i].style.setProperty("--t", t.toFixed(3));
    }

    // Step 4: integer scores
    for (let d = 0; d < 10; d++) {
      scoreCells[d].querySelector(".score-val").textContent = q.outputQ[d];
      scoreCells[d].classList.toggle("is-top", !blank && d === q.prediction);
    }

    // Precision experiment
    const bits = Number(bitsInput.value);
    $("bits-out").textContent = bits + " bits";
    document.querySelectorAll(".bit-row").forEach((r) => r.classList.toggle("is-current", Number(r.dataset.bits) === bits));
    if (blank) {
      $("bits-result").textContent = "Draw a digit or load a test image to try this.";
    } else {
      const lowered = fp32.infer(pixels, bits);
      const p = Int8Engine.softmax(lowered.logits);
      $("bits-result").textContent =
        `With ${bits}-bit weights the network reads your drawing as ${article(lowered.prediction)} (${pct(p[lowered.prediction])} confident).`;
    }
  }

  clearPad();
})();
