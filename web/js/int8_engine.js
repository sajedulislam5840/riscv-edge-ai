// Integer-only INT8 inference for the 784-128-10 MNIST network, written to match the C++
// engine (src/quantize.cpp, matmul_int8_fixed_point_fused) bit for bit.
//
// Inputs come from model_int8.js (the .qbin model) and quant_params.js (calibrated scales and
// requantization constants written by the C++ benchmark). Works in the browser and in Node.
(function (root) {
  "use strict";

  function base64ToBytes(b64) {
    if (typeof atob === "function") {
      const s = atob(b64);
      const out = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return Uint8Array.from(Buffer.from(b64, "base64"));
  }

  // .qbin: magic, tensor count, then per tensor rows, cols, scale (f32), zero point (i32), int8 data.
  function parseQbin(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.getUint32(0, true) !== 0x514e4554) throw new Error("not a .qbin model");
    const count = dv.getUint32(4, true);
    const tensors = [];
    let off = 8;
    for (let i = 0; i < count; i++) {
      const rows = dv.getUint32(off, true);
      const cols = dv.getUint32(off + 4, true);
      const scale = dv.getFloat32(off + 8, true);
      const zeroPoint = dv.getInt32(off + 12, true);
      off += 16;
      const data = new Int8Array(bytes.buffer, bytes.byteOffset + off, rows * cols);
      off += rows * cols;
      tensors.push({ rows, cols, scale, zeroPoint, data });
    }
    return tensors;
  }

  // round(x * multiplier * 2^-shift), as multiply_by_quantized_multiplier does it in int64.
  // All intermediate values stay below 2^53, so JavaScript numbers are exact here.
  function requantize(x, multiplier, shift) {
    const total = x * multiplier;
    if (shift <= 0) return total * Math.pow(2, -shift);
    const p = Math.pow(2, shift);
    return Math.floor((total + p / 2) / p);
  }

  // One fused dense layer: INT32 accumulate, add rescaled bias, requantize, optional ReLU.
  function dense(x, zx, W, b, l, zOut, relu) {
    const K = W.rows, N = W.cols, w = W.data;
    const out = new Int8Array(N);
    const acc = new Int32Array(N);
    for (let j = 0; j < N; j++) {
      let a = 0;
      for (let k = 0; k < K; k++) a += (x[k] - zx) * (w[k * N + j] - W.zeroPoint);
      a += requantize(b.data[j] - b.zeroPoint, l.biasMultiplier, l.biasShift);
      acc[j] = a;
      let r = requantize(a, l.multiplier, l.shift) + zOut;
      if (relu && r < zOut) r = zOut;
      out[j] = Math.max(-128, Math.min(127, r));
    }
    return { out, acc };
  }

  function argmax(v) {
    let best = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return best;
  }

  function createEngine(qbinBase64, params) {
    const [W1, b1, W2, b2] = parseQbin(base64ToBytes(qbinBase64));
    const P = params;
    return {
      params: P,
      layers: { W1, b1, W2, b2 },
      // pixels: 784 values in 0..255 (row-major 28x28, white digit on black).
      infer(pixels) {
        const x = new Int8Array(784);
        for (let i = 0; i < 784; i++) x[i] = pixels[i] - 128; // S = 1/255, Z = -128
        const h = dense(x, P.input.zeroPoint, W1, b1, P.layer1, P.hidden.zeroPoint, true);
        const o = dense(h.out, P.hidden.zeroPoint, W2, b2, P.layer2, P.output.zeroPoint, false);
        const logits = Array.from(o.out, (q) => (q - P.output.zeroPoint) * P.output.scale);
        return { input: x, hidden: h.out, hiddenAcc: h.acc, outputQ: o.out, logits, prediction: argmax(o.out) };
      },
    };
  }

  // FP32 reference forward pass with the trained float weights (weights.js).
  function createFp32(weightsBase64) {
    const bytes = base64ToBytes(weightsBase64);
    const f = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    const W1 = f.subarray(0, 100352), b1 = f.subarray(100352, 100480);
    const W2 = f.subarray(100480, 101760), b2 = f.subarray(101760, 101770);
    return {
      infer(pixels, weightBits) {
        const q = weightBits ? fakeQuant : (w) => w;
        const w1 = q(W1, weightBits, "W1"), c1 = q(b1, weightBits, "b1");
        const w2 = q(W2, weightBits, "W2"), c2 = q(b2, weightBits, "b2");
        const h = new Float32Array(128);
        for (let j = 0; j < 128; j++) {
          let s = c1[j];
          for (let k = 0; k < 784; k++) if (pixels[k]) s += (pixels[k] / 255) * w1[k * 128 + j];
          h[j] = s > 0 ? s : 0;
        }
        const logits = new Array(10);
        for (let j = 0; j < 10; j++) {
          let s = c2[j];
          for (let k = 0; k < 128; k++) s += h[k] * w2[k * 10 + j];
          logits[j] = s;
        }
        return { hidden: h, logits, prediction: argmax(logits) };
      },
    };
  }

  // Symmetric per-tensor rounding of weights to n bits (for the precision experiment).
  const fqCache = {};
  function fakeQuant(w, bits, key) {
    const k = key + ":" + bits;
    if (fqCache[k]) return fqCache[k];
    const qmax = Math.pow(2, bits - 1) - 1;
    let m = 0;
    for (let i = 0; i < w.length; i++) m = Math.max(m, Math.abs(w[i]));
    const s = m / qmax;
    const out = new Float32Array(w.length);
    for (let i = 0; i < w.length; i++) out[i] = Math.max(-qmax, Math.min(qmax, Math.round(w[i] / s))) * s;
    return (fqCache[k] = out);
  }

  function softmax(v) {
    const m = Math.max.apply(null, v);
    const e = v.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }

  root.Int8Engine = { createEngine, createFp32, softmax, parseQbin, requantize };
})(typeof window !== "undefined" ? window : globalThis);
