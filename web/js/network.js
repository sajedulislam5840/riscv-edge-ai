// Deterministic weight initialization
const ModelWeights = {
  W1: new Float32Array(784 * 128),
  b1: new Float32Array(128),
  W2: new Float32Array(128 * 10),
  b2: new Float32Array(10)
};

(function initializeWeights() {
  // Seeded RNG so weights are deterministic
  let seed = 1234567;
  const rnd = () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  // Each hidden node is a unique oriented Gabor patch. Centers tile the 28x28 image
  // in the same 16x8 layout as the activation grid, so the grid reads as a spatial map.
  for (let c = 0; c < 128; c++) {
    const cx = ((c % 16) + 0.5) * (28 / 16) + (rnd() - 0.5) * 3;
    const cy = (Math.floor(c / 16) + 0.5) * (28 / 8) + (rnd() - 0.5) * 3;
    const sigma = 3 + rnd() * 3;
    const theta = rnd() * Math.PI;
    const k = 0.5 + rnd() * 0.9;
    const phase = rnd() * Math.PI * 2;
    const ct = Math.cos(theta), st = Math.sin(theta);
    for (let r = 0; r < 784; r++) {
      const x = (r % 28) - cx;
      const y = Math.floor(r / 28) - cy;
      const env = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      ModelWeights.W1[r * 128 + c] = 0.35 * env * Math.cos(k * (x * ct + y * st) + phase);
    }
    ModelWeights.b1[c] = -0.15 + (rnd() - 0.5) * 0.1;
  }
  for (let r = 0; r < 128; r++) {
    for (let c = 0; c < 10; c++) ModelWeights.W2[r * 10 + c] = Math.sin((r + 1) * (c + 1) * 0.28) * 0.45;
  }
  for (let i = 0; i < 10; i++) ModelWeights.b2[i] = 0.0;
})();

class DigitCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    this.isDrawing = false;
    this.setupListeners();
    this.clear();
  }

  infer() {
    if (window.triggerInference) window.triggerInference();
  }

  setupListeners() {
    const c = this.canvas;

    c.addEventListener('pointerdown', (e) => {
      this.isDrawing = true;
      c.setPointerCapture(e.pointerId);
      c.classList.add('drawing');
      this.draw(e);
    });

    c.addEventListener('pointermove', (e) => this.draw(e));

    const stop = () => {
      if (!this.isDrawing) return;
      this.isDrawing = false;
      c.classList.remove('drawing');
      this.ctx.beginPath();
      this.infer();
    };
    c.addEventListener('pointerup', stop);
    c.addEventListener('pointercancel', stop);

    document.getElementById('btn-clear').addEventListener('click', () => {
      this.clear();
      this.flash();
      this.infer();
    });

    document.getElementById('btn-invert').addEventListener('click', () => {
      this.invert();
      this.flash();
      this.infer();
    });

    document.querySelectorAll('.action-btn').forEach((btn) => {
      btn.addEventListener('pointerdown', (e) => {
        const r = btn.getBoundingClientRect();
        const size = Math.max(r.width, r.height);
        const dot = document.createElement('span');
        dot.className = 'ripple';
        dot.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - r.left - size / 2}px;top:${e.clientY - r.top - size / 2}px`;
        btn.appendChild(dot);
        dot.addEventListener('animationend', () => dot.remove());
      });
    });
  }

  flash() {
    this.canvas.classList.remove('flash');
    void this.canvas.offsetWidth;
    this.canvas.classList.add('flash');
  }

  draw(e) {
    if (!this.isDrawing) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

    this.ctx.lineWidth = 14;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = '#ffffff';

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
    this.infer();
  }

  clear() {
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.beginPath();
  }

  invert() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255 - d[i];
      d[i + 1] = 255 - d[i + 1];
      d[i + 2] = 255 - d[i + 2];
    }
    this.ctx.putImageData(imgData, 0, 0);
  }

  getDownsampledPixels() {
    if (!this._off) {
      this._off = document.createElement('canvas');
      this._off.width = 28;
      this._off.height = 28;
      this._offCtx = this._off.getContext('2d', { willReadFrequently: true });
    }
    this._offCtx.drawImage(this.canvas, 0, 0, 28, 28);
    const img = this._offCtx.getImageData(0, 0, 28, 28).data;

    const tensor = new Float32Array(784);
    for (let i = 0; i < 784; i++) tensor[i] = img[i * 4] / 255; // 0..1, background = 0
    return tensor;
  }
}

window.DigitCanvas = DigitCanvas;
window.ModelWeights = ModelWeights;