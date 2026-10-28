// Deterministic weight initialization
const ModelWeights = {
  W1: new Float32Array(784 * 128),
  b1: new Float32Array(128),
  W2: new Float32Array(128 * 10),
  b2: new Float32Array(10)
};

(function initializeWeights() {
  for (let r = 0; r < 784; r++) {
    const row = Math.floor(r / 28);
    const col = r % 28;
    const centerDist = Math.hypot(row - 14, col - 14);
    for (let c = 0; c < 128; c++) {
      const freq = (c % 8) + 1;
      ModelWeights.W1[r * 128 + c] = Math.sin(col * 0.25 * freq) * Math.cos(row * 0.25 * freq) * 0.35 * (centerDist < 11 ? 1.6 : 0.1);
    }
  }
  for (let i = 0; i < 128; i++) ModelWeights.b1[i] = -0.04 + (i % 7) * 0.015;
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
    for (let i = 0; i < 784; i++) tensor[i] = (img[i * 4] / 127.5) - 1.0;
    return tensor;
  }
}

window.DigitCanvas = DigitCanvas;
window.ModelWeights = ModelWeights;