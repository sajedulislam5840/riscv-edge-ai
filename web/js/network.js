// Deterministic synthetic weights matching 784 -> 128 -> 10 MLP
const ModelWeights = {
  W1: new Float32Array(784 * 128),
  b1: new Float32Array(128),
  W2: new Float32Array(128 * 10),
  b2: new Float32Array(10)
};

(function initializeTrainedFeatures() {
  for (let r = 0; r < 784; r++) {
    const row = Math.floor(r / 28);
    const col = r % 28;
    const centerDist = Math.hypot(row - 14, col - 14);
    for (let c = 0; c < 128; c++) {
      const freq = (c % 8) + 1;
      ModelWeights.W1[r * 128 + c] = Math.sin(col * 0.2 * freq) * Math.cos(row * 0.2 * freq) * 0.4 * (centerDist < 12 ? 1.4 : 0.1);
    }
  }
  for (let i = 0; i < 128; i++) {
    ModelWeights.b1[i] = -0.05 + (i % 5) * 0.02;
  }
  for (let r = 0; r < 128; r++) {
    for (let c = 0; c < 10; c++) {
      ModelWeights.W2[r * 10 + c] = Math.sin((r + 1) * (c + 1) * 0.3) * 0.4;
    }
  }
  for (let i = 0; i < 10; i++) {
    ModelWeights.b2[i] = 0.0;
  }
})();

class DigitCanvas {
  constructor(canvasId, previewId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.previewCanvas = document.getElementById(previewId);
    this.previewCtx = this.previewCanvas.getContext('2d');
    this.isDrawing = false;
    this.setupListeners();
    this.clear();
  }

  setupListeners() {
    const start = (e) => {
      this.isDrawing = true;
      this.draw(e);
    };
    const stop = () => {
      if (!this.isDrawing) return;
      this.isDrawing = false;
      this.ctx.beginPath();
      this.updateDownsampled();
      if (window.triggerInference) window.triggerInference();
    };

    this.canvas.addEventListener('mousedown', start);
    this.canvas.addEventListener('mousemove', (e) => this.draw(e));
    window.addEventListener('mouseup', stop);

    this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); start(e.touches[0]); });
    this.canvas.addEventListener('touchmove', (e) => { e.preventDefault(); this.draw(e.touches[0]); });
    window.addEventListener('touchend', stop);

    document.getElementById('btn-clear').addEventListener('click', () => {
      this.clear();
      this.updateDownsampled();
      if (window.triggerInference) window.triggerInference();
    });

    document.getElementById('btn-invert').addEventListener('click', () => {
      this.invert();
      this.updateDownsampled();
      if (window.triggerInference) window.triggerInference();
    });
  }

  draw(e) {
    if (!this.isDrawing) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.ctx.lineWidth = 16;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = '#ffffff';

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
    this.updateDownsampled();
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

  updateDownsampled() {
    this.previewCtx.drawImage(this.canvas, 0, 0, 28, 28);
  }

  getDownsampledPixels() {
    this.updateDownsampled();
    const img = this.previewCtx.getImageData(0, 0, 28, 28).data;
    const tensor = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      tensor[i] = (img[i * 4] / 127.5) - 1.0;
    }
    return tensor;
  }
}

window.DigitCanvas = DigitCanvas;
window.ModelWeights = ModelWeights;
