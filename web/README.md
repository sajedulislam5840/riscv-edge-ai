# Web demo

Browser demo of the INT8 network: draw a digit on the canvas and the page runs the same fixed-point
inference as the C++ engine, showing the hidden-layer activations, FP32 vs INT8 logits and the
quantization error (SQNR, RMSE, max error). Plain HTML, CSS and JavaScript, plus Chart.js.

Live: https://sajedulislam5840.github.io/riscv-edge-ai/

## Run locally

```bash
cd web
python -m http.server 8080
```

Then open http://localhost:8080.

The weights in `js/weights.js` are written by `benchmarks/mnist/train_and_export.py`.
