# Web demo

Draw a digit and the page classifies it twice: with the trained FP32 weights, and with the
integer-only INT8 engine. The INT8 path in `js/int8_engine.js` is a port of the C++ engine and
reproduces it byte for byte (checked on all 10,000 MNIST test images). Plain HTML, CSS and
JavaScript, no build step and no libraries.

Live: https://sajedulislam5840.github.io/riscv-edge-ai/

## Files

- `js/weights.js`: trained FP32 weights, written by `benchmarks/mnist/train_and_export.py`
- `js/model_int8.js`: the INT8 model (same bytes as the `.qbin`), example test images and the
  bit-width table, written by `benchmarks/mnist/export_model_and_data.py`
- `js/quant_params.js`: calibrated scales and requantization constants, written by `mnist_benchmark`
- `js/int8_engine.js`: integer inference and the FP32 reference
- `js/app.js`: drawing pad, preprocessing and the page's live views

## Run locally

```bash
cd web
python -m http.server 8080
```

Then open http://localhost:8080.
