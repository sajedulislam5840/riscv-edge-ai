"""Quantize the trained MNIST MLP and export the data the C++ benchmark needs.

Inputs
  web/js/weights.js     trained FP32 weights, written by train_and_export.py
  mnist.npz             MNIST in the Keras format (x_train, y_train, x_test, y_test)

Outputs
  benchmarks/mnist/results/mnist_model.qbin   INT8 model, symmetric weights (zero point 0)
  benchmarks/mnist/data/fp32_weights.bin      the same weights in FP32, for the reference path
  benchmarks/mnist/data/mnist_model_asym.qbin asymmetric variant, for comparison only
  benchmarks/mnist/data/mnist_test.bin        the 10,000 test images and labels
  benchmarks/mnist/data/mnist_calib.bin       the first 2,000 training images (activation calibration)

Weights are quantized symmetrically per tensor: q = round(w / S), S = max|w| / 127, zero point 0.
That is what the SIMD MAC unit assumes: it multiplies raw signed bytes, and the activation zero
point is folded into the bias as a constant per output neuron.

Usage (from the repo root):  python benchmarks/mnist/export_model_and_data.py [path/to/mnist.npz]
"""
import base64
import os
import re
import struct
import sys

import numpy as np

QBIN_MAGIC = 0x514E4554
DATA_MAGIC = 0x54534E4D  # "MNST"

npz_path = sys.argv[1] if len(sys.argv) > 1 else "mnist.npz"
if not os.path.exists(npz_path):
    sys.exit(f"{npz_path} not found. Download it from "
             "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")

# ---- trained FP32 weights -------------------------------------------------
js = open("web/js/weights.js", encoding="utf-8").read()
b64 = re.search(r'TRAINED_WEIGHTS_B64 = "([^"]+)"', js).group(1)
flat = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
assert flat.size == 784 * 128 + 128 + 128 * 10 + 10, flat.size
W1 = flat[:100352].reshape(784, 128)
b1 = flat[100352:100480].reshape(1, 128)
W2 = flat[100480:101760].reshape(128, 10)
b2 = flat[101760:].reshape(1, 10)
layers = [W1, b1, W2, b2]


def symmetric(arr):
    s = float(np.max(np.abs(arr))) / 127.0
    s = s if s > 0 else 1.0
    return s, 0, np.clip(np.round(arr / s), -127, 127).astype(np.int8)


def asymmetric(arr):
    mn, mx = min(float(arr.min()), 0.0), max(float(arr.max()), 0.0)
    s = (mx - mn) / 255.0 if mx > mn else 1.0
    zp = int(np.clip(np.round(-128.0 - mn / s), -128, 127))
    return s, zp, np.clip(np.round(arr / s) + zp, -128, 127).astype(np.int8)


def write_qbin(path, quantizer):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", QBIN_MAGIC, len(layers)))
        for arr in layers:
            s, zp, q = quantizer(arr)
            f.write(struct.pack("<IIfi", arr.shape[0], arr.shape[1], s, zp))
            f.write(q.tobytes())


def write_images(path, images, labels):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", DATA_MAGIC, len(images), 28, 28))
        f.write(images.astype(np.uint8).tobytes())
        f.write(labels.astype(np.uint8).tobytes())


os.makedirs("benchmarks/mnist/results", exist_ok=True)
os.makedirs("benchmarks/mnist/data", exist_ok=True)

write_qbin("benchmarks/mnist/results/mnist_model.qbin", symmetric)
write_qbin("benchmarks/mnist/data/mnist_model_asym.qbin", asymmetric)
flat.astype("<f4").tofile("benchmarks/mnist/data/fp32_weights.bin")

d = np.load(npz_path)
write_images("benchmarks/mnist/data/mnist_test.bin", d["x_test"], d["y_test"])
write_images("benchmarks/mnist/data/mnist_calib.bin", d["x_train"][:2000], d["y_train"][:2000])

# Reference FP32 accuracy computed here, so the C++ result can be cross-checked.
x = d["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
logits = np.maximum(x @ W1 + b1, 0) @ W2 + b2
acc = (logits.argmax(1) == d["y_test"]).mean() * 100
print(f"FP32 test accuracy (numpy reference): {acc:.2f}%")
print("Wrote mnist_model.qbin (symmetric), fp32 weights, test set and calibration set.")
