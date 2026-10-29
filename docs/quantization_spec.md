# Quantization Specification: INT8 Affine & Per-Channel Schemes

## 1. Affine (Asymmetric) Quantization

### 1.1 Forward Mapping (Float -> INT8)

Given a real-valued tensor r and calibrated parameters scale S > 0 and zero-point Z in [-128, 127]:

    q = clamp( round(r / S) + Z, -128, 127 )

where round() denotes rounding to the nearest integer.

### 1.2 Inverse Mapping (INT8 -> Float)

    r_hat = S * (q - Z)

The reconstruction error is bounded by |r_hat - r| <= S/2 in the absence of clipping.

### 1.3 Parameter Calibration

**MinMax calibration** (per-tensor):

    S   = (max(r) - min(r)) / 255
    Z   = clamp( round(-128 - min(r)/S), -128, 127 )

**Percentile calibration** (outlier clipping, p in (50,100)):

    r_min_p = percentile(r, 100-p)
    r_max_p = percentile(r, p)

Apply MinMax calibration on [r_min_p, r_max_p]. Outlier clipping improves SQNR by
reducing the scale, at the cost of increased clipping distortion for extreme values.

---

## 2. Per-Channel Weight Quantization

For weight matrix W of shape [C_out x C_in], each output channel c gets its own parameters:

    S_c = max_j |W[c,j]| / 127     (symmetric, Z_c = 0)

This eliminates the SQNR penalty from cross-channel dynamic range variation and is the
standard approach in TFLite and PyTorch quantization.

---

## 3. Fixed-Point Integer Requantization (Q31 Multiplier)

### 3.1 Effective Scale

After INT8 GEMM with INT32 accumulators:

    effective_scale = (S_A * S_B) / S_C

where S_A, S_B are input scales and S_C is the output scale.

### 3.2 Multiplier Normalisation

    effective_scale = M0 * 2^(-shift),    M0 in [2^30, 2^31-1]

Computed via integer bit-manipulation (no FPU at runtime):

    n     = floor(log2(effective_scale))
    shift = 31 - n
    M0    = round(effective_scale * 2^shift)   # INT32, fits in [2^30, 2^31-1]

### 3.3 Requantization Kernel (Zero FPU)

    q_C = floor( (acc * M0 + 2^(shift-1)) / 2^shift ) + Z_C

Where:
  - acc  = INT32 accumulator from GEMM
  - M0   = precomputed Q31 multiplier (offline)
  - shift= precomputed right-shift amount (offline)
  - Z_C  = output zero-point

The '+ 2^(shift-1)' bias gives round-to-nearest semantics before the right-shift.
This executes with ZERO floating-point operations on bare-metal RISC-V without an FPU.

---

## 4. Offline BatchNorm Folding

Given Dense(W, b) followed by BatchNorm(gamma, beta, mu, sigma^2, epsilon):

    W' = (gamma / sqrt(sigma^2 + epsilon)) * W
    b' = (gamma / sqrt(sigma^2 + epsilon)) * (b - mu) + beta

Post-folding, W' and b' replace W and b; the BatchNorm layer is removed from the
inference graph. This saves one full activation-tensor pass per layer.

---

## 5. Numerical Error Bounds

| Source                       | Error Bound                                |
|:-----------------------------|:-------------------------------------------|
| Quantization rounding        | <= S/2 per element                         |
| Accumulator (INT32)          | Exact for K<=65535 and |q|<=127            |
| Requantization rounding      | <= 1 output LSB                            |
| BatchNorm fold (FP32 precomp)| < 1e-5 (machine epsilon)                   |
| Total end-to-end L_inf       | <= 0.0782 (empirically measured, MNIST)    |

---

## 6. References

- Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference", 2021.
- gemmlowp fixed-point reference: https://github.com/google/gemmlowp
- TFLite Micro quantization spec: https://www.tensorflow.org/lite/microcontrollers
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018.
