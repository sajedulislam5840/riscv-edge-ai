# riscv-edge-ai

Integer-only neural network inference for small RISC-V cores, with a custom 4-lane INT8 multiply-accumulate (MAC) instruction designed in Verilog.

The goal is simple: run useful AI models on cheap microcontrollers that have no floating-point unit. A 32-bit RISC-V register can hold four 8-bit values, but a plain RV32IM core multiplies them one at a time. This project builds the software side (INT8 quantization and a fixed-point runtime) and the hardware side (a SIMD MAC unit mapped to the RISC-V `custom-0` opcode), and verifies that both produce exactly the same results.

**Live demo:** https://sajedulislam5840.github.io/riscv-edge-ai/ (draw a digit and watch the INT8 network classify it in the browser)

---

## Status

| Part | State |
|---|---|
| FP32 reference kernels (GEMM, ReLU, bias, softmax) | Done, unit tested |
| INT8 quantization library (affine, per-channel, Q31 requantization, BN folding, pooling, `.qbin` format) | Done, unit tested on host |
| SIMD MAC RTL (`hardware/rtl/simd_mac.v`) + `custom-0` decoder wrapper | Done, synthesized and verified against golden vectors |
| Real MNIST test-set evaluation of the INT8 pipeline | In progress |
| Bare-metal RV32IM firmware + cycle counts on Spike | In progress |
| Integration of the MAC into a RISC-V core (PicoRV32) | Planned |
| Technical report | Planned |

Everything listed as done below can be reproduced with the commands in [Reproduce](#reproduce).

---

## Verified results

### SIMD MAC hardware

The unit takes two 32-bit registers, each holding four signed INT8 values, and computes

```
rd = (clear ? 0 : rd) + a0*b0 + a1*b1 + a2*b2 + a3*b3
```

in one clock cycle: four 8x8 signed multipliers, a two-level adder tree, a 32-bit accumulator and an overflow flag.

**Synthesis (Yosys, no placement/routing yet):**

| Target | Multipliers | Logic | Flip-flops | Carry |
|---|---|---|---|---|
| Xilinx 7-series (`synth_xilinx`) | 4 DSP48E1 | 33 LUTs | 34 | 8 CARRY4 |
| Lattice iCE40 UP5K (`synth_ice40 -dsp`) | 4 SB_MAC16 | 67 LUT4 | 34 | 31 SB_CARRY |

Fmax has not been measured yet. That needs a place-and-route run (nextpnr / Vivado), which is on the to-do list.

**RTL verification:** `tests/gen_simd_vectors.cpp` pairs the trained first-layer weights of the MNIST network with a synthetic activation pattern and generates **25,088 test vectors**: every 4-element chunk of the 128 first-layer dot products (784 inputs each, so 196 chunks per neuron), with the accumulator chained from chunk to chunk. The Verilog RTL, compiled to C++ with Yosys CXXRTL, matches the software golden model on **all 25,088 vectors (0 mismatches)**.

**ISA encoding** (R-type, opcode `0x0B` / `custom-0`):

| Instruction | funct7 | funct3 | Operation |
|---|---|---|---|
| `simd_mac` | `0000000` | `000` | `rd = rd + dot4(rs1, rs2)` |
| `simd_mac_clr` | `0000001` | `000` | `rd = dot4(rs1, rs2)` |

Software can emit it with the standard binutils `.insn` directive, without modifying the compiler:

```c
asm volatile(".insn r 0x0B, 0, 1, %0, %1, %2" : "=r"(acc) : "r"(a), "r"(b));
```

`hardware/include/simd_mac_isa.h` wraps this and provides a plain C fallback for non-RISC-V builds.

### Quantization library

- 7 test suites (FP32 kernels, quantization, stress/overflow cases) pass with `ctest`.
- INT8 weights take 4x less memory than FP32 (101.6 KB vs 406.4 KB for the 784-128-10 MLP).
- The trained FP32 model reaches 98.23% on the 10,000-image MNIST test set (reported by `benchmarks/mnist/train_and_export.py`).
- **INT8 accuracy on the real test set: pending.** The current benchmark harness feeds synthetic inputs and uses hand-set activation scales, so its numbers are not a meaningful accuracy measure. It is being replaced with a real test-set evaluation that uses calibrated scales.

Speed numbers will be reported as RISC-V cycle counts (Spike, then RTL simulation), not as x86 host timings. Host timings say nothing about the target hardware.

---

## Design notes

**Quantization.** Affine INT8 quantization `q = clamp(round(r/S) + Z, -128, 127)`. Requantization after each layer uses a precomputed Q31 multiplier and shift (`M0 * 2^-shift`), so inference needs no floating-point operations. Weights are symmetric (`Z_W = 0`), which keeps the hardware datapath to plain signed multiplies.

**Why a stateless R-type instruction (and its limitation).** The accumulate form `simd_mac` reads three registers: `rs1`, `rs2` and the old value of `rd`. Most small RISC-V cores, including PicoRV32's PCPI coprocessor port, only provide two source operands. For core integration I plan to move the accumulator inside the unit (`clear` / `accumulate` / `read` instructions), which needs only two read ports. I'll document this decision in `hardware/docs/simd_mac_spec.md` when I make the change.

**Overflow.** Four products of at most 2^14 in magnitude fit in 18 bits, so the adder tree cannot overflow. Only the 32-bit accumulator can, and the unit flags it.

---

## Reproduce

Requirements: CMake, a C++17 compiler, Python 3. For the hardware steps, also Yosys (or `pip install yowasp-yosys` and set `YOSYS=yowasp-yosys`).

```bash
# C++ library and unit tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# RTL co-simulation: all 25,088 golden vectors through the Verilog
bash hardware/scripts/cosim.sh

# Synthesis resource report (Xilinx 7-series and iCE40)
bash hardware/scripts/synth.sh

# Optional: event-driven Verilog testbenches with Icarus Verilog
bash hardware/scripts/run_sim.sh
```

To retrain the model and regenerate the weights: `python3 benchmarks/mnist/train_and_export.py` (needs PyTorch).

---

## Repository layout

```
include/engine/     tensor, quantization and kernel headers
src/                FP32 ops and INT8 quantization engine
tests/              unit tests + golden-vector generator for the RTL
benchmarks/mnist/   training/export scripts, benchmark harness, trained model (.qbin)
hardware/rtl/       simd_mac.v (datapath), simd_mac_wrapper.v (custom-0 decoder)
hardware/tb/        Verilog testbenches, CXXRTL co-sim, golden vectors
hardware/include/   C header for the custom instruction
hardware/docs/      microarchitecture spec, synthesis notes
web/                browser demo (vanilla JS, fixed-point INT8 inference)
docs/               quantization spec, benchmark notes, figures
```

---

## Roadmap

1. Evaluate the INT8 pipeline on the real MNIST test set with calibrated activation scales.
2. Freestanding C firmware for RV32IM (no heap, no exceptions, no standard library) running on Spike, with `rdcycle` measurements for three versions of the same GEMM: scalar, SIMD dot-product, and SIMD with an internal accumulator.
3. Attach the MAC unit to PicoRV32 and run full inference in RTL simulation. Report area and Fmax with and without the unit.
4. Short technical report.

Later: design-space exploration (2/4/8 lanes, saturation, pipelining), a small CNN workload, FPGA bring-up, and porting to other open RISC-V cores.

## Related work

This builds on well-known ideas: the RISC-V P (packed SIMD) extension has a similar 4-way INT8 multiply-accumulate (`smaqa`), and projects such as Google's CFU Playground and PULP's XpulpNN add custom ML instructions to open RISC-V cores. What this project adds is a small, fully open, end-to-end stack, from quantization to RTL, built so that every step can be checked against the previous one.

## Acknowledgements

I used AI coding assistants while developing this project. The design decisions, verification and measurements are my own, and I can explain every part of it.

## License

MIT, see [LICENSE](LICENSE).
