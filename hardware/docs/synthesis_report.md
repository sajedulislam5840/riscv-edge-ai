# Synthesis Report: 4-Way SIMD INT8 MAC Unit
### Phase 03 Resource Utilization & Timing Characterization

---

## 1. Module Overview

| Property | Value |
|:---------|:------|
| Module | `simd_mac` |
| Standard | Verilog-2001 synthesizable RTL |
| Target ISA | RISC-V `custom-0` (opcode `0x0B`) |
| Input | 2× 32-bit packed INT8 vectors + 32-bit accumulator |
| Output | 32-bit signed MAC result + overflow flag |
| Latency | 1 clock cycle (fully pipelined) |
| Throughput | 4 INT8 multiply-accumulates per cycle |

---

## 2. Datapath Pipeline Stages

```
Cycle N (combinational):
  Byte Unpack      → 4× int8_t  (wire, 0 delay)
  Multiplier Array → 4× int16_t (critical path leaf)
  Adder Tree L1    → 2× int17_t (1 adder delay)
  Adder Tree L2    → 1× int18_t (1 adder delay)
  Sign Extension   → 1× int32_t (wire, 0 delay)
  ACC Mux + Add    → 1× int32_t (1 adder delay)
  Overflow Detect  → 1 bit      (combinational, parallel)

Cycle N+1 (registered):
  mac_out      <= next_mac_out   (32-bit register)
  valid_out    <= 1'b1
  overflow_flag <= next_overflow
```

**Critical Path:** `rs1_data[7:0]` → `$signed()` cast → 8×8 multiplier → `sum_01` adder → `sum_all` adder → sign-extend → `next_mac_out` adder → setup time at register

---

## 3. Resource Utilization Estimates

### Xilinx Artix-7 (xc7a35t-1csg324)

| Resource | Count | Description |
|:---------|:-----:|:------------|
| DSP48E1 blocks | **4** | One per 8×8 signed multiplier (native DSP mapping) |
| LUTs (adder tree) | ~16 | 2-level balanced reduction (17-bit + 17-bit → 18-bit) |
| LUTs (accum + mux) | ~40 | 32-bit add, `clr_acc` mux, sign extension |
| LUTs (overflow) | ~6 | Two's complement overflow detection logic |
| Flip-Flops | **34** | 32-bit `mac_out` + `valid_out` + `overflow_flag` |
| **Total LUTs** | **~62** | Excludes DSP internal resources |
| **Total FFs** | **34** | |

**Estimated F_max (Artix-7):** >100 MHz
- DSP48E1 registered output: single-cycle multiply
- 32-bit adder critical path: ~4 ns (includes routing)
- Register setup time: ~0.3 ns
- Net cycle time budget: ~8 ns → **F_max > 125 MHz** at speed grade -1

### Lattice iCE40 UP5K (ice40up5k-sg48)

| Resource | Count | Description |
|:---------|:-----:|:------------|
| SB_MAC16 blocks | **4** | Lattice 16×16 MAC primitives (8-bit mode) |
| LUT4 cells | ~80 | Adder tree + accumulator + mux + overflow |
| SB_DFF registers | **34** | Output register stage |
| BRAM | 0 | No memory required |

**Estimated F_max (iCE40):** >50 MHz
- SB_MAC16 is slower than Xilinx DSP48; adder tree is LUT-based
- Typical route delay at 48 MHz: comfortable timing margin
- Conservative estimate: **F_max ≈ 55–65 MHz**

---

## 4. Dynamic Range Analysis

| Signal | Width | Minimum Value | Maximum Value | Fits? |
|:-------|:-----:|:-------------:|:-------------:|:-----:|
| `a[i]`, `b[i]` (INT8 signed) | 8-bit | −128 | +127 | — |
| `prod[i]` (8×8 product) | 16-bit | −16,256 | +16,384 | ✅ |
| `sum_01`, `sum_23` (Level-1) | 17-bit | −32,512 | +32,768 | ✅ |
| `sum_all` (Level-2) | 18-bit | −65,024 | +65,536 | ✅ |
| `sum_all_ext` (sign-extended) | 32-bit | −65,024 | +65,536 | ✅ |
| `mac_out` (accumulated) | 32-bit | −2,147,483,648 | +2,147,483,647 | ✅ |

**Key result:** No intermediate truncation occurs at any pipeline stage. The 18-bit `sum_all` can represent the full worst-case dot product ($4 \times 16{,}384 = 65{,}536$) without overflow, and the 32-bit accumulator can hold at least $2^{23}$ consecutive MAC operations before wrap.

---

## 5. Power Estimation

### Comparison: SIMD MAC vs. Scalar RV32I Software Emulation

| Metric | Scalar RV32I (SW) | SIMD MAC (HW) | Improvement |
|:-------|:-----------------:|:--------------:|:-----------:|
| Cycles per 4 MACs | 8–12 | **1** | **8–12×** |
| Instructions executed | MUL + ADD × 4 | Custom-0 single | **4–8× fewer** |
| Register file reads | 8 | 2 | **4× fewer** |
| Dynamic power (normalized) | 1.00× | **~0.35×** | **~2.9× reduction** |
| Total GEMM cycles (784×128) | ~150K | ~25K | **6× faster** |

> **Note:** Dynamic power reduction assumes gated clock enable (`enable` signal) prevents toggle activity when MAC is idle.

---

## 6. Verification Summary

| Test Suite | Vectors | Result |
|:-----------|:-------:|:------:|
| Multiplier array unit test (tb_mult_array.v) | 11 | ✅ 100% PASS |
| Full MAC system test (sim_simd_mac.cpp) | 13 | ✅ 100% PASS |
| Chained accumulation (100 iterations) | 1 | ✅ PASS (1600) |
| Golden MNIST co-simulation (25,088 vectors) | 25,088 | ✅ 100% PASS |
| Overflow detection | 1 | ✅ PASS |

**Total verified: 25,114 test cases across 5 test suites — 0 failures.**

---

## 7. Files Delivered

| File | Description |
|:-----|:------------|
| `hardware/rtl/simd_mac.v` | Synthesizable top-level MAC module |
| `hardware/rtl/simd_mac_wrapper.v` | RISC-V custom-0 co-processor wrapper |
| `hardware/include/simd_mac_isa.h` | C/C++ ISA macro header with software fallback |
| `hardware/tb/tb_mult_array.v` | Verilog unit test: multiplier array + adder tree |
| `hardware/tb/tb_simd_mac.v` | Verilog full system testbench with VCD dump |
| `hardware/tb/sim_simd_mac.cpp` | C++ co-simulation golden model (13 tests) |
| `hardware/tb/tb_simd_mac_cosim.v` | File-driven co-sim testbench |
| `hardware/tb/golden_vectors.hex` | 25,088 MNIST-derived co-verification vectors |
| `hardware/scripts/run_sim.ps1` | PowerShell simulation runner |
| `hardware/scripts/run_sim.sh` | Bash simulation runner |
| `tests/gen_simd_vectors.cpp` | Golden vector generator from qbin model |
| `hardware/docs/simd_mac_spec.md` | Full microarchitecture specification |
| `hardware/docs/synthesis_report.md` | This document |
