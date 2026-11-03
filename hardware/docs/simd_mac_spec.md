# Microarchitectural Specification: 4-Way SIMD INT8 Multiply-Accumulate (MAC) Unit
### RISC-V Custom-0 Co-Processor ISA Extension

---

## 1. Executive Summary

This document specifies the microarchitecture, register packing conventions, interface signals, and timing of the **4-Way SIMD INT8 Multiply-Accumulate (MAC)** accelerator unit. 

The core target is to accelerate General Matrix Multiply (GEMM) in quantized integer neural networks (such as 784→128→10 MLPs on MNIST) by processing four 8-bit multiply-accumulate operations in a single clock cycle, delivering a theoretical **4× arithmetic throughput improvement** over scalar RV32I integer execution while requiring zero floating-point hardware.

---

## 2. Microarchitecture Block Diagram

```
                             32-bit rs1 (Vector A)             32-bit rs2 (Vector B)
                                       │                                 │
                 ┌──────────┬──────────┼──────────┬──────────┐           │
                 ▼          ▼          ▼          ▼          │           │
            [Byte 3]   [Byte 2]   [Byte 1]   [Byte 0]        │           │
            [31:24]    [23:16]     [15:8]      [7:0]         │           │
               │          │          │          │            │           │
               │   ┌──────┴──────────┼──────────┴────────────┼───────────┘
               │   │                 │                       │
               ▼   ▼                 ▼                       ▼
            ┌─────────┐         ┌─────────┐             ┌─────────┐
            │ Mult 3  │         │ Mult 2  │  ...        │ Mult 0  │
            │  8 x 8  │         │  8 x 8  │             │  8 x 8  │
            └────┬────┘         └────┬────┘             └────┬────┘
         16-bit  │            16-bit │                16-bit │
         signed  ▼            signed ▼                signed ▼
               prod3               prod2                   prod0
                 │                   │                       │
                 └─────────┬─────────┘                       │
                           ▼                                 │
                   ┌──────────────┐                          │
                   │ Adder Tree L1│ (17-bit signed)          │
                   └───────┬──────┘                          │
                           │                                 │
                           └─────────────────┬───────────────┘
                                             ▼
                                     ┌──────────────┐
                                     │ Adder Tree L2│ (18-bit signed sum_all)
                                     └───────┬──────┘
                                             │
                                             ▼
                     32-bit acc_in ──► [clr_acc MUX]
                                             │
                                             ▼
                                     ┌──────────────┐
                                     │  32-bit ADD  │◄── 18-to-32 Sign Extension
                                     └───────┬──────┘
                                             │
                                    ┌────────┴────────┐
                                    ▼                 ▼
                             [mac_out 32-bit]   [overflow_flag]
```

---

## 3. Register Packing & Byte Conventions

RISC-V standard registers are 32 bits wide. Each 32-bit source register is treated as a SIMD vector holding four packed 8-bit signed two's complement integers ($[-128, 127]$):

| Register Bits | Byte Index | C++ Type | Value Range | Mathematical Role |
|:---:|:---:|:---:|:---:|:---|
| `[7:0]`   | Byte 0 (Least Significant) | `int8_t` | $-128 \dots +127$ | Element $k$ |
| `[15:8]`  | Byte 1                     | `int8_t` | $-128 \dots +127$ | Element $k+1$ |
| `[23:16]` | Byte 2                     | `int8_t` | $-128 \dots +127$ | Element $k+2$ |
| `[31:24]` | Byte 3 (Most Significant)  | `int8_t` | $-128 \dots +127$ | Element $k+3$ |

**Memory Consistency:** Standard little-endian byte ordering matches the software engine (`engine::matmul_int8`), where four consecutive memory bytes loaded with `LW` (Load Word) map directly into Byte 0, Byte 1, Byte 2, and Byte 3.

---

## 4. Mathematical Operation

Given:
- Vector $A \in \mathbb{Z}^4$ where $a_i = \text{rs1\_data}[8i+7 : 8i]$
- Vector $B \in \mathbb{Z}^4$ where $b_i = \text{rs2\_data}[8i+7 : 8i]$
- Accumulator base $C \in \mathbb{Z}^{32}$:
  $$\text{effective\_acc} = \begin{cases} 0 & \text{if } \text{clr\_acc} = 1 \\ \text{acc\_in} & \text{if } \text{clr\_acc} = 0 \end{cases}$$

The module computes:
$$\text{mac\_out} = \text{effective\_acc} + \sum_{i=0}^{3} (a_i \times b_i)$$

### Dynamic Range Analysis:
1. **Multiplication Stage:**
   - Maximum positive product: $-128 \times -128 = +16,384$ ($0\text{x}4000$)
   - Minimum negative product: $-128 \times +127 = -16,256$ ($0\text{x}C080$)
   - Fits completely in 16-bit signed integer (range $[-32,768, +32,767]$).
2. **Reduction Adder Stage:**
   - Maximum sum: $4 \times 16,384 = +65,536$ ($0\text{x}10000$)
   - Minimum sum: $4 \times (-16,256) = -65,024$
   - Fits completely in 18-bit signed integer (range $[-131,072, +131,071]$).
   - Zero intermediate overflow occurs prior to 32-bit accumulation.

---

## 5. Signal Interface Table

| Signal Name | Direction | Width | Description |
|:---|:---:|:---:|:---|
| `clk` | Input | 1 | System clock (target: 50–100 MHz) |
| `rst_n` | Input | 1 | Active-low synchronous reset |
| `enable` | Input | 1 | Clock enable / execution gate |
| `rs1_data` | Input | 32 | First 4-byte packed signed vector operand |
| `rs2_data` | Input | 32 | Second 4-byte packed signed vector operand |
| `acc_in` | Input | 32 | 32-bit accumulator base operand from `rd` |
| `clr_acc` | Input | 1 | Clear flag: $1 \implies$ initialize sum from 0, $0 \implies$ accumulate to `acc_in` |
| `mac_out` | Output | 32 | 32-bit accumulated signed result |
| `valid_out` | Output | 1 | Valid output strobe (asserted 1 cycle after enable) |
| `overflow_flag` | Output | 1 | High if 32-bit accumulation experienced signed wrap-around |

---

## 6. RISC-V Custom-0 Instruction Encoding

The module interfaces to the CPU pipeline as an R-type instruction in the standard unreserved `custom-0` space:

```
  31         25 24     20 19     15 14  12 11      7 6            0
 ┌─────────────┬─────────┬─────────┬──────┬─────────┬──────────────┐
 │   funct7    │   rs2   │   rs1   │funct3│   rd    │    opcode    │
 └─────────────┴─────────┴─────────┴──────┴─────────┴──────────────┘
     7 bits      5 bits    5 bits   3 bits  5 bits       7 bits
```

| Mnemonic | Opcode | funct3 | funct7 | Operation | Description |
|:---|:---:|:---:|:---:|:---|:---|
| `simd_mac rd, rs1, rs2` | `0001011` (`0x0B`) | `000` | `0000000` | $rd \leftarrow rd + \sum_{i=0}^3 a_i b_i$ | Accumulate with previous $rd$ |
| `simd_mac_clr rd, rs1, rs2` | `0001011` (`0x0B`) | `000` | `0000001` | $rd \leftarrow 0 + \sum_{i=0}^3 a_i b_i$ | Clear accumulator & start new dot product |

---

## 7. Pipeline & Timing Characteristics

- **Latency:** 1 clock cycle (registered output).
- **Throughput:** 1 operation / clock cycle (fully pipelined, Initiation Interval = 1).
- **Critical Path:**
  $T_{\text{comb}} = T_{\text{slice}} + T_{8\times8\text{-mult}} + 2 \times T_{\text{adder}} + T_{\text{acc32}} + T_{\text{setup}}$.
  Easily meets 100 MHz timing constraints on modern 28nm/40nm ASIC and 50–80 MHz on Xilinx 7-series FPGAs.

---

*Specification Version: 1.0 — Approved for Phase 03 RTL Implementation*
