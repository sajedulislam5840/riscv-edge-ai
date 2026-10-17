# Bare-Metal Neural Network Inference Engine on RISC-V Architecture
### A Hardware-Software Co-Design for Ultra-Low-Power Edge Silicon

<p align="left">
  <img src="https://img.shields.io/badge/C%2B%2B-17_Standard-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/ISA-RISC--V_RV32I%2FM-d33682?style=for-the-badge" alt="RISC-V"/>
  <img src="https://img.shields.io/badge/Hardware-Verilog_RTL-8B0000?style=for-the-badge" alt="Verilog"/>
  <img src="https://img.shields.io/badge/Quantization-Affine_INT8-268bd2?style=for-the-badge" alt="INT8"/>
  <img src="https://img.shields.io/badge/Design-HW%2FSW_Codesign-purple?style=for-the-badge" alt="Co-Design"/>
  <img src="https://img.shields.io/badge/Status-Active_Research-2ea44f?style=for-the-badge" alt="Status"/>
</p>

An academic-grade, freestanding C++ inference runtime and custom Verilog RTL accelerator engineered from first principles for cycle-accurate neural network execution on **RISC-V (RV32I/M)** microarchitectures.

This research project explores the boundary between systems engineering, microarchitecture, and embedded deep learning: eliminating floating-point unit (FPU) dependencies, resolving register bandwidth underutilization in scalar integer math, and measuring real execution cycles on constrained silicon.

---

## ⚡ Execution Pipeline Architecture

<p align="center">
  <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MDAgMjQwIiB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIyNDAiPgogIDxkZWZzPgogICAgPHN0eWxlPgogICAgICAuYmcgeyBmaWxsOiAjMGQxMTE3OyB9CiAgICAgIC5ib3ggeyBmaWxsOiAjMTYxYjIyOyBzdHJva2U6ICMzMDM2M2Q7IHN0cm9rZS13aWR0aDogMjsgcng6IDg7IH0KICAgICAgLmJveC1hY3RpdmUgeyBmaWxsOiAjMTYxYjIyOyBzdHJva2U6ICM1OGE2ZmY7IHN0cm9rZS13aWR0aDogMjsgcng6IDg7IH0KICAgICAgLnRpdGxlLXRleHQgeyBmb250LWZhbWlseTogLWFwcGxlLXN5c3RlbSwgQmxpbmtNYWNTeXN0ZW1Gb250LCAiU2Vnb2UgVUkiLCBIZWx2ZXRpY2EsIEFyaWFsLCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEzcHg7IGZvbnQtd2VpZ2h0OiA2MDA7IGZpbGw6ICM1OGE2ZmY7IH0KICAgICAgLmJvZHktdGV4dCB7IGZvbnQtZmFtaWx5OiAtYXBwbGUtc3lzdGVtLCBCbGlua01hY1N5c3RlbUZvbnQsICJTZWdvZSBVSSIsIEhlbHZldGljYSwgQXJpYWwsIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTFweDsgZmlsbDogIzhiOTQ5ZTsgfQogICAgICAuYXJyb3cgeyBzdHJva2U6ICM4Yjk0OWU7IHN0cm9rZS13aWR0aDogMjsgbWFya2VyLWVuZDogdXJsKCNhcnJvd2hlYWQpOyB9CiAgICA8L3N0eWxlPgogICAgPG1hcmtlciBpZD0iYXJyb3doZWFkIiBtYXJrZXJXaWR0aD0iOCIgbWFya2VySGVpZ2h0PSI2IiByZWZYPSI3IiByZWZZPSIzIiBvcmllbnQ9ImF1dG8iPgogICAgICA8cG9seWdvbiBwb2ludHM9IjAgMCwgOCAzLCAwIDYiIGZpbGw9IiM4Yjk0OWUiIC8+CiAgICA8L21hcmtlcj4KICA8L2RlZnM+CgogIDxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGNsYXNzPSJiZyIgLz4KCiAgPCEtLSBTdGFnZSAxIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLCA0MCkiPgogICAgPHJlY3Qgd2lkdGg9IjE4MCIgaGVpZ2h0PSIxNTAiIGNsYXNzPSJib3giIC8+CiAgICA8dGV4dCB4PSIyMCIgeT0iMjYiIGNsYXNzPSJ0aXRsZS10ZXh0Ij5TdGFnZSAxOiBGUDMyIEJhc2VsaW5lPC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9IjU1IiBjbGFzcz0iYm9keS10ZXh0Ij7igKIgQ29udGludW91cyAxRCBCdWZmZXI8L3RleHQ+CiAgICA8dGV4dCB4PSIyMCIgeT0iNzUiIGNsYXNzPSJib2R5LXRleHQiPuKAoiBOYWl2ZSBHRU1NIEtlcm5lbDwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSI5NSIgY2xhc3M9ImJvZHktdGV4dCI+4oCiIEVtdWxhdGVkIEZQVSBCYXNlbGluZTwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSIxMjUiIGNsYXNzPSJib2R5LXRleHQiIGZpbGw9IiNmODUxNDkiPltIaWdoIEN5Y2xlIE92ZXJoZWFkXTwvdGV4dD4KICA8L2c+CgogIDwhLS0gQXJyb3cgMSAtPiAyIC0tPgogIDxsaW5lIHgxPSIyMTAiIHkxPSIxMTUiIHgyPSIyNTAiIHkyPSIxMTUiIGNsYXNzPSJhcnJvdyIgLz4KCiAgPCEtLSBTdGFnZSAyIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDI1MCwgNDApIj4KICAgIDxyZWN0IHdpZHRoPSIxODAiIGhlaWdodD0iMTUwIiBjbGFzcz0iYm94LWFjdGl2ZSIgLz4KICAgIDx0ZXh0IHg9IjI0IiB5PSIyNiIgY2xhc3M9InRpdGxlLXRleHQiPlN0YWdlIDI6IFF1YW50aXplPC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9IjU1IiBjbGFzcz0iYm9keS10ZXh0Ij7igKIgQWZmaW5lIEZQMzIg4oaSIElOVDg8L3RleHQ+CiAgICA8dGV4dCB4PSIyMCIgeT0iNzUiIGNsYXNzPSJib2R5LXRleHQiPuKAoiA3NSUgRm9vdHByaW50IFJlZHVjdGlvbjwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSI5NSIgY2xhc3M9ImJvZHktdGV4dCI+4oCiIDMyLWJpdCBBY2N1bXVsYXRvcjwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSIxMjUiIGNsYXNzPSJib2R5LXRleHQiIGZpbGw9IiMzZmI5NTAiPltNZW1vcnkgQ29tcHJlc3NlZF08L3RleHQ+CiAgPC9nPgoKICA8IS0tIEFycm93IDIgLT4gMyAtLT4KICA8bGluZSB4MT0iNDMwIiB5MT0iMTE1IiB4Mj0iNDcwIiB5Mj0iMTE1IiBjbGFzcz0iYXJyb3ciIC8+CgogIDwhLS0gU3RhZ2UgMyAtLT4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0NzAsIDQwKSI+CiAgICA8cmVjdCB3aWR0aD0iMTgwIiBoZWlnaHQ9IjE1MCIgY2xhc3M9ImJveCIgLz4KICAgIDx0ZXh0IHg9IjE4IiB5PSIyNiIgY2xhc3M9InRpdGxlLXRleHQiPlN0YWdlIDM6IDQtV2F5IFNJTUQgUlRMPC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9IjU1IiBjbGFzcz0iYm9keS10ZXh0Ij7igKIgUmVnaXN0ZXIgUGFja2luZyAoMzJiKTwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSI3NSIgY2xhc3M9ImJvZHktdGV4dCI+4oCiIEN1c3RvbS0wIElTQSBJbnN0cnVjdGlvbjwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSI5NSIgY2xhc3M9ImJvZHktdGV4dCI+4oCiIDEtQ3ljbGUgUGFyYWxsZWwgTUFDPC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9IjEyNSIgY2xhc3M9ImJvZHktdGV4dCIgZmlsbD0iIzU4YTZmZiI+WzR4IENvbXB1dGUgRGVuc2l0eV08L3RleHQ+CiAgPC9nPgoKICA8IS0tIEFycm93IDMgLT4gNCAtLT4KICA8bGluZSB4MT0iNjUwIiB5MT0iMTE1IiB4Mj0iNjkwIiB5Mj0iMTE1IiBjbGFzcz0iYXJyb3ciIC8+CgogIDwhLS0gU3RhZ2UgNCAtLT4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2OTAsIDQwKSI+CiAgICA8cmVjdCB3aWR0aD0iMTgwIiBoZWlnaHQ9IjE1MCIgY2xhc3M9ImJveCIgLz4KICAgIDx0ZXh0IHg9IjIwIiB5PSIyNiIgY2xhc3M9InRpdGxlLXRleHQiPlN0YWdlIDQ6IEN5Y2xlIEF1ZGl0PC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9IjU1IiBjbGFzcz0iYm9keS10ZXh0Ij7igKIgQmFyZS1NZXRhbCBFeGVjdXRpb248L3RleHQ+CiAgICA8dGV4dCB4PSIyMCIgeT0iNzUiIGNsYXNzPSJib2R5LXRleHQiPuKAoiAzLVdheSBDeWNsZSBDb21wYXJpc29uPC90ZXh0PgogICAgPHRleHQgeD0iMjAiIHk9Ijk1IiBjbGFzcz0iYm9keS10ZXh0Ij7igKIgTGF0ZW5jeSAmYW1wOyBJUEMgTWV0cmljczwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwIiB5PSIxMjUiIGNsYXNzPSJib2R5LXRleHQiIGZpbGw9IiNkMjk5MjIiPltWZXJpZmllZCBTcGVlZHVwXTwvdGV4dD4KICA8L2c+Cjwvc3ZnPg==" alt="Architecture Pipeline" width="100%"/>
</p>
---

## 🔬 Core Architectural Pillars

### 1. Zero-Allocation Bare-Metal Memory Fabric (Software)
- **Contiguous Row-Major Buffering:** Eliminates nested pointer indirection (`vector<vector<T>>`), minimizing cache evictions and maximizing CPU spatial locality.
- **Freestanding Core:** Strictly freestanding runtime (`-nostdlib`, zero dynamic heap/`malloc`) to emulate deeply embedded micro-controller arena constraints.

### 2. Affine Fixed-Point INT8 Quantization (Software)
- **Mathematical Mapping:** Projects real-value activations and weights into 8-bit signed space:
  $$q = \text{clamp}\left(\left\lfloor \frac{r}{S} \right\rceil + Z, -128, 127\right)$$
- **Accumulator Guard:** Integrates wide 32-bit accumulators during matrix inner products to mathematically prevent numerical overflow before fixed-point rescale down-shifting.

### 3. Custom 4-Way SIMD Dot-Product Accelerator (Hardware RTL)
- **The Microarchitectural Bottleneck:** On standard 32-bit RISC-V cores, scalar INT8 arithmetic leaves the upper 24 bits of registers idle.
- **Dedicated RTL Processing Element:** A single-cycle Verilog MAC unit decoding a custom instruction (`custom-0`). It unpacks two 32-bit registers containing four signed 8-bit values each ($A_0..A_3, B_0..B_3$) and computes:
  $$\text{Result} = \sum_{i=0}^{3} (A_i \times B_i) + \text{Accumulator}$$
  featuring signed sign-extension and zero-overhead saturation protection.

### 4. Cycle-Accurate Microarchitectural Simulation & Profiling
- **Three-Way Performance Evaluation:** Granular auditing comparing **FP32 Software Scalar** vs. **INT8 Software Scalar** vs. **INT8 Hardware-Accelerated** execution cycles, memory bus traffic, and instruction mix.

---

## 📁 Repository Anatomy

```text
riscv-edge-ai/
├── docs/
│   ├── assets/                 # SVG schematics and visual flowcharts
│   └── architecture_spec.md    # ISA encoding and RTL datapath specification
│   ├── include/
│   │   └── engine/             # Contiguous tensor abstractions and operators
│   └── src/                    # Baseline kernels and custom-0 inline assembly
├── hardware/                   # Verilog RTL and verification testbenches
│   ├── rtl/                    # 4-way SIMD MAC PE and ALU wrapper
│   └── tb/                     # Self-checking testbenches and golden vectors
├── tests/                      # Unit tests and assertion checks
├── CMakeLists.txt              # Root build configuration
└── main.cpp                    # Host validation harness and benchmark entry
```

---

## 📊 40-Day Research & Engineering Roadmap

| Phase | Milestone Description | Target Timeline | Status |
| :--- | :--- | :--- | :--- |
| **Phase 01** | **Software Baseline:** Modular CMake harness, row-major tensor buffer, naive FP32 GEMM kernel | Days 01–07 | **`In Progress`** ⏳ |
| **Phase 02** | **Software Quantization:** Affine FP32 → INT8 mapping, 32-bit accumulator, divergence bounds | Days 08–14 | `Planned` 📅 |
| **Phase 03** | **Hardware RTL Co-Design:** 4-way SIMD MAC PE in Verilog, sign-extension tree, custom-0 ISA mapping | Days 15–23 | `Planned` 📅 |
| **Phase 04** | **Bare-Metal Integration:** Freestanding runtime, ELF binary loader, three-way cycle-accurate profiling | Days 24–33 | `Planned` 📅 |
| **Phase 05** | **Academic Artifacts:** IEEE-format technical report, LaTeX proofs, reproducible release | Days 34–40 | `Planned` 📅 |

---

## 🛠️ Verification & Build Instructions

Compile and run the test suite locally or on Google Colab with standard C++17 tooling:

```bash
# 1. Clone the repository if you have not already cloned it
git clone https://github.com/sajedulislam5840/riscv-edge-ai.git
cd riscv-edge-ai

# 2. Build via CMake
mkdir -p build
cd build
cmake ..
cmake --build . --parallel

# 3. Execute the host sanity runner
./engine_runner

# 4. Run the automated test suite
ctest --output-on-failure
```

---

## 🔧 Development Notes

- The host validation harness is intended for functional testing and benchmarking.
- Actual bare-metal execution requires a compatible RISC-V toolchain, target runtime, and simulator or physical hardware.
- Hardware acceleration results should be validated against a software reference implementation.
- Cycle counts must be measured on a defined simulation or hardware target; host wall-clock timings are not equivalent to RISC-V cycle counts.

---

## 📜 Research Scope

This project aims to demonstrate the design and evaluation of a lightweight neural network inference pipeline through software optimization, integer quantization, custom hardware datapaths, and reproducible performance analysis.

The final evaluation should report correctness, numerical deviation, instruction count, execution cycles, and resource utilization where the relevant tools and hardware targets are available.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
