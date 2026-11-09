#!/usr/bin/env bash
# Synthesize simd_mac.v with Yosys for Xilinx 7-series and Lattice iCE40 and print resource usage.
set -euo pipefail
cd "$(dirname "$0")/../rtl"
YOSYS=${YOSYS:-yosys}
echo "== Xilinx 7-series (synth_xilinx) =="
"$YOSYS" -q -p "read_verilog simd_mac.v; synth_xilinx -top simd_mac -family xc7; tee -o synth_xc7.txt stat" >/dev/null
grep -E "cells|DSP48E1|LUT|FDRE|CARRY4" synth_xc7.txt
echo "== Lattice iCE40 (synth_ice40 -dsp) =="
"$YOSYS" -q -p "read_verilog simd_mac.v; synth_ice40 -top simd_mac -dsp; tee -o synth_ice40.txt stat" >/dev/null
grep -E "cells|SB_" synth_ice40.txt
