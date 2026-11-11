#!/usr/bin/env bash
# Event-driven simulation of the self-checking Verilog testbench with Icarus Verilog.
# The main RTL check is hardware/scripts/cosim.sh (all golden vectors); this one runs the
# hand-written corner cases in hardware/tb/tb_simd_mac.v.
#   bash hardware/scripts/run_sim.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! command -v iverilog >/dev/null; then
    echo "iverilog not found (sudo apt install iverilog)" >&2
    exit 1
fi
mkdir -p build
iverilog -g2001 -Wall -o build/tb_simd_mac.vvp hardware/rtl/simd_mac.v hardware/tb/tb_simd_mac.v
vvp build/tb_simd_mac.vvp
