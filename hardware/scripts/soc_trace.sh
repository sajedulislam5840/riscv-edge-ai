#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
hardware/scripts/soc_sim.sh build >/dev/null
read -r START END < <(python3 scripts/make_trace.py --find-pcs firmware/build/pico_mnist.elf)
HIDDEN=$(sed -nE 's/#define HIDDEN_SIZE ([0-9]+)/\1/p' firmware/build/model_data.h)
mkdir -p build
BIN=firmware/build/pico_mnist.bin
build/soc/cxxrtl_soc "$BIN" 4000000000 0 --examples 1 --trace build/trace_scalar.tsv --trace-start "$START" --trace-end "$END" --trace-skip 0 >/dev/null
build/soc/cxxrtl_soc "$BIN" 4000000000 0 --examples 1 --trace build/trace_mac.tsv --trace-start "$START" --trace-end "$END" --trace-skip "$HIDDEN" >/dev/null
python3 scripts/make_trace.py build/trace_scalar.tsv build/trace_mac.tsv > web/js/trace.js
echo "wrote web/js/trace.js (neuron loop $START to $END)"
