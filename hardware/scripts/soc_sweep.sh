#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
hardware/scripts/soc_sim.sh build >/dev/null
WITH_MAC=0 hardware/scripts/soc_sim.sh build >/dev/null
BIN=firmware/build/pico_mnist.bin

echo "core without the MAC unit, scalar code only, 0 wait states:"
build/soc_nomac/cxxrtl_soc "$BIN" 4000000000 0 --examples 1 --scalar-only | grep "one inference"
echo
echo "core with the MAC unit (cycles per inference):"
echo "wait_states scalar mac_clr_add mac_internal opt_scalar opt_mac speedup_opt_mac_vs_opt_scalar"
for w in 0 1 2 4 8 16; do
  out=$(build/soc/cxxrtl_soc "$BIN" 4000000000 "$w" --examples 1)
  naive=$(echo "$out" | sed -nE 's/^cycles per inference: scalar ([0-9]+),.*/\1/p')
  clr=$(echo "$out" | sed -nE 's/.*simd_mac_clr \+ add ([0-9]+) .*/\1/p' | tail -1)
  int=$(echo "$out" | sed -nE 's/.*simd_macz\/simd_macc ([0-9]+) \(.*/\1/p')
  os=$(echo "$out" | sed -nE 's/^optimised scalar.*, ([0-9]+) cycles/\1/p')
  om=$(echo "$out" | sed -nE 's/^optimised MAC.*, ([0-9]+) cycles/\1/p')
  awk -v w="$w" -v a="$naive" -v b="$clr" -v c="$int" -v d="$os" -v e="$om" 'BEGIN { printf "%d %d %d %d %d %d %.2f\n", w, a, b, c, d, e, d / e }'
done
