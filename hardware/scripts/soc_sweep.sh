#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
hardware/scripts/soc_sim.sh build >/dev/null
WITH_MAC=0 hardware/scripts/soc_sim.sh build >/dev/null
BIN=firmware/build/pico_mnist.bin

echo "core without the MAC unit, scalar code only, 0 wait states:"
build/soc_nomac/cxxrtl_soc "$BIN" 4000000000 0 --examples 1 --scalar-only | grep "one inference"
echo
echo "core with the MAC unit:"
echo "wait_states scalar_cycles mac_clr_add_cycles mac_internal_cycles speedup_clr_add speedup_internal"
for w in 0 1 2 4 8 16; do
  line=$(build/soc/cxxrtl_soc "$BIN" 4000000000 "$w" --examples 1 | grep "cycles per inference")
  echo "$line" | awk -v w="$w" '{
    gsub(/[(),]/, " "); s=$5; c=$9; i=$(NF-5);
    n=split($0, f, " "); for (k=1;k<=n;k++) { if (f[k]=="add") c=f[k+1]; if (f[k]=="simd_macz/simd_macc") i=f[k+1]; if (f[k]=="scalar") s=f[k+1] }
    printf "%d %d %d %d %.2f %.2f\n", w, s, c, i, s/c, s/i }'
done
