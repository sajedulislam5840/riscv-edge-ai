#!/usr/bin/env bash
# Place and route simd_mac (inside hardware/fpga/simd_mac_fmax_top.v) and report its post-route
# maximum clock frequency on two open-toolchain FPGAs: Lattice iCE40 UP5K and Lattice ECP5-25k.
# Needs yosys + nextpnr, or: pip install yowasp-yosys yowasp-nextpnr-ice40 yowasp-nextpnr-ecp5
#   YOSYS=yowasp-yosys NEXTPNR_PREFIX=yowasp- bash hardware/scripts/fmax.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
YOSYS=${YOSYS:-yosys}
P=${NEXTPNR_PREFIX:-}
OUT=build/fmax
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p "$OUT"
cp hardware/rtl/simd_mac.v hardware/fpga/simd_mac_fmax_top.v "$OUT"/
cd "$OUT"

"$YOSYS" -q -p "read_verilog simd_mac.v simd_mac_fmax_top.v; synth_ice40 -top simd_mac_fmax_top -dsp -json ice40.json"
"$YOSYS" -q -p "read_verilog simd_mac.v simd_mac_fmax_top.v; synth_ecp5 -top simd_mac_fmax_top -json ecp5.json"

report() {  # $1 = label, rest = nextpnr command
    local label=$1; shift
    for s in $SEEDS; do
        "$@" --seed "$s" --timing-allow-fail 2> "pnr_${label}_$s.log" >/dev/null
        f=$(grep "Max frequency for clock" "pnr_${label}_$s.log" | tail -1 | sed -E 's/.*: ([0-9.]+) MHz.*/\1/')
        echo "$label seed $s: $f MHz"
    done
}
report ice40-up5k "${P}nextpnr-ice40" --up5k --package sg48 --json ice40.json --freq 100 --pcf-allow-unconstrained
report ecp5-25k   "${P}nextpnr-ecp5"  --25k --package CABGA256 --json ecp5.json --freq 200
