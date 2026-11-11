#!/usr/bin/env bash
# Place and route the MAC unit and report its post-route maximum clock frequency on two
# open-toolchain FPGAs, Lattice iCE40 UP5K and Lattice ECP5-25k, for two designs:
#   unit     simd_mac alone            (hardware/fpga/simd_mac_fmax_top.v)
#   wrapper  decoder + unit + feedback (hardware/fpga/simd_mac_wrapper_fmax_top.v)
# Needs yosys + nextpnr, or: pip install yowasp-yosys yowasp-nextpnr-ice40 yowasp-nextpnr-ecp5
#   YOSYS=yowasp-yosys NEXTPNR_PREFIX=yowasp- bash hardware/scripts/fmax.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
YOSYS=${YOSYS:-yosys}
P=${NEXTPNR_PREFIX:-}
OUT=build/fmax
SEEDS=${SEEDS:-"1 2 3"}
mkdir -p "$OUT"
cp hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v hardware/fpga/*_fmax_top.v "$OUT"/
cd "$OUT"

report() {  # $1 = label, rest = nextpnr command
    local label=$1; shift
    for s in $SEEDS; do
        local log="pnr_${label// /_}_$s.log"
        "$@" --seed "$s" --timing-allow-fail 2> "$log" >/dev/null
        f=$(grep "Max frequency for clock" "$log" | tail -1 | sed -E 's/.*: ([0-9.]+) MHz.*/\1/')
        echo "$label seed $s: $f MHz"
    done
}
for design in unit wrapper; do
    if [ "$design" = unit ]; then top=simd_mac_fmax_top; else top=simd_mac_wrapper_fmax_top; fi
    srcs="simd_mac.v simd_mac_wrapper.v $top.v"
    "$YOSYS" -q -p "read_verilog $srcs; synth_ice40 -top $top -dsp -json ice40_$design.json"
    "$YOSYS" -q -p "read_verilog $srcs; synth_ecp5 -top $top -json ecp5_$design.json"
    report "$design ice40-up5k" "${P}nextpnr-ice40" --up5k --package sg48 --json "ice40_$design.json" --freq 100 --pcf-allow-unconstrained
    report "$design ecp5-25k"   "${P}nextpnr-ecp5"  --25k --package CABGA256 --json "ecp5_$design.json" --freq 200
done
