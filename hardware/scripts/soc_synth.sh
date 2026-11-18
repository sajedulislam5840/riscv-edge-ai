#!/usr/bin/env bash
# Area and clock speed of PicoRV32 with and without the MAC unit, and of the unit alone.
#   ECP5 (LFE5U-25F): synthesis + place-and-route, three placement seeds.
#   iCE40 (UP5K style, with SB_MAC16): synthesis only. The core needs more pins than the UP5K
#   package has, and the HX8K has no DSP blocks, so there is no meaningful place-and-route.
# Requires: yowasp-yosys and yowasp-nextpnr-ecp5 (pip install yowasp-yosys yowasp-nextpnr-ecp5).
set -euo pipefail
cd "$(dirname "$0")/../.."
YOSYS=${YOSYS:-yowasp-yosys}
NEXTPNR=${NEXTPNR:-yowasp-nextpnr-ecp5}
OUT=build/synth
mkdir -p "$OUT"
CORE="hardware/third_party/picorv32/picorv32.v hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v hardware/soc/pcpi_simd_mac.v hardware/soc/pico_mac_soc.v"
UNIT="hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v hardware/soc/pcpi_simd_mac.v"

count() {  # count <log> <cell-regex>: total of the final statistics block
  awk '/Printing statistics/{p=1} p' "$1" | grep -E "^\s+[0-9]+\s+($2)\s*$" | awk '{a[$2]=$1} END{for(k in a) s+=a[k]; print s+0}'
}
report() {  # report <label> <log> <lut-regex> <ff-regex> <dsp-regex>
  printf "%-28s LUT4 %5d   FF %4d   DSP %d\n" "$1" "$(count "$2" "$3")" "$(count "$2" "$4")" "$(count "$2" "$5")"
}

echo "== ECP5 =="
for w in 0 1; do
  "$YOSYS" -q -l "$OUT/ecp5_w$w.log" -p "read_verilog $CORE; chparam -set WITH_MAC $w pico_mac_soc; synth_ecp5 -top pico_mac_soc -json $OUT/ecp5_w$w.json"
  label="core, no MAC unit"; [ "$w" = 1 ] && label="core + MAC unit"
  report "$label" "$OUT/ecp5_w$w.log" "LUT4" "TRELLIS_FF" "MULT18X18D"
  for seed in 1 2 3; do
    f=$("$NEXTPNR" --25k --package CABGA256 --json "$OUT/ecp5_w$w.json" --freq 40 --seed $seed --lpf-allow-unconstrained 2>&1 \
        | grep "Max frequency for clock" | tail -1 | sed -E 's/.*: ([0-9.]+) MHz.*/\1/')
    echo "    seed $seed: $f MHz"
  done
done
"$YOSYS" -q -l "$OUT/unit_ecp5.log" -p "read_verilog $UNIT; synth_ecp5 -top pcpi_simd_mac"
report "MAC unit alone" "$OUT/unit_ecp5.log" "LUT4" "TRELLIS_FF" "MULT18X18D"

echo "== iCE40 (synthesis only) =="
for w in 0 1; do
  "$YOSYS" -q -l "$OUT/ice40_w$w.log" -p "read_verilog $CORE; chparam -set WITH_MAC $w pico_mac_soc; synth_ice40 -dsp -top pico_mac_soc"
  label="core, no MAC unit"; [ "$w" = 1 ] && label="core + MAC unit"
  report "$label" "$OUT/ice40_w$w.log" "SB_LUT4" "SB_DFF\w*" "SB_MAC16"
done
"$YOSYS" -q -l "$OUT/unit_ice40.log" -p "read_verilog $UNIT; synth_ice40 -dsp -top pcpi_simd_mac"
report "MAC unit alone" "$OUT/unit_ice40.log" "SB_LUT4" "SB_DFF\w*" "SB_MAC16"
