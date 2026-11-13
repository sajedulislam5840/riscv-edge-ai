#!/usr/bin/env bash
# Simulate PicoRV32 + the SIMD MAC unit (hardware/soc/) in RTL and run a program on it.
#   hardware/scripts/soc_sim.sh isa     custom-instruction self-test (seconds)
#   hardware/scripts/soc_sim.sh mnist   full MNIST inference, scalar and with the MAC (about a minute)
# Requires: yowasp-yosys (or yosys), g++, and the RISC-V toolchain on PATH (firmware/setup_toolchain.sh).
# Optional: WAIT_STATES=N adds N wait states to every memory access.
set -euo pipefail
cd "$(dirname "$0")/../.."
WHAT=${1:-isa}
YOSYS=${YOSYS:-yowasp-yosys}
BUILD=build/soc
mkdir -p "$BUILD"

"$YOSYS" -q -p "read_verilog hardware/third_party/picorv32/picorv32.v hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v hardware/soc/pcpi_simd_mac.v hardware/soc/pico_mac_soc.v; hierarchy -top pico_mac_soc; proc; flatten; write_cxxrtl $BUILD/soc_cxxrtl.cc"
CXXRTL_INC="$(yosys-config --datdir 2>/dev/null || true)/include/backends/cxxrtl/runtime"
if [ ! -d "$CXXRTL_INC" ]; then
  CXXRTL_INC=$(python3 -c "import yowasp_yosys,os;print(os.path.join(os.path.dirname(yowasp_yosys.__file__),'share/include/backends/cxxrtl/runtime'))")
fi
g++ -std=c++14 -O2 -I"$CXXRTL_INC" -I"$BUILD" hardware/tb/cxxrtl_soc.cpp -o "$BUILD/cxxrtl_soc"

python3 firmware/gen_model_header.py >/dev/null
make -s -C firmware pico
case "$WHAT" in
  isa)   "$BUILD/cxxrtl_soc" firmware/build/pico_isa_test.bin 200000000 "${WAIT_STATES:-0}" ;;
  mnist) "$BUILD/cxxrtl_soc" firmware/build/pico_mnist.bin 4000000000 "${WAIT_STATES:-0}" ;;
  *) echo "usage: $0 isa|mnist"; exit 2 ;;
esac
