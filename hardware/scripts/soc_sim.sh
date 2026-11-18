#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
WHAT=${1:-isa}
YOSYS=${YOSYS:-yowasp-yosys}
WITH_MAC=${WITH_MAC:-1}
BUILD=build/soc
[ "$WITH_MAC" = 1 ] || BUILD=build/soc_nomac
mkdir -p "$BUILD"

"$YOSYS" -q -p "read_verilog hardware/third_party/picorv32/picorv32.v hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v hardware/soc/pcpi_simd_mac.v hardware/soc/pico_mac_soc.v; chparam -set WITH_MAC $WITH_MAC pico_mac_soc; hierarchy -top pico_mac_soc; proc; flatten; write_cxxrtl $BUILD/soc_cxxrtl.cc"
CXXRTL_INC="$(yosys-config --datdir 2>/dev/null || true)/include/backends/cxxrtl/runtime"
if [ ! -d "$CXXRTL_INC" ]; then
  CXXRTL_INC=$(python3 -c "import yowasp_yosys,os;print(os.path.join(os.path.dirname(yowasp_yosys.__file__),'share/include/backends/cxxrtl/runtime'))")
fi
g++ -std=c++14 -O2 -I"$CXXRTL_INC" -I"$BUILD" hardware/tb/cxxrtl_soc.cpp -o "$BUILD/cxxrtl_soc"

python3 firmware/gen_model_header.py >/dev/null
make -s -C firmware pico
case "$WHAT" in
  build) ;;
  isa)   "$BUILD/cxxrtl_soc" firmware/build/pico_isa_test.bin 200000000 "${WAIT_STATES:-0}" ;;
  mnist) "$BUILD/cxxrtl_soc" firmware/build/pico_mnist.bin 4000000000 "${WAIT_STATES:-0}" ;;
  *) echo "usage: $0 isa|mnist|build"; exit 2 ;;
esac
