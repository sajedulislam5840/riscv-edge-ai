#!/usr/bin/env bash
# Compile the RTL to C++ with Yosys CXXRTL and run all golden vectors through it:
# first the bare MAC unit (simd_mac.v), then the custom-0 decoder wrapper (simd_mac_wrapper.v).
# Requires: yosys (or `pip install yowasp-yosys`, then YOSYS=yowasp-yosys), g++.
set -euo pipefail
cd "$(dirname "$0")/../.."
YOSYS=${YOSYS:-yosys}
BUILD=build/cosim
mkdir -p "$BUILD"
"$YOSYS" -q -p "read_verilog hardware/rtl/simd_mac.v; hierarchy -top simd_mac; proc; flatten; write_cxxrtl $BUILD/simd_mac_cxxrtl.cc"
CXXRTL_INC="$(yosys-config --datdir 2>/dev/null || true)/include/backends/cxxrtl/runtime"
if [ ! -d "$CXXRTL_INC" ]; then
  CXXRTL_INC=$(python3 -c "import yowasp_yosys,os;print(os.path.join(os.path.dirname(yowasp_yosys.__file__),'share/include/backends/cxxrtl/runtime'))")
fi
g++ -std=c++14 -O1 -I"$CXXRTL_INC" -I"$BUILD" hardware/tb/cxxrtl_cosim.cpp -o "$BUILD/cxxrtl_cosim"
"$BUILD/cxxrtl_cosim" hardware/tb/golden_vectors.hex

"$YOSYS" -q -p "read_verilog hardware/rtl/simd_mac.v hardware/rtl/simd_mac_wrapper.v; hierarchy -top simd_mac_wrapper; proc; flatten; write_cxxrtl $BUILD/simd_mac_wrapper_cxxrtl.cc"
g++ -std=c++14 -O1 -I"$CXXRTL_INC" -I"$BUILD" hardware/tb/cxxrtl_cosim_wrapper.cpp -o "$BUILD/cxxrtl_cosim_wrapper"
"$BUILD/cxxrtl_cosim_wrapper" hardware/tb/golden_vectors.hex
