#!/usr/bin/env bash
# Install the RISC-V toolchain used by the firmware: the xPack riscv-none-elf-gcc cross compiler
# and the Spike instruction-set simulator (built from source). Linux or WSL on Windows.
#   bash firmware/setup_toolchain.sh            installs into ~/.riscv-edge-ai-tools
#   RV_PREFIX=/some/dir bash firmware/setup_toolchain.sh
# Then: export PATH="$HOME/.riscv-edge-ai-tools/bin:$PATH"
set -euo pipefail
PREFIX=${RV_PREFIX:-$HOME/.riscv-edge-ai-tools}
GCC_VER=14.2.0-3
SPIKE_TAG=v1.1.0
JOBS=$(nproc 2>/dev/null || echo 2)
mkdir -p "$PREFIX/bin" "$PREFIX/src"
export PATH="$PREFIX/bin:$PATH"
cd "$PREFIX/src"

if ! command -v riscv-none-elf-gcc >/dev/null; then
  echo "== riscv-none-elf-gcc $GCC_VER"
  curl -sSL -o gcc.tgz "https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v$GCC_VER/xpack-riscv-none-elf-gcc-$GCC_VER-linux-x64.tar.gz"
  tar xzf gcc.tgz && rm gcc.tgz
  for f in "xpack-riscv-none-elf-gcc-$GCC_VER"/bin/*; do ln -sf "$PWD/$f" "$PREFIX/bin/"; done
fi

# Spike's build needs dtc (device-tree compiler); dtc's build needs flex.
if ! command -v dtc >/dev/null; then
  if ! command -v flex >/dev/null; then
    echo "== flex"
    curl -sSL https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz | tar xz
    (cd flex-2.6.4 && ./configure --prefix="$PREFIX" -q && make -j"$JOBS" -s && make install -s) >/dev/null
  fi
  echo "== dtc"
  rm -rf dtc && git clone -q --depth 1 --branch v1.7.2 https://github.com/dgibson/dtc.git dtc
  make -C dtc -j"$JOBS" -s NO_PYTHON=1 NO_YAML=1 NO_VALGRIND=1 dtc >/dev/null
  cp dtc/dtc "$PREFIX/bin/"
fi

if ! command -v spike >/dev/null; then
  echo "== spike $SPIKE_TAG (takes a few minutes)"
  rm -rf spike && git clone -q --depth 1 --branch "$SPIKE_TAG" https://github.com/riscv-software-src/riscv-isa-sim.git spike
  mkdir -p spike/build && cd spike/build
  # -include cstdint: the release predates newer compilers' header cleanups.
  ../configure --prefix="$PREFIX" CXXFLAGS="-O2 -include cstdint" >/dev/null
  make -j"$JOBS" >/dev/null && make install >/dev/null
fi

echo "Ready. Add to PATH:  export PATH=\"$PREFIX/bin:\$PATH\""
riscv-none-elf-gcc --version | head -1
command -v spike
