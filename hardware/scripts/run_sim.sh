#!/usr/bin/env bash
# =============================================================================
# run_sim.sh — SIMD MAC Simulation Runner (Linux / macOS / WSL)
# =============================================================================
# Usage:
#   cd <repo_root>
#   bash hardware/scripts/run_sim.sh
#
# Returns exit code 0 on all-tests-pass, 1 on failure.
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RTL_DIR="$REPO_ROOT/hardware/rtl"
TB_DIR="$REPO_ROOT/hardware/tb"

echo "============================================"
echo " SIMD MAC Simulation Runner (Bash)         "
echo "============================================"
echo "Repo root : $REPO_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Detect simulator
# ---------------------------------------------------------------------------
HAS_IVERILOG=$(command -v iverilog 2>/dev/null && echo "yes" || echo "no")
HAS_VERILATOR=$(command -v verilator 2>/dev/null && echo "yes" || echo "no")
HAS_GXX=$(command -v g++ 2>/dev/null && echo "yes" || echo "no")

echo "Tool detection:"
echo "  iverilog  : $HAS_IVERILOG"
echo "  verilator : $HAS_VERILATOR"
echo "  g++       : $HAS_GXX"
echo ""

# ---------------------------------------------------------------------------
# Run with iverilog if available (preferred)
# ---------------------------------------------------------------------------
if [ "$HAS_IVERILOG" = "yes" ]; then
    echo "--- Running via Icarus Verilog (iverilog) ---"
    OUT_VVP="/tmp/simd_mac_sim.vvp"
    iverilog -g2001 -Wall \
        -o "$OUT_VVP" \
        "$RTL_DIR/simd_mac.v" \
        "$TB_DIR/tb_simd_mac.v"
    vvp "$OUT_VVP"
    EXIT=$?
    rm -f "$OUT_VVP"
    if [ $EXIT -eq 0 ]; then
        echo ""
        echo "[SUCCESS] All tests passed (iverilog)."
    else
        echo ""
        echo "[FAILURE] Tests failed — check output above."
    fi
    exit $EXIT
fi

# ---------------------------------------------------------------------------
# Run with verilator if available
# ---------------------------------------------------------------------------
if [ "$HAS_VERILATOR" = "yes" ]; then
    echo "--- Running via Verilator ---"
    BUILD_DIR="/tmp/verilator_simd_mac"
    mkdir -p "$BUILD_DIR"
    verilator --cc --exe --build \
        --top-module tb_simd_mac \
        -Mdir "$BUILD_DIR" \
        "$RTL_DIR/simd_mac.v" \
        "$TB_DIR/tb_simd_mac.v"
    "$BUILD_DIR/Vtb_simd_mac"
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        echo ""
        echo "[SUCCESS] All tests passed (verilator)."
    else
        echo ""
        echo "[FAILURE] Tests failed."
    fi
    exit $EXIT
fi

# ---------------------------------------------------------------------------
# Fallback: C++ golden model
# ---------------------------------------------------------------------------
echo "--- Falling back to C++ golden model co-simulation ---"
CPP_SRC="$TB_DIR/sim_simd_mac.cpp"
CPP_EXE="/tmp/sim_simd_mac"

if [ ! -f "$CPP_SRC" ]; then
    echo "[ERROR] C++ co-sim source not found: $CPP_SRC"
    exit 1
fi

g++ -std=c++17 -Wall -Wextra -O2 -o "$CPP_EXE" "$CPP_SRC"
"$CPP_EXE"
EXIT=$?
rm -f "$CPP_EXE"

if [ $EXIT -eq 0 ]; then
    echo ""
    echo "[SUCCESS] All tests passed (C++ model)."
else
    echo ""
    echo "[FAILURE] Tests failed — check output above."
fi
exit $EXIT
