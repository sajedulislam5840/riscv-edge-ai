// =============================================================================
// sim_simd_mac.cpp — C++ Bit-Exact Co-Simulation for simd_mac RTL
// Description: Software golden model replicating hardware simd_mac behavior
//              Used as simulation fallback when iverilog/verilator not available.
// Compile:     g++ -std=c++17 -Wall -O2 -o sim_simd_mac sim_simd_mac.cpp
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <array>
#include <string>

// ---------------------------------------------------------------------------
// Bit-exact software model of simd_mac hardware
// ---------------------------------------------------------------------------
struct SimdMacState {
    int32_t  mac_out      = 0;
    bool     valid_out    = false;
    bool     overflow_flag = false;
};

static inline int8_t extract_i8(uint32_t packed, int lane) {
    return static_cast<int8_t>((packed >> (lane * 8)) & 0xFF);
}

SimdMacState simd_mac_model(
    uint32_t rs1_data,
    uint32_t rs2_data,
    int32_t  acc_in,
    bool     clr_acc,
    bool     enable,
    bool     rst_n,
    const SimdMacState& prev_state
) {
    SimdMacState next = prev_state;

    if (!rst_n) {
        next.mac_out       = 0;
        next.valid_out     = false;
        next.overflow_flag = false;
        return next;
    }

    if (!enable) {
        next.valid_out = false;
        return next;
    }

    // 1. Byte unpack and 8x8 signed multiply
    int16_t prod[4];
    for (int i = 0; i < 4; i++) {
        prod[i] = static_cast<int16_t>(extract_i8(rs1_data, i))
                * static_cast<int16_t>(extract_i8(rs2_data, i));
    }

    // 2. Balanced 2-level adder tree
    int32_t sum_01  = static_cast<int32_t>(prod[0]) + static_cast<int32_t>(prod[1]); // 17-bit
    int32_t sum_23  = static_cast<int32_t>(prod[2]) + static_cast<int32_t>(prod[3]); // 17-bit
    int32_t sum_all = sum_01 + sum_23;                                                 // 18-bit

    // 3. Accumulator mux
    int32_t effective_acc = clr_acc ? 0 : acc_in;

    // 4. 32-bit addition (C++ uses wrapping two's complement for int32_t)
    int64_t raw64        = static_cast<int64_t>(effective_acc) + static_cast<int64_t>(sum_all);
    int32_t raw_result   = static_cast<int32_t>(raw64 & 0xFFFFFFFF); // wrap

    // 5. Overflow detection (same logic as RTL)
    bool pos_ovf = (effective_acc > 0) && (sum_all > 0) && (raw_result < 0);
    bool neg_ovf = (effective_acc < 0) && (sum_all < 0) && (raw_result > 0);

    next.mac_out       = raw_result;
    next.valid_out     = true;
    next.overflow_flag = pos_ovf || neg_ovf;
    return next;
}

// ---------------------------------------------------------------------------
// Self-checking test runner
// ---------------------------------------------------------------------------
static int g_errors     = 0;
static int g_test_count = 0;

void check(const char* name,
           uint32_t rs1, uint32_t rs2, int32_t acc_in, bool clr_acc,
           int32_t expected)
{
    SimdMacState prev;
    prev.mac_out       = acc_in;   // Not used directly; model uses acc_in param
    prev.valid_out     = false;
    prev.overflow_flag = false;

    SimdMacState result = simd_mac_model(rs1, rs2, acc_in, clr_acc, true, true, prev);
    g_test_count++;

    if (result.mac_out != expected) {
        printf("[FAIL] %-50s : expected=%d, got=%d\n", name, expected, result.mac_out);
        g_errors++;
    } else {
        printf("[PASS] %-50s : mac_out=%-10d  ovf=%d\n", name, result.mac_out, result.overflow_flag);
    }
}

int main() {
    printf("==================================================\n");
    printf("  SIMD MAC C++ Co-Simulation (Software Model)    \n");
    printf("==================================================\n\n");

    // --- Group 1: Reset state ---
    printf("--- Group 1: Reset State ---\n");
    {
        SimdMacState s = simd_mac_model(0, 0, 0, false, false, false, {});
        g_test_count++;
        if (s.mac_out == 0 && !s.valid_out && !s.overflow_flag)
            printf("[PASS] Reset state: mac_out=0, valid=0, overflow=0\n");
        else {
            printf("[FAIL] Reset state: mac_out=%d valid=%d ovf=%d\n",
                   s.mac_out, s.valid_out, s.overflow_flag);
            g_errors++;
        }
    }

    // --- Group 2: Arithmetic corner cases ---
    printf("\n--- Group 2: Arithmetic Corner Cases ---\n");
    check("All Zeros",                     0x00000000, 0x00000000,      0, true,    0);
    check("Identity: 4 * 1*1 = 4",         0x01010101, 0x01010101,      0, true,    4);
    check("MaxNeg*MaxNeg: 4*16384=65536",  0x80808080, 0x80808080,      0, true,    65536);
    check("MaxPos*MaxPos: 4*16129=64516",  0x7F7F7F7F, 0x7F7F7F7F,      0, true,    64516);
    check("MaxNeg DotProduct: -65024",     0x80808080, 0x7F7F7F7F,      0, true,   -65024);

    // --- Group 3: Accumulator mode ---
    printf("\n--- Group 3: Accumulator Mode Tests ---\n");
    check("clr_acc=1 ignores acc_in=100",  0x01010101, 0x01010101,    100, true,    4);
    check("clr_acc=0 acc=100 + 4 = 104",  0x01010101, 0x01010101,    100, false,   104);
    check("Zero dot + acc_in=12345",       0x00000000, 0x00000000,  12345, false,   12345);

    // --- Group 4: Chained accumulation ---
    printf("\n--- Group 4: Chained Accumulation (100 iterations) ---\n");
    {
        // 100 × ([2,2,2,2] · [2,2,2,2]) = 100 × 16 = 1600
        int32_t acc = 0;
        SimdMacState s;
        for (int i = 0; i < 100; i++) {
            s = simd_mac_model(0x02020202, 0x02020202, acc, (i == 0), true, true, s);
            acc = s.mac_out;
        }
        g_test_count++;
        if (acc == 1600)
            printf("[PASS] 100-step chain: final=%d (expected 1600)\n", acc);
        else {
            printf("[FAIL] 100-step chain: got=%d, expected=1600\n", acc);
            g_errors++;
        }
    }

    // --- Group 5: Mixed signs ---
    printf("\n--- Group 5: Mixed Signs ---\n");
    // [127,-128,50,-50] · [1,1,1,1] = 127-128+50-50 = -1
    // rs1 = {-50(0xCE), 50(0x32), -128(0x80), 127(0x7F)} = 0xCE32807F
    check("Mixed: 127-128+50-50 = -1",    0xCE32807F, 0x01010101,      0, true,   -1);
    // Cancellation: [25,-25,25,-25]·[40,40,20,20] = 1000-1000+500-500 = 0
    check("Cancellation: 1000-1000+... = 0", 0xE719E719, 0x14142828,   0, true,    0);

    // --- Group 6: Overflow detection ---
    printf("\n--- Group 6: Overflow Detection ---\n");
    {
        // INT32_MAX + 65536 wraps negative — overflow should be true
        SimdMacState s;
        s = simd_mac_model(0x80808080, 0x80808080, INT32_MAX, false, true, true, s);
        g_test_count++;
        // 4*16384 = 65536 (sum_all), effective_acc=INT32_MAX
        // Both positive, result wraps negative → overflow
        if (s.overflow_flag && s.mac_out < 0)
            printf("[PASS] Overflow detected: INT32_MAX + 65536 wraps to %d, ovf=1\n", s.mac_out);
        else {
            printf("[FAIL] Overflow not detected: mac_out=%d, ovf=%d\n", s.mac_out, s.overflow_flag);
            g_errors++;
        }
    }

    // --- Final ---
    printf("\n==================================================\n");
    if (g_errors == 0) {
        printf(">> [SUCCESS] All %d Tests PASSED! 0 errors. <<\n", g_test_count);
    } else {
        printf(">> [FAILURE] %d / %d Tests FAILED! <<\n", g_errors, g_test_count);
    }
    printf("==================================================\n");

    return (g_errors == 0) ? 0 : 1;
}
