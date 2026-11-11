// Host test for hardware/include/simd_mac_isa.h (software model of the custom instructions).
// Checks that simd_macz/simd_macc chains, simd_mac_clr/simd_mac chains and a scalar loop give
// the same dot products, including the extreme values -128 and 127.
#include "simd_mac_isa.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(-128, 127);
    int failures = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        const int n = 4 * (1 + trial % 200);
        std::vector<int8_t> a(n), b(n);
        for (int i = 0; i < n; ++i) {
            a[i] = static_cast<int8_t>(trial < 10 ? -128 : byte(rng));
            b[i] = static_cast<int8_t>(trial < 5 ? -128 : trial < 10 ? 127 : byte(rng));
        }
        int32_t scalar = 0;
        for (int i = 0; i < n; ++i) scalar += int32_t(a[i]) * int32_t(b[i]);

        int32_t internal = 12345, via_rd = 0;
        for (int i = 0; i < n; i += 4) {
            const uint32_t pa = simd_pack_i8(a[i], a[i + 1], a[i + 2], a[i + 3]);
            const uint32_t pb = simd_pack_i8(b[i], b[i + 1], b[i + 2], b[i + 3]);
            if (i == 0) { SIMD_MACZ(internal, pa, pb); SIMD_MAC_CLR(via_rd, pa, pb); }
            else        { internal = -1; SIMD_MACC(internal, pa, pb); SIMD_MAC(via_rd, pa, pb); }  // rd value is ignored by MACC
        }
        if (internal != scalar || via_rd != scalar || simd_dot4(a.data(), b.data(), n) != scalar) {
            if (failures < 5) std::printf("trial %d: scalar %d macz/macc %d mac_clr/mac %d\n", trial, scalar, internal, via_rd);
            ++failures;
        }
    }
    if (failures) std::printf("SIMD ISA model: %d failures\n", failures);
    else std::printf("SIMD ISA model: 2000 random dot products match\n");
    return failures ? 1 : 0;
}
