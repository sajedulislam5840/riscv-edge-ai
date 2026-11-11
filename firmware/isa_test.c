// Self-test for the custom-0 MAC instructions on Spike (with firmware/spike_ext loaded).
// Runs each instruction through the macros in hardware/include/simd_mac_isa.h and checks the
// result against a plain C reference: random operands, long simd_macz/simd_macc chains with
// other code in between, the rd-accumulator forms, extreme values, and 32-bit wraparound.
#define SIMD_MAC_HW_ENABLE 1
#include <stdint.h>
#include "htif.h"
#include "simd_mac_isa.h"

static uint32_t state = 0x12345678u;
static uint32_t rnd(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }

static int32_t ref_dot4(uint32_t a, uint32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 4; i++) s += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return s;
}

static int failures;
static void check(const char *what, int32_t got, int32_t want) {
    if (got != want) {
        if (failures < 5) { htif_puts("FAIL "); htif_puts(what); htif_puts(": got "); htif_put_i32(got);
                            htif_puts(" want "); htif_put_i32(want); htif_puts("\n"); }
        failures++;
    }
}

int main(void) {
    htif_puts("simd_mac ISA self-test on Spike\n");
    uint32_t checks = 0;

    for (int i = 0; i < 2000; i++, checks += 2) {           // single instructions
        uint32_t a = rnd(), b = rnd();
        int32_t r;
        SIMD_MACZ(r, a, b);     check("simd_macz", r, ref_dot4(a, b));
        SIMD_MAC_CLR(r, a, b);  check("simd_mac_clr", r, ref_dot4(a, b));
    }

    for (int chain = 0; chain < 50; chain++) {              // simd_macz/simd_macc, other code in between
        uint32_t a = rnd(), b = rnd();
        int32_t want = ref_dot4(a, b), r;
        SIMD_MACZ(r, a, b);
        for (int k = 0; k < 100; k++, checks++) {
            a = rnd(); b = rnd();
            volatile uint32_t noise = a * 7u + b; (void)noise;
            want += ref_dot4(a, b);
            SIMD_MACC(r, a, b);
            check("simd_macc chain", r, want);
        }
    }

    for (int chain = 0; chain < 50; chain++) {              // simd_mac_clr/simd_mac, total in rd
        uint32_t a = rnd(), b = rnd();
        int32_t want = ref_dot4(a, b), r;
        SIMD_MAC_CLR(r, a, b);
        for (int k = 0; k < 100; k++, checks++) {
            a = rnd(); b = rnd();
            want += ref_dot4(a, b);
            SIMD_MAC(r, a, b);
            check("simd_mac chain", r, want);
        }
    }

    {   // the rd forms also write the unit's result register, as in the RTL,
        // so simd_macc continues from their result
        int32_t r, t;
        uint32_t a = rnd(), b = rnd(), c = rnd(), d = rnd();
        SIMD_MAC_CLR(r, a, b);
        SIMD_MACC(t, c, d);
        check("simd_macc after simd_mac_clr", t, ref_dot4(a, b) + ref_dot4(c, d));
        checks++;
    }

    {   // one long internal chain with nothing else touching the unit
        int32_t want = 0, r = 0;
        uint32_t a = rnd(), b = rnd();
        SIMD_MACZ(r, a, b); want = ref_dot4(a, b);
        for (int k = 0; k < 5000; k++, checks++) {
            a = rnd(); b = rnd();
            want += ref_dot4(a, b);
            SIMD_MACC(r, a, b);
            check("long simd_macc chain", r, want);
        }
    }

    {   // extreme values
        int32_t r;
        SIMD_MACZ(r, 0x80808080u, 0x80808080u); check("-128*-128 x4", r, 65536);
        SIMD_MACZ(r, 0x7F7F7F7Fu, 0x80808080u); check("127*-128 x4", r, -65024);
        SIMD_MACZ(r, 0x7F7F7F7Fu, 0x7F7F7F7Fu); check("127*127 x4", r, 64516);
        checks += 3;
    }

    {   // 32-bit wraparound: 32768 x 65536 = 2^31 wraps to INT32_MIN, as in the RTL adder
        int32_t r;
        SIMD_MACZ(r, 0x80808080u, 0x80808080u);
        for (int k = 1; k < 32768; k++) SIMD_MACC(r, 0x80808080u, 0x80808080u);
        check("wraparound", r, (int32_t)0x80000000u);
        checks++;
    }

    htif_puts("checks: "); htif_put_u32(checks);
    htif_puts(failures ? "\nRESULT: FAIL\n" : "\nRESULT: all custom-0 instructions match the C reference\n");
    return failures ? 1 : 0;
}
