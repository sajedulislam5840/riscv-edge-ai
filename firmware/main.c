// Bare-metal MNIST inference on RV32IM: integer-only, no heap, no standard library.
//
// Runs the 784-128-10 INT8 network on the ten example test images in three versions and checks
// every output byte of each against the expected values computed on the host (same integer
// arithmetic as the C++ engine):
//
//   scalar               plain RV32IM, one 8x8 multiply per multiply instruction
//   simd_mac_clr + add   the custom instruction computes four products and their sum;
//                        an ordinary add keeps the running total
//   simd_macz/simd_macc  the running total stays inside the MAC unit
//
// It reports the instruction count of one inference for each version (Spike counts one cycle
// per instruction, so these are instruction counts, not pipeline cycles).
#include <stdint.h>
#include "htif.h"
#include "model_data.h"

#define SIMD_MAC_HW_ENABLE 1
#include "simd_mac_isa.h"

// round(x * mult * 2^-shift) in 64-bit integer arithmetic, as the engine does it.
static int32_t requantize(int32_t x, int32_t mult, int shift) {
    int64_t total = (int64_t)x * mult;
    if (shift <= 0) return (int32_t)(total << -shift);
    return (int32_t)((total + ((int64_t)1 << (shift - 1))) >> shift);
}

static int8_t clamp8(int32_t v) { return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v); }

// Bias, requantization and ReLU for one neuron, given its accumulator sum((x - Z_x) * w).
static int8_t finish(int32_t acc, int8_t b, int32_t zb, int32_t mult, int shift, int32_t bmult, int bshift,
                     int32_t zout, int relu) {
    acc += requantize((int32_t)b - zb, bmult, bshift);
    int32_t r = requantize(acc, mult, shift) + zout;
    if (relu && r < zout) r = zout;
    return clamp8(r);
}

// ---------------------------------------------------------------------------- scalar
// One dense layer. Weights are [K][N] row-major with zero point 0 (symmetric).
static void dense(const int8_t *x, int32_t zx, const int8_t *w, const int8_t *b, int32_t zb,
                  int K, int N, int32_t mult, int shift, int32_t bmult, int bshift,
                  int32_t zout, int relu, int8_t *out) {
    for (int j = 0; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) acc += ((int32_t)x[k] - zx) * (int32_t)w[k * N + j];
        out[j] = finish(acc, b[j], zb, mult, shift, bmult, bshift, zout, relu);
    }
}

static int8_t input[IN_SIZE], hidden[HIDDEN_SIZE], output[OUT_SIZE];

static int argmax(const int8_t *v) {
    int best = 0;
    for (int d = 1; d < OUT_SIZE; d++) if (v[d] > v[best]) best = d;
    return best;
}

static int infer_scalar(const uint8_t *pixels) {
    for (int i = 0; i < IN_SIZE; i++) input[i] = (int8_t)((int32_t)pixels[i] - 128);
    dense(input, ZP_IN, W1, B1, B1_ZP, IN_SIZE, HIDDEN_SIZE, L1_MULT, L1_SHIFT, L1_BIAS_MULT, L1_BIAS_SHIFT,
          ZP_HID, 1, hidden);
    dense(hidden, ZP_HID, W2, B2, B2_ZP, HIDDEN_SIZE, OUT_SIZE, L2_MULT, L2_SHIFT, L2_BIAS_MULT, L2_BIAS_SHIFT,
          ZP_OUT, 0, output);
    return argmax(output);
}

// ---------------------------------------------------------------------------- custom MAC
// Activations as bytes and as packed words (GCC defines union type punning).
typedef union { int8_t b[IN_SIZE]; uint32_t w[IN_SIZE / 4]; } in_buf_t;
typedef union { int8_t b[HIDDEN_SIZE]; uint32_t w[HIDDEN_SIZE / 4]; } hid_buf_t;
static in_buf_t in_q;
static hid_buf_t hid_q;
static int8_t out_q[OUT_SIZE];

enum { MAC_CLR_ADD, MAC_INTERNAL };

// One dense layer with the custom instruction. w holds K/4 packed words per neuron; zcorr is
// the per-neuron constant -Z_x * sum(w), so acc + zcorr = sum((x - Z_x) * w).
static void dense_mac(const uint32_t *x, const uint32_t *w, const int32_t *zcorr, const int8_t *b, int32_t zb,
                      int K, int N, int32_t mult, int shift, int32_t bmult, int bshift,
                      int32_t zout, int relu, int8_t *out, int mode) {
    const int words = K / 4;
    for (int j = 0; j < N; j++) {
        const uint32_t *wj = w + j * words;
        int32_t acc;
        if (mode == MAC_INTERNAL) {
            SIMD_MACZ(acc, x[0], wj[0]);
            for (int c = 1; c < words; c++) SIMD_MACC(acc, x[c], wj[c]);
        } else {
            acc = 0;
            for (int c = 0; c < words; c++) {
                int32_t dot;
                SIMD_MAC_CLR(dot, x[c], wj[c]);
                acc += dot;
            }
        }
        out[j] = finish(acc + zcorr[j], b[j], zb, mult, shift, bmult, bshift, zout, relu);
    }
}

static int infer_mac(const uint8_t *pixels, int mode) {
    for (int i = 0; i < IN_SIZE; i++) in_q.b[i] = (int8_t)((int32_t)pixels[i] - 128);
    dense_mac(in_q.w, W1_PACKED, ZCORR1, B1, B1_ZP, IN_SIZE, HIDDEN_SIZE, L1_MULT, L1_SHIFT, L1_BIAS_MULT,
              L1_BIAS_SHIFT, ZP_HID, 1, hid_q.b, mode);
    dense_mac(hid_q.w, W2_PACKED, ZCORR2, B2, B2_ZP, HIDDEN_SIZE, OUT_SIZE, L2_MULT, L2_SHIFT, L2_BIAS_MULT,
              L2_BIAS_SHIFT, ZP_OUT, 0, out_q, mode);
    return argmax(out_q);
}

// ---------------------------------------------------------------------------- main
static int matches(const int8_t *got, int e) {
    for (int d = 0; d < OUT_SIZE; d++) if (got[d] != EXPECTED[e][d]) return 0;
    return 1;
}

static void put_ratio_x100(uint32_t num, uint32_t den) {  // prints num/den with two decimals
    uint32_t r = (uint32_t)(((uint64_t)num * 100 + den / 2) / den);
    htif_put_u32(r / 100); htif_puts(".");
    if (r % 100 < 10) htif_puts("0");
    htif_put_u32(r % 100);
}

int main(void) {
    htif_puts("riscv-edge-ai: bare-metal INT8 MNIST on RV32IM\n");
    int failures = 0;
    uint32_t c0 = 0, i0 = 0, i_clr = 0, i_int = 0;
    for (int e = 0; e < NUM_EXAMPLES; e++) {
        uint32_t cs = rdcycle(), is = rdinstret();
        int pred = infer_scalar(IMAGES[e]);
        uint32_t cycles = rdcycle() - cs, instrs = rdinstret() - is;
        int ok = matches(output, e);

        is = rdinstret();
        int pred_clr = infer_mac(IMAGES[e], MAC_CLR_ADD);
        uint32_t n_clr = rdinstret() - is;
        ok &= matches(out_q, e) && pred_clr == pred;

        is = rdinstret();
        int pred_int = infer_mac(IMAGES[e], MAC_INTERNAL);
        uint32_t n_int = rdinstret() - is;
        ok &= matches(out_q, e) && pred_int == pred;

        if (e == 0) { c0 = cycles; i0 = instrs; i_clr = n_clr; i_int = n_int; }
        failures += !ok;
        htif_puts("test image "); htif_put_u32(TEST_INDEX[e]);
        htif_puts(": label "); htif_put_u32(LABELS[e]);
        htif_puts(", predicted "); htif_put_u32((uint32_t)pred);
        htif_puts(ok ? ", outputs match host\n" : ", OUTPUT MISMATCH\n");
    }
    htif_puts("one inference: "); htif_put_u32(i0); htif_puts(" instructions, ");
    htif_put_u32(c0); htif_puts(" cycles (rdcycle)\n");
    htif_puts("with simd_mac_clr + add: "); htif_put_u32(i_clr); htif_puts(" instructions, ");
    put_ratio_x100(i0, i_clr); htif_puts("x fewer\n");
    htif_puts("with simd_macz/simd_macc: "); htif_put_u32(i_int); htif_puts(" instructions, ");
    put_ratio_x100(i0, i_int); htif_puts("x fewer\n");
    htif_puts("multiply-accumulates per inference: ");
    htif_put_u32((uint32_t)(IN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUT_SIZE)); htif_puts("\n");
    htif_puts(failures ? "RESULT: FAIL\n" : "RESULT: all outputs match the C++ engine\n");
    return failures ? 1 : 0;
}
