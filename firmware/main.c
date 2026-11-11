// Bare-metal MNIST inference on RV32IM: integer-only, no heap, no standard library.
//
// Runs the 784-128-10 INT8 network on the ten example test images, checks every output byte
// against the expected values computed on the host (same integer arithmetic as the C++ engine),
// and reports the cycle and instruction counts for one inference.
//
// This is the scalar baseline: one 8x8 multiply per instruction. The SIMD MAC version will run
// the same layer with the custom instruction, four multiplies at a time.
#include <stdint.h>
#include "htif.h"
#include "model_data.h"

// round(x * mult * 2^-shift) in 64-bit integer arithmetic, as the engine does it.
static int32_t requantize(int32_t x, int32_t mult, int shift) {
    int64_t total = (int64_t)x * mult;
    if (shift <= 0) return (int32_t)(total << -shift);
    return (int32_t)((total + ((int64_t)1 << (shift - 1))) >> shift);
}

static int8_t clamp8(int32_t v) { return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v); }

// One dense layer. Weights are [K][N] row-major with zero point 0 (symmetric).
static void dense(const int8_t *x, int32_t zx, const int8_t *w, const int8_t *b, int32_t zb,
                  int K, int N, int32_t mult, int shift, int32_t bmult, int bshift,
                  int32_t zout, int relu, int8_t *out) {
    for (int j = 0; j < N; j++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) acc += ((int32_t)x[k] - zx) * (int32_t)w[k * N + j];
        acc += requantize((int32_t)b[j] - zb, bmult, bshift);
        int32_t r = requantize(acc, mult, shift) + zout;
        if (relu && r < zout) r = zout;
        out[j] = clamp8(r);
    }
}

static int8_t input[IN_SIZE], hidden[HIDDEN_SIZE], output[OUT_SIZE];

static int infer(const uint8_t *pixels) {
    for (int i = 0; i < IN_SIZE; i++) input[i] = (int8_t)((int32_t)pixels[i] - 128);
    dense(input, ZP_IN, W1, B1, B1_ZP, IN_SIZE, HIDDEN_SIZE, L1_MULT, L1_SHIFT, L1_BIAS_MULT, L1_BIAS_SHIFT,
          ZP_HID, 1, hidden);
    dense(hidden, ZP_HID, W2, B2, B2_ZP, HIDDEN_SIZE, OUT_SIZE, L2_MULT, L2_SHIFT, L2_BIAS_MULT, L2_BIAS_SHIFT,
          ZP_OUT, 0, output);
    int best = 0;
    for (int d = 1; d < OUT_SIZE; d++) if (output[d] > output[best]) best = d;
    return best;
}

int main(void) {
    htif_puts("riscv-edge-ai: bare-metal INT8 MNIST on RV32IM (scalar baseline)\n");
    int failures = 0;
    uint32_t c0 = 0, i0 = 0;
    for (int e = 0; e < NUM_EXAMPLES; e++) {
        uint32_t cs = rdcycle(), is = rdinstret();
        int pred = infer(IMAGES[e]);
        uint32_t cycles = rdcycle() - cs, instrs = rdinstret() - is;
        if (e == 0) { c0 = cycles; i0 = instrs; }
        int match = 1;
        for (int d = 0; d < OUT_SIZE; d++) if (output[d] != EXPECTED[e][d]) match = 0;
        failures += !match;
        htif_puts("test image "); htif_put_u32(TEST_INDEX[e]);
        htif_puts(": label "); htif_put_u32(LABELS[e]);
        htif_puts(", predicted "); htif_put_u32((uint32_t)pred);
        htif_puts(match ? ", outputs match host\n" : ", OUTPUT MISMATCH\n");
        (void)cycles; (void)instrs;
    }
    htif_puts("one inference: "); htif_put_u32(i0); htif_puts(" instructions, ");
    htif_put_u32(c0); htif_puts(" cycles (rdcycle)\n");
    htif_puts("multiply-accumulates per inference: ");
    htif_put_u32((uint32_t)(IN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUT_SIZE)); htif_puts("\n");
    htif_puts(failures ? "RESULT: FAIL\n" : "RESULT: all outputs match the C++ engine\n");
    return failures ? 1 : 0;
}
