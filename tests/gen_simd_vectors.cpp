// Golden test vectors for the simd_mac RTL.
//
// Takes the INT8 MNIST model and one real test image, splits every first-layer dot product
// into 4-byte chunks exactly as firmware would feed them to the custom instruction, and writes
// one line per instruction to hardware/tb/golden_vectors.hex:
//
//   <rs1> <rs2> <acc_in> <clr_acc> <expected_rd>      (hex, hex, hex, 0/1, hex)
//
// rs1 holds four input bytes (q = pixel - 128), rs2 four weight bytes of one neuron.
// The MAC multiplies raw signed bytes. Because weights are symmetric (zero point 0), the
// layer's accumulator is  sum((q_x - Z_x) * w) = sum(q_x * w) - Z_x * sum(w),  so the
// activation zero point becomes a per-neuron constant that firmware folds into the bias.
// This program checks that identity against the engine for every neuron.
//
// Run from the repo root (after export_model_and_data.py):
//   ./build/gen_simd_vectors [model.qbin] [test_images.bin] [image_index] [out.hex]

#include "engine/quantize.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t pack4(const int8_t *v) {
    return uint32_t(uint8_t(v[0])) | uint32_t(uint8_t(v[1])) << 8 |
           uint32_t(uint8_t(v[2])) << 16 | uint32_t(uint8_t(v[3])) << 24;
}

// Bit-exact software model of one simd_mac instruction.
static int32_t simd_mac_sw(int32_t acc, uint32_t rs1, uint32_t rs2) {
    int32_t sum = 0;
    for (int i = 0; i < 4; ++i)
        sum += int32_t(int8_t(rs1 >> (8 * i))) * int32_t(int8_t(rs2 >> (8 * i)));
    return acc + sum;
}

static std::vector<int8_t> load_input(const std::string &path, uint32_t index, uint8_t &label) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (run export_model_and_data.py first)");
    uint32_t h[4];
    f.read(reinterpret_cast<char *>(h), sizeof(h));
    if (h[0] != 0x54534E4D || index >= h[1]) throw std::runtime_error("bad data file or index");
    std::vector<uint8_t> px(784);
    f.seekg(16 + std::streamoff(index) * 784);
    f.read(reinterpret_cast<char *>(px.data()), 784);
    f.seekg(16 + std::streamoff(h[1]) * 784 + index);
    f.read(reinterpret_cast<char *>(&label), 1);
    std::vector<int8_t> q(784);
    for (int k = 0; k < 784; ++k) q[k] = int8_t(int(px[k]) - 128);  // S = 1/255, Z = -128
    return q;
}

int main(int argc, char **argv) {
    try {
        const std::string qbin = argc > 1 ? argv[1] : "benchmarks/mnist/results/mnist_model.qbin";
        const std::string data = argc > 2 ? argv[2] : "benchmarks/mnist/data/mnist_test.bin";
        const uint32_t index = argc > 3 ? uint32_t(std::stoul(argv[3])) : 0;
        const std::string out_path = argc > 4 ? argv[4] : "hardware/tb/golden_vectors.hex";

        const auto model = engine::load_quantized_model(qbin);
        const engine::QuantizedTensor &W1 = model.at(0);
        if (W1.params().zero_point != 0)
            throw std::runtime_error("weights must be symmetric (zero point 0) for the SIMD MAC");
        const size_t K = W1.rows(), N = W1.cols();
        if (K % 4 != 0) throw std::runtime_error("input size must be a multiple of 4");

        uint8_t label = 0;
        const std::vector<int8_t> x = load_input(data, index, label);
        const int32_t zx = -128;

        std::ofstream out(out_path);
        if (!out) throw std::runtime_error("cannot write " + out_path);
        // No header lines: the Verilog testbench reads the file with $fscanf and expects data only.

        std::vector<int8_t> w(K);
        size_t vectors = 0;
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) w[k] = W1(k, j);

            int32_t acc = 0, sum_w = 0, reference = 0;
            for (size_t c = 0; c < K / 4; ++c) {
                const uint32_t rs1 = pack4(&x[4 * c]), rs2 = pack4(&w[4 * c]);
                const int clr = (c == 0);
                const int32_t acc_in = clr ? 0 : acc;
                acc = simd_mac_sw(acc_in, rs1, rs2);
                char line[64];
                std::snprintf(line, sizeof(line), "%08x %08x %08x %d %08x\n", rs1, rs2,
                              uint32_t(acc_in), clr, uint32_t(acc));
                out << line;
                ++vectors;
            }
            for (size_t k = 0; k < K; ++k) {
                sum_w += w[k];
                reference += (int32_t(x[k]) - zx) * int32_t(w[k]);  // what the engine computes
            }
            if (acc - zx * sum_w != reference)
                throw std::runtime_error("zero-point folding mismatch at neuron " + std::to_string(j));
        }
        std::printf("Wrote %zu vectors (%zu neurons x %zu chunks) from test image %u (label %d) to %s\n",
                    vectors, N, K / 4, index, int(label), out_path.c_str());
        std::printf("Zero-point folding check passed for all %zu neurons\n", N);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
