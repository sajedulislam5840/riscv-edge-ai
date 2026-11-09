// =============================================================================
// gen_simd_vectors.cpp — Golden SIMD MAC Vector Generator
// Description: Loads the trained MNIST model (mnist_model.qbin), slices
//              input activations and weights into 4-byte INT8 chunks,
//              computes dot products using the software MAC model, and writes
//              hardware/tb/golden_vectors.hex ready for Verilog co-simulation.
//
// Output format (one vector per line):
//   <rs1_hex8> <rs2_hex8> <acc_in_hex8> <clr_acc_bit> <expected_rd_hex8>
//
// Compile:  g++ -std=c++17 -O2 -o gen_simd_vectors gen_simd_vectors.cpp
// Run:      ./gen_simd_vectors
//           (run from repo root; reads benchmarks/mnist/results/mnist_model.qbin)
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// QBIN model structure (matches Phase 02 export_weights.py format)
// ---------------------------------------------------------------------------
struct QbinModel {
    char     magic[4];    // "TENQ"
    int32_t  version;
    int32_t  in_size;     // 784
    int32_t  hid_size;    // 128
    float    scale_w1;
    int32_t  out_size;    // ignored for dot product generation
    std::vector<int8_t>  W1;      // shape: [in_size, hid_size]
    std::vector<int32_t> b1;      // shape: [hid_size]
};

static QbinModel load_qbin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    QbinModel m;
    f.read(m.magic, 4);
    if (std::strncmp(m.magic, "TENQ", 4) != 0)
        throw std::runtime_error("Invalid TENQ magic");

    f.read(reinterpret_cast<char*>(&m.version),  4);
    f.read(reinterpret_cast<char*>(&m.in_size),  4);
    f.read(reinterpret_cast<char*>(&m.hid_size), 4);
    f.read(reinterpret_cast<char*>(&m.scale_w1), 4);
    f.read(reinterpret_cast<char*>(&m.out_size), 4);

    // W1: in_size × hid_size int8
    m.W1.resize(static_cast<size_t>(m.in_size) * m.hid_size);
    f.read(reinterpret_cast<char*>(m.W1.data()), m.W1.size());

    // b1: hid_size int32
    m.b1.resize(m.hid_size);
    f.read(reinterpret_cast<char*>(m.b1.data()), m.hid_size * 4);

    return m;
}

// ---------------------------------------------------------------------------
// Pack four int8 values into uint32 (little-endian)
// ---------------------------------------------------------------------------
static uint32_t pack4(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return static_cast<uint32_t>(static_cast<uint8_t>(b0))
         | (static_cast<uint32_t>(static_cast<uint8_t>(b1)) << 8)
         | (static_cast<uint32_t>(static_cast<uint8_t>(b2)) << 16)
         | (static_cast<uint32_t>(static_cast<uint8_t>(b3)) << 24);
}

// ---------------------------------------------------------------------------
// Software SIMD MAC model (bit-exact to RTL)
// ---------------------------------------------------------------------------
static int32_t simd_mac_sw(int32_t acc, uint32_t rs1, uint32_t rs2) {
    int32_t sum = 0;
    for (int i = 0; i < 4; i++) {
        int8_t  a = static_cast<int8_t>((rs1 >> (i * 8)) & 0xFF);
        int8_t  b = static_cast<int8_t>((rs2 >> (i * 8)) & 0xFF);
        sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
    }
    return acc + sum;
}

// ---------------------------------------------------------------------------
// Generate vectors: for each output neuron j, stream all 784/4=196 chunks
// of the input activation row (use a fixed quantized test pattern) dot W1[:,j]
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string qbin_path = "benchmarks/mnist/results/mnist_model.qbin";
    std::string out_path  = "hardware/tb/golden_vectors.hex";

    if (argc > 1) qbin_path = argv[1];
    if (argc > 2) out_path  = argv[2];

    printf("Loading model: %s\n", qbin_path.c_str());
    QbinModel model = load_qbin(qbin_path);
    printf("Model: in=%d, hid=%d, scale_w1=%.6f\n",
           model.in_size, model.hid_size, model.scale_w1);

    // Generate a fixed quantized activation pattern:
    // Use MNIST-like pixel values scaled to INT8 range
    // Pattern: activation[i] = (i % 256) - 128  (synthetic but realistic spread)
    std::vector<int8_t> activation(model.in_size);
    for (int i = 0; i < model.in_size; i++) {
        activation[i] = static_cast<int8_t>((i * 7 + 13) % 128);
    }

    std::ofstream out(out_path);
    if (!out) {
        fprintf(stderr, "Cannot open output: %s\n", out_path.c_str());
        return 1;
    }

    // Write header comment
    out << "# SIMD MAC Golden Vectors — Generated from MNIST mnist_model.qbin\n";
    out << "# Format: rs1_hex rs2_hex acc_in_hex clr_acc expected_rd_hex\n";
    out << "# rs1 = 4 packed INT8 activations (little-endian)\n";
    out << "# rs2 = 4 packed INT8 weights     (little-endian)\n";
    out << "# clr_acc = 1 on first chunk of each dot product, 0 for subsequent chunks\n";
    out << "#\n";

    int    vectors_written = 0;
    int    chunks_per_dot  = model.in_size / 4;   // 196 for 784-dim input

    // For each of hid_size (128) output neurons
    for (int j = 0; j < model.hid_size; j++) {
        int32_t acc_gold = 0;

        for (int chunk = 0; chunk < chunks_per_dot; chunk++) {
            int    base  = chunk * 4;
            int    clr   = (chunk == 0) ? 1 : 0;

            // Pack 4 activations into rs1
            uint32_t rs1 = pack4(
                activation[base + 0],
                activation[base + 1],
                activation[base + 2],
                activation[base + 3]
            );

            // Pack 4 weights (W1 is row-major: W1[in_idx, j])
            uint32_t rs2 = pack4(
                model.W1[(base + 0) * model.hid_size + j],
                model.W1[(base + 1) * model.hid_size + j],
                model.W1[(base + 2) * model.hid_size + j],
                model.W1[(base + 3) * model.hid_size + j]
            );

            // Compute expected result using software model
            int32_t acc_in = clr ? 0 : acc_gold;
            acc_gold = simd_mac_sw(acc_in, rs1, rs2);

            // Write vector line
            out << std::hex;
            char line[128];
            std::snprintf(line, sizeof(line),
                "%08x %08x %08x %d %08x\n",
                rs1,
                rs2,
                static_cast<uint32_t>(acc_in),
                clr,
                static_cast<uint32_t>(acc_gold));
            out << line;
            vectors_written++;
        }
    }

    out.close();
    printf("Written %d vectors (%d dot products × %d chunks) to: %s\n",
           vectors_written, model.hid_size, chunks_per_dot, out_path.c_str());

    return 0;
}
