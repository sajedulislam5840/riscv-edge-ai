// MNIST evaluation: FP32 reference vs. the integer-only INT8 engine.
//
// Reads the files written by benchmarks/mnist/export_model_and_data.py:
//   benchmarks/mnist/results/mnist_model.qbin  INT8 weights (symmetric, zero point 0)
//   benchmarks/mnist/data/fp32_weights.bin     the same weights in FP32
//   benchmarks/mnist/data/mnist_test.bin       10,000 test images + labels
//   benchmarks/mnist/data/mnist_calib.bin      2,000 training images for calibration
//
// Activation scales are calibrated on the training images only. The test set is used
// for nothing except the final accuracy numbers.
//
// Also writes web/js/quant_params.js: the calibrated scales and the integer requantization
// constants, so the web demo runs exactly the same integer arithmetic as this engine.
//
// Run from the repo root:  ./build/mnist_benchmark [model.qbin]

#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Images {
    uint32_t count = 0;
    std::vector<uint8_t> pixels;  // count * 784
    std::vector<uint8_t> labels;  // count
};

Images load_images(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (run export_model_and_data.py first)");
    uint32_t header[4];
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (header[0] != 0x54534E4D || header[2] != 28 || header[3] != 28)
        throw std::runtime_error("bad header in " + path);
    Images im;
    im.count = header[1];
    im.pixels.resize(static_cast<size_t>(im.count) * 784);
    im.labels.resize(im.count);
    f.read(reinterpret_cast<char *>(im.pixels.data()), static_cast<std::streamsize>(im.pixels.size()));
    f.read(reinterpret_cast<char *>(im.labels.data()), static_cast<std::streamsize>(im.labels.size()));
    if (!f) throw std::runtime_error("truncated file " + path);
    return im;
}

struct Fp32Model {
    engine::Tensor<float> W1{784, 128}, b1{1, 128}, W2{128, 10}, b2{1, 10};
};

Fp32Model load_fp32(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    Fp32Model m;
    for (engine::Tensor<float> *t : {&m.W1, &m.b1, &m.W2, &m.b2})
        f.read(reinterpret_cast<char *>(t->data()), static_cast<std::streamsize>(t->size() * sizeof(float)));
    if (!f) throw std::runtime_error("truncated file " + path);
    return m;
}

Fp32Model dequantized(const std::vector<engine::QuantizedTensor> &q) {
    Fp32Model m;
    m.W1 = engine::dequantize_tensor(q[0]);
    m.b1 = engine::dequantize_tensor(q[1]);
    m.W2 = engine::dequantize_tensor(q[2]);
    m.b2 = engine::dequantize_tensor(q[3]);
    return m;
}

engine::Tensor<float> image_to_float(const Images &im, size_t i) {
    engine::Tensor<float> x(1, 784);
    for (size_t k = 0; k < 784; ++k) x.data()[k] = im.pixels[i * 784 + k] / 255.0f;
    return x;
}

// FP32 forward pass. Writes hidden activations and logits.
void forward_fp32(const Fp32Model &m, const engine::Tensor<float> &x,
                  engine::Tensor<float> &h, engine::Tensor<float> &out) {
    engine::matmul_fp32(x, m.W1, h);
    engine::bias_add(h, m.b1);
    engine::relu_inplace(h);
    engine::matmul_fp32(h, m.W2, out);
    engine::bias_add(out, m.b2);
}

template <typename T>
size_t argmax(const T *v, size_t n) {
    return static_cast<size_t>(std::max_element(v, v + n) - v);
}

struct Int8Pipeline {
    engine::QuantParams in, hidden, out;
    int32_t m1 = 0, m2 = 0;
    int s1 = 0, s2 = 0;
};

Int8Pipeline make_pipeline(const std::vector<engine::QuantizedTensor> &q,
                           engine::QuantParams hidden, engine::QuantParams out) {
    Int8Pipeline p;
    // Pixels p in 0..255 map exactly to q = p - 128 with S = 1/255, Z = -128.
    p.in = {1.0f / 255.0f, -128};
    p.hidden = hidden;
    p.out = out;
    // Requantization multipliers are computed once, offline; inference is integer-only.
    engine::quantize_multiplier_and_shift(
        double(p.in.scale) * double(q[0].params().scale) / double(p.hidden.scale), p.m1, p.s1);
    engine::quantize_multiplier_and_shift(
        double(p.hidden.scale) * double(q[2].params().scale) / double(p.out.scale), p.m2, p.s2);
    return p;
}

size_t predict_int8(const std::vector<engine::QuantizedTensor> &q, const Int8Pipeline &p,
                    const Images &im, size_t i) {
    engine::QuantizedTensor x(1, 784, p.in);
    for (size_t k = 0; k < 784; ++k) x.data()[k] = static_cast<int8_t>(int(im.pixels[i * 784 + k]) - 128);
    engine::QuantizedTensor h(1, 128, p.hidden);
    engine::matmul_int8_fixed_point_fused(x, q[0], q[1], h, p.m1, p.s1, /*apply_relu=*/true);
    engine::QuantizedTensor out(1, 10, p.out);
    engine::matmul_int8_fixed_point_fused(h, q[2], q[3], out, p.m2, p.s2, /*apply_relu=*/false);
    return argmax(out.data(), 10);  // same scale for all logits, so argmax on INT8 is exact
}

double accuracy_int8(const std::vector<engine::QuantizedTensor> &q, const Int8Pipeline &p,
                     const Images &test, std::vector<size_t> *preds = nullptr) {
    size_t correct = 0;
    for (size_t i = 0; i < test.count; ++i) {
        size_t y = predict_int8(q, p, test, i);
        if (preds) (*preds)[i] = y;
        correct += (y == test.labels[i]);
    }
    return 100.0 * double(correct) / double(test.count);
}

double accuracy_fp32(const Fp32Model &m, const Images &test, std::vector<size_t> *preds = nullptr) {
    engine::Tensor<float> h(1, 128), out(1, 10);
    size_t correct = 0;
    for (size_t i = 0; i < test.count; ++i) {
        forward_fp32(m, image_to_float(test, i), h, out);
        size_t y = argmax(out.data(), 10);
        if (preds) (*preds)[i] = y;
        correct += (y == test.labels[i]);
    }
    return 100.0 * double(correct) / double(test.count);
}

// Run the FP32 model on the calibration images and collect hidden activations and logits.
void calibrate(const Fp32Model &m, const Images &calib, engine::Tensor<float> &hidden_all,
               engine::Tensor<float> &logits_all) {
    engine::Tensor<float> h(1, 128), out(1, 10);
    for (size_t i = 0; i < calib.count; ++i) {
        forward_fp32(m, image_to_float(calib, i), h, out);
        std::copy(h.data(), h.data() + 128, hidden_all.data() + i * 128);
        std::copy(out.data(), out.data() + 10, logits_all.data() + i * 10);
    }
}

// Bias multiplier exactly as matmul_int8_fixed_point_fused computes it.
void bias_multiplier(const engine::QuantizedTensor &bias, float in_scale, float w_scale, int32_t &m, int &shift) {
    const double mult = static_cast<double>(bias.params().scale) / static_cast<double>(in_scale * w_scale);
    engine::quantize_multiplier_and_shift(mult, m, shift);
}

void write_web_params(const std::string &path, const std::vector<engine::QuantizedTensor> &q,
                      const Int8Pipeline &p, double int8_acc, double agreement) {
    int32_t bm1, bm2;
    int bs1, bs2;
    bias_multiplier(q[1], p.in.scale, q[0].params().scale, bm1, bs1);
    bias_multiplier(q[3], p.hidden.scale, q[2].params().scale, bm2, bs2);
    std::ofstream f(path);
    if (!f) {
        std::fprintf(stderr, "warning: could not write %s\n", path.c_str());
        return;
    }
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "// Calibrated INT8 parameters for the web demo. Generated by benchmarks/mnist/run_benchmark.cpp.\n"
                  "window.QUANT_PARAMS = {\n"
                  "  input:  { scale: %.9g, zeroPoint: %d },\n"
                  "  hidden: { scale: %.9g, zeroPoint: %d },\n"
                  "  output: { scale: %.9g, zeroPoint: %d },\n"
                  "  layer1: { multiplier: %d, shift: %d, biasMultiplier: %d, biasShift: %d },\n"
                  "  layer2: { multiplier: %d, shift: %d, biasMultiplier: %d, biasShift: %d },\n"
                  "  int8TestAccuracy: %.2f,\n"
                  "  agreementWithFp32: %.2f\n"
                  "};\n",
                  p.in.scale, p.in.zero_point, p.hidden.scale, p.hidden.zero_point, p.out.scale,
                  p.out.zero_point, p.m1, p.s1, bm1, bs1, p.m2, p.s2, bm2, bs2, int8_acc, agreement);
    f << buf;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const std::string qbin = argc > 1 ? argv[1] : "benchmarks/mnist/results/mnist_model.qbin";
        const auto q = engine::load_quantized_model(qbin);
        if (q.size() != 4) throw std::runtime_error("expected 4 tensors in " + qbin);
        const Fp32Model fp = load_fp32("benchmarks/mnist/data/fp32_weights.bin");
        const Images test = load_images("benchmarks/mnist/data/mnist_test.bin");
        const Images calib = load_images("benchmarks/mnist/data/mnist_calib.bin");

        std::printf("Model:        %s\n", qbin.c_str());
        std::printf("Weight zero points: W1 %d, W2 %d\n", q[0].params().zero_point, q[2].params().zero_point);
        std::printf("Test images:  %u   Calibration images (train set): %u\n\n", test.count, calib.count);

        engine::Tensor<float> hid(calib.count, 128), logit(calib.count, 10);
        calibrate(fp, calib, hid, logit);

        // Primary configuration, fixed in advance: min/max calibration.
        const auto hidden_mm = engine::calculate_quant_params_calibrated(hid, engine::CalibrationMethod::MinMax);
        const auto out_mm = engine::calculate_quant_params_calibrated(logit, engine::CalibrationMethod::MinMax);
        // Comparison: 99.99th-percentile clipping of the hidden layer.
        const auto hidden_pc =
            engine::calculate_quant_params_calibrated(hid, engine::CalibrationMethod::Percentile, 99.99f);

        std::vector<size_t> pred_fp(test.count), pred_q(test.count);
        const double acc_fp = accuracy_fp32(fp, test, &pred_fp);
        const double acc_wq = accuracy_fp32(dequantized(q), test);
        const Int8Pipeline p_mm = make_pipeline(q, hidden_mm, out_mm);
        const double acc_q = accuracy_int8(q, p_mm, test, &pred_q);
        const double acc_pc = accuracy_int8(q, make_pipeline(q, hidden_pc, out_mm), test);

        size_t agree = 0;
        for (size_t i = 0; i < test.count; ++i) agree += (pred_fp[i] == pred_q[i]);

        std::printf("Hidden activation scale %.6f (zero point %d), output scale %.6f (zero point %d)\n\n",
                    hidden_mm.scale, hidden_mm.zero_point, out_mm.scale, out_mm.zero_point);
        std::printf("%-46s %8s\n", "Configuration", "Accuracy");
        std::printf("%-46s %7.2f%%\n", "FP32 reference", acc_fp);
        std::printf("%-46s %7.2f%%\n", "INT8 weights, FP32 activations", acc_wq);
        std::printf("%-46s %7.2f%%\n", "INT8 end to end (min/max calibration)", acc_q);
        std::printf("%-46s %7.2f%%\n", "INT8 end to end (99.99th pct hidden clipping)", acc_pc);
        std::printf("\nINT8 vs FP32 top-1 agreement: %.2f%% (%zu of %u test images)\n",
                    100.0 * double(agree) / double(test.count), agree, test.count);

        write_web_params("web/js/quant_params.js", q, p_mm, acc_q,
                         100.0 * double(agree) / double(test.count));

        std::ofstream s("benchmarks/mnist/results/summary.txt");
        s << "model " << qbin << "\n"
          << "fp32_accuracy " << acc_fp << "\n"
          << "int8_weights_fp32_activations_accuracy " << acc_wq << "\n"
          << "int8_accuracy_minmax " << acc_q << "\n"
          << "int8_accuracy_pct99.99 " << acc_pc << "\n"
          << "agreement " << 100.0 * double(agree) / double(test.count) << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
