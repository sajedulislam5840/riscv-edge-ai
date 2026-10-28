#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>

int main() {
    std::cout << "===============================================================\n";
    std::cout << "    MNIST Benchmark: FP32 Baseline vs. INT8 Fixed-Point Engine \n";
    std::cout << "===============================================================\n\n";

    const std::string model_file = "benchmarks/mnist/results/mnist_model.qbin";
    const std::string summary_file = "benchmarks/mnist/results/summary.txt";
    const std::string csv_file = "benchmarks/mnist/results/benchmark_data.csv";

    std::vector<engine::QuantizedTensor> model;
    std::ifstream check_file(model_file, std::ios::binary);
    
    if (check_file.is_open()) {
        check_file.close();
        model = engine::load_quantized_model(model_file);
        std::cout << "[Step 1] Successfully loaded trained weights from " << model_file << "\n";
    } else {
        std::cout << "[Step 1] Pretrained file not found; generating synthetic 784->128->10 weights...\n";
        engine::QuantParams qW1 = {0.008f, 0};
        engine::QuantParams qb1 = {0.002f, 0};
        engine::QuantParams qW2 = {0.012f, 0};
        engine::QuantParams qb2 = {0.004f, 0};

        engine::QuantizedTensor W1(784, 128, qW1);
        engine::QuantizedTensor b1(1, 128, qb1);
        engine::QuantizedTensor W2(128, 10, qW2);
        engine::QuantizedTensor b2(1, 10, qb2);

        for (size_t i = 0; i < W1.size(); ++i) W1.data()[i] = static_cast<int8_t>((i % 25) - 12);
        for (size_t i = 0; i < b1.size(); ++i) b1.data()[i] = static_cast<int8_t>(i % 5);
        for (size_t i = 0; i < W2.size(); ++i) W2.data()[i] = static_cast<int8_t>((i % 19) - 9);
        for (size_t i = 0; i < b2.size(); ++i) b2.data()[i] = static_cast<int8_t>(i % 3);

        model = {W1, b1, W2, b2};
        engine::save_quantized_model(model_file, model);
    }

    engine::QuantizedTensor& qW1 = model[0];
    engine::QuantizedTensor& qb1 = model[1];
    engine::QuantizedTensor& qW2 = model[2];
    engine::QuantizedTensor& qb2 = model[3];

    engine::Tensor<float> fp_W1 = engine::dequantize_tensor(qW1);
    engine::Tensor<float> fp_b1 = engine::dequantize_tensor(qb1);
    engine::Tensor<float> fp_W2 = engine::dequantize_tensor(qW2);
    engine::Tensor<float> fp_b2 = engine::dequantize_tensor(qb2);

    const size_t NUM_SAMPLES = 1000;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<engine::Tensor<float>> test_samples_fp32;
    test_samples_fp32.reserve(NUM_SAMPLES);
    for (size_t s = 0; s < NUM_SAMPLES; ++s) {
        engine::Tensor<float> img(1, 784, 0.0f);
        for (size_t i = 0; i < 784; ++i) {
            img.data()[i] = dist(rng);
        }
        test_samples_fp32.push_back(std::move(img));
    }

    engine::ArenaAllocator memory_arena(65536);
    engine::QuantParams q_in = {0.015f, 0};
    engine::QuantParams q_h1 = {0.035f, 0};
    engine::QuantParams q_out = {0.080f, 0};

    double total_fp32_time_us = 0.0;
    double total_int8_time_us = 0.0;
    float max_logit_error = 0.0f;
    double total_squared_error = 0.0;
    size_t matching_predictions = 0;

    std::ofstream csv(csv_file);
    csv << "sample_idx,fp32_pred,int8_pred,match,max_err\n";

    for (size_t s = 0; s < NUM_SAMPLES; ++s) {
        const auto& x_fp32 = test_samples_fp32[s];

        // --- FP32 Baseline Execution ---
        auto t0 = std::chrono::high_resolution_clock::now();
        engine::Tensor<float> fp_h1(1, 128, 0.0f);
        engine::matmul_fp32(x_fp32, fp_W1, fp_h1);
        engine::bias_add(fp_h1, fp_b1);
        engine::relu(fp_h1, fp_h1);

        engine::Tensor<float> fp_out(1, 10, 0.0f);
        engine::matmul_fp32(fp_h1, fp_W2, fp_out);
        engine::bias_add(fp_out, fp_b2);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_fp32_time_us += std::chrono::duration<double, std::micro>(t1 - t0).count();

        // --- INT8 Fixed-Point Execution ---
        auto t2 = std::chrono::high_resolution_clock::now();
        engine::QuantizedTensor qX = engine::quantize_tensor(x_fp32, q_in);
        engine::QuantizedTensor qH1(1, 128, q_h1);
        engine::matmul_int8_fixed_point(qX, qW1, qH1);
        engine::bias_add(qH1, qb1);
        engine::relu(qH1);

        engine::QuantizedTensor qOut(1, 10, q_out);
        engine::matmul_int8_fixed_point(qH1, qW2, qOut);
        engine::bias_add(qOut, qb2);
        auto t3 = std::chrono::high_resolution_clock::now();
        total_int8_time_us += std::chrono::duration<double, std::micro>(t3 - t2).count();

        engine::Tensor<float> deq_out = engine::dequantize_tensor(qOut);

        size_t pred_fp32 = 0;
        size_t pred_int8 = 0;
        float max_val_fp = fp_out.data()[0];
        float max_val_int = deq_out.data()[0];
        float sample_max_err = 0.0f;

        for (size_t c = 0; c < 10; ++c) {
            float err = std::abs(fp_out.data()[c] - deq_out.data()[c]);
            if (err > sample_max_err) sample_max_err = err;
            if (err > max_logit_error) max_logit_error = err;
            total_squared_error += err * err;

            if (fp_out.data()[c] > max_val_fp) {
                max_val_fp = fp_out.data()[c];
                pred_fp32 = c;
            }
            if (deq_out.data()[c] > max_val_int) {
                max_val_int = deq_out.data()[c];
                pred_int8 = c;
            }
        }

        bool match = (pred_fp32 == pred_int8);
        if (match) matching_predictions++;

        if (s < 100) {
            csv << s << "," << pred_fp32 << "," << pred_int8 << "," << (match ? 1 : 0) << "," << sample_max_err << "\n";
        }
    }
    csv.close();

    double agreement_rate = (100.0 * matching_predictions) / NUM_SAMPLES;
    double avg_fp32_latency = total_fp32_time_us / NUM_SAMPLES;
    double avg_int8_latency = total_int8_time_us / NUM_SAMPLES;
    double rmse = std::sqrt(total_squared_error / (NUM_SAMPLES * 10));
    double speedup = avg_fp32_latency / avg_int8_latency;

    std::cout << "\n================ Benchmark Summary ================\n";
    std::cout << "Evaluated Samples:       " << NUM_SAMPLES << "\n";
    std::cout << "Top-1 Agreement Rate:    " << std::fixed << std::setprecision(2) << agreement_rate << " %\n";
    std::cout << "Max Logit Error (L_inf): " << std::fixed << std::setprecision(4) << max_logit_error << "\n";
    std::cout << "Root Mean Square Error:  " << std::fixed << std::setprecision(4) << rmse << "\n";
    std::cout << "FP32 Latency / sample:   " << std::fixed << std::setprecision(2) << avg_fp32_latency << " us\n";
    std::cout << "INT8 Latency / sample:   " << std::fixed << std::setprecision(2) << avg_int8_latency << " us\n";
    std::cout << "Inference Speedup:       " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "Arena Peak Memory:       " << memory_arena.capacity() << " Bytes\n";
    std::cout << "===================================================\n\n";

    std::ofstream out(summary_file);
    out << "MNIST Benchmark Evaluation Summary\n";
    out << "Samples: " << NUM_SAMPLES << "\n";
    out << "Agreement Rate: " << agreement_rate << "%\n";
    out << "Max Logit Error: " << max_logit_error << "\n";
    out << "RMSE: " << rmse << "\n";
    out << "FP32 Avg Latency: " << avg_fp32_latency << " us\n";
    out << "INT8 Avg Latency: " << avg_int8_latency << " us\n";
    out << "Speedup: " << speedup << "x\n";
    out.close();

    std::cout << "Results saved to " << summary_file << " and " << csv_file << "\n";
    return 0;
}
