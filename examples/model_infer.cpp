#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main()
{
    std::cout << "===============================================================\n";
    std::cout << " Bare-Metal Neural Inference Engine: Pretrained Model Pipeline \n";
    std::cout << "===============================================================\n\n";

    const std::string model_path = "model_weights.qbin";

    // 1. Export Pretrained Quantized Model to Binary
    {
        engine::QuantParams qW1 = {0.015f, 0};
        engine::QuantParams qb1 = {0.005f, 0};
        engine::QuantParams qW2 = {0.020f, 0};
        engine::QuantParams qb2 = {0.008f, 0};

        engine::QuantizedTensor W1(4, 8, qW1);
        engine::QuantizedTensor b1(1, 8, qb1);
        engine::QuantizedTensor W2(8, 3, qW2);
        engine::QuantizedTensor b2(1, 3, qb2);

        for (size_t i = 0; i < W1.size(); ++i)
            W1.data()[i] = static_cast<int8_t>((i % 25) - 12);
        for (size_t i = 0; i < b1.size(); ++i)
            b1.data()[i] = static_cast<int8_t>(i * 2);
        for (size_t i = 0; i < W2.size(); ++i)
            W2.data()[i] = static_cast<int8_t>((i % 17) - 8);
        for (size_t i = 0; i < b2.size(); ++i)
            b2.data()[i] = static_cast<int8_t>(i - 1);

        engine::save_quantized_model(model_path, {W1, b1, W2, b2});
        std::cout << "[Step 1] Pretrained weights serialized to: " << model_path << "\n";
    }

    // 2. Load Binary Model into Engine
    std::vector<engine::QuantizedTensor> model = engine::load_quantized_model(model_path);
    std::cout << "[Step 2] Loaded " << model.size() << " tensors from disk.\n";

    engine::QuantizedTensor &W1 = model[0];
    engine::QuantizedTensor &b1 = model[1];
    engine::QuantizedTensor &W2 = model[2];
    engine::QuantizedTensor &b2 = model[3];

    // 3. Static Memory Arena Planning (Zero dynamic heap allocations during inference)
    engine::ArenaAllocator memory_arena(4096);
    std::cout << "[Step 3] Initialized static inference arena (" << memory_arena.capacity() << " bytes)\n";

    // 4. Run Model Forward Inference
    engine::QuantParams q_in = {0.025f, 0};
    engine::QuantParams q_h1 = {0.050f, 0};
    engine::QuantParams q_out = {0.100f, 0};

    engine::Tensor<float> raw_input(1, 4, {1.2f, -0.8f, 2.5f, -1.5f});
    engine::QuantizedTensor qX = engine::quantize_tensor(raw_input, q_in);

    // Layer 1: GEMM -> Bias -> LeakyReLU
    engine::QuantizedTensor qH1(1, 8, q_h1);
    engine::matmul_int8_fixed_point(qX, W1, qH1);
    engine::bias_add(qH1, b1);
    engine::leaky_relu(qH1, 0.1f);

    // Layer 2: GEMM -> Bias
    engine::QuantizedTensor qLogits(1, 3, q_out);
    engine::matmul_int8_fixed_point(qH1, W2, qLogits);
    engine::bias_add(qLogits, b2);

    // 5. Inspect Logits and Probabilities
    engine::Tensor<float> logits = engine::dequantize_tensor(qLogits);
    engine::Tensor<float> probs = logits;
    engine::softmax(probs);

    std::cout << "\n--- Classification Output ---\n";
    for (size_t c = 0; c < 3; ++c)
    {
        std::cout << " Class [" << c << "] -> Logit: " << std::fixed << std::setprecision(4)
                  << std::setw(8) << logits.data()[c]
                  << " | Probability: " << std::setw(6) << (probs.data()[c] * 100.0f) << "%\n";
    }

    std::cout << "\n>> [SUCCESS] Pretrained Model Inference Execution Complete! <<\n";
    return 0;
}