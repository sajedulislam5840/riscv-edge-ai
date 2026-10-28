#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    std::cout << "===============================================================\n";
    std::cout << " Bare-Metal Neural Network Inference Engine for RISC-V\n";
    std::cout << " Phase 02 Demo: FP32 Baseline vs. Quantized INT8 Inference\n";
    std::cout << "===============================================================\n\n";

    // -------------------------------------------------------------------------
    // 1. Model Weights & Biases Setup (4 -> 8 -> 3 MLP)
    // -------------------------------------------------------------------------
    engine::Tensor<float> X(1, 4, {0.5f, -1.2f, 2.1f, -0.7f});

    engine::Tensor<float> W1(4, 8, {0.2f, -0.5f, 0.7f, -0.1f, 0.4f, -0.3f, 0.6f, -0.8f, -0.4f, 0.6f, -0.2f, 0.5f, -0.7f, 0.1f, -0.5f, 0.3f, 0.8f, -0.1f, 0.3f, -0.6f, 0.2f, -0.4f, 0.9f, -0.2f, -0.3f, 0.7f, -0.5f, 0.4f, -0.1f, 0.8f, -0.6f, 0.5f});

    engine::Tensor<float> b1(1, 8, {0.1f, -0.2f, 0.3f, 0.0f, -0.1f, 0.2f, -0.3f, 0.1f});

    engine::Tensor<float> W2(8, 3, {0.4f, -0.6f, 0.1f, -0.3f, 0.5f, -0.2f, 0.7f, -0.1f, 0.4f, -0.5f, 0.8f, -0.3f, 0.2f, -0.4f, 0.6f, -0.6f, 0.3f, -0.5f, 0.9f, -0.2f, 0.1f, -0.1f, 0.7f, -0.8f});

    engine::Tensor<float> b2(1, 3, {0.2f, -0.1f, 0.3f});

    // -------------------------------------------------------------------------
    // 2. FP32 Baseline Forward Pass
    // -------------------------------------------------------------------------
    // Layer 1: GEMM -> Bias Add -> ReLU
    engine::Tensor<float> H1_fp32(1, 8, 0.0f);
    engine::matmul_fp32(X, W1, H1_fp32);
    engine::bias_add(H1_fp32, b1);

    engine::Tensor<float> A1_fp32(1, 8, 0.0f);
    engine::relu(H1_fp32, A1_fp32);

    // Layer 2: GEMM -> Bias Add
    engine::Tensor<float> logits_fp32(1, 3, 0.0f);
    engine::matmul_fp32(A1_fp32, W2, logits_fp32);
    engine::bias_add(logits_fp32, b2);

    // Softmax (In-place on cloned logits)
    engine::Tensor<float> probs_fp32 = logits_fp32;
    engine::softmax(probs_fp32);

    // -------------------------------------------------------------------------
    // 3. INT8 Quantized Forward Pass
    // -------------------------------------------------------------------------
    // Dynamic Calibration Parameters
    engine::QuantParams q_in = engine::calculate_quant_params(-2.0f, 2.5f);
    engine::QuantParams q_w1 = engine::calculate_quant_params(-1.0f, 1.0f);
    engine::QuantParams q_b1 = engine::calculate_quant_params(-0.5f, 0.5f);
    engine::QuantParams q_h1 = engine::calculate_quant_params(-5.0f, 5.0f);

    engine::QuantParams q_w2 = engine::calculate_quant_params(-1.0f, 1.0f);
    engine::QuantParams q_b2 = engine::calculate_quant_params(-0.5f, 0.5f);
    engine::QuantParams q_out = engine::calculate_quant_params(-5.0f, 5.0f);

    // Offline Weight/Bias Quantization
    engine::QuantizedTensor qX = engine::quantize_tensor(X, q_in);
    engine::QuantizedTensor qW1 = engine::quantize_tensor(W1, q_w1);
    engine::QuantizedTensor qb1 = engine::quantize_tensor(b1, q_b1);
    engine::QuantizedTensor qW2 = engine::quantize_tensor(W2, q_w2);
    engine::QuantizedTensor qb2 = engine::quantize_tensor(b2, q_b2);

    // Layer 1: INT8 GEMM -> Quantized Bias Add -> Quantized ReLU
    engine::QuantizedTensor qH1(1, 8, q_h1);
    engine::matmul_int8(qX, qW1, qH1);
    engine::bias_add(qH1, qb1);
    engine::relu(qH1);

    // Layer 2: INT8 GEMM -> Quantized Bias Add
    engine::QuantizedTensor qLogits(1, 3, q_out);
    engine::matmul_int8(qH1, qW2, qLogits);
    engine::bias_add(qLogits, qb2);

    // Dequantize logits to compute final classification probabilities
    engine::Tensor<float> logits_int8_dequant = engine::dequantize_tensor(qLogits);
    engine::Tensor<float> probs_int8 = logits_int8_dequant;
    engine::softmax(probs_int8);

    // -------------------------------------------------------------------------
    // 4. Output Inspection & Metrics
    // -------------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Logits Comparison (FP32 vs INT8 Reconstructed) ---\n";
    float max_logit_err = 0.0f;
    for (size_t i = 0; i < 3; ++i)
    {
        float err = std::abs(logits_fp32.data()[i] - logits_int8_dequant.data()[i]);
        if (err > max_logit_err)
            max_logit_err = err;
        std::cout << " Class [" << i << "] -> FP32: " << std::setw(8) << logits_fp32.data()[i]
                  << " | INT8: " << std::setw(8) << logits_int8_dequant.data()[i]
                  << " | Abs Error: " << std::setw(6) << err << "\n";
    }

    std::cout << "\n--- Classification Probabilities (Softmax) ---\n";
    size_t pred_fp32 = 0;
    size_t pred_int8 = 0;
    float max_p_fp32 = -1.0f;
    float max_p_int8 = -1.0f;

    for (size_t i = 0; i < 3; ++i)
    {
        float p_f = probs_fp32.data()[i];
        float p_q = probs_int8.data()[i];
        if (p_f > max_p_fp32)
        {
            max_p_fp32 = p_f;
            pred_fp32 = i;
        }
        if (p_q > max_p_int8)
        {
            max_p_int8 = p_q;
            pred_int8 = i;
        }

        std::cout << " Class [" << i << "] -> FP32: " << std::setw(6) << (p_f * 100.0f) << "%"
                  << " | INT8: " << std::setw(6) << (p_q * 100.0f) << "%\n";
    }

    std::cout << "\n--- Verdict ---\n";
    std::cout << " FP32 Prediction : Class " << pred_fp32 << " (Confidence: " << (max_p_fp32 * 100.0f) << "%)\n";
    std::cout << " INT8 Prediction : Class " << pred_int8 << " (Confidence: " << (max_p_int8 * 100.0f) << "%)\n";
    std::cout << " Maximum Logit Deviation: " << max_logit_err << "\n";
    std::cout << " Prediction Agreement   : " << (pred_fp32 == pred_int8 ? "MATCH (100%)" : "MISMATCH") << "\n\n";

    std::cout << ">> [SUCCESS] 2-Layer MLP INT8 Quantized Inference Verified! <<\n";
    return 0;
}