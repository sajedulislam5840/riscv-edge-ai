#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_probabilities(const engine::Tensor<float>& probs, const std::string* labels) {
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < probs.cols(); ++i) {
        std::cout << "  " << labels[i] << ": " << (probs(0, i) * 100.0f) << "%\n";
    }
}

int main() {
    print_separator("RISC-V Edge AI: 2-Layer Neural Network Demo");
    std::cout << "\nArchitecture: Input(4) → Dense(8) → ReLU → Dense(3) → Softmax\n";
    std::cout << "Task: Multi-class classification (3 classes)\n";
    std::cout << "Precision: FP32 Baseline (Phase 01)\n";

    // ========== Layer 1: Dense + ReLU ==========
    print_separator("Layer 1: Fully Connected (4 → 8) + ReLU");

    // Input: Single sample with 4 features
    std::cout << "\nInput Features (1x4):\n";
    engine::Tensor<float> input(1, 4, {0.5f, -1.2f, 0.8f, 2.1f});
    input.print();

    // Layer 1 Weights: 4x8
    std::cout << "\nLayer 1 Weights (4x8): [Initialized with small random-like values]\n";
    engine::Tensor<float> W1(4, 8, {
        0.12f,  0.45f, -0.33f,  0.21f,  0.67f, -0.15f,  0.89f, -0.42f,
       -0.25f,  0.71f,  0.09f, -0.58f,  0.34f,  0.81f, -0.19f,  0.52f,
        0.38f, -0.62f,  0.14f,  0.76f, -0.41f,  0.28f,  0.93f, -0.07f,
       -0.51f,  0.16f,  0.84f, -0.29f,  0.47f, -0.73f,  0.05f,  0.61f
    });
    std::cout << "  (Shape: 4 rows × 8 cols)\n";

    // Layer 1 Bias: 1x8
    engine::Tensor<float> b1(1, 8, {
        0.01f, -0.02f, 0.03f, -0.01f, 0.02f, 0.01f, -0.03f, 0.02f
    });

    // Forward pass Layer 1: Z1 = input * W1 + b1
    engine::Tensor<float> Z1(1, 8, 0.0f);
    engine::matmul_fp32(input, W1, Z1);
    engine::bias_add(Z1, b1);

    std::cout << "\nPre-Activation (Z1 = X·W1 + b1):\n";
    Z1.print();

    // Apply ReLU activation
    engine::relu_inplace(Z1);
    std::cout << "\nPost-Activation (A1 = ReLU(Z1)):\n";
    Z1.print();

    // ========== Layer 2: Dense + Softmax ==========
    print_separator("Layer 2: Fully Connected (8 → 3) + Softmax");

    // Layer 2 Weights: 8x3
    std::cout << "\nLayer 2 Weights (8x3): [Initialized for classification]\n";
    engine::Tensor<float> W2(8, 3, {
        0.31f, -0.44f,  0.22f,
       -0.17f,  0.58f,  0.09f,
        0.64f, -0.21f,  0.37f,
        0.13f,  0.79f, -0.46f,
       -0.52f,  0.08f,  0.91f,
        0.27f, -0.63f,  0.14f,
        0.48f,  0.35f, -0.71f,
       -0.09f,  0.56f,  0.19f
    });
    std::cout << "  (Shape: 8 rows × 3 cols)\n";

    // Layer 2 Bias: 1x3
    engine::Tensor<float> b2(1, 3, {0.1f, -0.05f, 0.15f});

    // Forward pass Layer 2: Z2 = A1 * W2 + b2
    engine::Tensor<float> Z2(1, 3, 0.0f);
    engine::matmul_fp32(Z1, W2, Z2);
    engine::bias_add(Z2, b2);

    std::cout << "\nLogits (Z2 = A1·W2 + b2):\n";
    Z2.print();

    // Apply Softmax to get class probabilities
    engine::softmax(Z2);
    std::cout << "\nClass Probabilities (Softmax(Z2)):\n";
    Z2.print();

    // ========== Results ==========
    print_separator("Classification Results");

    std::string class_labels[] = {"Class A (Setosa)", "Class B (Versicolor)", "Class C (Virginica)"};
    std::cout << "\nPredicted Probabilities:\n";
    print_probabilities(Z2, class_labels);

    // Find predicted class (argmax)
    size_t predicted_class = 0;
    float max_prob = Z2(0, 0);
    for (size_t i = 1; i < Z2.cols(); ++i) {
        if (Z2(0, i) > max_prob) {
            max_prob = Z2(0, i);
            predicted_class = i;
        }
    }

    std::cout << "\n>> PREDICTED CLASS: " << class_labels[predicted_class];
    std::cout << " (Confidence: " << std::fixed << std::setprecision(2)
              << (max_prob * 100.0f) << "%)\n";

    // ========== Performance Metrics ==========
    print_separator("Phase 01 Completion Status");

    std::cout << "\n✅ Tensor Data Structure (Contiguous Memory Layout)\n";
    std::cout << "✅ Matrix Multiplication (Naive GEMM Kernel)\n";
    std::cout << "✅ ReLU Activation (In-place & Out-of-place)\n";
    std::cout << "✅ Bias Addition (Broadcasting)\n";
    std::cout << "✅ Softmax Activation (Numerically Stable)\n";
    std::cout << "✅ 2-Layer MLP Forward Pass\n";
    std::cout << "✅ Multi-class Classification Demo\n";

    std::cout << "\n🎯 Phase 01 COMPLETE - Ready for Phase 02 (INT8 Quantization)\n";

    print_separator("End of Demo");
    std::cout << "\n";

    return 0;
}
