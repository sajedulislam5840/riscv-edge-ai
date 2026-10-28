#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

void test_softmax_single_row() {
    // Single row: raw logits
    engine::Tensor<float> X(1, 3, {1.0f, 2.0f, 3.0f});

    engine::softmax(X);

    // Expected probabilities (computed with numpy for reference)
    // exp(1) = 2.718, exp(2) = 7.389, exp(3) = 20.086
    // sum = 30.193
    // softmax = [0.09003057, 0.24472847, 0.66524096]

    assert(float_eq(X(0, 0), 0.09003f, 1e-4f));
    assert(float_eq(X(0, 1), 0.24473f, 1e-4f));
    assert(float_eq(X(0, 2), 0.66524f, 1e-4f));

    // Check probabilities sum to 1.0
    float sum = X(0, 0) + X(0, 1) + X(0, 2);
    assert(float_eq(sum, 1.0f, 1e-4f));

    std::cout << "  [PASS] Softmax Single Row\n";
}

void test_softmax_multi_row() {
    // Two rows (batch of 2)
    engine::Tensor<float> X(2, 3, {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f
    });

    engine::softmax(X);

    // Row 0: same as above
    assert(float_eq(X(0, 0), 0.09003f, 1e-4f));
    assert(float_eq(X(0, 1), 0.24473f, 1e-4f));
    assert(float_eq(X(0, 2), 0.66524f, 1e-4f));

    // Row 1: uniform logits should give uniform probabilities
    assert(float_eq(X(1, 0), 0.33333f, 1e-4f));
    assert(float_eq(X(1, 1), 0.33333f, 1e-4f));
    assert(float_eq(X(1, 2), 0.33333f, 1e-4f));

    // Check each row sums to 1.0
    float sum_row0 = X(0, 0) + X(0, 1) + X(0, 2);
    float sum_row1 = X(1, 0) + X(1, 1) + X(1, 2);
    assert(float_eq(sum_row0, 1.0f, 1e-4f));
    assert(float_eq(sum_row1, 1.0f, 1e-4f));

    std::cout << "  [PASS] Softmax Multi-Row (Batch Processing)\n";
}

void test_softmax_numerical_stability() {
    // Large values that could cause overflow without max subtraction
    engine::Tensor<float> X(1, 3, {1000.0f, 1001.0f, 1002.0f});

    engine::softmax(X);

    // Should still work and produce valid probabilities
    float sum = X(0, 0) + X(0, 1) + X(0, 2);
    assert(float_eq(sum, 1.0f, 1e-4f));

    // Max value should get highest probability
    assert(X(0, 2) > X(0, 1));
    assert(X(0, 1) > X(0, 0));

    std::cout << "  [PASS] Softmax Numerical Stability (Large Values)\n";
}

void test_softmax_uniform_distribution() {
    // All equal logits should produce uniform distribution
    engine::Tensor<float> X(1, 4, {5.0f, 5.0f, 5.0f, 5.0f});

    engine::softmax(X);

    assert(float_eq(X(0, 0), 0.25f, 1e-4f));
    assert(float_eq(X(0, 1), 0.25f, 1e-4f));
    assert(float_eq(X(0, 2), 0.25f, 1e-4f));
    assert(float_eq(X(0, 3), 0.25f, 1e-4f));

    std::cout << "  [PASS] Softmax Uniform Distribution\n";
}

int main() {
    std::cout << "[TEST SUITE] Running Day 6 Softmax Activation Tests...\n";
    test_softmax_single_row();
    test_softmax_multi_row();
    test_softmax_numerical_stability();
    test_softmax_uniform_distribution();
    std::cout << "[TEST SUITE] All softmax tests passed successfully!\n";
    return 0;
}
