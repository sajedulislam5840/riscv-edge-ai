#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_bias_add_row_vector() {
    // Matrix: 2x3
    engine::Tensor<float> X(2, 3, {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });

    // Bias: 1x3
    engine::Tensor<float> bias(1, 3, {0.1f, 0.2f, 0.3f});

    engine::bias_add(X, bias);

    // Expected: each row gets bias added
    assert(float_eq(X(0, 0), 1.1f));
    assert(float_eq(X(0, 1), 2.2f));
    assert(float_eq(X(0, 2), 3.3f));
    assert(float_eq(X(1, 0), 4.1f));
    assert(float_eq(X(1, 1), 5.2f));
    assert(float_eq(X(1, 2), 6.3f));

    std::cout << "  [PASS] Bias Add with Row Vector (1xN)\n";
}

void test_bias_add_broadcast() {
    engine::Tensor<float> X(3, 2, {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    });

    engine::Tensor<float> bias(1, 2, {10.0f, 20.0f});

    engine::bias_add(X, bias);

    assert(float_eq(X(0, 0), 11.0f));
    assert(float_eq(X(0, 1), 22.0f));
    assert(float_eq(X(1, 0), 13.0f));
    assert(float_eq(X(1, 1), 24.0f));
    assert(float_eq(X(2, 0), 15.0f));
    assert(float_eq(X(2, 1), 26.0f));

    std::cout << "  [PASS] Bias Add Broadcasting\n";
}

void test_bias_add_dimension_mismatch() {
    engine::Tensor<float> X(2, 3, 0.0f);
    engine::Tensor<float> bias(1, 5, 0.0f);  // Wrong size

    bool caught = false;
    try {
        engine::bias_add(X, bias);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Expected std::invalid_argument on dimension mismatch");
    std::cout << "  [PASS] Bias Add Dimension Mismatch Exception\n";
}

int main() {
    std::cout << "[TEST SUITE] Running Day 6 Bias Addition Tests...\n";
    test_bias_add_row_vector();
    test_bias_add_broadcast();
    test_bias_add_dimension_mismatch();
    std::cout << "[TEST SUITE] All bias_add tests passed successfully!\n";
    return 0;
}
