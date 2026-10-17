#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_out_of_place_relu() {
    engine::Tensor<float> X(2, 3, {-2.5f, 0.0f, 3.14f, -0.01f, 10.0f, -100.0f});
    engine::Tensor<float> Y(2, 3, 0.0f);

    engine::relu(X, Y);

    assert(float_eq(Y(0, 0), 0.0f));
    assert(float_eq(Y(0, 1), 0.0f));
    assert(float_eq(Y(0, 2), 3.14f));
    assert(float_eq(Y(1, 0), 0.0f));
    assert(float_eq(Y(1, 1), 10.0f));
    assert(float_eq(Y(1, 2), 0.0f));

    // Verify input tensor X remained untouched
    assert(float_eq(X(0, 0), -2.5f));
    assert(float_eq(X(1, 2), -100.0f));

    std::cout << "  [PASS] Out-of-Place ReLU\n";
}

void test_in_place_relu() {
    engine::Tensor<float> X(2, 2, {-1.0f, 5.0f, -0.0f, -42.0f});

    engine::relu_inplace(X);

    assert(float_eq(X(0, 0), 0.0f));
    assert(float_eq(X(0, 1), 5.0f));
    assert(float_eq(X(1, 0), 0.0f));
    assert(float_eq(X(1, 1), 0.0f));

    std::cout << "  [PASS] In-Place ReLU\n";
}

int main() {
    std::cout << "[TEST SUITE] Running Day 5 ReLU Activation Tests...\n";
    test_out_of_place_relu();
    test_in_place_relu();
    std::cout << "[TEST SUITE] All ReLU tests passed successfully!\n";
    return 0;
}
