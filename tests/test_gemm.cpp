#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_identity_multiplication() {
    engine::Tensor<float> A(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    engine::Tensor<float> I(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
    engine::Tensor<float> C(2, 2);

    engine::matmul_fp32(A, I, C);

    assert(float_eq(C(0, 0), 1.0f));
    assert(float_eq(C(0, 1), 2.0f));
    assert(float_eq(C(1, 0), 3.0f));
    assert(float_eq(C(1, 1), 4.0f));
    std::cout << "[PASS] Identity Matrix Multiplication\n";
}

void test_rectangular_gemm() {
    // A: 2x3
    // [1, 2, 3]
    // [4, 5, 6]
    engine::Tensor<float> A(2, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // B: 3x2
    // [7,  8]
    // [9,  1]
    // [2,  3]
    engine::Tensor<float> B(3, 2, {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f});

    engine::Tensor<float> C(2, 2);
    engine::matmul_fp32(A, B, C);

    // C(0,0) = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    // C(0,1) = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    // C(1,0) = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    // C(1,1) = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    assert(float_eq(C(0, 0), 31.0f));
    assert(float_eq(C(0, 1), 19.0f));
    assert(float_eq(C(1, 0), 85.0f));
    assert(float_eq(C(1, 1), 55.0f));
    std::cout << "[PASS] Rectangular Matrix Multiplication (2x3 * 3x2)\n";
}

void test_dimension_mismatch() {
    engine::Tensor<float> A(2, 3);
    engine::Tensor<float> B(4, 2);
    engine::Tensor<float> C(2, 2);

    bool caught = false;
    try {
        engine::matmul_fp32(A, B, C);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught && "Expected std::invalid_argument on mismatched inner dimensions");
    std::cout << "[PASS] Dimension Mismatch Exception Handling\n";
}

int main() {
    std::cout << "--- Running Day 3 & 4 GEMM Verification Suite ---\n";
    test_identity_multiplication();
    test_rectangular_gemm();
    test_dimension_mismatch();
    std::cout << "All GEMM tests completed successfully!\n";
    return 0;
}
