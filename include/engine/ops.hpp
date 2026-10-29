#ifndef ENGINE_OPS_HPP
#define ENGINE_OPS_HPP

#include "engine/tensor.hpp"
#include <stdexcept>
#include <cstdint>

namespace engine {

// C = A * B
// A: (M x K), B: (K x N), C: (M x N)
void matmul_fp32(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);

// Out-of-place ReLU: Y = max(0, X)
void relu(const Tensor<float>& X, Tensor<float>& Y);

// In-place ReLU: X = max(0, X)
void relu_inplace(Tensor<float>& X);

// Bias addition: X = X + b (broadcast bias vector across rows)
// X: (M x N), b: (1 x N) or (N,), modifies X in-place
void bias_add(Tensor<float>& X, const Tensor<float>& bias);

// Softmax activation (in-place, row-wise): X = exp(X) / sum(exp(X))
// Numerically stable with max subtraction trick
void softmax(Tensor<float>& X);

// Symmetric INT8 matrix multiplication with 32-bit accumulator (Z_A = Z_B = 0)
// Target kernel for custom RISC-V 4-way SIMD MAC instruction
void matmul_int8(
    const int8_t* A, size_t M, size_t K,
    const int8_t* B, size_t N,
    int32_t* C
);

// Asymmetric INT8 matrix multiplication with zero-point offsets
void matmul_int8_asymmetric(
    const int8_t* A, size_t M, size_t K, int32_t zp_A,
    const int8_t* B, size_t N, int32_t zp_B,
    int32_t* C
);

} // namespace engine

#endif // ENGINE_OPS_HPP
