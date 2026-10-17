#ifndef ENGINE_OPS_HPP
#define ENGINE_OPS_HPP

#include "engine/tensor.hpp"
#include <stdexcept>

namespace engine {

// C = A * B
// A: (M x K), B: (K x N), C: (M x N)
void matmul_fp32(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);

// Out-of-place ReLU: Y = max(0, X)
void relu(const Tensor<float>& X, Tensor<float>& Y);

// In-place ReLU: X = max(0, X)
void relu_inplace(Tensor<float>& X);

} // namespace engine

#endif // ENGINE_OPS_HPP
