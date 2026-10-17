#ifndef ENGINE_OPS_HPP
#define ENGINE_OPS_HPP

#include "engine/tensor.hpp"
#include <stdexcept>

namespace engine {

// C = A * B
// A: (M x K), B: (K x N), C: (M x N)
void matmul_fp32(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C);

} // namespace engine

#endif // ENGINE_OPS_HPP
