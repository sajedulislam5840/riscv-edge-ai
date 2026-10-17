#include "engine/ops.hpp"
#include <algorithm>

namespace engine {

void matmul_fp32(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols must equal B.rows.");
    }

    size_t M = A.rows();
    size_t K = A.cols();
    size_t N = B.cols();

    if (C.rows() != M || C.cols() != N) {
        throw std::invalid_argument("Output matrix C dimensions mismatch.");
    }

    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }
}

void relu(const Tensor<float>& X, Tensor<float>& Y) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) {
        throw std::invalid_argument("Dimension mismatch between input X and output Y for ReLU.");
    }

    const float* in = X.data();
    float* out = Y.data();
    size_t total = X.size();

    for (size_t i = 0; i < total; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

void relu_inplace(Tensor<float>& X) {
    float* data = X.data();
    size_t total = X.size();

    for (size_t i = 0; i < total; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

} // namespace engine
