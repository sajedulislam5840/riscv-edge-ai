#include "engine/ops.hpp"

namespace engine {

void matmul_fp32(const Tensor<float>& A, const Tensor<float>& B, Tensor<float>& C) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimension mismatch: A.cols must equal B.rows.");
    }

    size_t M = A.rows();
    size_t K = A.cols();
    size_t N = B.cols();

    if (C.rows() != M || C.cols() != N) {
        throw std::invalid_argument("Output matrix C has incorrect dimensions.");
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

} // namespace engine
