#include "engine/ops.hpp"
#include <algorithm>
#include <cmath>

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

void bias_add(Tensor<float>& X, const Tensor<float>& bias) {
    // X: (M x N), bias: (1 x N) or (N,)
    size_t M = X.rows();
    size_t N = X.cols();

    // Support both (1, N) and (N, 1) bias shapes
    bool valid = (bias.rows() == 1 && bias.cols() == N) ||
                 (bias.rows() == N && bias.cols() == 1) ||
                 (bias.rows() == 1 && bias.cols() == 1 && bias.size() == N);

    if (!valid) {
        throw std::invalid_argument("Bias dimensions must be (1 x N) or match columns of X.");
    }

    float* x_data = X.data();
    const float* b_data = bias.data();

    // Broadcast bias across all rows
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            x_data[i * N + j] += b_data[j];
        }
    }
}

void softmax(Tensor<float>& X) {
    // Row-wise softmax with numerical stability (max subtraction trick)
    size_t M = X.rows();
    size_t N = X.cols();
    float* data = X.data();

    for (size_t i = 0; i < M; ++i) {
        float* row = data + i * N;

        // Find max value in the row for numerical stability
        float max_val = row[0];
        for (size_t j = 1; j < N; ++j) {
            max_val = std::max(max_val, row[j]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        // Normalize
        for (size_t j = 0; j < N; ++j) {
            row[j] /= sum;
        }
    }
}

// INT8 matrix multiplication with 32-bit accumulator
void matmul_int8(
    const int8_t* A, size_t M, size_t K,
    const int8_t* B, size_t N,
    int32_t* C
) {
    // C[i,j] = sum(A[i,k] * B[k,j]) for k in [0, K)
    // Use INT32 accumulator to prevent overflow
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                // Multiply two INT8 values, result fits in INT32
                int32_t a_val = static_cast<int32_t>(A[i * K + k]);
                int32_t b_val = static_cast<int32_t>(B[k * N + j]);
                sum += a_val * b_val;
            }
            C[i * N + j] = sum;
        }
    }
}

} // namespace engine
