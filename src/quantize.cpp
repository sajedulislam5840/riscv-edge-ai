#include "engine/quantize.hpp"

namespace engine
{

    QuantParams calculate_quant_params(float min_val, float max_val)
    {
        if (min_val > max_val)
        {
            std::swap(min_val, max_val);
        }

        // Ensure range includes zero
        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

        if (std::abs(max_val - min_val) < 1e-7f)
        {
            return {1.0f, 0};
        }

        // Map [min_val, max_val] to signed INT8 range [-128, 127]
        float scale = (max_val - min_val) / 255.0f;
        float initial_zp = -128.0f - (min_val / scale);

        int32_t zero_point = static_cast<int32_t>(std::round(initial_zp));
        zero_point = std::clamp(zero_point, -128, 127);

        return {scale, zero_point};
    }

    int8_t quantize_val(float val, const QuantParams &params)
    {
        float scaled = val / params.scale;
        float shifted = scaled + static_cast<float>(params.zero_point);
        int32_t clamped = std::clamp(static_cast<int32_t>(std::round(shifted)), -128, 127);
        return static_cast<int8_t>(clamped);
    }

    float dequantize_val(int8_t val, const QuantParams &params)
    {
        return (static_cast<float>(val) - static_cast<float>(params.zero_point)) * params.scale;
    }

    QuantizedTensor quantize_tensor(const Tensor<float> &input, const QuantParams &params)
    {
        QuantizedTensor output(input.rows(), input.cols(), params);
        const float *in_data = input.data();
        int8_t *out_data = output.data();
        size_t total = input.size();

        for (size_t i = 0; i < total; ++i)
        {
            out_data[i] = quantize_val(in_data[i], params);
        }
        return output;
    }

    Tensor<float> dequantize_tensor(const QuantizedTensor &input)
    {
        Tensor<float> output(input.rows(), input.cols(), 0.0f);
        const int8_t *in_data = input.data();
        float *out_data = output.data();
        size_t total = input.size();
        const QuantParams &params = input.params();

        for (size_t i = 0; i < total; ++i)
        {
            out_data[i] = dequantize_val(in_data[i], params);
        }
        return output;
    }

    // Quantized matrix multiplication with 32-bit accumulator
    void matmul_int8(
        const QuantizedTensor &A,
        const QuantizedTensor &B,
        QuantizedTensor &C)
    {
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Matrix dimension mismatch for INT8 GEMM");
        }

        size_t M = A.rows();
        size_t K = A.cols();
        size_t N = B.cols();

        if (C.rows() != M || C.cols() != N)
        {
            throw std::invalid_argument("Output matrix C has incorrect dimensions");
        }

        const int8_t *a_data = A.data();
        const int8_t *b_data = B.data();
        int8_t *c_data = C.data();

        const QuantParams &a_params = A.params();
        const QuantParams &b_params = B.params();
        const QuantParams &c_params = C.params();

        // Compute scale factors for requantization
        float output_scale = (a_params.scale * b_params.scale) / c_params.scale;

        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                // Use 32-bit accumulator to prevent overflow
                int32_t acc = 0;

                for (size_t k = 0; k < K; ++k)
                {
                    int8_t a_val = a_data[i * K + k];
                    int8_t b_val = b_data[k * N + j];

                    // Accumulate in INT32
                    acc += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                }

                // Requantize: apply scale and zero-point correction
                float acc_fp = static_cast<float>(acc) * output_scale;
                acc_fp += static_cast<float>(c_params.zero_point);

                // Clamp to INT8 range
                int32_t result = std::clamp(static_cast<int32_t>(std::round(acc_fp)), -128, 127);
                c_data[i * N + j] = static_cast<int8_t>(result);
            }
        }
    }

} // namespace engine