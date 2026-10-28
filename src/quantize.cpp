#include "engine/quantize.hpp"

namespace engine
{

    QuantParams calculate_quant_params(float min_val, float max_val)
    {
        if (min_val > max_val)
        {
            std::swap(min_val, max_val);
        }

        // Ensure range includes zero to preserve zero-point representation
        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

        // Edge case: constant or near-zero range
        if (std::abs(max_val - min_val) < 1e-7f)
        {
            return {1.0f, 0};
        }

        float scale = (max_val - min_val) / 255.0f;
        float initial_zp = -128.0f - (min_val / scale);

        int32_t zero_point = static_cast<int32_t>(std::round(initial_zp));
        zero_point = std::clamp(zero_point, -128, 127);

        return {scale, zero_point};
    }

    int8_t quantize_val(float val, const QuantParams &params)
    {
        if (params.scale <= 0.0f)
        {
            throw std::invalid_argument("Scale must be positive in quantize_val.");
        }
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
        if (input.empty())
        {
            return output;
        }

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
        if (input.empty())
        {
            return output;
        }

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

    void matmul_int8(const QuantizedTensor &A, const QuantizedTensor &B, QuantizedTensor &C)
    {
        if (A.empty() || B.empty())
        {
            throw std::invalid_argument("Cannot perform matmul_int8 on empty tensor.");
        }
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8: A.cols must match B.rows.");
        }
        if (C.rows() != A.rows() || C.cols() != B.cols())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8: Output tensor C dimension invalid.");
        }

        const size_t M = A.rows();
        const size_t K = A.cols();
        const size_t N = B.cols();

        const int32_t zp_A = A.params().zero_point;
        const int32_t zp_B = B.params().zero_point;
        const int32_t zp_C = C.params().zero_point;

        const float effective_scale = (A.params().scale * B.params().scale) / C.params().scale;

        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                int32_t acc = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    int32_t val_A = static_cast<int32_t>(A(i, k)) - zp_A;
                    int32_t val_B = static_cast<int32_t>(B(k, j)) - zp_B;
                    acc += val_A * val_B;
                }

                float scaled_acc = static_cast<float>(acc) * effective_scale;
                int32_t res = static_cast<int32_t>(std::round(scaled_acc)) + zp_C;
                C(i, j) = static_cast<int8_t>(std::clamp(res, -128, 127));
            }
        }
    }

    void bias_add(QuantizedTensor &X, const QuantizedTensor &bias)
    {
        if (X.empty())
        {
            return;
        }
        if (bias.rows() != 1 || bias.cols() != X.cols())
        {
            throw std::invalid_argument("Dimension mismatch in quantized bias_add: bias must be 1 x X.cols().");
        }

        const size_t rows = X.rows();
        const size_t cols = X.cols();

        const float S_x = X.params().scale;
        const int32_t Z_x = X.params().zero_point;
        const float S_b = bias.params().scale;
        const int32_t Z_b = bias.params().zero_point;

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                float real_x = (static_cast<float>(X(r, c)) - Z_x) * S_x;
                float real_b = (static_cast<float>(bias(0, c)) - Z_b) * S_b;
                float sum = real_x + real_b;

                int32_t q_res = static_cast<int32_t>(std::round(sum / S_x)) + Z_x;
                X(r, c) = static_cast<int8_t>(std::clamp(q_res, -128, 127));
            }
        }
    }

    void relu(QuantizedTensor &X)
    {
        if (X.empty())
        {
            return;
        }
        int8_t *ptr = X.data();
        const size_t total = X.size();
        const int8_t zp = static_cast<int8_t>(X.params().zero_point);

        for (size_t i = 0; i < total; ++i)
        {
            if (ptr[i] < zp)
            {
                ptr[i] = zp;
            }
        }
    }

    void relu(const QuantizedTensor &input, QuantizedTensor &output)
    {
        if (input.rows() != output.rows() || input.cols() != output.cols())
        {
            throw std::invalid_argument("Dimension mismatch in out-of-place quantized relu.");
        }
        if (input.empty())
        {
            return;
        }

        output.set_params(input.params());
        const int8_t *in_ptr = input.data();
        int8_t *out_ptr = output.data();
        const size_t total = input.size();
        const int8_t zp = static_cast<int8_t>(input.params().zero_point);

        for (size_t i = 0; i < total; ++i)
        {
            out_ptr[i] = (in_ptr[i] < zp) ? zp : in_ptr[i];
        }
    }

} // namespace engine