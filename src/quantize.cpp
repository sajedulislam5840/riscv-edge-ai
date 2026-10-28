#include "engine/quantize.hpp"

namespace engine
{

    QuantParams calculate_quant_params(float min_val, float max_val)
    {
        if (min_val > max_val)
        {
            std::swap(min_val, max_val);
        }

        min_val = std::min(min_val, 0.0f);
        max_val = std::max(max_val, 0.0f);

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

    QuantParams calculate_quant_params_calibrated(
        const Tensor<float> &input,
        CalibrationMethod method,
        float percentile)
    {
        if (input.size() == 0)
        {
            return {1.0f, 0};
        }

        if (method == CalibrationMethod::MinMax)
        {
            const float *d = input.data();
            float min_val = d[0];
            float max_val = d[0];
            for (size_t i = 1; i < input.size(); ++i)
            {
                if (d[i] < min_val)
                    min_val = d[i];
                if (d[i] > max_val)
                    max_val = d[i];
            }
            return calculate_quant_params(min_val, max_val);
        }

        if (percentile <= 50.0f || percentile >= 100.0f)
        {
            throw std::invalid_argument("Percentile must be between 50.0 and 100.0.");
        }

        std::vector<float> sorted_vals(input.data(), input.data() + input.size());
        std::sort(sorted_vals.begin(), sorted_vals.end());

        float lower_pct = (100.0f - percentile) / 100.0f;
        float upper_pct = percentile / 100.0f;

        size_t low_idx = static_cast<size_t>(std::floor(lower_pct * static_cast<float>(sorted_vals.size() - 1)));
        size_t high_idx = static_cast<size_t>(std::ceil(upper_pct * static_cast<float>(sorted_vals.size() - 1)));

        float min_clipped = sorted_vals[low_idx];
        float max_clipped = sorted_vals[high_idx];

        return calculate_quant_params(min_clipped, max_clipped);
    }

    PerChannelQuantParams calculate_per_channel_params(const Tensor<float> &weight, bool per_column)
    {
        if (weight.size() == 0)
        {
            return {{}, {}};
        }

        PerChannelQuantParams result;
        const size_t channels = per_column ? weight.cols() : weight.rows();
        result.scales.resize(channels);
        result.zero_points.resize(channels);

        if (per_column)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                float min_v = weight(0, c);
                float max_v = weight(0, c);
                for (size_t r = 1; r < weight.rows(); ++r)
                {
                    float v = weight(r, c);
                    if (v < min_v)
                        min_v = v;
                    if (v > max_v)
                        max_v = v;
                }
                QuantParams p = calculate_quant_params(min_v, max_v);
                result.scales[c] = p.scale;
                result.zero_points[c] = p.zero_point;
            }
        }
        else
        {
            for (size_t r = 0; r < channels; ++r)
            {
                float min_v = weight(r, 0);
                float max_v = weight(r, 0);
                for (size_t c = 1; c < weight.cols(); ++c)
                {
                    float v = weight(r, c);
                    if (v < min_v)
                        min_v = v;
                    if (v > max_v)
                        max_v = v;
                }
                QuantParams p = calculate_quant_params(min_v, max_v);
                result.scales[r] = p.scale;
                result.zero_points[r] = p.zero_point;
            }
        }
        return result;
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

    // Commit 15: Decompose double multiplier into Q31 integer multiplier and right shift
    void quantize_multiplier_and_shift(double double_multiplier, int32_t &quantized_multiplier, int &shift)
    {
        if (double_multiplier <= 0.0)
        {
            quantized_multiplier = 0;
            shift = 0;
            return;
        }

        int exponent;
        double significand = std::frexp(double_multiplier, &exponent);
        int64_t q = static_cast<int64_t>(std::round(significand * (1LL << 31)));

        if (q == (1LL << 31))
        {
            q /= 2;
            exponent++;
        }

        quantized_multiplier = static_cast<int32_t>(q);
        shift = -exponent;
    }

    // Fixed-point multiplication with symmetric integer rounding
    int32_t multiply_by_quantized_multiplier(int32_t x, int32_t quantized_multiplier, int shift)
    {
        int64_t total = static_cast<int64_t>(x) * static_cast<int64_t>(quantized_multiplier);
        int64_t rounding = 1LL << (shift - 1);
        int64_t result = (total + rounding) >> shift;
        return static_cast<int32_t>(result);
    }

    QuantizedTensor quantize_tensor(const Tensor<float> &input, const QuantParams &params)
    {
        QuantizedTensor output(input.rows(), input.cols(), params);
        if (input.size() == 0)
            return output;

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
        if (input.size() == 0)
            return output;

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

    QuantizedTensor quantize_tensor_per_channel(const Tensor<float> &input, const PerChannelQuantParams &params)
    {
        if (params.num_channels() != input.cols())
        {
            throw std::invalid_argument("Per-channel params count must match input columns.");
        }

        QuantParams fallback = {params.scales[0], params.zero_points[0]};
        QuantizedTensor output(input.rows(), input.cols(), fallback);
        if (input.size() == 0)
            return output;

        for (size_t r = 0; r < input.rows(); ++r)
        {
            for (size_t c = 0; c < input.cols(); ++c)
            {
                QuantParams ch_param = {params.scales[c], params.zero_points[c]};
                output(r, c) = quantize_val(input(r, c), ch_param);
            }
        }
        return output;
    }

    Tensor<float> dequantize_tensor_per_channel(const QuantizedTensor &input, const PerChannelQuantParams &params)
    {
        if (params.num_channels() != input.cols())
        {
            throw std::invalid_argument("Per-channel params count must match input columns.");
        }

        Tensor<float> output(input.rows(), input.cols(), 0.0f);
        if (input.size() == 0)
            return output;

        for (size_t r = 0; r < input.rows(); ++r)
        {
            for (size_t c = 0; c < input.cols(); ++c)
            {
                QuantParams ch_param = {params.scales[c], params.zero_points[c]};
                output(r, c) = dequantize_val(input(r, c), ch_param);
            }
        }
        return output;
    }

    void matmul_int8(const QuantizedTensor &A, const QuantizedTensor &B, QuantizedTensor &C)
    {
        if (A.size() == 0 || B.size() == 0)
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

    // Commit 15: Integer-only Fixed-Point GEMM (Zero FPU instructions)
    void matmul_int8_fixed_point(const QuantizedTensor &A, const QuantizedTensor &B, QuantizedTensor &C)
    {
        if (A.size() == 0 || B.size() == 0)
        {
            throw std::invalid_argument("Cannot perform matmul on empty tensor.");
        }
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_fixed_point: A.cols must match B.rows.");
        }
        if (C.rows() != A.rows() || C.cols() != B.cols())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_fixed_point: Output tensor C dimension invalid.");
        }

        const size_t M = A.rows();
        const size_t K = A.cols();
        const size_t N = B.cols();

        const int32_t zp_A = A.params().zero_point;
        const int32_t zp_B = B.params().zero_point;
        const int32_t zp_C = C.params().zero_point;

        double real_multiplier = static_cast<double>(A.params().scale * B.params().scale) / static_cast<double>(C.params().scale);
        int32_t q_multiplier;
        int shift;
        quantize_multiplier_and_shift(real_multiplier, q_multiplier, shift);

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

                // Fixed point requantization without float operations
                int32_t scaled_acc = multiply_by_quantized_multiplier(acc, q_multiplier, shift);
                int32_t res = scaled_acc + zp_C;
                C(i, j) = static_cast<int8_t>(std::clamp(res, -128, 127));
            }
        }
    }

    void matmul_int8_per_channel(const QuantizedTensor &A,
                                 const QuantizedTensor &B,
                                 const PerChannelQuantParams &B_params,
                                 QuantizedTensor &C)
    {
        if (A.size() == 0 || B.size() == 0)
        {
            throw std::invalid_argument("Cannot perform matmul on empty tensor.");
        }
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_per_channel: A.cols must match B.rows.");
        }
        if (B_params.num_channels() != B.cols())
        {
            throw std::invalid_argument("Per-channel parameters count must equal B.cols().");
        }
        if (C.rows() != A.rows() || C.cols() != B.cols())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_per_channel: Output tensor C dimension invalid.");
        }

        const size_t M = A.rows();
        const size_t K = A.cols();
        const size_t N = B.cols();

        const int32_t zp_A = A.params().zero_point;
        const float scale_A = A.params().scale;
        const int32_t zp_C = C.params().zero_point;
        const float scale_C = C.params().scale;

        for (size_t j = 0; j < N; ++j)
        {
            const int32_t zp_B = B_params.zero_points[j];
            const float effective_scale = (scale_A * B_params.scales[j]) / scale_C;

            for (size_t i = 0; i < M; ++i)
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
        if (X.size() == 0)
            return;
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
        if (X.size() == 0)
            return;
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
        if (input.size() == 0)
            return;

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