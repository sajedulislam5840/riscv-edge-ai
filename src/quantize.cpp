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
        shift = 31 - exponent;
    }

    int32_t multiply_by_quantized_multiplier(int32_t x, int32_t quantized_multiplier, int shift)
    {
        int64_t total = static_cast<int64_t>(x) * static_cast<int64_t>(quantized_multiplier);
        if (shift <= 0)
        {
            return static_cast<int32_t>(total << (-shift));
        }
        int64_t round_val = 1LL << (shift - 1);
        int64_t result = (total + round_val) >> shift;
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

    QuantizedTensorView slice_tensor(
        QuantizedTensor &tensor,
        size_t start_row,
        size_t start_col,
        size_t num_rows,
        size_t num_cols)
    {
        if (start_row + num_rows > tensor.rows() || start_col + num_cols > tensor.cols())
        {
            throw std::out_of_range("Slice bounds exceed tensor dimensions.");
        }

        int8_t *base_ptr = tensor.data() + (start_row * tensor.cols() + start_col);
        return QuantizedTensorView(base_ptr, num_rows, num_cols, tensor.cols(), tensor.params());
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

                int32_t scaled_acc = multiply_by_quantized_multiplier(acc, q_multiplier, shift);
                int32_t res = scaled_acc + zp_C;
                C(i, j) = static_cast<int8_t>(std::clamp(res, -128, 127));
            }
        }
    }

    void matmul_int8_view(const QuantizedTensorView &A, const QuantizedTensorView &B, QuantizedTensorView &C)
    {
        if (A.cols() != B.rows())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_view: A.cols must match B.rows.");
        }
        if (C.rows() != A.rows() || C.cols() != B.cols())
        {
            throw std::invalid_argument("Dimension mismatch in matmul_int8_view: Output view dimension invalid.");
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

    void fold_batch_norm(
        Tensor<float> &weight,
        Tensor<float> &bias,
        const Tensor<float> &mean,
        const Tensor<float> &var,
        const Tensor<float> &gamma,
        const Tensor<float> &beta,
        float epsilon)
    {
        size_t out_channels = weight.cols();
        if (mean.size() != out_channels || var.size() != out_channels ||
            gamma.size() != out_channels || beta.size() != out_channels ||
            bias.size() != out_channels)
        {
            throw std::invalid_argument("Dimension mismatch in fold_batch_norm parameter vectors.");
        }

        const float *mu = mean.data();
        const float *v = var.data();
        const float *g = gamma.data();
        const float *b = beta.data();
        float *b_out = bias.data();

        for (size_t c = 0; c < out_channels; ++c)
        {
            float inv_std = 1.0f / std::sqrt(v[c] + epsilon);
            float scale_factor = g[c] * inv_std;

            for (size_t r = 0; r < weight.rows(); ++r)
            {
                weight(r, c) *= scale_factor;
            }

            b_out[c] = (b_out[c] - mu[c]) * scale_factor + b[c];
        }
    }

    void leaky_relu(QuantizedTensor &X, float alpha)
    {
        if (X.size() == 0)
            return;
        int8_t *ptr = X.data();
        const size_t total = X.size();
        const int32_t zp = X.params().zero_point;

        int32_t q_multiplier;
        int shift;
        quantize_multiplier_and_shift(static_cast<double>(alpha), q_multiplier, shift);

        for (size_t i = 0; i < total; ++i)
        {
            int32_t val = static_cast<int32_t>(ptr[i]);
            if (val < zp)
            {
                int32_t diff = val - zp;
                int32_t scaled = multiply_by_quantized_multiplier(diff, q_multiplier, shift);
                int32_t clamped = std::clamp(zp + scaled, -128, 127);
                ptr[i] = static_cast<int8_t>(clamped);
            }
        }
    }

    void leaky_relu(const QuantizedTensor &input, QuantizedTensor &output, float alpha)
    {
        if (input.rows() != output.rows() || input.cols() != output.cols())
        {
            throw std::invalid_argument("Dimension mismatch in out-of-place leaky_relu.");
        }
        output.set_params(input.params());
        const int8_t *in_ptr = input.data();
        int8_t *out_ptr = output.data();
        const size_t total = input.size();
        const int32_t zp = input.params().zero_point;

        int32_t q_multiplier;
        int shift;
        quantize_multiplier_and_shift(static_cast<double>(alpha), q_multiplier, shift);

        for (size_t i = 0; i < total; ++i)
        {
            int32_t val = static_cast<int32_t>(in_ptr[i]);
            if (val < zp)
            {
                int32_t diff = val - zp;
                int32_t scaled = multiply_by_quantized_multiplier(diff, q_multiplier, shift);
                int32_t clamped = std::clamp(zp + scaled, -128, 127);
                out_ptr[i] = static_cast<int8_t>(clamped);
            }
            else
            {
                out_ptr[i] = in_ptr[i];
            }
        }
    }

    // Commit 19: Quantized Element-Wise Addition (Integer-only)
    void add_int8(const QuantizedTensor &A, const QuantizedTensor &B, QuantizedTensor &C)
    {
        if (A.rows() != B.rows() || A.cols() != B.cols() ||
            A.rows() != C.rows() || A.cols() != C.cols())
        {
            throw std::invalid_argument("Dimension mismatch in add_int8.");
        }
        if (A.size() == 0)
            return;

        const double real_mult_A = static_cast<double>(A.params().scale) / static_cast<double>(C.params().scale);
        const double real_mult_B = static_cast<double>(B.params().scale) / static_cast<double>(C.params().scale);

        int32_t mult_A, mult_B;
        int shift_A, shift_B;
        quantize_multiplier_and_shift(real_mult_A, mult_A, shift_A);
        quantize_multiplier_and_shift(real_mult_B, mult_B, shift_B);

        const int32_t zp_A = A.params().zero_point;
        const int32_t zp_B = B.params().zero_point;
        const int32_t zp_C = C.params().zero_point;

        const size_t total = A.size();
        const int8_t *a_ptr = A.data();
        const int8_t *b_ptr = B.data();
        int8_t *c_ptr = C.data();

        for (size_t i = 0; i < total; ++i)
        {
            int32_t val_A = static_cast<int32_t>(a_ptr[i]) - zp_A;
            int32_t val_B = static_cast<int32_t>(b_ptr[i]) - zp_B;

            int32_t scaled_A = multiply_by_quantized_multiplier(val_A, mult_A, shift_A);
            int32_t scaled_B = multiply_by_quantized_multiplier(val_B, mult_B, shift_B);

            int32_t sum = scaled_A + scaled_B + zp_C;
            c_ptr[i] = static_cast<int8_t>(std::clamp(sum, -128, 127));
        }
    }

    // Commit 20: Quantized 2D Max Pooling
    void maxpool2d_int8(
        const QuantizedTensor &input,
        QuantizedTensor &output,
        size_t kernel_h,
        size_t kernel_w,
        size_t stride_h,
        size_t stride_w)
    {
        size_t out_h = (input.rows() - kernel_h) / stride_h + 1;
        size_t out_w = (input.cols() - kernel_w) / stride_w + 1;

        if (output.rows() != out_h || output.cols() != out_w)
        {
            throw std::invalid_argument("Output tensor dimension mismatch in maxpool2d_int8.");
        }
        output.set_params(input.params());

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                size_t start_r = oh * stride_h;
                size_t start_c = ow * stride_w;
                int8_t max_val = -128;

                for (size_t kh = 0; kh < kernel_h; ++kh)
                {
                    for (size_t kw = 0; kw < kernel_w; ++kw)
                    {
                        int8_t val = input(start_r + kh, start_c + kw);
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output(oh, ow) = max_val;
            }
        }
    }

    // Commit 20: Quantized 2D Average Pooling
    void avgpool2d_int8(
        const QuantizedTensor &input,
        QuantizedTensor &output,
        size_t kernel_h,
        size_t kernel_w,
        size_t stride_h,
        size_t stride_w)
    {
        size_t out_h = (input.rows() - kernel_h) / stride_h + 1;
        size_t out_w = (input.cols() - kernel_w) / stride_w + 1;

        if (output.rows() != out_h || output.cols() != out_w)
        {
            throw std::invalid_argument("Output tensor dimension mismatch in avgpool2d_int8.");
        }
        output.set_params(input.params());

        const int32_t zp = input.params().zero_point;
        const double inv_window = 1.0 / static_cast<double>(kernel_h * kernel_w);
        int32_t q_mult;
        int shift;
        quantize_multiplier_and_shift(inv_window, q_mult, shift);

        for (size_t oh = 0; oh < out_h; ++oh)
        {
            for (size_t ow = 0; ow < out_w; ++ow)
            {
                size_t start_r = oh * stride_h;
                size_t start_c = ow * stride_w;
                int32_t sum = 0;

                for (size_t kh = 0; kh < kernel_h; ++kh)
                {
                    for (size_t kw = 0; kw < kernel_w; ++kw)
                    {
                        sum += (static_cast<int32_t>(input(start_r + kh, start_c + kw)) - zp);
                    }
                }

                int32_t avg = multiply_by_quantized_multiplier(sum, q_mult, shift);
                int32_t res = avg + zp;
                output(oh, ow) = static_cast<int8_t>(std::clamp(res, -128, 127));
            }
        }
    }

    // Commit 21: Model Parameter Serialization / Deserialization
    void save_quantized_model(const std::string &filepath, const std::vector<QuantizedTensor> &tensors)
    {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }

        const uint32_t magic = 0x514E4554; // "QNET"
        out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

        uint32_t num_tensors = static_cast<uint32_t>(tensors.size());
        out.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

        for (const auto &t : tensors)
        {
            uint32_t rows = static_cast<uint32_t>(t.rows());
            uint32_t cols = static_cast<uint32_t>(t.cols());
            float scale = t.params().scale;
            int32_t zp = t.params().zero_point;

            out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
            out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
            out.write(reinterpret_cast<const char *>(&scale), sizeof(scale));
            out.write(reinterpret_cast<const char *>(&zp), sizeof(zp));

            out.write(reinterpret_cast<const char *>(t.data()), t.size() * sizeof(int8_t));
        }
    }

    std::vector<QuantizedTensor> load_quantized_model(const std::string &filepath)
    {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Failed to open file for reading: " + filepath);
        }

        uint32_t magic = 0;
        in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic != 0x514E4554)
        {
            throw std::runtime_error("Invalid file header: not a valid QNET model binary.");
        }

        uint32_t num_tensors = 0;
        in.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

        std::vector<QuantizedTensor> result;
        result.reserve(num_tensors);

        for (uint32_t i = 0; i < num_tensors; ++i)
        {
            uint32_t rows = 0;
            uint32_t cols = 0;
            float scale = 0.0f;
            int32_t zp = 0;

            in.read(reinterpret_cast<char *>(&rows), sizeof(rows));
            in.read(reinterpret_cast<char *>(&cols), sizeof(cols));
            in.read(reinterpret_cast<char *>(&scale), sizeof(scale));
            in.read(reinterpret_cast<char *>(&zp), sizeof(zp));

            QuantParams qp = {scale, zp};
            QuantizedTensor t(rows, cols, qp);
            in.read(reinterpret_cast<char *>(t.data()), t.size() * sizeof(int8_t));
            result.push_back(std::move(t));
        }

        return result;
    }

} // namespace engine