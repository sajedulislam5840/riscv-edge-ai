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

} // namespace engine