#include "engine/quantize.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace engine {

// Calculate quantization parameters from floating-point data
QuantizationParams calculate_quantization_params(
    const float* data,
    size_t size,
    QuantizationScheme scheme
) {
    if (size == 0) {
        return QuantizationParams(1.0f, 0);
    }

    // Find min and max values in the data
    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 1; i < size; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    QuantizationParams params;

    if (scheme == QuantizationScheme::SYMMETRIC) {
        // Symmetric quantization: zero-point = 0
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        params.scale = abs_max / 127.0f;
        params.zero_point = 0;

        // Handle edge case: all zeros
        if (abs_max == 0.0f) {
            params.scale = 1.0f;
        }
    } else {
        // Asymmetric quantization: full range [-128, 127]
        float range = max_val - min_val;
        params.scale = range / 255.0f;

        // Handle edge case: all same value
        if (range == 0.0f) {
            params.scale = 1.0f;
            params.zero_point = 0;
        } else {
            // Calculate zero-point to map min_val to -128
            float zero_point_fp = -128.0f - (min_val / params.scale);
            params.zero_point = static_cast<int32_t>(std::round(zero_point_fp));

            // Clamp zero-point to valid range
            params.zero_point = std::max(-128, std::min(127, params.zero_point));
        }
    }

    return params;
}

// Quantize FP32 value to INT8 using given parameters
int8_t quantize_value(float value, const QuantizationParams& params) {
    // q = clamp(round(r / S) + Z, -128, 127)
    float quantized_fp = (value / params.scale) + static_cast<float>(params.zero_point);
    int32_t quantized_int = static_cast<int32_t>(std::round(quantized_fp));

    // Clamp to INT8 range
    quantized_int = std::max(-128, std::min(127, quantized_int));

    return static_cast<int8_t>(quantized_int);
}

// Quantize array of FP32 values to INT8
void quantize_array(
    const float* input,
    int8_t* output,
    size_t size,
    const QuantizationParams& params
) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = quantize_value(input[i], params);
    }
}

// Dequantize INT8 value back to FP32
float dequantize_value(int8_t quantized, const QuantizationParams& params) {
    // r = S * (q - Z)
    return params.scale * (static_cast<float>(quantized) - static_cast<float>(params.zero_point));
}

// Dequantize array of INT8 values to FP32
void dequantize_array(
    const int8_t* input,
    float* output,
    size_t size,
    const QuantizationParams& params
) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = dequantize_value(input[i], params);
    }
}

} // namespace engine
