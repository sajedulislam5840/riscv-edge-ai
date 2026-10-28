#ifndef ENGINE_QUANTIZE_HPP
#define ENGINE_QUANTIZE_HPP

#include <cstdint>
#include <cstddef>

namespace engine {

// Quantization parameters for affine INT8 mapping
struct QuantizationParams {
    float scale;        // Scale factor: S in q = (r / S) + Z
    int32_t zero_point; // Zero-point offset: Z in q = (r / S) + Z

    QuantizationParams() : scale(1.0f), zero_point(0) {}
    QuantizationParams(float s, int32_t z) : scale(s), zero_point(z) {}
};

// Quantization scheme types
enum class QuantizationScheme {
    SYMMETRIC,   // Zero-point = 0, symmetric range [-127, 127]
    ASYMMETRIC   // Full range [-128, 127] with learned zero-point
};

// Calculate quantization parameters from floating-point data
QuantizationParams calculate_quantization_params(
    const float* data,
    size_t size,
    QuantizationScheme scheme = QuantizationScheme::SYMMETRIC
);

// Quantize FP32 value to INT8 using given parameters
int8_t quantize_value(float value, const QuantizationParams& params);

// Quantize array of FP32 values to INT8
void quantize_array(
    const float* input,
    int8_t* output,
    size_t size,
    const QuantizationParams& params
);

// Dequantize INT8 value back to FP32
float dequantize_value(int8_t quantized, const QuantizationParams& params);

// Dequantize array of INT8 values to FP32
void dequantize_array(
    const int8_t* input,
    float* output,
    size_t size,
    const QuantizationParams& params
);

} // namespace engine

#endif // ENGINE_QUANTIZE_HPP
