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

} // namespace engine

#endif // ENGINE_QUANTIZE_HPP
