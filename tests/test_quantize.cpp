#include "engine/quantize.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void test_symmetric_quantization_params() {
    float data[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    size_t size = 5;

    engine::QuantizationParams params = engine::calculate_quantization_params(
        data, size, engine::QuantizationScheme::SYMMETRIC
    );

    // For symmetric quantization with range [-10, 10]:
    // scale = 10 / 127 ≈ 0.0787
    // zero_point = 0
    assert(float_eq(params.scale, 10.0f / 127.0f, 0.001f));
    assert(params.zero_point == 0);

    std::cout << "  [PASS] Symmetric quantization parameter calculation\n";
}

void test_asymmetric_quantization_params() {
    float data[] = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};
    size_t size = 5;

    engine::QuantizationParams params = engine::calculate_quantization_params(
        data, size, engine::QuantizationScheme::ASYMMETRIC
    );

    // For asymmetric quantization with range [0, 8]:
    // scale = 8 / 255 ≈ 0.0314
    // zero_point should map 0.0f to -128
    assert(float_eq(params.scale, 8.0f / 255.0f, 0.001f));
    assert(params.zero_point == -128);

    std::cout << "  [PASS] Asymmetric quantization parameter calculation\n";
}

void test_quantize_dequantize_roundtrip() {
    engine::QuantizationParams params(0.1f, 0);

    // Test several values
    float test_values[] = {-12.7f, -5.0f, 0.0f, 5.0f, 12.7f};

    for (float val : test_values) {
        int8_t quantized = engine::quantize_value(val, params);
        float dequantized = engine::dequantize_value(quantized, params);

        // Dequantized value should be close to original (within quantization error)
        assert(float_eq(val, dequantized, params.scale * 2.0f));
    }

    std::cout << "  [PASS] Quantize-dequantize roundtrip\n";
}

void test_quantize_array() {
    float input[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    int8_t output[4];
    size_t size = 4;

    engine::QuantizationParams params(0.01f, 0);
    engine::quantize_array(input, output, size, params);

    // Check quantized values
    assert(output[0] == -100);  // -1.0 / 0.01 = -100
    assert(output[1] == 0);     //  0.0 / 0.01 = 0
    assert(output[2] == 100);   //  1.0 / 0.01 = 100

    std::cout << "  [PASS] Array quantization\n";
}

void test_dequantize_array() {
    int8_t input[] = {-100, 0, 100};
    float output[3];
    size_t size = 3;

    engine::QuantizationParams params(0.01f, 0);
    engine::dequantize_array(input, output, size, params);

    // Check dequantized values
    assert(float_eq(output[0], -1.0f));
    assert(float_eq(output[1], 0.0f));
    assert(float_eq(output[2], 1.0f));

    std::cout << "  [PASS] Array dequantization\n";
}

void test_edge_case_all_zeros() {
    float data[] = {0.0f, 0.0f, 0.0f};
    size_t size = 3;

    engine::QuantizationParams params = engine::calculate_quantization_params(
        data, size, engine::QuantizationScheme::SYMMETRIC
    );

    // Should handle gracefully without division by zero
    assert(params.scale == 1.0f);
    assert(params.zero_point == 0);

    std::cout << "  [PASS] Edge case: all zeros\n";
}

int main() {
    std::cout << "[TEST SUITE] Running Quantization Unit Tests...\n";
    test_symmetric_quantization_params();
    test_asymmetric_quantization_params();
    test_quantize_dequantize_roundtrip();
    test_quantize_array();
    test_dequantize_array();
    test_edge_case_all_zeros();
    std::cout << "[TEST SUITE] All quantization tests passed successfully!\n";
    return 0;
}
