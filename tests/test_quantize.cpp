#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept>

void test_quant_param_calculation()
{
    std::cout << "[Test 1] Testing parameter calculation... ";
    engine::QuantParams p = engine::calculate_quant_params(-10.0f, 30.0f);
    assert(p.scale > 0.0f);
    assert(p.zero_point >= -128 && p.zero_point <= 127);

    // Constant range edge case
    engine::QuantParams p_const = engine::calculate_quant_params(2.0f, 2.0f);
    assert(p_const.scale > 0.0f);
    std::cout << "PASSED\n";
}

void test_quantize_dequantize_roundtrip()
{
    std::cout << "[Test 2] Testing FP32 <-> INT8 roundtrip conversion... ";
    engine::Tensor<float> input(1, 4, {-10.0f, 0.0f, 15.0f, 30.0f});
    engine::QuantParams params = engine::calculate_quant_params(-10.0f, 30.0f);

    engine::QuantizedTensor q = engine::quantize_tensor(input, params);
    engine::Tensor<float> reconstructed = engine::dequantize_tensor(q);

    for (size_t i = 0; i < input.size(); ++i)
    {
        float err = std::abs(input.data()[i] - reconstructed.data()[i]);
        assert(err <= params.scale + 1e-4f);
    }
    std::cout << "PASSED\n";
}

void test_quantized_gemm()
{
    std::cout << "[Test 3] Testing INT8 GEMM kernel vs FP32 baseline... ";
    engine::Tensor<float> A_fp32(2, 3, {1.5f, -2.0f, 0.5f, -1.0f, 3.0f, 2.5f});
    engine::Tensor<float> B_fp32(3, 2, {0.5f, -1.0f, 2.0f, 1.5f, -0.5f, 2.0f});

    engine::Tensor<float> C_ref(2, 2, 0.0f);
    engine::matmul_fp32(A_fp32, B_fp32, C_ref);

    engine::QuantParams qA = engine::calculate_quant_params(-3.0f, 3.0f);
    engine::QuantParams qB = engine::calculate_quant_params(-3.0f, 3.0f);
    engine::QuantParams qC = engine::calculate_quant_params(-10.0f, 15.0f);

    engine::QuantizedTensor A_int8 = engine::quantize_tensor(A_fp32, qA);
    engine::QuantizedTensor B_int8 = engine::quantize_tensor(B_fp32, qB);
    engine::QuantizedTensor C_int8(2, 2, qC);

    engine::matmul_int8(A_int8, B_int8, C_int8);
    engine::Tensor<float> C_dequant = engine::dequantize_tensor(C_int8);

    for (size_t i = 0; i < C_ref.size(); ++i)
    {
        float err = std::abs(C_ref.data()[i] - C_dequant.data()[i]);
        assert(err < 0.25f);
    }
    std::cout << "PASSED\n";
}

void test_quantized_bias_add()
{
    std::cout << "[Test 4] Testing Quantized Bias Add... ";
    engine::Tensor<float> input_fp32(2, 3, {1.2f, -2.4f, 0.8f, -0.5f, 2.1f, -1.7f});
    engine::Tensor<float> bias_fp32(1, 3, {0.5f, -0.8f, 1.2f});

    engine::Tensor<float> ref_out = input_fp32;
    engine::bias_add(ref_out, bias_fp32);

    engine::QuantParams q_in = engine::calculate_quant_params(-4.0f, 4.0f);
    engine::QuantParams q_b = engine::calculate_quant_params(-2.0f, 2.0f);

    engine::QuantizedTensor input_int8 = engine::quantize_tensor(input_fp32, q_in);
    engine::QuantizedTensor bias_int8 = engine::quantize_tensor(bias_fp32, q_b);

    engine::bias_add(input_int8, bias_int8);
    engine::Tensor<float> dequant = engine::dequantize_tensor(input_int8);

    for (size_t i = 0; i < ref_out.size(); ++i)
    {
        float err = std::abs(ref_out.data()[i] - dequant.data()[i]);
        assert(err < 0.25f);
    }
    std::cout << "PASSED\n";
}

void test_quantized_relu()
{
    std::cout << "[Test 5] Testing Quantized ReLU (in-place & out-of-place)... ";
    engine::Tensor<float> input_fp32(2, 3, {-3.5f, 0.0f, 2.8f, 1.4f, -0.9f, -5.2f});

    engine::Tensor<float> ref_out(2, 3, 0.0f);
    engine::relu(input_fp32, ref_out);

    engine::QuantParams qparams = engine::calculate_quant_params(-6.0f, 4.0f);
    engine::QuantizedTensor q_in = engine::quantize_tensor(input_fp32, qparams);

    engine::QuantizedTensor q_inplace = q_in;
    engine::relu(q_inplace);

    engine::QuantizedTensor q_out(2, 3, qparams);
    engine::relu(q_in, q_out);

    for (size_t i = 0; i < q_inplace.size(); ++i)
    {
        assert(q_inplace.data()[i] == q_out.data()[i]);
    }

    engine::Tensor<float> dequant = engine::dequantize_tensor(q_inplace);
    for (size_t i = 0; i < ref_out.size(); ++i)
    {
        float err = std::abs(ref_out.data()[i] - dequant.data()[i]);
        assert(err <= qparams.scale + 1e-4f);
    }
    std::cout << "PASSED\n";
}

void test_edge_cases()
{
    std::cout << "[Test 6] Testing edge cases & dimension guards... ";
    engine::QuantParams qp = {0.1f, 0};

    // 1. Index bounds checking
    bool caught_oob = false;
    try
    {
        engine::QuantizedTensor t(2, 2, qp);
        t(2, 2) = 1;
    }
    catch (const std::out_of_range &)
    {
        caught_oob = true;
    }
    assert(caught_oob);

    // 2. GEMM dimension mismatch
    bool caught_gemm = false;
    try
    {
        engine::QuantizedTensor A(2, 3, qp);
        engine::QuantizedTensor B(4, 2, qp);
        engine::QuantizedTensor C(2, 2, qp);
        engine::matmul_int8(A, B, C);
    }
    catch (const std::invalid_argument &)
    {
        caught_gemm = true;
    }
    assert(caught_gemm);

    // 3. Bias dimension mismatch
    bool caught_bias = false;
    try
    {
        engine::QuantizedTensor X(2, 4, qp);
        engine::QuantizedTensor bias(1, 3, qp);
        engine::bias_add(X, bias);
    }
    catch (const std::invalid_argument &)
    {
        caught_bias = true;
    }
    assert(caught_bias);

    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "====================================================\n";
    std::cout << "     Running Comprehensive Quantization Tests       \n";
    std::cout << "====================================================\n";

    test_quant_param_calculation();
    test_quantize_dequantize_roundtrip();
    test_quantized_gemm();
    test_quantized_bias_add();
    test_quantized_relu();
    test_edge_cases();

    std::cout << "\n>> All 6 Quantization Test Suites Passed Successfully! <<\n";
    return 0;
}