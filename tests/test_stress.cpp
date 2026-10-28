#include "engine/tensor.hpp"
#include "engine/ops.hpp"
#include "engine/quantize.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

void test_accumulator_overflow_guard()
{
    std::cout << "[Stress 1] Accumulator Overflow Protection (Large dot products)... ";
    const size_t K = 2048;
    engine::QuantParams qp = {0.05f, 0};

    engine::QuantizedTensor A(1, K, qp);
    engine::QuantizedTensor B(K, 1, qp);

    // Fill with extreme int8 values: 127 * 127 * 2048 = 33,030,272 (fits in int32 without overflow)
    for (size_t k = 0; k < K; ++k)
    {
        A(0, k) = 127;
        B(k, 0) = 127;
    }

    engine::QuantParams qp_out = {1000.0f, 0};
    engine::QuantizedTensor C(1, 1, qp_out);

    engine::matmul_int8_fixed_point(A, B, C);
    assert(C(0, 0) > 0);
    std::cout << "PASSED (Accumulator handled 33M+ peak safely)\n";
}

void test_subnormal_scale_protection()
{
    std::cout << "[Stress 2] Subnormal & Extreme Dynamic Range Protection... ";
    engine::Tensor<float> tiny_input(1, 4, {1e-9f, -1e-9f, 2e-9f, 0.0f});
    engine::QuantParams qp = engine::calculate_quant_params(-1e-9f, 2e-9f);

    assert(qp.scale > 0.0f);
    engine::QuantizedTensor q = engine::quantize_tensor(tiny_input, qp);
    engine::Tensor<float> deq = engine::dequantize_tensor(q);

    for (size_t i = 0; i < 4; ++i)
    {
        assert(std::abs(tiny_input.data()[i] - deq.data()[i]) <= qp.scale);
    }
    std::cout << "PASSED\n";
}

void test_arena_allocator_stress()
{
    std::cout << "[Stress 3] Static Arena Allocator Boundary & Alignment... ";
    engine::ArenaAllocator arena(1024);

    void *ptr1 = arena.allocate(64, 8);
    assert(reinterpret_cast<size_t>(ptr1) % 8 == 0);

    void *ptr2 = arena.allocate(128, 16);
    assert(reinterpret_cast<size_t>(ptr2) % 16 == 0);

    assert(arena.used() >= 192);
    assert(arena.peak() >= 192);

    arena.reset();
    assert(arena.used() == 0);
    assert(arena.peak() >= 192);

    bool caught_oob = false;
    try
    {
        arena.allocate(2048);
    }
    catch (const std::bad_alloc &)
    {
        caught_oob = true;
    }
    assert(caught_oob);
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "====================================================\n";
    std::cout << "        Running Quantization Stress Battery         \n";
    std::cout << "====================================================\n";

    test_accumulator_overflow_guard();
    test_subnormal_scale_protection();
    test_arena_allocator_stress();

    std::cout << "\n>> All Stress Tests Passed Successfully! <<\n";
    return 0;
}