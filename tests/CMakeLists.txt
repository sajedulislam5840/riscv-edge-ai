set(ENGINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/ops.cpp
    ${CMAKE_SOURCE_DIR}/src/quantize.cpp
)

# ── FP32 reference kernels ───────────────────────────────────────────
add_executable(test_tensor    test_main.cpp    ${ENGINE_SOURCES})
add_executable(test_gemm      test_gemm.cpp    ${ENGINE_SOURCES})
add_executable(test_relu      test_relu.cpp    ${ENGINE_SOURCES})
add_executable(test_bias_add  test_bias_add.cpp ${ENGINE_SOURCES})
add_executable(test_softmax   test_softmax.cpp  ${ENGINE_SOURCES})

# ── INT8 quantization ───────────────────────────────────────
add_executable(test_quantize  test_quantize.cpp ${ENGINE_SOURCES})

# ── Accumulator overflow and arena stress tests ─────────────────────────────
add_executable(test_stress    test_stress.cpp   ${ENGINE_SOURCES})

# Register all tests with ctest
add_executable(test_simd_isa  test_simd_isa.cpp)
target_include_directories(test_simd_isa PRIVATE ${CMAKE_SOURCE_DIR}/hardware/include)
add_test(NAME TensorSanityTest      COMMAND test_tensor)
add_test(NAME SimdIsaModelTest      COMMAND test_simd_isa)
add_test(NAME GemmCorrectnessTest   COMMAND test_gemm)
add_test(NAME ReluActivationTest    COMMAND test_relu)
add_test(NAME BiasAddTest           COMMAND test_bias_add)
add_test(NAME SoftmaxTest           COMMAND test_softmax)
add_test(NAME QuantizationTest      COMMAND test_quantize)
add_test(NAME QuantStressTest       COMMAND test_stress)
