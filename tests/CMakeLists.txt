set(ENGINE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/ops.cpp
    ${CMAKE_SOURCE_DIR}/src/quantize.cpp
)

# ── Phase 01: FP32 baseline tests ───────────────────────────────────────────
add_executable(test_tensor    test_main.cpp    ${ENGINE_SOURCES})
add_executable(test_gemm      test_gemm.cpp    ${ENGINE_SOURCES})
add_executable(test_relu      test_relu.cpp    ${ENGINE_SOURCES})
add_executable(test_bias_add  test_bias_add.cpp ${ENGINE_SOURCES})
add_executable(test_softmax   test_softmax.cpp  ${ENGINE_SOURCES})

# ── Phase 02: INT8 quantization tests ───────────────────────────────────────
add_executable(test_quantize  test_quantize.cpp ${ENGINE_SOURCES})

# ── Phase 02: Accumulator / arena stress battery ─────────────────────────────
add_executable(test_stress    test_stress.cpp   ${ENGINE_SOURCES})

# Register all tests with ctest
add_test(NAME TensorSanityTest      COMMAND test_tensor)
add_test(NAME GemmCorrectnessTest   COMMAND test_gemm)
add_test(NAME ReluActivationTest    COMMAND test_relu)
add_test(NAME BiasAddTest           COMMAND test_bias_add)
add_test(NAME SoftmaxTest           COMMAND test_softmax)
add_test(NAME QuantizationTest      COMMAND test_quantize)
add_test(NAME QuantStressTest       COMMAND test_stress)
