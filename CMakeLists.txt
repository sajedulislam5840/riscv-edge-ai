cmake_minimum_required(VERSION 3.14)
project(riscv_edge_ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -O2)

include_directories(include)

# Core engine library
add_library(engine_core
    src/ops.cpp
)

# Sanity host runner
add_executable(engine_runner main.cpp)
target_link_libraries(engine_runner PRIVATE engine_core)

# Automated testing suite
enable_testing()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp")
    add_executable(test_tensor tests/test_main.cpp)
    target_link_libraries(test_tensor PRIVATE engine_core)
    add_test(NAME TensorSanityTest COMMAND test_tensor)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gemm.cpp")
    add_executable(test_gemm tests/test_gemm.cpp)
    target_link_libraries(test_gemm PRIVATE engine_core)
    add_test(NAME GemmCorrectnessTest COMMAND test_gemm)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_relu.cpp")
    add_executable(test_relu tests/test_relu.cpp)
    target_link_libraries(test_relu PRIVATE engine_core)
    add_test(NAME ReluActivationTest COMMAND test_relu)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bias_add.cpp")
    add_executable(test_bias_add tests/test_bias_add.cpp)
    target_link_libraries(test_bias_add PRIVATE engine_core)
    add_test(NAME BiasAddTest COMMAND test_bias_add)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_softmax.cpp")
    add_executable(test_softmax tests/test_softmax.cpp)
    target_link_libraries(test_softmax PRIVATE engine_core)
    add_test(NAME SoftmaxTest COMMAND test_softmax)
endif()
