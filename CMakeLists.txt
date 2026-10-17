cmake_minimum_required(VERSION 3.14)
project(riscv_edge_ai VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce strict compiler diagnostics
add_compile_options(-Wall -Wextra -Wpedantic -O2)

include_directories(include)

# Core engine library
add_library(engine_core
    src/tensor.cpp
    src/ops.cpp
)

# Sanity host executable
add_executable(engine_runner main.cpp)
target_link_libraries(engine_runner PRIVATE engine_core)

# Automated testing suite
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine_core)
add_test(NAME TensorSanityTest COMMAND test_tensor)

add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE engine_core)
add_test(NAME GemmCorrectnessTest COMMAND test_gemm)
