#include <iostream>
#include "engine/tensor.hpp"

int main() {
    std::cout << "========================================\n";
    std::cout << "  Bare-Metal Neural Engine Initialized   \n";
    std::cout << "  Target ISA: RISC-V RV32I / Custom SIM \n";
    std::cout << "========================================\n";

    engine::Tensor<float> dummy_weight(2, 3, 0.5f);
    dummy_weight(0, 1) = 1.25f;
    dummy_weight(1, 2) = -0.75f;

    std::cout << "\n[INFO] Initialized 2x3 Weight Tensor:\n";
    dummy_weight.print();

    return 0;
}
