#include <iostream>
#include <cassert>
#include "engine/tensor.hpp"

void test_tensor_allocation() {
    engine::Tensor<float> t(2, 3, 1.5f);
    assert(t.rows() == 2);
    assert(t.cols() == 3);
    assert(t.size() == 6);
    assert(t(0, 0) == 1.5f);
    assert(t(1, 2) == 1.5f);
    std::cout << "  [PASS] Tensor allocation and initialization test\n";
}

void test_row_major_indexing() {
    engine::Tensor<int> t(2, 2);
    t(0, 0) = 10;
    t(0, 1) = 20;
    t(1, 0) = 30;
    t(1, 1) = 40;

    // মেমরিতে এক লাইনে ক্রমান্বয়ে সাজানো আছে কি না চেক
    assert(t[0] == 10);
    assert(t[1] == 20);
    assert(t[2] == 30);
    assert(t[3] == 40);
    std::cout << "  [PASS] Row-major continuous memory layout test\n";
}

int main() {
    std::cout << "[TEST SUITE] Running Day 2 Tensor Tests...\n";
    test_tensor_allocation();
    test_row_major_indexing();
    std::cout << "[TEST SUITE] All Day 2 tests passed successfully!\n";
    return 0;
}
