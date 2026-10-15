#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>

namespace engine {

template <typename T>
class Tensor {
public:
    
    Tensor(size_t rows, size_t cols, T initial_val = T())
        : rows_(rows), cols_(cols), data_(rows * cols, initial_val) {}

 
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }


    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

 
    T& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const T& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    // 1D ডিরেক্ট ইনডেক্সিং
    T& operator[](size_t idx) {
        return data_[idx];
    }

    const T& operator[](size_t idx) const {
        return data_[idx];
    }

    void print() const {
        for (size_t i = 0; i < rows_; ++i) {
            std::cout << "[ ";
            for (size_t j = 0; j < cols_; ++j) {
                std::cout << static_cast<double>((*this)(i, j)) << " ";
            }
            std::cout << "]\n";
        }
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_; // 1D Continuous Flat Buffer
};

} // namespace engine

#endif // ENGINE_TENSOR_HPP
