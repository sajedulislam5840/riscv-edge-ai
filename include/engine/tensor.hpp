#ifndef ENGINE_TENSOR_HPP
#define ENGINE_TENSOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>

namespace engine {

template <typename T>
class Tensor {
public:
    Tensor(size_t rows, size_t cols, T initial_val = T())
        : rows_(rows), cols_(cols), data_(rows * cols, initial_val) {}

    Tensor(size_t rows, size_t cols, std::initializer_list<T> list)
        : rows_(rows), cols_(cols), data_(list) {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument("Initializer list size does not match tensor dimensions.");
        }
    }

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return data_.size(); }

    T& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const T& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;
};

} // namespace engine

#endif // ENGINE_TENSOR_HPP
