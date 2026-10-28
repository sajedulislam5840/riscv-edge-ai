#ifndef ENGINE_QUANTIZED_TENSOR_HPP
#define ENGINE_QUANTIZED_TENSOR_HPP

#include "engine/tensor.hpp"
#include "engine/quantize.hpp"
#include <vector>
#include <cstdint>

namespace engine {

// Quantized tensor storing INT8 values with quantization parameters
class QuantizedTensor {
public:
    QuantizedTensor(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0) {}

    QuantizedTensor(size_t rows, size_t cols, const QuantizationParams& params)
        : rows_(rows), cols_(cols), data_(rows * cols, 0), params_(params) {}

    // Construct from FP32 tensor with automatic quantization
    QuantizedTensor(const Tensor<float>& fp32_tensor, QuantizationScheme scheme = QuantizationScheme::SYMMETRIC)
        : rows_(fp32_tensor.rows()), cols_(fp32_tensor.cols()), data_(fp32_tensor.size()) {

        // Calculate quantization parameters
        params_ = calculate_quantization_params(fp32_tensor.data(), fp32_tensor.size(), scheme);

        // Quantize the data
        quantize_array(fp32_tensor.data(), data_.data(), fp32_tensor.size(), params_);
    }

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return data_.size(); }

    int8_t& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const int8_t& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    int8_t& operator[](size_t idx) {
        return data_[idx];
    }

    const int8_t& operator[](size_t idx) const {
        return data_[idx];
    }

    int8_t* data() noexcept { return data_.data(); }
    const int8_t* data() const noexcept { return data_.data(); }

    const QuantizationParams& get_params() const noexcept { return params_; }
    void set_params(const QuantizationParams& params) { params_ = params; }

    // Convert back to FP32 tensor
    Tensor<float> dequantize() const {
        Tensor<float> result(rows_, cols_, 0.0f);
        dequantize_array(data_.data(), result.data(), size(), params_);
        return result;
    }

    void print() const {
        for (size_t r = 0; r < rows_; ++r) {
            std::cout << "[ ";
            for (size_t c = 0; c < cols_; ++c) {
                std::cout << static_cast<int>((*this)(r, c)) << " ";
            }
            std::cout << "]\n";
        }
    }

private:
    size_t rows_;
    size_t cols_;
    std::vector<int8_t> data_;
    QuantizationParams params_;
};

} // namespace engine

#endif // ENGINE_QUANTIZED_TENSOR_HPP
