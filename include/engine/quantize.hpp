#ifndef ENGINE_QUANTIZE_HPP
#define ENGINE_QUANTIZE_HPP

#include "engine/tensor.hpp"
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace engine
{

    enum class CalibrationMethod
    {
        MinMax,
        Percentile
    };

    struct QuantParams
    {
        float scale;
        int32_t zero_point;
    };

    // Per-channel quantization parameters (per output column/filter)
    struct PerChannelQuantParams
    {
        std::vector<float> scales;
        std::vector<int32_t> zero_points;

        size_t num_channels() const { return scales.size(); }
    };

    QuantParams calculate_quant_params(float min_val, float max_val);

    // Commit 14: Dynamic Range Calibration Modes
    QuantParams calculate_quant_params_calibrated(
        const Tensor<float> &input,
        CalibrationMethod method = CalibrationMethod::MinMax,
        float percentile = 99.9f);

    PerChannelQuantParams calculate_per_channel_params(const Tensor<float> &weight, bool per_column = true);

    int8_t quantize_val(float val, const QuantParams &params);
    float dequantize_val(int8_t val, const QuantParams &params);

    class QuantizedTensor
    {
    public:
        QuantizedTensor(size_t rows, size_t cols, QuantParams params)
            : rows_(rows), cols_(cols), params_(params), data_(rows * cols, 0) {}

        QuantizedTensor(size_t rows, size_t cols, QuantParams params, const std::vector<int8_t> &values)
            : rows_(rows), cols_(cols), params_(params), data_(values)
        {
            if (values.size() != rows * cols)
            {
                throw std::invalid_argument("Size of initial values does not match tensor dimensions.");
            }
        }

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t size() const { return data_.size(); }

        const QuantParams &params() const { return params_; }
        void set_params(const QuantParams &params) { params_ = params; }

        int8_t *data() { return data_.data(); }
        const int8_t *data() const { return data_.data(); }

        int8_t &operator()(size_t r, size_t c)
        {
            if (r >= rows_ || c >= cols_)
            {
                throw std::out_of_range("QuantizedTensor index out of bounds.");
            }
            return data_[r * cols_ + c];
        }

        const int8_t &operator()(size_t r, size_t c) const
        {
            if (r >= rows_ || c >= cols_)
            {
                throw std::out_of_range("QuantizedTensor index out of bounds.");
            }
            return data_[r * cols_ + c];
        }

        void print() const
        {
            std::cout << "QuantizedTensor (" << rows_ << "x" << cols_
                      << ", scale=" << params_.scale << ", zp=" << params_.zero_point << "):\n";
            for (size_t r = 0; r < rows_; ++r)
            {
                std::cout << "  [ ";
                for (size_t c = 0; c < cols_; ++c)
                {
                    std::cout << static_cast<int>((*this)(r, c)) << " ";
                }
                std::cout << "]\n";
            }
        }

    private:
        size_t rows_;
        size_t cols_;
        QuantParams params_;
        std::vector<int8_t> data_;
    };

    // Conversions
    QuantizedTensor quantize_tensor(const Tensor<float> &input, const QuantParams &params);
    Tensor<float> dequantize_tensor(const QuantizedTensor &input);

    // Per-channel tensor quantization & dequantization
    QuantizedTensor quantize_tensor_per_channel(const Tensor<float> &input, const PerChannelQuantParams &params);
    Tensor<float> dequantize_tensor_per_channel(const QuantizedTensor &input, const PerChannelQuantParams &params);

    // Standard Kernels
    void matmul_int8(const QuantizedTensor &A, const QuantizedTensor &B, QuantizedTensor &C);
    void bias_add(QuantizedTensor &X, const QuantizedTensor &bias);
    void relu(QuantizedTensor &X);
    void relu(const QuantizedTensor &input, QuantizedTensor &output);

    // Per-channel GEMM
    void matmul_int8_per_channel(const QuantizedTensor &A,
                                 const QuantizedTensor &B,
                                 const PerChannelQuantParams &B_params,
                                 QuantizedTensor &C);

} // namespace engine

#endif // ENGINE_QUANTIZE_HPP