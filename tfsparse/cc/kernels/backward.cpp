//
// Created by erik on 16.2.2023.
//

#include "common.h"

template<typename Float, typename Int, bool Transpose>
void weight_grad_for_range(std::int64_t first, std::int64_t last,
                           const MatrixType<const Float>& features,
                           const MatrixType<const Int>& lookup,
                           const MatrixType<const Float>& weights,
                           const MatrixType<const Float>& output,
                           const MatrixType<Float>& gradient)
{
    std::int64_t fan_in = weights.dimension(1);
    std::int64_t batch_size = Transpose ? features.dimension(1) : features.dimension(0);

    auto feature_at = [&](int batch_index, int feature_index) {
        if constexpr (Transpose) {
            return features(feature_index, batch_index);
        } else {
            return features(batch_index, feature_index);
        }
    };

    for(std::int64_t unit_index = first; unit_index < last; ++unit_index) {
        for(int i = 0; i < fan_in; ++i) {
            Int source = lookup(unit_index, i);
            Float result = 0;
            for(std::int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
                result += feature_at(batch_index, source) * output(batch_index, unit_index);
            }
            gradient(unit_index, i) = result;
        }
    }
}

template<typename Float, typename Int, bool Transpose>
void feature_grad_for_range(std::int64_t first, std::int64_t last,
                            const MatrixType<const Float>& features,
                            const MatrixType<const Int>& lookup,
                            const MatrixType<const Float>& weights,
                            const MatrixType<const Float>& output,
                            const MatrixType<Float>& gradient)
{
    std::int64_t num_features = Transpose ? features.dimension(0) : features.dimension(1);
    std::int64_t num_units = lookup.dimension(0);
    std::int64_t fan_in = weights.dimension(1);

    auto grad_at = [&](std::int32_t batch_index, std::int32_t feature_index) -> Float& {
        if constexpr (Transpose) {
            return gradient(feature_index, batch_index);
        } else {
            return gradient(batch_index, feature_index);
        }
    };

    for(std::int64_t batch_index = first; batch_index < last; ++batch_index) {
        for(std::int64_t feature_index = 0; feature_index < num_features; ++feature_index) {
            grad_at(batch_index, feature_index) = Float(0);
        }
    }

    for(std::int64_t batch_index = first; batch_index < last; ++batch_index) {
        for(std::int64_t unit_index = 0; unit_index < num_units; ++unit_index) {
            Float out = output(batch_index, unit_index);
            for(int i = 0; i < fan_in; ++i) {
                Int source = lookup(unit_index, i);
                grad_at(batch_index, source) += weights(unit_index, i) * out;
            }
        }
    }
}


template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulGradKernel<CPUDevice, Float, Int, BackwardImplementations::CPU, Transpose> {
    void operator()(const CPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& output,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient
                    ) {

        std::int64_t batch_size = Transpose ? features.dimension(1) : features.dimension(0);
        std::int64_t num_features = Transpose ? features.dimension(0) : features.dimension(1);
        std::int64_t num_units = lookup.dimension(0);
        std::int64_t fan_in = weights.dimension(1);

        auto wg_for_sub_batch = [&](std::int64_t first, std::int64_t last){
            weight_grad_for_range<Float, Int, Transpose>(first, last, features, lookup, weights, output, weight_gradient);
        };

        device.parallelFor(
            num_units,
            Eigen::TensorOpCost(batch_size * fan_in, fan_in, batch_size * fan_in),
            wg_for_sub_batch);

        auto fg_for_sub_batch = [&](std::int64_t first, std::int64_t last){
            feature_grad_for_range<Float, Int, Transpose>(first, last, features, lookup, weights, output, feature_gradient);
        };

        device.parallelFor(
            batch_size,
            Eigen::TensorOpCost(num_units * fan_in, batch_size * num_features, num_units * fan_in),
            fg_for_sub_batch);
    }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::int32_t, BackwardImplementations::CPU, false>;
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::uint32_t, BackwardImplementations::CPU, false>;
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::int32_t, BackwardImplementations::CPU, true>;
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::uint32_t, BackwardImplementations::CPU, true>;