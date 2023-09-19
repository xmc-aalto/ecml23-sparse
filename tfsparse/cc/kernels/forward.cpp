//
// Created by erik on 30.3.2022.
//

#include "common.h"

// Implement the actual computation for a range of outputs
template<typename Float, typename Int, bool Transpose>
void matmul_for_range(std::int64_t first, std::int64_t last,
                      const MatrixType<const Float>& features,
                      const MatrixType<const Int>& lookup,
                      const MatrixType<const Float>& weights,
                      const MatrixType<Float>& output,
                      std::bool_constant<Transpose> transpose)
{
    std::int64_t num_batches = Transpose ? features.dimension(1) : features.dimension(0);
    std::int64_t fan_in = weights.dimension(1);

    auto feature_at = [&](int batch_index, int feature_index) {
        if constexpr (Transpose) {
            return features(feature_index, batch_index);
        } else {
            return features(batch_index, feature_index);
        }
    };

    for(std::int64_t unit_index = first; unit_index < last; ++unit_index) {
        for(std::int64_t batch_index = 0; batch_index < num_batches; ++batch_index) {
            Float values[4] = {};
            std::int32_t i = 0;
            for(; i + 4 <= fan_in; i += 4) {
                for(int j = 0; j < 4; ++j) {
                    Int source = lookup(unit_index, i + j);
                    values[j] += feature_at(batch_index, source) * weights(unit_index, i + j);
                }
            }
            for (; i < fan_in; ++i) {
                Int source = lookup(unit_index, i);
                values[0] += feature_at(batch_index, source) * weights(unit_index, i);
            }
            output(batch_index, unit_index) = values[0] + values[1] + values[2] + values[3];
        }
    }
}

template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulKernel<CPUDevice, Float, Int, ForwardImplementations::CPU, Transpose> {
    void operator()(const CPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    MatrixType<Float> output) {

        std::int64_t batch_size = Transpose ? features.dimension(1) : features.dimension(0);
        std::int64_t num_units = lookup.dimension(0);
        std::int64_t fan_in = weights.dimension(1);

        auto run_for_sub_batch = [&](std::int64_t first, std::int64_t last){
            matmul_for_range(first, last, features, lookup, weights, output, std::bool_constant<Transpose>{});
        };

        device.parallelFor(
            num_units,
            Eigen::TensorOpCost(batch_size * fan_in, batch_size, batch_size * fan_in),
            run_for_sub_batch);
    }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::int32_t, ForwardImplementations::CPU, false>;
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::uint32_t, ForwardImplementations::CPU, false>;
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::int32_t, ForwardImplementations::CPU, true>;
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::uint32_t, ForwardImplementations::CPU, true>;
