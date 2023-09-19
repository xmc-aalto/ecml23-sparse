//
// Created by erik on 12.3.2023.
//

#ifndef TFSPARSE_CC_KERNELS_BACKWARD_CU_H
#define TFSPARSE_CC_KERNELS_BACKWARD_CU_H

#include "kernel_data.cu.h"

template<typename Float, typename Int, bool Transpose>
class BackwardDataHelper : public KernelDataHelper<Float, Int, false, Transpose> {
public:
    using Base = KernelDataHelper<Float, Int, false, Transpose>;
    BackwardDataHelper(const Float* __restrict__ features, const Int* __restrict__ lookup,
                       const Float* __restrict__ weights, const Float* __restrict__ outputs,
                       Float* __restrict__ feature_grad, Float* __restrict__ weight_grad,
                       std::int32_t batch_size, std::int32_t input_size, std::int32_t output_size,
                       std::int32_t lookup_size);

    __device__ Float& feature_grad_at(std::int32_t batch, std::int32_t feature) const {
        this->validate_batch_index(batch);
        this->validate_feature_index(feature);
        if constexpr (Transpose) {
            return mFeatureGradPtr[feature * this->batch_size() + batch];
        } else {
            return mFeatureGradPtr[batch * this->input_size() + feature];
        }
    }

    __device__ Float& weight_grad_at(std::int32_t unit, std::int32_t index) const {
        this->validate_unit_index(unit);
        this->validate_weight_index(index);
        return mWeightGradPtr[unit * this->lookup_size() + index];
    }
private:
    Float* __restrict__ mFeatureGradPtr = nullptr;
    Float* __restrict__ mWeightGradPtr  = nullptr;
};

template<typename Float, typename Int, bool Transpose>
inline BackwardDataHelper<Float, Int, Transpose>::BackwardDataHelper(
    const Float* __restrict__ features, const Int* __restrict__ lookup,
    const Float* __restrict__ weights, const Float* __restrict__ outputs,
    Float* __restrict__ feature_grad, Float* __restrict__ weight_grad,
    std::int32_t batch_size, std::int32_t input_size, std::int32_t output_size,
    std::int32_t lookup_size) : Base(features, lookup, weights, outputs,
                                     batch_size, input_size, output_size, lookup_size),
                                mFeatureGradPtr(feature_grad), mWeightGradPtr(weight_grad) {

}

enum class EBackwardStrategy {
    SCALAR, VECTOR
};

template<typename Float, typename Int, bool Transpose>
struct BackwardKernelHelpers {
    using BackwardKernelData = BackwardDataHelper<Float, Int, Transpose>;
    using int_vec_t = typename BackwardKernelData::int_vec_t;
    using float_vec_t = typename BackwardKernelData::float_vec_t;

    static __device__ void weight_grad_scalar(BackwardKernelData& data,
                                              std::int32_t unit_index,
                                              std::int32_t weight_index
    ) {
        data.validate_weight_index(weight_index);
        data.validate_unit_index(unit_index);

        Int source = data.lookup_at(unit_index, weight_index);
        Float result = 0;
        for(std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
            Float out = data.output_at(batch_index, unit_index);
            if(out == 0) continue;

            Float feature = data.feature_at(batch_index, source);
            result += feature * out;
        }
        data.weight_grad_at(unit_index, weight_index) = result;
    }

    static __device__ void weight_grad_vector(BackwardKernelData& data,
                                                     std::int32_t unit_index,
                                                     std::int32_t weight_index
    ) {
        data.validate_weight_index(weight_index);
        data.validate_unit_index(unit_index);
        gsl_AssertAudit( weight_index % 4 == 0 );

        auto source4 = data.lookup_vec_at(unit_index, weight_index);
        float_vec_t result{0, 0, 0, 0};

        for(std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
            Float out = data.output_at(batch_index, unit_index);
            if(out == 0) continue;

            for (int k = 0; k < 4; ++k) {
                Float feature = data.feature_at(batch_index, read(source4, k));
                ref(result, k) += feature * out;
            }
        }

        for(int k = 0; k < 4; ++k) {
            data.weight_grad_at(unit_index, weight_index + k) = read(result, k);
        }
    }

    static __device__ void feature_grad_scalar(BackwardKernelData& data,
                                               std::int32_t unit_index,
                                               std::int32_t weight_index
    ) {
        data.validate_weight_index(weight_index);
        data.validate_unit_index(unit_index);

        Int source = data.lookup_at(unit_index, weight_index);
        Float weight = data.weight_at(unit_index, weight_index);
        for(std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
            Float out = data.output_at(batch_index, unit_index);
            Float increment = weight * out;
            atomicAdd(&data.feature_grad_at(batch_index, source), increment);
        }
    }

    static __device__ void feature_grad_scalar_tp(BackwardKernelData& data,
                                                  std::int32_t batch_index,
                                                  std::int32_t unit_index
    ) {
        data.validate_batch_index(batch_index);
        data.validate_unit_index(unit_index);

        Float out = data.output_at(batch_index, unit_index);
        if(out == 0) {
            return;
        }

        for(std::int32_t weight_index = 0; weight_index < data.lookup_size(); ++weight_index) {
            Int source = data.lookup_at(unit_index, weight_index);
            Float weight = data.weight_at(unit_index, weight_index);
            Float increment = weight * out;
            atomicAdd(&data.feature_grad_at(batch_index, source), increment);
        }
    }

    static __device__ void feature_grad_vector_tp(BackwardKernelData& data,
                                                  std::int32_t batch_index,
                                                  std::int32_t unit_index
    ) {
        data.validate_batch_index(batch_index);
        data.validate_unit_index(unit_index);

        Float out = data.output_at(batch_index, unit_index);
        if(out == 0) {
            return;
        }

        for(std::int32_t weight_index = 0; weight_index < data.lookup_size(); weight_index += 4) {
            auto source4 = data.lookup_vec_at(unit_index, weight_index);
            auto weight4 = data.weight_vec_at(unit_index, weight_index);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                Float increment = read(weight4, k) * out;
                atomicAdd(&data.feature_grad_at(batch_index, read(source4, k)), increment);
            }
        }
    }


    static __device__ void feature_grad_vector(BackwardKernelData& data,
                                               std::int32_t unit_index,
                                               std::int32_t weight_index
    ) {
        data.validate_weight_index(weight_index);
        data.validate_unit_index(unit_index);
        gsl_AssertAudit( weight_index % 4 == 0 );

        auto source4 = data.lookup_vec_at(unit_index, weight_index);
        auto weight4 = data.weight_vec_at(unit_index, weight_index);
        for(std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
            Float out = data.output_at(batch_index, unit_index);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                Float increment = read(weight4, k) * out;
                atomicAdd(&data.feature_grad_at(batch_index, read(source4, k)), increment);
            }
        }
    }

    //

    static __device__ void weight_grad_kernel(BackwardKernelData& data, make_const<EBackwardStrategy::SCALAR>) {
        std::int32_t weight_index = blockIdx.x * blockDim.x + threadIdx.x;
        std::int32_t unit_index = blockIdx.y * blockDim.y + threadIdx.y;
        if(unit_index >= data.output_size() || weight_index >= data.lookup_size()) {
            return;
        }

        weight_grad_scalar(data, unit_index, weight_index);
    }

    static __device__ void weight_grad_kernel(BackwardKernelData& data, make_const<EBackwardStrategy::VECTOR>) {
        std::int32_t weight_index = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
        std::int32_t unit_index = blockIdx.y * blockDim.y + threadIdx.y;
        if(unit_index >= data.output_size() || weight_index >= data.lookup_size()) {
            return;
        }

        weight_grad_vector(data, unit_index, weight_index);
    }

    static __device__ void feature_grad_kernel(BackwardKernelData& data, make_const<EBackwardStrategy::SCALAR>) {
        std::int32_t weight_index = blockIdx.x * blockDim.x + threadIdx.x;

        if(weight_index >= data.lookup_size()) {
            return;
        }

        std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_block_size = blockDim.y * gridDim.y;

        for(std::int32_t unit_index = unit_index_offset; unit_index < data.output_size(); unit_index += unit_block_size) {
            feature_grad_scalar(data, unit_index, weight_index);
        }
    }

    static __device__ void feature_grad_kernel_tp(BackwardKernelData& data, make_const<EBackwardStrategy::SCALAR>) {
        std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_block_size = blockDim.y * gridDim.y;
        std::int32_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;

        if(batch_index >= data.batch_size()) {
            return;
        }

        for(std::int32_t unit_index = unit_index_offset; unit_index < data.output_size(); unit_index += unit_block_size) {
            feature_grad_scalar_tp(data, batch_index, unit_index);
        }
    }

    static __device__ void feature_grad_kernel(BackwardKernelData& data, make_const<EBackwardStrategy::VECTOR>) {
        std::int32_t weight_index = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
        if(weight_index >= data.lookup_size()) {
            return;
        }

        std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_block_size = blockDim.y * gridDim.y;

        for(std::int32_t unit_index = unit_index_offset; unit_index < data.output_size(); unit_index += unit_block_size) {
            feature_grad_vector(data, unit_index, weight_index);
        }
    }

    static __device__ void feature_grad_kernel_tp(BackwardKernelData& data, make_const<EBackwardStrategy::VECTOR>) {
        gsl_AssertAudit( data.lookup_size() % 4 == 0 );
        std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_block_size = blockDim.y * gridDim.y;
        std::int32_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;

        if(batch_index >= data.batch_size()) {
            return;
        }

        for(std::int32_t unit_index = unit_index_offset; unit_index < data.output_size(); unit_index += unit_block_size) {
            feature_grad_vector_tp(data, batch_index, unit_index);
        }
    }

};

template<typename Float, typename Int, bool Transpose>
__global__ void backward_weights_scalar(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::weight_grad_kernel(data, make_const<EBackwardStrategy::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void backward_features_scalar(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::feature_grad_kernel(data, make_const<EBackwardStrategy::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void backward_features_scalar_tp(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::feature_grad_kernel_tp(data, make_const<EBackwardStrategy::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void backward_features_vector_tp(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::feature_grad_kernel_tp(data, make_const<EBackwardStrategy::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void backward_weights_vector(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::weight_grad_kernel(data, make_const<EBackwardStrategy::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void backward_features_vector(BackwardDataHelper<Float, Int, Transpose> data) {
    BackwardKernelHelpers<Float, Int, Transpose>::feature_grad_kernel(data, make_const<EBackwardStrategy::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__host__ void weight_grad_launch_scalar(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    auto grid = data.get_2d_grid_dim(block, EShapeIdentifier::OutputSize, EShapeIdentifier::LookupSize);
    backward_weights_scalar<<<grid, block, 0, stream>>>(data);
}

template<typename Float, typename Int, bool Transpose>
__host__ void feature_grad_launch_scalar(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    auto grid = data.get_2d_grid_dim(block, EShapeIdentifier::OutputSize, EShapeIdentifier::LookupSize);
    backward_features_scalar<<<grid, block, 0, stream>>>(data);
}

template<typename Float, typename Int, bool Transpose>
__host__ void feature_grad_launch_scalar_tp(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    auto grid = data.get_2d_grid_dim(block,  EShapeIdentifier::OutputSize, EShapeIdentifier::BatchSize);
    backward_features_scalar_tp<<<grid, block, 0, stream>>>(data);
}

template<typename Float, typename Int, bool Transpose>
__host__ void feature_grad_launch_vector_tp(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    gsl_Expects(data.lookup_size() % 4 == 0);
    auto grid = data.get_2d_grid_dim(block,  EShapeIdentifier::OutputSize, EShapeIdentifier::BatchSize);
    backward_features_vector_tp<<<grid, block, 0, stream>>>(data);
}

template<typename Float, typename Int, bool Transpose>
__host__ void weight_grad_launch_vector_weights(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    gsl_Expects(data.lookup_size() % 4 == 0);
    auto grid = data.get_grid_dim(dim3(data.lookup_size() / 4, data.output_size(), 1), block);
    backward_weights_vector<<<grid, block, 0, stream>>>(data);
}

template<typename Float, typename Int, bool Transpose>
__host__ void feature_grad_launch_vector_weights(BackwardDataHelper<Float, Int, Transpose>& data, dim3 block, cudaStream_t stream) {
    gsl_Expects(data.lookup_size() % 4 == 0);
    auto grid = data.get_grid_dim(dim3(data.lookup_size() / 4, data.output_size(), 1), block);
    backward_features_vector<<<grid, block, 0, stream>>>(data);
}

#endif //TFSPARSE_CC_KERNELS_BACKWARD_CU_H
