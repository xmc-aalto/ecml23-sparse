//
// Created by erik on 7.3.2023.
//

#ifndef TFSPARSE_CC_KERNELS_FORWARD_CU_H
#define TFSPARSE_CC_KERNELS_FORWARD_CU_H

#include "kernel_data.cu.h"
#include "gpu_vec.cu.h"
#include "strategies.h"
#include "gsl/gsl-lite.hpp"
#include <cstdint>
#include <cassert>

enum class EInnerBatchMode {
    SCALAR, VECTOR, UNROLLED
};

enum class EOuterBatchMode {
    SCALAR, VECTOR
};

/*!
 * For a detailed description of the implementation, see \ref forwardkernels
 */
template<typename Float, typename Int, bool Transpose>
struct ForwardKernelHelpers {
    using ForwardKernelData = KernelDataHelper<Float, Int, true, Transpose>;
    using int_vec_t = typename ForwardKernelData::int_vec_t;
    using float_vec_t = typename ForwardKernelData::float_vec_t;

    // computation functions
    static __device__ Float calc_single_scalar(const ForwardKernelData& data,
                                               std::int32_t batch_index,
                                               std::int32_t unit_index) {
        //! [simple_kernel_batch_outer]
        data.validate_batch_index(batch_index);
        data.validate_unit_index(unit_index);

        float value = 0.f;
        for(std::int32_t i = 0; i < data.lookup_size(); ++i) {
            Int source = data.lookup_at(unit_index, i);
            Float feature = data.feature_at(batch_index, source);
            value += feature * data.weight_at(unit_index, i);
        }
        return value;
        //! [simple_kernel_batch_outer]
    }

    static __device__ Float calc_single_vec(const ForwardKernelData& data,
                                            std::int32_t batch_index,
                                            std::int32_t unit_index) {
        //! [vectorized_kernel_batch_outer]
        data.validate_batch_index(batch_index);
        data.validate_unit_index(unit_index);
        gsl_ExpectsAudit( data.lookup_size() % 4 == 0 );

        Float value = 0.f;
        for(std::int32_t i = 0; i < data.lookup_size(); i += 4) {
            int_vec_t indices = data.lookup_vec_at(unit_index, i);
            float_vec_t weights = data.weight_vec_at(unit_index, i);
#pragma unroll
            for(int j = 0; j < 4; ++j) {
                Float feature = data.feature_at(batch_index, read(indices, j));
                value += feature * read(weights, j);
            }
        }

        return value;
        //! [vectorized_kernel_batch_outer]
    }

    static __device__ void calc_batch_scalar(ForwardKernelData& data,
                                             std::int32_t unit_index,
                                             std::int32_t start=0) {
        //! [simple_kernel_batch_inner]
        data.validate_unit_index(unit_index);
        data.validate_weight_index(start);

        for (std::int32_t i = start; i < data.lookup_size(); ++i) {
            Int source = data.lookup_at(unit_index, i);
            Float weight = data.weight_at(unit_index, i);
            for (std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
                Float feature = data.feature_at(batch_index, source);
                data.output_at(batch_index, unit_index) += feature * weight;
            }
        }
        //! [simple_kernel_batch_inner]
    }

    static __device__ void calc_batch_vec(ForwardKernelData& data,
                                          std::int32_t unit_index,
                                          std::int32_t start) {
        //! [vectorized_kernel_batch_inner]
        data.validate_unit_index(unit_index);
        data.validate_weight_index(start);

        gsl_ExpectsAudit( data.lookup_size() % 4 == 0 );
        gsl_ExpectsAudit( start % 4 == 0 );

        for (std::int32_t i = start; i < data.lookup_size(); i += 4) {
            int_vec_t indices = data.lookup_vec_at(unit_index, i);
            float_vec_t weights = data.weight_vec_at(unit_index, i);
            for (std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
                Float local_accumulator = 0;
#pragma unroll
                for(int j = 0; j < 4; ++j) {
                    local_accumulator += data.feature_at(batch_index, read(indices, j)) * read(weights, j);
                }
                data.output_at(batch_index, unit_index) += local_accumulator;
            }
        }
        //! [vectorized_kernel_batch_inner]
    }


    template<int Steps>
    static __device__ void calc_batch_unrolled(ForwardKernelData& data,
                                               std::integral_constant<int, Steps> steps,
                                               std::int32_t unit_index,
                                               std::int32_t start) {
        //! [unrolled_kernel_batch_inner-prologue]
        static_assert(Steps % 4 == 0, "Steps need to be a multiple of 4.");

        data.validate_unit_index(unit_index);
        data.validate_weight_index(start);

        gsl_ExpectsAudit( data.lookup_size() % 4 == 0 );
        gsl_ExpectsAudit( start % 4 == 0 );

        //! [unrolled_kernel_batch_inner-prologue]

        //! [unrolled_kernel_batch_inner-loading]
        Int cached_source[Steps];
        Float cached_weights[Steps];

#pragma unroll
        for(std::int32_t offset = 0; offset < Steps; offset += 4) {
            int_vec_t source_vec = data.lookup_vec_at(unit_index, start + offset);
            float_vec_t weight_vec = data.weight_vec_at(unit_index, start + offset);
#pragma unroll
            for(int k = 0; k < 4; ++k) {
                cached_source[offset + k] = read(source_vec, k);
                cached_weights[offset + k] = read(weight_vec, k);
            }
        }
        //! [unrolled_kernel_batch_inner-loading]

        //! [unrolled_kernel_batch_inner-loop]
        for (std::int32_t batch_index = 0; batch_index < data.batch_size(); ++batch_index) {
            Float accumulator = 0.0;
#pragma unroll
            for(std::int32_t offset = 0; offset < Steps; ++offset) {
                accumulator += data.feature_at(batch_index, cached_source[offset]) * cached_weights[offset];
            }
            data.output_at(batch_index, unit_index) += accumulator;
        }
        //! [unrolled_kernel_batch_inner-loop]
    }

    // -----------------------------------------------------------------------------------------------------------------
    //      kernels

    template<EOuterBatchMode Mode>
    static __device__ void outer_batch_kernel(ForwardKernelData& data, std::integral_constant<EOuterBatchMode, Mode> mode) {
        //! [simple_kernel_batch_outer-global]
        std::int32_t batch_index = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_index = blockIdx.x * blockDim.x + threadIdx.x;

        if(batch_index >= data.batch_size() || unit_index >= data.output_size()) {
            return;
        }

        if constexpr (Mode == EOuterBatchMode::VECTOR) {
            data.output_at(batch_index, unit_index) = calc_single_vec(data, batch_index, unit_index);
        } else {
            data.output_at(batch_index, unit_index) = calc_single_scalar(data, batch_index, unit_index);
        }
        //! [simple_kernel_batch_outer-global]
    }

    template<EOuterBatchMode Mode>
    static __device__ void outer_batch_kernel_tp(ForwardKernelData& data, std::integral_constant<EOuterBatchMode, Mode> mode) {
        //! [simple_kernel_batch_outer-global]
        std::int32_t unit_index_offset = blockIdx.y * blockDim.y + threadIdx.y;
        std::int32_t unit_block_size = blockDim.y * gridDim.y;
        std::int32_t batch_index = blockIdx.x * blockDim.x + threadIdx.x;

        if(batch_index >= data.batch_size()) {
            return;
        }

        for(std::int32_t unit_index = unit_index_offset; unit_index < data.output_size(); unit_index += unit_block_size) {
            if constexpr (Mode == EOuterBatchMode::VECTOR) {
                data.output_at(batch_index, unit_index) = calc_single_vec(data, batch_index, unit_index);
            } else {
                data.output_at(batch_index, unit_index) = calc_single_scalar(data, batch_index, unit_index);
            }
        }
        //! [simple_kernel_batch_outer-global]
    }

    template<EInnerBatchMode Mode>
    static __device__ void inner_batch_kernel(ForwardKernelData& data, std::integral_constant<EInnerBatchMode, Mode> mode) {
        //! [simple_kernel_batch_outer-global]
        std::int64_t unit_index = blockIdx.x * blockDim.x + threadIdx.x;
        if(unit_index >= data.output_size()) {
            return;
        }

        if constexpr (Mode == EInnerBatchMode::SCALAR) {
            calc_batch_scalar(data, unit_index, 0);
        } else if (Mode == EInnerBatchMode::VECTOR) {
            calc_batch_vec(data, unit_index, 0);
        } else {
            // Fan in needs to be a multiple of 4, otherwise we cannot use the vectorized load in the optimized kernel
            int mul32_fan_in = (data.lookup_size() / 32) * 32;
            for (int i = 0; i < mul32_fan_in; i += 32) {
                calc_batch_unrolled(data, std::integral_constant<int, 32>{}, unit_index, i);
            }
            calc_batch_vec(data, unit_index, mul32_fan_in);
        }
        //! [simple_kernel_batch_outer-global]
    }

};

// ---------------------------------------------------------------------------------------------------------------------


template<typename Float, typename Int, bool Transpose>
__global__ void forward_ob_scalar(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::outer_batch_kernel(data, make_const<EOuterBatchMode::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ob_vector(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::outer_batch_kernel(data, make_const<EOuterBatchMode::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ob_scalar_tp(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::outer_batch_kernel_tp(data, make_const<EOuterBatchMode::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ob_vector_tp(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::outer_batch_kernel_tp(data, make_const<EOuterBatchMode::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ib_scalar(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::inner_batch_kernel(data, make_const<EInnerBatchMode::SCALAR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ib_vector(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::inner_batch_kernel(data, make_const<EInnerBatchMode::VECTOR>{});
}

template<typename Float, typename Int, bool Transpose>
__global__ void forward_ib_unrolled(KernelDataHelper<Float, Int, true, Transpose> data) {
    ForwardKernelHelpers<Float, Int, Transpose>::inner_batch_kernel(data, make_const<EInnerBatchMode::UNROLLED>{});
}
template<typename Float, typename Int, bool Transpose, ForwardImplementations Strategy>
__host__ void launch_forward(KernelDataHelper<Float, Int, true, Transpose>& data,
                             FwdImpTag<Strategy> strategy,
                             dim3 block,
                             cudaStream_t stream) {
    if constexpr (is_vectorized(Strategy)) {
        gsl_Expects(data.lookup_size() % 4 == 0);
    }

    // dispatch the implementations
    auto grid = data.get_grid_dim(block, get_grid(strategy));
    if constexpr (Strategy == ForwardImplementations::GPU_Fast) {
        forward_ib_unrolled<<<grid, block, 0, stream>>>(data);
    } else if constexpr (Strategy == ForwardImplementations::GPU_InnerBatchVectorized) {
        forward_ib_vector<<<grid, block, 0, stream>>>(data);
    } else if constexpr (Strategy == ForwardImplementations::GPU_InnerBatchNaive) {
        forward_ib_scalar<<<grid, block, 0, stream>>>(data);
    } else if constexpr (Strategy == ForwardImplementations::GPU_OuterBatchNaive) {
        forward_ob_scalar<<<grid, block, 0, stream>>>(data);
    } else if constexpr (Strategy == ForwardImplementations::GPU_OuterBatchVectorized) {
        forward_ob_vector<<<grid, block, 0, stream>>>(data);
    } else if constexpr (Strategy == ForwardImplementations::GPU_TP_OuterBatchNaive) {
        forward_ob_scalar_tp<<<grid, block, 0, stream>>>(data);
    }
}


#endif //TFSPARSE_CC_KERNELS_FORWARD_CU_H
