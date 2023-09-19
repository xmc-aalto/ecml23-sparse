//
// Created by erik on 12.3.2023.
//

#ifndef TFSPARSE_CC_KERNELS_KERNEL_DATA_CU_H
#define TFSPARSE_CC_KERNELS_KERNEL_DATA_CU_H

#include <cstdint>
#include <gsl/gsl-lite.hpp>
#include "gpu_vec.cu.h"
#include "strategies.h"

/*!
 * \brief Base class for \ref ForwardKernelHelper
 * \details This class extracts out of \ref ForwardKernelHelper template all the computations that can be performed
 * independently of the actual types involved. Thus, this class manages the shape-related information for the forward
 * kernels.
 */
class ShapeDataHelper {
public:
    ShapeDataHelper(std::int32_t batch_size, std::int32_t input_size,
                    std::int32_t output_size, std::int32_t lookup_size);

    // index validator functions
    /// Checks that `batch` is a valid batch index in debug mode, and a no-op for release.
    __host__ __device__ void validate_batch_index(std::int32_t batch [[maybe_unused]]) const {
        gsl_ExpectsAudit(0 <= batch && batch < batch_size());
    }

    /// Checks that `batch` is a valid input feature index in debug mode, and a no-op for release.
    __host__ __device__ void validate_feature_index(std::int32_t feature [[maybe_unused]]) const {
        gsl_ExpectsAudit(0 <= feature && feature < input_size());
    }

    /// Checks that `unit` is a valid output index in debug mode, and a no-op for release.
    __host__ __device__ void validate_unit_index(std::int32_t unit [[maybe_unused]]) const {
        gsl_ExpectsAudit(0 <= unit && unit < output_size());
    }

    /// Checks that `weight` is a valid weight lookup index in debug mode, and a no-op for release.
    __host__ __device__ void validate_weight_index(std::int32_t weight [[maybe_unused]]) const {
        gsl_ExpectsAudit(0 <= weight && weight < lookup_size());
    }

    // size info functions
    /// Gets the batch size.
    __host__ __device__ std::int32_t batch_size() const  { return mBatchSize; }

    /// Gets the size (feature dimension) of the input.
    __host__ __device__ std::int32_t input_size() const  { return mInputSize; }

    /// Gets the size (number of units) of the output.
    __host__ __device__ std::int32_t output_size() const { return mOutputSize; }

    /// Gets the size (number of structural non-zeros) of the weight values and corresponding indices.
    __host__ __device__ std::int32_t lookup_size() const { return mLookupSize; }

    /// Gets the size based on a runtime selector
    __host__ std::int32_t get_size_of(EShapeIdentifier direction) const {
        switch (direction) {
            case EShapeIdentifier::BatchSize:
                return batch_size();
            case EShapeIdentifier::InputSize:
                return input_size();
            case EShapeIdentifier::OutputSize:
                return output_size();
            case EShapeIdentifier::LookupSize:
                return lookup_size();
        }
        gsl_FailFast();
    }

    // launch config helpers
    static __host__ dim3 get_grid_dim(dim3 grid_size, dim3 block_size);

    __host__ dim3 get_1d_grid_dim(dim3 block, EShapeIdentifier direction) const;
    __host__ dim3 get_2d_grid_dim(dim3 block, EShapeIdentifier y_dir, EShapeIdentifier x_dir) const;

    template<class... T>
    __host__ dim3 get_grid_dim(dim3 block, std::tuple<T...> shape) const;

private:
    // sizes
    std::int32_t mBatchSize = -1;
    std::int32_t mInputSize = -1;
    std::int32_t mOutputSize = -1;
    std::int32_t mLookupSize = -1;
};

inline ShapeDataHelper::ShapeDataHelper(std::int32_t batch_size, std::int32_t input_size,
                                 std::int32_t output_size, std::int32_t lookup_size) :
    mBatchSize(batch_size), mInputSize(input_size), mOutputSize(output_size), mLookupSize(lookup_size) {

    gsl_Expects(batch_size > 0);
    gsl_Expects(input_size > 0);
    gsl_Expects(output_size > 0);
    gsl_Expects(lookup_size > 0);
}

inline __host__ dim3 ShapeDataHelper::get_1d_grid_dim(dim3 block, EShapeIdentifier direction) const {
    gsl_Expects(block.y == 1);
    gsl_Expects(block.z == 1);
    std::int32_t grid_size = get_size_of(direction);
    return get_grid_dim({(unsigned)grid_size, 1, 1}, block);
}

inline __host__ dim3 ShapeDataHelper::get_2d_grid_dim(dim3 block, EShapeIdentifier y_dir, EShapeIdentifier x_dir) const {
    gsl_Expects(block.z == 1);
    std::int32_t x_grid = get_size_of(x_dir);
    std::int32_t y_grid = get_size_of(y_dir);
    return get_grid_dim({(unsigned)x_grid, (unsigned)y_grid, 1}, block);
}

template<class... T>
inline __host__ dim3 ShapeDataHelper::get_grid_dim(dim3 block, std::tuple<T...> shape) const {
    if constexpr (sizeof...(T) == 1) {
        return get_1d_grid_dim(block, std::get<0>(shape));
    } else if constexpr (sizeof...(T) == 2) {
        return get_2d_grid_dim(block, std::get<0>(shape), std::get<1>(shape));
    } else {
        gsl_FailFast();
    }
}

inline __host__ dim3 ShapeDataHelper::get_grid_dim(dim3 grid_size, dim3 block_size) {
    dim3 grid_dim = {
        (grid_size.x + block_size.x - 1) / block_size.x,
        (grid_size.y + block_size.y - 1) / block_size.y,
        (grid_size.z + block_size.z - 1) / block_size.z
    };

    // ensure that y-grid does not overflow.
    /// TODO we need to make sure this does not result in wrong calculations,
    /// as it changes the assumptions about the grid.
    if(grid_dim.y > 65'535) {
        grid_dim.y = 65'535;
    }

    gsl_Ensures(grid_dim.x != 0);
    gsl_Ensures(grid_dim.y != 0 && grid_dim.y <= 65'535);
    gsl_Ensures(grid_dim.z != 0 && grid_dim.z <= 65'535);
    return grid_dim;
}

/*!
 * \brief Creates a template function with two arguments of name `FUN` that `static_assert`s if called with a
 * non-integer parameter.
 * \details This macro creates a new function that takes its two arguments as template parameters, but is disabled
 * by SFINAE if both are integers. If the function actually gets instantiated, it triggers a `static_assert` that
 * warns about the non-integer parameters. The usage is as follows: For a function `FUN` that takes two integer
 * parameters, we additionally provide the overload with this macro. If, by accident, the function is called with a
 * non-integer, this template is a better match, and we get an error message instead of broken code.
 *
 * Note that we cannot simply use `-Werror=float-conversion`, as nvcc does not understand this.
 */
#define DISABLE_NON_INTEGER_ARGUMENTS(FUN) \
template<class T, class U, class Sfinae=std::enable_if_t<!std::is_integral_v<T> || !std::is_integral_v<U>, void>> \
constexpr void FUN(T a, U b) const { \
    static_assert(std::is_integral_v<T> && std::is_integral_v<U>,\
                  #FUN "has been called with a non-integer argument");                                             \
  }

/*!
 * \brief Helper class to manage access to the data arrays.
 * \tparam Float Type to be used for floating point values.
 * \tparam Int Type to be used for the index integers in the sparse lookup.
 * \tparam WriteOutput Whether the `output` is writable (forward pass) or constant (backward pass)
 *
 * \details This class serves as a helper for implementing the CUDA kernels for the
 * the stratified-sparse matrix multiplication. It serves two purposes:
 * 1) By only accessing the underlying arrays through the `feature_at()`, `lookup_at()`, `weight_at()`
 * and `output_at()` accessors, we can make sure that index calculations are always correct (of course, the indices
 * themselves might be wrong still).
 * 2) In contrast to having multiple Tensor-like objects as parameters to a CUDA kernel, which would also prevent
 * index calculation errors, having this helper makes is easier for the compiler to know that certain objects have
 * identical shape. This saves both constant memory (as the shape need not be copied multiple times) and reduces
 * index calculations, as e.g. the `lookup` and `weights` arrays can share the same offsets.
 *
 */
template<typename Float, typename Int, bool WriteOutput, bool Transpose>
class KernelDataHelper : public ShapeDataHelper {
private:
    using OutFloat = std::conditional_t<WriteOutput, Float, const Float>;
public:
    KernelDataHelper(const Float* __restrict__ features, const Int* __restrict__ lookup,
                     const Float* __restrict__ weights, OutFloat* __restrict__ outputs,
                     std::int32_t batch_size, std::int32_t input_size, std::int32_t output_size,
                     std::int32_t lookup_size);

    // vectorization helpers
    /// Vector data type comprised of four `Int`s
    using int_vec_t = cuda_vector_type_t<Int, 4>;
    /// Vector data type comprised of four `Float`s
    using float_vec_t = cuda_vector_type_t<Float, 4>;

    // lookup helpers so that we don't make any indexing errors
    __device__ const Float& feature_at(std::int32_t batch, std::int32_t feature) const {
        validate_batch_index(batch);
        validate_feature_index(feature);
        if constexpr (Transpose) {
            return mFeaturePtr[feature * batch_size() + batch];
        } else {
            return mFeaturePtr[batch * input_size() + feature];
        }
    }

    __device__ const Int& lookup_at(std::int32_t unit, std::int32_t index) const {
        validate_unit_index(unit);
        validate_weight_index(index);
        const Int& looked_up = mLookupPtr[unit * lookup_size() + index];
        validate_feature_index(looked_up);
        return looked_up;
    }

    __device__ const Float& weight_at(std::int32_t unit, std::int32_t index) const {
        validate_unit_index(unit);
        validate_weight_index(index);
        return mWeightPtr[unit * lookup_size() + index];
    }

    __device__ OutFloat& output_at(std::int32_t batch, std::int32_t unit) {
        validate_batch_index(batch);
        validate_unit_index(unit);
        return mOutputPtr[batch * output_size() + unit];
    }

    // vectorized look-ups
    __device__ const int_vec_t& lookup_vec_at(std::int32_t unit, std::int32_t index) const {
        validate_unit_index(unit);
        validate_weight_index(index);
        gsl_ExpectsAudit(index % 4 == 0);
        gsl_ExpectsAudit(lookup_size() % 4 == 0);
        const Int* base_ptr = mLookupPtr + (unit * lookup_size() + index);
        return *reinterpret_cast<const int_vec_t*>(base_ptr);
    }

    __device__ const float_vec_t& weight_vec_at(std::int32_t unit, std::int32_t index) const {
        validate_unit_index(unit);
        validate_weight_index(index);
        gsl_ExpectsAudit(index % 4 == 0);
        gsl_ExpectsAudit(lookup_size() % 4 == 0);
        const Float* base_ptr = mWeightPtr + (unit * lookup_size() + index);
        return *reinterpret_cast<const float_vec_t*>(base_ptr);
    }

    // ensure we get errors for invalid function arguments
    DISABLE_NON_INTEGER_ARGUMENTS(feature_at);
    DISABLE_NON_INTEGER_ARGUMENTS(lookup_at);
    DISABLE_NON_INTEGER_ARGUMENTS(weight_at);
    DISABLE_NON_INTEGER_ARGUMENTS(output_at);
    DISABLE_NON_INTEGER_ARGUMENTS(lookup_vec_at);
    DISABLE_NON_INTEGER_ARGUMENTS(weight_vec_at);

private:
    // pointers to the data arrays
    const Float* __restrict__ mFeaturePtr = nullptr;
    const Int*   __restrict__ mLookupPtr  = nullptr;
    const Float* __restrict__ mWeightPtr  = nullptr;
    OutFloat*    __restrict__ mOutputPtr  = nullptr;
};

template<typename Float, typename Int, bool WriteOutput, bool Transpose>
KernelDataHelper<Float, Int, WriteOutput, Transpose>::KernelDataHelper(
    const Float* __restrict__ features, const Int* __restrict__ lookup,
    const Float* __restrict__ weights, OutFloat * __restrict__ outputs,
    std::int32_t batch_size, std::int32_t input_size,
    std::int32_t output_size, std::int32_t lookup_size) :
    ShapeDataHelper(batch_size, input_size, output_size, lookup_size),
    mFeaturePtr(features), mLookupPtr(lookup), mWeightPtr(weights), mOutputPtr(outputs)
{
    gsl_Expects(mFeaturePtr != nullptr);
    gsl_Expects(mLookupPtr  != nullptr);
    gsl_Expects(mWeightPtr  != nullptr);
    gsl_Expects(mOutputPtr  != nullptr);
}

#endif //TFSPARSE_CC_KERNELS_KERNEL_DATA_CU_H
