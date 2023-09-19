//
// Created by schulte1 on 2/16/23.
//

#include "common.h"
#include "backward.cu.h"

template<bool Transpose, typename Float, typename Int>
BackwardDataHelper<Float, Int, Transpose> makeBKH(
        MatrixType<const Float> features,
        MatrixType<const Int> lookup,
        MatrixType<const Float> weights,
        MatrixType<const Float> out_grad,
        MatrixType<Float> feature_gradient,
        MatrixType<Float> weight_gradient) {

    // verify compatibility of dimensions
    std::int32_t batch_size, input_size;
    if constexpr (Transpose) {
        batch_size = features.dimension(1);
        input_size = features.dimension(0);
    } else {
        batch_size = features.dimension(0);
        input_size = features.dimension(1);
    }

    gsl_Expects( batch_size          == out_grad.dimension(0) );
    gsl_Expects( lookup.dimension(0) == out_grad.dimension(1) );
    gsl_Expects( lookup.dimension(0) == weights.dimension(0) );
    gsl_Expects( lookup.dimension(1) == weights.dimension(1) );
    gsl_Expects( features.dimension(0) == feature_gradient.dimension(0) );
    gsl_Expects( features.dimension(1) == feature_gradient.dimension(1) );
    gsl_Expects( weights.dimension(0) == weight_gradient.dimension(0) );
    gsl_Expects( weights.dimension(1) == weight_gradient.dimension(1) );

    // ensure that all sizes fit into 32-bit integers
    gsl_Expects( features.dimension(0) < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( features.dimension(1) < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( out_grad.dimension(1) < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( lookup.dimension(1) < std::numeric_limits<std::int32_t>::max() );

    return BackwardDataHelper<Float, Int, Transpose>(
            features.data(), lookup.data(), weights.data(), out_grad.data(),
            feature_gradient.data(), weight_gradient.data(),
            batch_size, input_size,
            out_grad.dimension(1), lookup.dimension(1) );
}

template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulGradKernel<GPUDevice, Float, Int, BackwardImplementations::GPU_Scalar, Transpose> {
    void operator()(const GPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& out_grad,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient) {


        device.memset(feature_gradient.data(), 0, feature_gradient.size() * sizeof(Float));

        auto data = makeBKH<Transpose>(features, lookup, weights, out_grad, feature_gradient, weight_gradient);

        feature_grad_launch_scalar(data, {32, 32}, device.stream());
        weight_grad_launch_scalar(data, {32, 32}, device.stream());
    }
};

template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulGradKernel<GPUDevice, Float, Int, BackwardImplementations::GPU_Scalar_TP, Transpose> {
    void operator()(const GPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& out_grad,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient) {


        device.memset(feature_gradient.data(), 0, feature_gradient.size() * sizeof(Float));

        auto data = makeBKH<Transpose>(features, lookup, weights, out_grad, feature_gradient, weight_gradient);

        feature_grad_launch_scalar_tp(data, {16, 64}, device.stream());
        weight_grad_launch_scalar(data, {32, 32}, device.stream());
    }
};

template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulGradKernel<GPUDevice, Float, Int, BackwardImplementations::GPU_Vector, Transpose> {
    void operator()(const GPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& out_grad,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient) {

        device.memset(feature_gradient.data(), 0, feature_gradient.size() * sizeof(Float));

        auto data = makeBKH<Transpose>(features, lookup, weights, out_grad, feature_gradient, weight_gradient);


        if(data.lookup_size() % 4 == 0) {
            feature_grad_launch_vector_weights(data, {8, 128}, device.stream());
            weight_grad_launch_vector_weights(data, {8, 128}, device.stream());
        } else {
            feature_grad_launch_scalar(data, {32, 32}, device.stream());
            weight_grad_launch_scalar(data, {32, 32}, device.stream());
        }
    }
};

template<typename Float, typename Int, bool Transpose>
struct FixedFanInSparseMatmulGradKernel<GPUDevice, Float, Int, BackwardImplementations::GPU_Vector_TP, Transpose> {
    void operator()(const GPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& out_grad,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient) {


        device.memset(feature_gradient.data(), 0, feature_gradient.size() * sizeof(Float));

        auto data = makeBKH<Transpose>(features, lookup, weights, out_grad, feature_gradient, weight_gradient);

        if(data.lookup_size() % 4 == 0) {
            feature_grad_launch_vector_tp(data, {16, 64}, device.stream());
            weight_grad_launch_vector_weights(data, {8, 128}, device.stream());
        } else {
            feature_grad_launch_scalar(data, {32, 32}, device.stream());
            weight_grad_launch_scalar(data, {32, 32}, device.stream());
        }
    }
};


// Explicitly instantiate functors for the types of OpKernels registered.
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Scalar, false>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Scalar_TP, false>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Vector_TP, false>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Vector, false>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::uint32_t, BackwardImplementations::GPU_Vector, false>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Scalar, true>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Scalar_TP, true>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Vector_TP, true>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::uint32_t, BackwardImplementations::GPU_Vector_TP, true>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, BackwardImplementations::GPU_Vector, true>;
template struct FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::uint32_t, BackwardImplementations::GPU_Vector, true>;
