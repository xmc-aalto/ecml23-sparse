#include "forward.cu.h"
#include "common.h"

template<bool Transpose, typename Float, typename Int>
KernelDataHelper<Float, Int, true, Transpose> makeFKH(MatrixType<const Float> features,
                                        MatrixType<const Int> lookup,
                                        MatrixType<const Float> weights,
                                        MatrixType<Float> output) {
    // verify compatibility of dimensions
    std::int32_t batch_size, input_size;
    if constexpr (Transpose) {
        batch_size = features.dimension(1);
        input_size = features.dimension(0);
    } else {
        batch_size = features.dimension(0);
        input_size = features.dimension(1);
    }

    gsl_Expects( batch_size          == output.dimension(0));
    gsl_Expects( lookup.dimension(0) == output.dimension(1) );
    gsl_Expects( lookup.dimension(0) == weights.dimension(0) );
    gsl_Expects( lookup.dimension(1) == weights.dimension(1) );

    // ensure that all sizes fit into 32-bit integers
    gsl_Expects( batch_size < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( input_size < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( output.dimension(1) < std::numeric_limits<std::int32_t>::max() );
    gsl_Expects( lookup.dimension(1) < std::numeric_limits<std::int32_t>::max() );

    return KernelDataHelper<Float, Int, true, Transpose>(
            features.data(), lookup.data(), weights.data(), output.data(),
            batch_size, input_size, output.dimension(1), lookup.dimension(1) );
}

template<typename Float, typename Int, ForwardImplementations Choice, bool Transposed>
struct FixedFanInSparseMatmulKernel<GPUDevice, Float, Int, Choice, Transposed> {
    void operator()(const GPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    MatrixType<Float> output) {

        auto data = makeFKH<Transposed>(features, lookup, weights, output);

        // If we ask for a vectorized method, but don't have matching shape,
        // fall back to a slower implementation
        if constexpr (Choice == ForwardImplementations::GPU_OuterBatchVectorized ||
                      Choice == ForwardImplementations::GPU_InnerBatchVectorized ||
                      Choice == ForwardImplementations::GPU_Fast) {
            if(data.lookup_size() % 4 != 0) {
                launch_forward(data, FwdImpTag<ForwardImplementations::GPU_OuterBatchNaive>{}, {384}, device.stream());
                return;
            }
        }

        switch(Choice) {
            case ForwardImplementations::GPU_OuterBatchNaive:
            case ForwardImplementations::GPU_OuterBatchVectorized:
                launch_forward(data, FwdImpTag<Choice>{}, {384}, device.stream());
                break;
            case ForwardImplementations::GPU_InnerBatchNaive:
            case ForwardImplementations::GPU_InnerBatchVectorized:
            case ForwardImplementations::GPU_Fast:
                device.memset(output.data(), 0, output.size() * sizeof(Float));
                launch_forward(data, FwdImpTag<Choice>{}, {512}, device.stream());
                break;
            case ForwardImplementations::GPU_TP_OuterBatchNaive:
                // TODO the efficiency goes down for small batches
                launch_forward(data, FwdImpTag<Choice>{}, {16, 24}, device.stream());
                break;
            default:
                gsl_FailFast();
        }
    }
};

#define EXPLICIT_INSTANTIATION_FOR(Float, Int, Strategy) \
template struct FixedFanInSparseMatmulKernel<GPUDevice, Float, Int, ForwardImplementations::Strategy, true>; \
template struct FixedFanInSparseMatmulKernel<GPUDevice, Float, Int, ForwardImplementations::Strategy, false>;

// Explicitly instantiate functors for the types of OpKernels registered.
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_Fast);
EXPLICIT_INSTANTIATION_FOR(float, std::uint32_t, GPU_Fast);
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_OuterBatchNaive);
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_OuterBatchVectorized);
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_InnerBatchNaive);
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_InnerBatchVectorized);
EXPLICIT_INSTANTIATION_FOR(float, std::int32_t, GPU_TP_OuterBatchNaive);
EXPLICIT_INSTANTIATION_FOR(float, std::uint32_t, GPU_TP_OuterBatchNaive);

