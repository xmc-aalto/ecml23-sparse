//
// Created by erik on 17.3.2023.
//

#ifndef TFSPARSE_TFSPARSE_CC_KERNELS_STRATEGIES_H
#define TFSPARSE_TFSPARSE_CC_KERNELS_STRATEGIES_H

#include <tuple>

template<auto Value>
using make_const = std::integral_constant<decltype(Value), Value>;

enum class EShapeIdentifier {
    BatchSize,
    InputSize,
    OutputSize,
    LookupSize
};


enum class ForwardImplementations {
    Reference,
    CPU,
    GPU_OuterBatchNaive,
    GPU_OuterBatchVectorized,
    GPU_InnerBatchNaive,
    GPU_InnerBatchVectorized,
    GPU_Fast,
    GPU_TP_OuterBatchNaive,
};

enum class BackwardImplementations {
    Reference,
    CPU,
    GPU_Scalar,
    GPU_Scalar_TP,
    GPU_Vector,
    GPU_Vector_TP
};


template<ForwardImplementations Imp>
struct FwdImpTraits;

template<>
struct FwdImpTraits<ForwardImplementations::Reference> {
    static constexpr const bool IsGpuKernel = false;
    static constexpr const bool IsVectorized = false;
    static constexpr const bool IsTransposed = false;
};

template<>
struct FwdImpTraits<ForwardImplementations::CPU> {
    static constexpr const bool IsGpuKernel = false;
    static constexpr const bool IsVectorized = false;
    static constexpr const bool IsTransposed = false;
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_OuterBatchNaive> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = false;
    static constexpr const bool IsTransposed = false;
    static constexpr const std::tuple<EShapeIdentifier, EShapeIdentifier> Grid = {EShapeIdentifier::BatchSize, EShapeIdentifier::OutputSize};
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_OuterBatchVectorized> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = true;
    static constexpr const bool IsTransposed = false;
    static constexpr const std::tuple<EShapeIdentifier, EShapeIdentifier> Grid = {EShapeIdentifier::BatchSize, EShapeIdentifier::OutputSize};
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_InnerBatchNaive> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = false;
    static constexpr const bool IsTransposed = false;
    static constexpr const std::tuple<EShapeIdentifier> Grid = {EShapeIdentifier::OutputSize};
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_InnerBatchVectorized> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = true;
    static constexpr const bool IsTransposed = false;
    static constexpr const std::tuple<EShapeIdentifier> Grid = {EShapeIdentifier::OutputSize};
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_Fast> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = true;
    static constexpr const bool IsTransposed = false;
    static constexpr const std::tuple<EShapeIdentifier> Grid = {EShapeIdentifier::OutputSize};
};

template<>
struct FwdImpTraits<ForwardImplementations::GPU_TP_OuterBatchNaive> {
    static constexpr const bool IsGpuKernel = true;
    static constexpr const bool IsVectorized = false;
    static constexpr const bool IsTransposed = true;
    static constexpr const std::tuple<EShapeIdentifier, EShapeIdentifier> Grid =  {EShapeIdentifier::OutputSize, EShapeIdentifier::BatchSize};
};

template<ForwardImplementations Value>
struct FwdImpTag {
    static constexpr ForwardImplementations value = Value;
};


template<ForwardImplementations Imp>
constexpr bool is_on_gpu(FwdImpTag<Imp> value) {
    return FwdImpTraits<Imp>::IsGpuKernel;
}

template<ForwardImplementations Imp>
constexpr bool is_vectorized(FwdImpTag<Imp> value) {
    return FwdImpTraits<Imp>::IsVectorized;
}

template<ForwardImplementations Imp>
constexpr bool is_transposed(FwdImpTag<Imp> value) {
    return FwdImpTraits<Imp>::IsTransposed;
}

template<ForwardImplementations Imp>
constexpr auto get_grid(FwdImpTag<Imp> value) {
    return FwdImpTraits<Imp>::Grid;
}

constexpr const char* get_name(ForwardImplementations value) {
    switch (value) {
        case ForwardImplementations::Reference: return "Reference";
        case ForwardImplementations::CPU: return "CPU";
        case ForwardImplementations::GPU_OuterBatchNaive: return "GPU_OuterBatchNaive";
        case ForwardImplementations::GPU_OuterBatchVectorized: return "GPU_OuterBatchVectorized";
        case ForwardImplementations::GPU_InnerBatchNaive: return "GPU_InnerBatchNaive";
        case ForwardImplementations::GPU_InnerBatchVectorized: return "GPU_InnerBatchVectorized";
        case ForwardImplementations::GPU_Fast: return "GPU_Fast";
        case ForwardImplementations::GPU_TP_OuterBatchNaive: return "GPU_TP_OuterBatchNaive";
    }
}

constexpr const char* get_name(BackwardImplementations value) {
    switch (value) {
        case BackwardImplementations::Reference: return "Reference";
        case BackwardImplementations::CPU: return "CPU";
        case BackwardImplementations::GPU_Scalar: return "GPU_Scalar";
        case BackwardImplementations::GPU_Vector: return "GPU_Vector";
        case BackwardImplementations::GPU_Scalar_TP: return "GPU_Scalar_TP";
        case BackwardImplementations::GPU_Vector_TP: return "GPU_Vector_TP";
    }
}


constexpr bool is_on_gpu(BackwardImplementations value) {
    if(value == BackwardImplementations::Reference || value == BackwardImplementations::CPU) {
        return false;
    }
    return true;
}

constexpr bool is_vectorized(ForwardImplementations value) {
    switch(value) {
        case ForwardImplementations::GPU_OuterBatchVectorized:
        case ForwardImplementations::GPU_InnerBatchVectorized:
        case ForwardImplementations::GPU_Fast:
            return true;
        default:
            return false;
    }
}

template<BackwardImplementations Value>
struct BwdImpTag {
    static constexpr BackwardImplementations value = Value;
};


#endif //TFSPARSE_TFSPARSE_CC_KERNELS_STRATEGIES_H
