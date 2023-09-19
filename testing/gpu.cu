//
// Created by erik on 15.2.2023.
//
#include "helpers.h"

template<class T, class Source>
struct KernelArgHelper {
public:
    KernelArgHelper(Eigen::GpuDevice& device, Source& src) : mDevice(&device), mSource(&src) {
        void* mem = device.allocate(mSource->size() * sizeof(T));
        mGpuPtr = reinterpret_cast<T*>(mem);
        device.memcpyHostToDevice(mem, mSource->data(), mSource->size() * sizeof(T));
        mDevice->synchronize();
    }

    ~KernelArgHelper() {
        mDevice->synchronize();

        if constexpr (!std::is_const_v<T>) {
            mDevice->memcpyDeviceToHost(mSource->data(), mGpuPtr, sizeof(float) * mSource->size());
        }
        mDevice->deallocate(const_cast<std::remove_const_t<T>*>(mGpuPtr));
    }

    operator MatrixType<T>() {
        return MatrixType<T>(mGpuPtr, {mSource->rows(), mSource->cols()});
    }
private:
    Eigen::GpuDevice* mDevice = nullptr;
    T* mGpuPtr = nullptr;
    Source* mSource = nullptr;
};

template<class T>
auto kernel_arg_gpu(Eigen::GpuDevice& device, const BasicMatrix<T>& val) {
    return KernelArgHelper<const T, const BasicMatrix<T>>(device, val);
}

template<class T>
auto kernel_arg_gpu(Eigen::GpuDevice& device, BasicMatrix<T>& val) {
    return KernelArgHelper<T, BasicMatrix<T>>(device, val);
}

template<class Kernel, class... Args>
void run_gpu_kernel(const std::function<void(std::function<void()>)>& f, Kernel kernel, Args&... args) {
    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice device(&stream);

    auto call = [&](auto&&... vargs) {
        f([&](){
            kernel(vargs...);
            device.synchronize();
        });
    };

    call(device, kernel_arg_gpu(device, args)...);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorName(err) << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template<ForwardImplementations Imp, bool Transposed>
struct run_forward_imp<Imp, true, Transposed> {
    static void run(
        std::function<void(std::function<void()>)> f,
        const BasicMatrix<float>& features,
        const BasicMatrix<std::int32_t>& lookup,
        const BasicMatrix<float>& weights,
        BasicMatrix<float>& output) {
        FixedFanInSparseMatmulKernel<GPUDevice, float, std::int32_t, Imp, Transposed> kernel;
        run_gpu_kernel(f, kernel, features, lookup, weights, output);
    }
};

#define EXPLICIT_INSTANTIATION(STRATEGY) \
template struct run_forward_imp<ForwardImplementations::STRATEGY, true, false>; \
template struct run_forward_imp<ForwardImplementations::STRATEGY, true, true>;

EXPLICIT_INSTANTIATION(GPU_OuterBatchNaive);
EXPLICIT_INSTANTIATION(GPU_OuterBatchVectorized);
EXPLICIT_INSTANTIATION(GPU_InnerBatchNaive);
EXPLICIT_INSTANTIATION(GPU_InnerBatchVectorized);
EXPLICIT_INSTANTIATION(GPU_Fast);
EXPLICIT_INSTANTIATION(GPU_TP_OuterBatchNaive);

template<BackwardImplementations Imp, bool Transposed>
struct run_backward_imp<Imp, true, Transposed> {
    static void run(
        std::function<void(std::function<void()>)> f,
        const BasicMatrix<float>& features,
        const BasicMatrix<std::int32_t>& lookup,
        const BasicMatrix<float>& weights,
        const BasicMatrix<float>& out_grad,
        BasicMatrix<float>& ftr_grad,
        BasicMatrix<float>& wgt_grad) {
        FixedFanInSparseMatmulGradKernel<GPUDevice, float, std::int32_t, Imp, Transposed> kernel;
        run_gpu_kernel(f, kernel, features, lookup, weights, out_grad, ftr_grad, wgt_grad);
    }
};

template struct run_backward_imp<BackwardImplementations::GPU_Scalar, true, false>;
template struct run_backward_imp<BackwardImplementations::GPU_Vector, true, false>;
template struct run_backward_imp<BackwardImplementations::GPU_Scalar, true, true>;
template struct run_backward_imp<BackwardImplementations::GPU_Vector, true, true>;
template struct run_backward_imp<BackwardImplementations::GPU_Scalar_TP, true, true>;
template struct run_backward_imp<BackwardImplementations::GPU_Scalar_TP, true, false>;
template struct run_backward_imp<BackwardImplementations::GPU_Vector_TP, true, true>;
template struct run_backward_imp<BackwardImplementations::GPU_Vector_TP, true, false>;
