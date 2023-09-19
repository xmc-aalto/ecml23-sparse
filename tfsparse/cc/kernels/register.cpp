//
// Created by erik on 29.3.2022.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "common.h"
using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename Device, typename Float, typename Int, ForwardImplementations Strategy, bool Transposed>
class FixedFanInSparseMatmulOp : public OpKernel {
public:
    explicit FixedFanInSparseMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& feature_tensor = context->input(0);
        const Tensor& lookup_tensor = context->input(1);
        const Tensor& weights_tensor = context->input(2);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        std::int64_t batch_size = Transposed ? feature_tensor.dim_size(1) : feature_tensor.dim_size(0);
        std::int64_t num_units = lookup_tensor.dim_size(0);

        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, num_units}), &output_tensor));

        FixedFanInSparseMatmulKernel<Device, Float, Int, Strategy, Transposed> kernel;
        kernel(
            context->eigen_device<Device>(),
            feature_tensor.matrix<Float>(),
            lookup_tensor.matrix<Int>(),
            weights_tensor.matrix<Float>(),
            output_tensor->matrix<Float>()
        );
    }
};


template<class Device, class Float, class Int, BackwardImplementations Strategy, bool Transposed>
class FixedFanInSparseMatmulGradOp : public OpKernel {
public:
    explicit FixedFanInSparseMatmulGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& feature_tensor = context->input(0);
        const Tensor& lookup_tensor = context->input(1);
        const Tensor& weights_tensor = context->input(2);
        const Tensor& out_tensor = context->input(3);

        // Create an output tensor
        Tensor* feature_gradient_tensor = nullptr;
        Tensor* weight_gradient_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, feature_tensor.shape(), &feature_gradient_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_tensor.shape(), &weight_gradient_tensor));

        FixedFanInSparseMatmulGradKernel<Device, Float, Int, Strategy, Transposed> kernel;
        kernel(
            context->eigen_device<Device>(),
            feature_tensor.matrix<Float>(),
            lookup_tensor.matrix<Int>(),
            weights_tensor.matrix<Float>(),
            out_tensor.matrix<Float>(),
            feature_gradient_tensor->matrix<Float>(),
            weight_gradient_tensor->matrix<Float>()
        );
    }
};

#define REGISTER_FORWARD(_DEVICE, Float, Int, Strat)    \
REGISTER_KERNEL_BUILDER(Name("FanInSparseMatmul").Device(DEVICE##_##_DEVICE).TypeConstraint<Float>("T").TypeConstraint<Int>("I"), \
                        FixedFanInSparseMatmulOp<_DEVICE##Device, Float, Int, Strat, false>);

REGISTER_FORWARD(CPU, float, std::int32_t, ForwardImplementations::CPU);
REGISTER_FORWARD(CPU, float, std::uint32_t, ForwardImplementations::CPU);
REGISTER_FORWARD(GPU, float, std::int32_t, ForwardImplementations::GPU_Fast);
REGISTER_FORWARD(GPU, float, std::uint32_t, ForwardImplementations::GPU_Fast);

#define REGISTER_FORWARD_TP(_DEVICE, Float, Int, Strat)    \
REGISTER_KERNEL_BUILDER(Name("FanInSparseMatmulTp").Device(DEVICE##_##_DEVICE).TypeConstraint<Float>("T").TypeConstraint<Int>("I"), \
                        FixedFanInSparseMatmulOp<_DEVICE##Device, Float, Int, Strat, true>);

REGISTER_FORWARD_TP(CPU, float, std::int32_t, ForwardImplementations::CPU);
REGISTER_FORWARD_TP(CPU, float, std::uint32_t, ForwardImplementations::CPU);
REGISTER_FORWARD_TP(GPU, float, std::int32_t, ForwardImplementations::GPU_TP_OuterBatchNaive);
REGISTER_FORWARD_TP(GPU, float, std::uint32_t, ForwardImplementations::GPU_TP_OuterBatchNaive);

#define REGISTER_BACKWARD(_DEVICE, Float, Int, Strat)    \
REGISTER_KERNEL_BUILDER(Name("FanInSparseMatmulGrad").Device(DEVICE##_##_DEVICE).TypeConstraint<Float>("T").TypeConstraint<Int>("I"), \
                        FixedFanInSparseMatmulGradOp<_DEVICE##Device, Float, Int, Strat, false>);

REGISTER_BACKWARD(CPU, float, std::int32_t, BackwardImplementations::CPU);
REGISTER_BACKWARD(CPU, float, std::uint32_t, BackwardImplementations::CPU);
REGISTER_BACKWARD(GPU, float, std::int32_t, BackwardImplementations::GPU_Vector);
REGISTER_BACKWARD(GPU, float, std::uint32_t, BackwardImplementations::GPU_Vector);

#define REGISTER_BACKWARD_TP(_DEVICE, Float, Int, Strat)    \
REGISTER_KERNEL_BUILDER(Name("FanInSparseMatmulGradTp").Device(DEVICE##_##_DEVICE).TypeConstraint<Float>("T").TypeConstraint<Int>("I"), \
                        FixedFanInSparseMatmulGradOp<_DEVICE##Device, Float, Int, Strat, true>);

REGISTER_BACKWARD_TP(CPU, float, std::int32_t, BackwardImplementations::CPU);
REGISTER_BACKWARD_TP(CPU, float, std::uint32_t, BackwardImplementations::CPU);
REGISTER_BACKWARD_TP(GPU, float, std::int32_t, BackwardImplementations::GPU_Vector_TP);
REGISTER_BACKWARD_TP(GPU, float, std::uint32_t, BackwardImplementations::GPU_Vector_TP);
