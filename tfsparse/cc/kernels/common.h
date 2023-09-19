//
// Created by erik on 30.3.2022.
//

#ifndef TFSPARSE__COMMON_H
#define TFSPARSE__COMMON_H

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "strategies.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template<typename T>
using MatrixType = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>;

template<typename Device, typename Float, typename Int, ForwardImplementations Strategy, bool Transposed>
struct FixedFanInSparseMatmulKernel {
    void operator()(const Device& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    MatrixType<Float> output);
};

template<typename Device, typename Float, typename Int, BackwardImplementations Strategy, bool Transposed>
struct FixedFanInSparseMatmulGradKernel {
    void operator()(const Device& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& output,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient
                    );
};

#endif //TFSPARSE__COMMON_H
