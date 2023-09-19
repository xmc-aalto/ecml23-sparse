//
// Created by erik on 16.3.2023.
//
#include "common.h"

template<class T>
using BasicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<class O, class T>
constexpr BasicMatrix<O> make_mat(T& mat) {
    Eigen::Map<const BasicMatrix<O>> mapping(mat.data(), mat.dimension(0), mat.dimension(1));
    return mapping;
}

template<class Float, class Int>
BasicMatrix<Float>
make_dense(const BasicMatrix<Float>& features, const BasicMatrix<Int>& lookup, const BasicMatrix<Float>& weights) {
    BasicMatrix<float> dense_op = BasicMatrix<float>::Zero(features.cols(), lookup.rows());
    for(int i = 0; i < lookup.rows(); ++i) {
        for(int j = 0; j < lookup.cols(); ++j) {
            int index = lookup(i, j);
            Float weight = weights(i, j);
            dense_op(index, i) += weight;
        }
    }
    return dense_op;
}

template<typename Float, typename Int, bool Transposed>
struct FixedFanInSparseMatmulKernel<CPUDevice, Float, Int, ForwardImplementations::Reference, Transposed> {
    void operator()(const CPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    MatrixType<Float> output) {
        BasicMatrix<Float> features_mat = make_mat<Float>(features);
        if constexpr (Transposed) {
            features_mat.transposeInPlace();
        }
        BasicMatrix<Int> lookup_mat = make_mat<Int>(lookup);
        BasicMatrix<Float> weights_mat = make_mat<Float>(weights);
        auto dense_op = make_dense(features_mat, lookup_mat, weights_mat);
        BasicMatrix<Float> output_mat = features_mat * dense_op;
        output = MatrixType<Float>(output_mat.data(), {output_mat.rows(), output_mat.cols()});
    }
};

template<typename Float, typename Int, bool Transposed>
struct FixedFanInSparseMatmulGradKernel<CPUDevice, Float, Int, BackwardImplementations::Reference, Transposed> {
    void operator()(const CPUDevice& device, const MatrixType<const Float>& features,
                    const MatrixType<const Int>& lookup,
                    const MatrixType<const Float>& weights,
                    const MatrixType<const Float>& out_grad,
                    MatrixType<Float> feature_gradient,
                    MatrixType<Float> weight_gradient
    ) {
        // x = Az => dl/dA = dl/dx dx/dA = dl/dx z
        //        => dl/dz = dl/dx dx/dz = dl/dx A
        BasicMatrix<Float> features_mat = make_mat<Float>(features);
        if constexpr (Transposed) {
            features_mat.transposeInPlace();
        }

        BasicMatrix<Int> lookup_mat = make_mat<Int>(lookup);
        BasicMatrix<Float> weights_mat = make_mat<Float>(weights);
        BasicMatrix<Float> out_mat = make_mat<Float>(out_grad);
        auto dense_op = make_dense(features_mat, lookup_mat, weights_mat);

        BasicMatrix<Float> ftr_grad_mat = out_mat * dense_op.transpose();
        if constexpr (Transposed) {
            ftr_grad_mat.transposeInPlace();
        }
        feature_gradient = MatrixType<Float>(ftr_grad_mat.data(), {ftr_grad_mat.rows(), ftr_grad_mat.cols()});

        BasicMatrix<Float> dense_wgt_grad = features_mat.transpose() * out_mat;
        for (int i = 0; i < lookup_mat.rows(); ++i) {
            for (int j = 0; j < lookup_mat.cols(); ++j) {
                int index = lookup_mat(i, j);
                weight_gradient(i, j) = dense_wgt_grad(index, i);
            }
        }
    }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::int32_t, BackwardImplementations::Reference, false>;
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::int32_t, ForwardImplementations::Reference, false>;
template struct FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::int32_t, BackwardImplementations::Reference, true>;
template struct FixedFanInSparseMatmulKernel<CPUDevice, float, std::int32_t, ForwardImplementations::Reference, true>;
