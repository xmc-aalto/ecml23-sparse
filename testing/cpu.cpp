//
// Created by erik on 16.2.2023.
//

#include "helpers.h"
#include "gsl/gsl-lite.hpp"
#include <thread>

template<class T, class O>
constexpr MatrixType<O> make_tensor_map_imp(T& mat) {
    return MatrixType<O>(mat.data(), {mat.rows(), mat.cols()});
}

template<class T>
constexpr auto make_tensor_map(BasicMatrix<T>& mat) {
    return make_tensor_map_imp<BasicMatrix<T>, T>(mat);
}

template<class T>
constexpr auto make_const_tensor_map(const BasicMatrix<T>& mat) {
    return make_tensor_map_imp<const BasicMatrix<T>, const T>(mat);
}

template<class T>
auto apply_map(const BasicMatrix<T>& val) {
    return make_const_tensor_map(val);
}

template<class T>
auto apply_map(BasicMatrix<T>& val) {
    return make_tensor_map(val);
}

template<class Kernel, class... Args>
void run_cpu_kernel(Kernel kernel, Args&... args) {
    Eigen::ThreadPool pool(std::thread::hardware_concurrency());
    CPUDevice device(&pool, std::thread::hardware_concurrency());
    kernel(device, apply_map(args)...);
}


BasicMatrix<float>
make_dense(const BasicMatrix<float>& features, const BasicMatrix<int32_t>& lookup, const BasicMatrix<float>& weights) {
    BasicMatrix<float> dense_op = BasicMatrix<float>::Zero(features.cols(), lookup.rows());
    for(int i = 0; i < lookup.rows(); ++i) {
        for(int j = 0; j < lookup.cols(); ++j) {
            int index = lookup(i, j);
            float weight = weights(i, j);
            dense_op(index, i) += weight;
        }
    }
    return dense_op;
}


template<ForwardImplementations Imp, bool Transpose>
struct run_forward_imp<Imp, false, Transpose> {
    static void run(
        std::function<void(std::function<void()>)> f,
        const BasicMatrix<float>& features,
        const BasicMatrix<std::int32_t>& lookup,
        const BasicMatrix<float>& weights,
        BasicMatrix<float>& output) {
        f([&]() {
            FixedFanInSparseMatmulKernel<CPUDevice, float, std::int32_t, Imp, Transpose> kernel;
            run_cpu_kernel(kernel, features, lookup, weights, output);
        });
    }
};

template struct run_forward_imp<ForwardImplementations::Reference, false, false>;
template struct run_forward_imp<ForwardImplementations::Reference, false, true>;
template struct run_forward_imp<ForwardImplementations::CPU, false, false>;
template struct run_forward_imp<ForwardImplementations::CPU, false, true>;

template<BackwardImplementations Imp, bool Transpose>
struct run_backward_imp<Imp, false, Transpose> {
    static void run(
        std::function<void(std::function<void()>)> f,
        const BasicMatrix<float>& features,
        const BasicMatrix<std::int32_t>& lookup,
        const BasicMatrix<float>& weights,
        const BasicMatrix<float>& out_grad,
        BasicMatrix<float>& ftr_grad,
        BasicMatrix<float>& wgt_grad) {
        f([&]() {
            FixedFanInSparseMatmulGradKernel<CPUDevice, float, std::int32_t, Imp, Transpose> kernel;
            run_cpu_kernel(kernel, features, lookup, weights, out_grad, ftr_grad, wgt_grad);
        });
    }
};

template struct run_backward_imp<BackwardImplementations::Reference, false, false>;
template struct run_backward_imp<BackwardImplementations::Reference, false, true>;
template struct run_backward_imp<BackwardImplementations::CPU, false, true>;
template struct run_backward_imp<BackwardImplementations::CPU, false, false>;