//
// Created by erik on 16.2.2023.
//

#ifndef TFSPARSE_TESTING_HELPERS_H
#define TFSPARSE_TESTING_HELPERS_H

#undef NDEBUG
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#define NDEBUG
#include "common.h"

template<class T>
using BasicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template<auto Value>
using make_const = std::integral_constant<decltype(Value), Value>;

template<ForwardImplementations T, bool GPU, bool Transposed>
struct run_forward_imp {
    static void run(std::function<void(std::function<void()>)> f,
                    const BasicMatrix<float>& features,
                    const BasicMatrix<std::int32_t>& lookup,
                    const BasicMatrix<float>& weights,
                    BasicMatrix<float>& output);
};

template<ForwardImplementations Value, bool Transpose, class... Args>
void run_forward(FwdImpTag<Value> tag,
                 std::bool_constant<Transpose> tp,
                 std::function<void(std::function<void()>)> f,
                 Args&&... args) {
    run_forward_imp<Value, is_on_gpu(tag), Transpose>::run(f, std::forward<Args>(args)...);
}

template<BackwardImplementations T, bool GPU, bool Transposed>
struct run_backward_imp {
    static void run(std::function<void(std::function<void()>)> f,
                    const BasicMatrix<float>& features,
                    const BasicMatrix<std::int32_t>& lookup,
                    const BasicMatrix<float>& weights,
                    const BasicMatrix<float>& out_grad,
                    BasicMatrix<float>& ftr_grad,
                    BasicMatrix<float>& wgt_grad);
};

template<BackwardImplementations Value, bool Transpose, class... Args>
void run_backward(BwdImpTag<Value> tag,
                  std::bool_constant<Transpose> tp,
                  Args&&... args) {
    run_backward_imp<Value, is_on_gpu(Value), Transpose>::run(std::forward<Args>(args)...);
}


#endif //TFSPARSE_TESTING_HELPERS_H
