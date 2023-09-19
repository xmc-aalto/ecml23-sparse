//
// Created by erik on 17.2.2023.
//

#ifndef TFSPARSE_TESTING_TEST_CASE_TEMPLATES_H
#define TFSPARSE_TESTING_TEST_CASE_TEMPLATES_H

#include "helpers.h"
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"
#include <iostream>
#include <nanobench.h>

#define DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container)                                  \
    static size_t _doctest_subcase_idx = 0;                                                     \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {           \
        DOCTEST_SUBCASE(to_string(in).c_str()) { data = in; }   \
    });                                                                                         \
    _doctest_subcase_idx = 0

struct sSizeConfig {
    int InRows;
    int InCols;
    int OutCols;
    int NNZ;
};

std::string to_string(sSizeConfig c) {
    std::ostringstream str;
    str << "(" << c.InRows << ", " << c.InCols << ") x ([" << c.NNZ << "], " << c.OutCols << ")";
    return str.str();
}

void just_call(std::function<void()> f) {
    f();
}


template<ForwardImplementations Implementation, bool Transpose, class Bench>
void run_forward_test(FwdImpTag<Implementation> strategy,
                      sSizeConfig config,
                      std::bool_constant<Transpose> transpose,
                      Bench benchmark) {
    int in_rows = config.InRows;
    int in_cols = config.InCols;
    int out_cols = config.OutCols;
    int nnz = config.NNZ;
    DOCTEST_CAPTURE(in_rows);
    DOCTEST_CAPTURE(in_cols);
    DOCTEST_CAPTURE(out_cols);
    DOCTEST_CAPTURE(nnz);
    DOCTEST_CAPTURE(get_name(Implementation));

    BasicMatrix<float> features = BasicMatrix<float>::Random(in_rows, in_cols);
    if(transpose) {
        features.transposeInPlace();
    }

    BasicMatrix<std::int32_t> lookup = Eigen::MatrixX<std::int32_t>::Zero(out_cols, nnz);
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::int32_t> dist(0, in_cols-1);
    for(auto& coeff : lookup.reshaped()) {
        coeff = dist(rng);
    }

    BasicMatrix<float> weights = BasicMatrix<float>::Random(out_cols, nnz);

    BasicMatrix<float> output_bsl = BasicMatrix<float>::Random(in_rows, out_cols);
    BasicMatrix<float> output_dev = BasicMatrix<float>::Random(in_rows, out_cols);

    if constexpr (!std::is_same_v<Bench, std::nullptr_t>) {
        auto call_with_bm = [&](auto f) {
            benchmark->run(get_name(Implementation), [&] {
                f();
                ankerl::nanobench::detail::doNotOptimizeAway(output_dev.coeff(0, 0));
            });
        };
        run_forward(strategy, transpose, call_with_bm, features, lookup, weights, output_dev);
    } else {
        run_forward(FwdImpTag<ForwardImplementations::Reference>{}, transpose, just_call, features, lookup, weights, output_bsl);
        run_forward(strategy, transpose, just_call, features, lookup, weights, output_dev);
        DOCTEST_CHECK((output_bsl - output_dev).maxCoeff() < 1e-5);
    }
}

template<BackwardImplementations Implementation, bool Transpose, class Bench>
void run_backward_test(BwdImpTag<Implementation> strategy,
                      sSizeConfig config,
                      std::bool_constant<Transpose> transpose,
                      Bench benchmark) {
    int in_rows = config.InRows;
    int in_cols = config.InCols;
    int out_cols = config.OutCols;
    int nnz = config.NNZ;
    DOCTEST_CAPTURE(in_rows);
    DOCTEST_CAPTURE(in_cols);
    DOCTEST_CAPTURE(out_cols);
    DOCTEST_CAPTURE(nnz);

    BasicMatrix<float> features = BasicMatrix<float>::Random(in_rows, in_cols);
    if(transpose) {
        features.transposeInPlace();
    }

    BasicMatrix<std::int32_t> lookup = Eigen::MatrixX<std::int32_t>::Zero(out_cols, nnz);
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::int32_t> dist(0, in_cols-1);
    for(auto& coeff : lookup.reshaped()) {
        coeff = dist(rng);
    }

    BasicMatrix<float> weights = BasicMatrix<float>::Random(out_cols, nnz);
    BasicMatrix<float> out_grad = BasicMatrix<float>::Random(in_rows, out_cols);
    out_grad = out_grad.unaryExpr([](auto f){ return f < 0 ? 0 : f; });

    BasicMatrix<float> ftr_grad_bsl = BasicMatrix<float>::Random(in_rows, in_cols);
    BasicMatrix<float> wgt_grad_bsl = BasicMatrix<float>::Random(out_cols, nnz);
    BasicMatrix<float> ftr_grad_dev = BasicMatrix<float>::Random(in_rows, in_cols);
    BasicMatrix<float> wgt_grad_dev = BasicMatrix<float>::Random(out_cols, nnz);
    if(transpose) {
        ftr_grad_dev.transposeInPlace();
        ftr_grad_bsl.transposeInPlace();
    }

    if constexpr (!std::is_same_v<Bench, std::nullptr_t>) {
        auto call_with_bm = [&](auto f) {
            benchmark->run(get_name(Implementation), [&] {
                f();
                ankerl::nanobench::detail::doNotOptimizeAway(ftr_grad_dev.coeff(0, 0));
                ankerl::nanobench::detail::doNotOptimizeAway(wgt_grad_dev.coeff(0, 0));
            });
        };
        run_backward(strategy, transpose, call_with_bm, features, lookup, weights, out_grad, ftr_grad_dev, wgt_grad_dev);
    } else {
        run_backward(BwdImpTag<BackwardImplementations::Reference>{}, transpose, just_call, features, lookup, weights, out_grad, ftr_grad_bsl, wgt_grad_bsl);
        run_backward(strategy, transpose, just_call, features, lookup, weights, out_grad, ftr_grad_dev, wgt_grad_dev);
        DOCTEST_CHECK_EQ(ftr_grad_bsl.rows(), ftr_grad_dev.rows());
        DOCTEST_CHECK_EQ(ftr_grad_bsl.cols(), ftr_grad_dev.cols());
        DOCTEST_CHECK((ftr_grad_bsl - ftr_grad_dev).maxCoeff() < 1.5e-5);
        DOCTEST_CHECK((wgt_grad_bsl - wgt_grad_dev).maxCoeff() < 1.5e-5);
    }
}


#define RUN_ALL_FORWARD(name)                                           \
DOCTEST_TEST_CASE_TEMPLATE(name, Implementation,                        \
    FwdImpTag<ForwardImplementations::CPU>,                             \
    FwdImpTag<ForwardImplementations::GPU_Fast>,                        \
    FwdImpTag<ForwardImplementations::GPU_OuterBatchNaive>,             \
    FwdImpTag<ForwardImplementations::GPU_OuterBatchVectorized>,        \
    FwdImpTag<ForwardImplementations::GPU_InnerBatchNaive>,             \
    FwdImpTag<ForwardImplementations::GPU_InnerBatchVectorized>,        \
    FwdImpTag<ForwardImplementations::GPU_TP_OuterBatchNaive>           \
)

#define RUN_ALL_BACKWARD(name)                                          \
DOCTEST_TEST_CASE_TEMPLATE(name, Implementation,                        \
    BwdImpTag<BackwardImplementations::CPU>,                            \
    BwdImpTag<BackwardImplementations::GPU_Scalar>,                     \
    BwdImpTag<BackwardImplementations::GPU_Scalar_TP>,                  \
    BwdImpTag<BackwardImplementations::GPU_Vector>,                     \
    BwdImpTag<BackwardImplementations::GPU_Vector_TP>                   \
)


#endif //TFSPARSE_TESTING_TEST_CASE_TEMPLATES_H
