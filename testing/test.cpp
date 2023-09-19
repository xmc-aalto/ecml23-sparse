//
// Created by erik on 15.2.2023.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"
#include <iostream>

#include "test_case_templates.h"
#include <nanobench.h>

RUN_ALL_FORWARD("fixed-input forward") {
    BasicMatrix<float> features = BasicMatrix<float>{
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {0.0, 1.0, 1.0}
    };
    bool transpose = false;
    DOCTEST_SUBCASE("Transposed") {
        transpose = true;
        features.transposeInPlace();
    }
    DOCTEST_SUBCASE("Non-Transposed") {}

    BasicMatrix<std::int32_t> lookup = Eigen::MatrixX<std::int32_t>{{1, 2}, {0, 2}};
    BasicMatrix<float> weights = BasicMatrix<float>{{1.0, 2.0}, {3.0, 4.0}};

    BasicMatrix<float> expected = BasicMatrix<float>{
        {2.0 + 3.0 * 2.0, 3.0 + 3.0 * 4.0},
        {5.0 + 6.0 * 2.0, 4.0 * 3.0 + 6.0 * 4.0},
        {1.0 + 1.0 * 2.0, 0.0 + 1.0 * 4.0}
    };

    BasicMatrix<float> output = BasicMatrix<float>::Zero(expected.rows(), expected.cols());

    if(transpose) {
        run_forward(Implementation{}, std::true_type{}, just_call, features, lookup, weights, output);
    } else {
        run_forward(Implementation{}, std::false_type{}, just_call, features, lookup, weights, output);
    }

    bool b = output == expected;
    if(!b) {
        std::cout << output << "\n\n";
        std::cout << expected << "\n";
    }
    DOCTEST_CHECK(b);
}


std::vector<sSizeConfig> TestConfigurations = {
    {3, 5, 4, 2},
    {12, 8, 8, 4},
    {127, 254, 513, 4},
    {12, 8, 32, 16},
    {17, 12, 63, 44},
    {127, 254, 513, 73},
    {127, 254, 513, 80},
    {1, 32, 4'000'000, 4}
};

RUN_ALL_FORWARD("forward pass with random input") {
    sSizeConfig config;
    DOCTEST_VALUE_PARAMETERIZED_DATA(config, TestConfigurations);

    {
        DOCTEST_INFO("Non-transposed kernel");
        run_forward_test(Implementation{}, config, std::false_type{}, nullptr);
    }

    {
        DOCTEST_INFO("Transposed kernel");
        run_forward_test(Implementation{}, config, std::true_type{}, nullptr);
    }
}

RUN_ALL_BACKWARD("backward pass with random input") {
    sSizeConfig config;
    DOCTEST_VALUE_PARAMETERIZED_DATA(config, TestConfigurations);

    {
        DOCTEST_INFO("Non-transposed kernel");
        run_backward_test(Implementation{}, config, std::false_type{}, nullptr);
    }

    {
        DOCTEST_INFO("Transposed kernel");
        run_backward_test(Implementation{}, config, std::true_type{}, nullptr);
    }
}

RUN_ALL_FORWARD("forward check transpose") {
    sSizeConfig config;
    DOCTEST_VALUE_PARAMETERIZED_DATA(config, TestConfigurations);

    int in_rows = config.InRows;
    int in_cols = config.InCols;
    int out_cols = config.OutCols;
    int nnz = config.NNZ;
    DOCTEST_CAPTURE(in_rows);
    DOCTEST_CAPTURE(in_cols);
    DOCTEST_CAPTURE(out_cols);
    DOCTEST_CAPTURE(nnz);

    BasicMatrix<float> features = BasicMatrix<float>::Random(in_rows, in_cols);
    BasicMatrix<float> features_tp =  features.transpose();
    BasicMatrix<std::int32_t> lookup = Eigen::MatrixX<std::int32_t>::Zero(out_cols, nnz);
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::int32_t> dist(0, in_cols-1);
    for(auto& coeff : lookup.reshaped()) {
        coeff = dist(rng);
    }

    BasicMatrix<float> weights = BasicMatrix<float>::Random(out_cols, nnz);
    BasicMatrix<float> out = BasicMatrix<float>::Random(in_rows, out_cols);
    BasicMatrix<float> out_tp = BasicMatrix<float>::Random(in_rows, out_cols);

    run_forward(Implementation{}, std::false_type{}, just_call, features, lookup, weights, out);
    run_forward(Implementation{}, std::true_type{}, just_call, features_tp, lookup, weights, out_tp);

    /*if((out - out_tp).maxCoeff() > 1.5e-5) {
        std::cout << out << "\n\n";
        std::cout << out_tp << "\n";
    }*/

    DOCTEST_CHECK((out - out_tp).maxCoeff() < 1.5e-5);
}

RUN_ALL_BACKWARD("backward check transpose") {
    sSizeConfig config;
    DOCTEST_VALUE_PARAMETERIZED_DATA(config, TestConfigurations);

    int in_rows = config.InRows;
    int in_cols = config.InCols;
    int out_cols = config.OutCols;
    int nnz = config.NNZ;
    DOCTEST_CAPTURE(in_rows);
    DOCTEST_CAPTURE(in_cols);
    DOCTEST_CAPTURE(out_cols);
    DOCTEST_CAPTURE(nnz);

    BasicMatrix<float> features = BasicMatrix<float>::Random(in_rows, in_cols);
    BasicMatrix<std::int32_t> lookup = Eigen::MatrixX<std::int32_t>::Zero(out_cols, nnz);
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::int32_t> dist(0, in_cols-1);
    for(auto& coeff : lookup.reshaped()) {
        coeff = dist(rng);
    }

    BasicMatrix<float> weights = BasicMatrix<float>::Random(out_cols, nnz);
    BasicMatrix<float> out_grad = BasicMatrix<float>::Random(in_rows, out_cols);
    out_grad = out_grad.unaryExpr([](auto f){ return f < 0 ? 0 : f; });

    BasicMatrix<float> ftr_grad = BasicMatrix<float>::Random(in_rows, in_cols);
    BasicMatrix<float> wgt_grad= BasicMatrix<float>::Random(out_cols, nnz);
    BasicMatrix<float> ftr_grad_tp = ftr_grad.transpose();
    BasicMatrix<float> wgt_grad_tp = BasicMatrix<float>::Random(out_cols, nnz);

    run_backward(Implementation{}, std::false_type{}, just_call, features, lookup, weights, out_grad, ftr_grad, wgt_grad);
    run_backward(Implementation{}, std::true_type{}, just_call, features.transpose(), lookup, weights, out_grad, ftr_grad_tp, wgt_grad_tp);
    ftr_grad_tp.transposeInPlace();

    DOCTEST_CHECK_EQ(ftr_grad.rows(), ftr_grad_tp.rows());
    DOCTEST_CHECK_EQ(wgt_grad.cols(), wgt_grad_tp.cols());
    DOCTEST_CHECK((ftr_grad - ftr_grad_tp).maxCoeff() < 1.5e-5);
    DOCTEST_CHECK((wgt_grad - wgt_grad_tp).maxCoeff() < 1.5e-5);
}