//
// Created by erik on 15.2.2023.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"
#include <nanobench.h>

#include "test_case_templates.h"

RUN_ALL_FORWARD("forward pass") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Forward Pass");
    run_forward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::false_type{}, &bench);
}

RUN_ALL_FORWARD("transposed forward pass") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed Forward Pass");
    run_forward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::true_type{}, &bench);
}


RUN_ALL_BACKWARD("backward pass benchmark") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Backward Pass");
    run_backward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::false_type{}, &bench);
}

RUN_ALL_BACKWARD("transposed backward pass benchmark") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed Backward Pass");
    run_backward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::true_type{}, &bench);
}
