//
// Created by erik on 25.2.2023.
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"
#include <iostream>

template<class F>
void run_benchmark([[maybe_unused]] std::string name, F&& f) {
    f();
}

struct ProfBench {
    template<class F>
    void run(std::string, F&& f) {
        f();
    }
};

#include "test_case_templates.h"


RUN_ALL_FORWARD("forward pass") {
    ProfBench bench;
    run_forward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::false_type{}, &bench);
}

RUN_ALL_FORWARD("transposed forward pass") {
    ProfBench bench;
    run_forward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::true_type{}, &bench);
}

RUN_ALL_BACKWARD("backward pass benchmark") {
    ProfBench bench;
    run_backward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::false_type{}, &bench);
}

RUN_ALL_BACKWARD("transposed backward pass benchmark") {
    ProfBench bench;
    run_backward_test(Implementation{}, {50, 32'768, 670'000, 32}, std::true_type{}, &bench);
}
