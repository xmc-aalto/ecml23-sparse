//
// Created by erik on 8.3.2023.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"
#include <nanobench.h>

ankerl::nanobench::Bench* active_benchmark = nullptr;

template<class F>
void run_benchmark(std::string name, F&& f) {
    active_benchmark->run(name, std::forward<F>(f));
}

#include "test_case_templates.h"

void save_results(const ankerl::nanobench::Bench& bench, const std::string& kind, const std::string& implementation) {
    std::ofstream csv_out(std::string("benchmarks/") + kind + implementation + ".csv");
    ankerl::nanobench::render(ankerl::nanobench::templates::csv(), bench, csv_out);

    std::ofstream json_out(std::string("benchmarks/") + kind + implementation + ".json");
    ankerl::nanobench::render(ankerl::nanobench::templates::json(), bench, json_out);
}

RUN_ALL_FORWARD("forward pass - batch") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Forward pass batch dependency");
    for(int batch_size : std::initializer_list<int>{4, 8, 10, 16, 25, 32, 50, 64, 100}) {
        bench.batch(batch_size).unit("instance").complexityN(batch_size);
        run_forward_test(Implementation{}, {batch_size, 32'768, 670'000, 32}, std::false_type{}, &bench);
    }

    save_results(bench, "forward-batch", get_name(Implementation::value));
}

RUN_ALL_FORWARD("transposed forward pass - batch") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed forward pass batch dependency");
    for(int batch_size : std::initializer_list<int>{4, 8, 10, 16, 25, 32, 50, 64, 100}) {
        bench.batch(batch_size).unit("instance").complexityN(batch_size);
        run_forward_test(Implementation{}, {batch_size, 32'768, 670'000, 32}, std::true_type{}, &bench);
    }

    save_results(bench, "transposed-forward-batch", get_name(Implementation::value));
}

RUN_ALL_FORWARD("forward pass - hidden") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Forward pass hidden dependency");
    for(int hidden_size : std::initializer_list<int>{768, 1024, 2048, 4096, 8192, 16'384, 24'576, 32'768, 49'152, 65'536}) {
        bench.batch(hidden_size).unit("hidden").complexityN(hidden_size);
        run_forward_test(Implementation{}, {32, hidden_size, 670'000, 32}, std::false_type{}, &bench);
    }

    save_results(bench, "forward-hidden", get_name(Implementation::value));
}

RUN_ALL_FORWARD("transposed forward pass - hidden") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed forward pass hidden dependency");
    for(int hidden_size : std::initializer_list<int>{768, 1024, 2048, 4096, 8192, 16'384, 24'576, 32'768, 49'152, 65'536}) {
        bench.batch(hidden_size).unit("hidden").complexityN(hidden_size);
        run_forward_test(Implementation{}, {32, hidden_size, 670'000, 32}, std::true_type{}, &bench);
    }

    save_results(bench, "transposed-forward-hidden", get_name(Implementation::value));
}

RUN_ALL_FORWARD("forward pass - outputs") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Forward pass output dependency");
    for(int output_size : std::initializer_list<int>{4'000, 10'000, 32'000, 64'000, 100'000, 256'000, 670'000, 1'000'000}) {
        bench.batch(output_size).unit("outputs").complexityN(output_size);
        run_forward_test(Implementation{}, {32, 32'768, output_size, 32}, std::false_type{}, &bench);
    }

    save_results(bench, "forward-out", get_name(Implementation::value));
}

RUN_ALL_FORWARD("transposed forward pass - outputs") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed forward pass output dependency");
    for(int output_size : std::initializer_list<int>{4'000, 10'000, 32'000, 64'000, 100'000, 256'000, 670'000, 1'000'000}) {
        bench.batch(output_size).unit("outputs").complexityN(output_size);
        run_forward_test(Implementation{}, {32, 32'768, output_size, 32}, std::true_type{}, &bench);
    }

    save_results(bench, "transposed-forward-out", get_name(Implementation::value));
}

RUN_ALL_FORWARD("forward pass - weights") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Forward pass output dependency");
    for(int weight_size : std::initializer_list<int>{4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128}) {
        bench.batch(weight_size).unit("weights").complexityN(weight_size);
        run_forward_test(Implementation{}, {32, 32'768, 670'000, weight_size}, std::false_type{}, &bench);
    }

    save_results(bench, "forward-weights", get_name(Implementation::value));
}

RUN_ALL_FORWARD("transposed forward pass - weights") {
    ankerl::nanobench::Bench bench;
    bench.warmup(1).minEpochIterations(2).title("Transposed forward pass output dependency");
    for(int weight_size : std::initializer_list<int>{4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128}) {
        bench.batch(weight_size).unit("weights").complexityN(weight_size);
        run_forward_test(Implementation{}, {32, 32'768, 670'000, weight_size}, std::true_type{}, &bench);
    }

    save_results(bench, "transposed-forward-weights", get_name(Implementation::value));
}
