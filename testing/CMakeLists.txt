include(FetchContent)
FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest
        GIT_TAG v2.4.9
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(doctest)

FetchContent_Declare(
        nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG v4.1.0
        GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

add_library(test-common gpu.cu cpu.cpp ../tfsparse/cc/kernels/reference.cpp)

target_link_libraries(test-common PUBLIC tf-compile-options doctest::doctest nanobench gsl-lite)
target_include_directories(test-common PUBLIC ../tfsparse/cc/kernels)
target_compile_definitions(test-common PUBLIC EIGEN_USE_THREADS NDEBUG)
# target_compile_options(test-common PUBLIC -Werror=float-conversion)
# target_compile_options(test-common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings -diag-suppress 611>)
foreach(KF IN LISTS KERNEL_SOURCES)
    target_sources(test-common PUBLIC ../${KF})
endforeach()

add_executable(test-ops test.cpp)
target_link_libraries(test-ops PUBLIC test-common)
target_compile_definitions(test-ops PUBLIC gsl_CONFIG_CONTRACT_CHECKING_AUDIT)
target_compile_options(test-ops PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=128>)

add_executable(quick-bench quick-bench.cpp)
target_link_libraries(quick-bench PUBLIC test-common)

add_executable(detailed-bench detailed-bench.cpp)
target_link_libraries(detailed-bench PUBLIC test-common)

add_executable(profile-ops profile.cpp)
target_link_libraries(profile-ops PUBLIC test-common)


target_compile_options(test-common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--resource-usage;--generate-line-info>)
target_compile_options(test-common INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
target_compile_definitions(test-common INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:EIGEN_USE_GPU=1;GOOGLE_CUDA=1>)
