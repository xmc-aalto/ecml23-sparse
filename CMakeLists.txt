cmake_minimum_required(VERSION 3.21)
project(TfSparse)
enable_language(CUDA)

include(cmake/tensorflow.cmake)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        gsl-lite
        GIT_REPOSITORY https://github.com/gsl-lite/gsl-lite.git
        GIT_TAG v0.40.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(gsl-lite)

set(KERNEL_SOURCES
        tfsparse/cc/kernels/forward.cpp
        tfsparse/cc/kernels/forward.cu
        tfsparse/cc/kernels/backward.cpp
        tfsparse/cc/kernels/backward.cu
        )

add_library(sparseops SHARED
        tfsparse/cc/ops/fixed_fan_in_ops.cpp
        tfsparse/cc/kernels/register.cpp
        ${KERNEL_SOURCES}
        )
target_link_libraries(sparseops PUBLIC tensorflow gsl-lite)
target_compile_definitions(sparseops PUBLIC EIGEN_USE_THREADS)

add_subdirectory(testing)

find_package(Doxygen)
if(${DOXYGEN_FOUND})
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_ANON_NSPACES YES)
    set(DOXYGEN_INTERNAL_DOCS YES)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_REFERENCED_BY_RELATION YES)
    set(DOXYGEN_REFERENCES_RELATION YES)
    set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
    set(DOXYGEN_COLLABORATION_GRAPH YES)
    set(DOXYGEN_TEMPLATE_RELATIONS YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_INTERACTIVE_SVG YES)

    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_EXTRACT_PRIV_VIRTUAL YES)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)

    set(DOXYGEN_EXAMPLE_PATH ${PROJECT_SOURCE_DIR}/tfsparse)

    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
    doxygen_add_docs(
            doxygen
            ${PROJECT_SOURCE_DIR}/tfsparse
            doc/forward.md
            COMMENT "Generate HTML docs"
    )
endif()