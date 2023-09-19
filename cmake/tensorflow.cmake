
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# workaround for TF searching for cuda in "third_party" directory: We create a symbolic link of the actual CUDA headers
# in a search path that looks as tensorflow is expecting it. This is done locally, we do not actually write into
# tensorflow's `third_party` directory.
set(INCLUDE_FIX_HELPER_DIR ${CMAKE_BINARY_DIR}/include-fix)
if(NOT EXISTS ${INCLUDE_FIX_HELPER_DIR})
    find_file(CUDA_HEADERS_SAMPLE_FILE "cuda_fp16.h" PATH_SUFFIXES include)
    get_filename_component(CUDA_HEADERS_DIRECTORY  ${CUDA_HEADERS_SAMPLE_FILE} DIRECTORY)
    file(MAKE_DIRECTORY "${INCLUDE_FIX_HELPER_DIR}/third_party/gpus/cuda")
    file(CREATE_LINK "${CUDA_HEADERS_DIRECTORY}" "${INCLUDE_FIX_HELPER_DIR}/third_party/gpus/cuda/include" SYMBOLIC)
endif()

macro(extract_tf_info)
    if(NOT DEFINED ${ARGV0} )
        execute_process(
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tfcfg.py" ${ARGV1}
                OUTPUT_VARIABLE ${ARGV0}
                COMMAND_ECHO STDOUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY
        )
        set(${ARGV0} "${${ARGV0}}" CACHE STRING "${ARGV2}")
    endif()
endmacro()

extract_tf_info(TF_INCLUDE_DIRS "includes" "include directories for tensorflow")
extract_tf_info(TF_COMPILE_DEFINITIONS "defines" "defines for compiling tensorflow")
extract_tf_info(TF_COMPILE_OPTIONS "other-compile" "other compiler options required for tensorflow")
extract_tf_info(TF_LINK_DIRS "link-dirs" "linker include directories for tensorflow")
extract_tf_info(TF_LINK_LIBS "link-libs" "linked libraries for tensorflow")

add_library(tf-compile-options INTERFACE)
target_compile_options(tf-compile-options INTERFACE ${TF_COMPILE_OPTIONS})
target_compile_definitions(tf-compile-options INTERFACE ${TF_COMPILE_DEFINITIONS})
target_include_directories(tf-compile-options INTERFACE ${INCLUDE_FIX_HELPER_DIR} ${TF_INCLUDE_DIRS})
target_compile_options(tf-compile-options INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(tf-compile-options INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:EIGEN_USE_GPU=1;GOOGLE_CUDA=1>)

add_library(tensorflow INTERFACE)
target_link_directories(tensorflow INTERFACE ${TF_LINK_DIRS})
target_link_libraries(tensorflow INTERFACE tf-compile-options ${TF_LINK_LIBS})


