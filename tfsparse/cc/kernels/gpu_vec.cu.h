//
// Created by erik on 25.2.2023.
//

#ifndef TFSPARSE_CC_KERNELS_GPU_VEC_H
#define TFSPARSE_CC_KERNELS_GPU_VEC_H

#include "vector_types.h"

// Vectorization helpers
template<class T, int N>
struct cuda_vector_type_helper;

template<>
struct cuda_vector_type_helper<float, 4> {
    using type = float4;
};

template<>
struct cuda_vector_type_helper<double, 4> {
    using type = double4;
};

template<>
struct cuda_vector_type_helper<int, 4> {
    using type = int4;
};

template<>
struct cuda_vector_type_helper<unsigned int, 4> {
    using type = uint4;
};

template<>
struct cuda_vector_type_helper<long, 4> {
    using type = long4;
};

template<>
struct cuda_vector_type_helper<long long, 4> {
    using type = longlong4;
};

template<class T, int N>
using cuda_vector_type_t = typename cuda_vector_type_helper<T, N>::type;

template<class Vector>
struct scalar_type_helper {
    using type = std::decay_t<decltype(std::declval<Vector>().x)>;
};

template<class Vector>
using cuda_scalar_type_t = typename scalar_type_helper<Vector>::type;

template<class Vector, class Scalar = cuda_scalar_type_t<Vector>>
__device__ Scalar read(Vector vec, int k) {
    return reinterpret_cast<Scalar*>(&vec)[k];
}

template<class Vector, class Scalar = cuda_scalar_type_t<Vector>>
__device__ Scalar& ref(Vector& vec, int k) {
    return reinterpret_cast<Scalar*>(&vec)[k];
}

#endif //TFSPARSE_CC_KERNELS_GPU_VEC_H
