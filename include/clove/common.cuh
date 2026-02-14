#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace clove {

// CUDA error checking
#define CLOVE_CHECK_CUDA(call)                                              \
    do {                                                                     \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

#define CLOVE_CHECK_LAST() CLOVE_CHECK_CUDA(cudaGetLastError())

#define CLOVE_SYNC_CHECK()                  \
    do {                                     \
        CLOVE_CHECK_CUDA(cudaDeviceSynchronize()); \
        CLOVE_CHECK_LAST();                  \
    } while (0)

// Ceiling division
constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// Data types
enum class DType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4
};

// Bytes per element
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::INT8: return 1;
        case DType::INT4: return 0;  // special: 2 values per byte
        default: return 0;
    }
}

// Type traits for kernel templates
template<typename T> struct CudaType;
template<> struct CudaType<float> { using type = float; static constexpr DType dtype = DType::FP32; };
template<> struct CudaType<half> { using type = half; static constexpr DType dtype = DType::FP16; };
template<> struct CudaType<int8_t> { using type = int8_t; static constexpr DType dtype = DType::INT8; };

// Vector types for coalesced memory access
using half4 = struct { half x, y, z, w; };
using half8 = struct { half4 lo, hi; };

__device__ __forceinline__ half4 make_half4(half a, half b, half c, half d) {
    return {a, b, c, d};
}

// Warp-level primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);

    return val;
}

}  // namespace clove
