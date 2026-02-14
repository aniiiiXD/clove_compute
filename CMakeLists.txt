cmake_minimum_required(VERSION 3.24)
project(clove_compute LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture - RTX 3xxx=86, RTX 4xxx=89
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  # For profiling

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Kernel library
add_library(clove_kernels STATIC
    src/kernels/gemm_fp16.cu
    src/kernels/gemm_int8.cu
    src/kernels/gemm_int4.cu
    src/kernels/attention.cu
    src/kernels/rope.cu
    src/kernels/layernorm.cu
    src/kernels/activations.cu
    src/kernels/softmax.cu
)

target_link_libraries(clove_kernels
    CUDA::cudart
    CUDA::cublas
)

# Runtime library
add_library(clove_runtime STATIC
    src/runtime/tensor.cu
    src/runtime/model_loader.cpp
    src/runtime/kv_cache.cu
    src/runtime/inference.cu
)

target_link_libraries(clove_runtime
    clove_kernels
    CUDA::cudart
)

# Main executable
add_executable(clove_compute src/main.cpp)
target_link_libraries(clove_compute
    clove_runtime
    clove_kernels
    CUDA::cudart
    CUDA::cublas
)

# Benchmark executable
add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm
    clove_kernels
    CUDA::cudart
    CUDA::cublas
)

# Tests
enable_testing()
add_executable(test_gemm tests/test_gemm.cu)
target_link_libraries(test_gemm
    clove_kernels
    CUDA::cudart
    CUDA::cublas
)
add_test(NAME test_gemm COMMAND test_gemm)
