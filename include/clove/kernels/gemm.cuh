#pragma once

#include <clove/common.cuh>
#include <cuda_fp16.h>

namespace clove {

// =============================================================================
// GEMM Configuration
// =============================================================================

// Tile sizes for FP16 GEMM (tuned for RTX 3xxx/4xxx)
struct GemmConfig {
    static constexpr int BM = 128;  // Block tile M
    static constexpr int BN = 128;  // Block tile N
    static constexpr int BK = 32;   // Block tile K

    static constexpr int WM = 64;   // Warp tile M
    static constexpr int WN = 64;   // Warp tile N

    static constexpr int TM = 8;    // Thread tile M
    static constexpr int TN = 8;    // Thread tile N

    // Number of threads per block
    static constexpr int NUM_THREADS = 256;

    // Tensor core sizes (WMMA)
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;
};

// =============================================================================
// FP16 GEMM (using Tensor Cores via WMMA)
// =============================================================================

// C = alpha * A @ B + beta * C
// A: [M, K] row-major
// B: [K, N] row-major
// C: [M, N] row-major
void gemm_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

// =============================================================================
// INT8 GEMM (Quantized)
// =============================================================================

// C = (A_q - A_zero) * A_scale @ (B_q - B_zero) * B_scale
// Per-tensor quantization
void gemm_int8(
    const int8_t* A_q,
    const int8_t* B_q,
    half* C,
    int M, int N, int K,
    float A_scale, int8_t A_zero,
    float B_scale, int8_t B_zero,
    cudaStream_t stream = 0
);

// =============================================================================
// INT4 GEMM (Quantized, Group-wise)
// =============================================================================

// Group size for INT4 quantization
constexpr int INT4_GROUP_SIZE = 128;

// A_q: [M, K/2] packed INT4
// B_q: [K, N/2] packed INT4
// Scales/zeros: per-group
void gemm_int4(
    const uint8_t* A_q,
    const half* A_scale,
    const half* A_zero,
    const uint8_t* B_q,
    const half* B_scale,
    const half* B_zero,
    half* C,
    int M, int N, int K,
    cudaStream_t stream = 0
);

// =============================================================================
// Batched GEMM
// =============================================================================

void gemm_fp16_batched(
    const half* A,    // [B, M, K]
    const half* B,    // [B, K, N]
    half* C,          // [B, M, N]
    int batch, int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = 0
);

}  // namespace clove
