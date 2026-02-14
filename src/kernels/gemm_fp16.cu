#include <clove/kernels/gemm.cuh>
#include <mma.h>

using namespace nvcuda::wmma;

namespace clove {

// =============================================================================
// FP16 GEMM Kernel using Tensor Cores (WMMA)
// =============================================================================

// Tile sizes
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warps per block
constexpr int WARPS_M = 4;  // BM / (WMMA_M * 2) = 128 / 32 = 4
constexpr int WARPS_N = 4;  // BN / (WMMA_N * 2) = 128 / 32 = 4
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 512

// Each warp computes 2x2 WMMA tiles = 32x32 output
constexpr int WARP_TILES_M = 2;
constexpr int WARP_TILES_N = 2;

__global__ void __launch_bounds__(NUM_THREADS)
gemm_fp16_wmma_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Block position
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Warp position within block
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpM = warpId / WARPS_N;
    const int warpN = warpId % WARPS_N;

    // Shared memory for tiles - using double buffering
    __shared__ half As[2][BK][BM + 8];  // +8 to avoid bank conflicts
    __shared__ half Bs[2][BK][BN + 8];

    // WMMA fragments for this warp's 2x2 output tiles
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[WARP_TILES_M][WARP_TILES_N];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            fill_fragment(acc[i][j], __float2half(0.0f));
        }
    }

    // Global memory positions
    const int aRow = by * BM;
    const int bCol = bx * BN;

    // Threads cooperatively load tiles
    const int loadIterA = (BM * BK) / NUM_THREADS;  // Loads per thread for A
    const int loadIterB = (BK * BN) / NUM_THREADS;  // Loads per thread for B

    // Precompute load positions
    const int loadIdxA = threadIdx.x;
    const int loadIdxB = threadIdx.x;

    int buffer = 0;

    // Load first tile
    #pragma unroll
    for (int i = 0; i < loadIterA; i++) {
        int idx = loadIdxA + i * NUM_THREADS;
        int loadK = idx / BM;
        int loadM = idx % BM;
        int globalM = aRow + loadM;
        int globalK = loadK;

        half val = (globalM < M && globalK < K) ?
                   A[globalM * K + globalK] : __float2half(0.0f);
        As[0][loadK][loadM] = val;
    }

    #pragma unroll
    for (int i = 0; i < loadIterB; i++) {
        int idx = loadIdxB + i * NUM_THREADS;
        int loadK = idx / BN;
        int loadN = idx % BN;
        int globalK = loadK;
        int globalN = bCol + loadN;

        half val = (globalK < K && globalN < N) ?
                   B[globalK * N + globalN] : __float2half(0.0f);
        Bs[0][loadK][loadN] = val;
    }

    __syncthreads();

    // Main loop over K dimension
    for (int k = 0; k < K; k += BK) {
        int nextK = k + BK;
        int nextBuffer = 1 - buffer;

        // Prefetch next tile (if not last iteration)
        if (nextK < K) {
            #pragma unroll
            for (int i = 0; i < loadIterA; i++) {
                int idx = loadIdxA + i * NUM_THREADS;
                int loadK = idx / BM;
                int loadM = idx % BM;
                int globalM = aRow + loadM;
                int globalK = nextK + loadK;

                half val = (globalM < M && globalK < K) ?
                           A[globalM * K + globalK] : __float2half(0.0f);
                As[nextBuffer][loadK][loadM] = val;
            }

            #pragma unroll
            for (int i = 0; i < loadIterB; i++) {
                int idx = loadIdxB + i * NUM_THREADS;
                int loadK = idx / BN;
                int loadN = idx % BN;
                int globalK = nextK + loadK;
                int globalN = bCol + loadN;

                half val = (globalK < K && globalN < N) ?
                           B[globalK * N + globalN] : __float2half(0.0f);
                Bs[nextBuffer][loadK][loadN] = val;
            }
        }

        // Compute on current tile using WMMA
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            // Load A and B fragments
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, col_major> a_frag[WARP_TILES_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag[WARP_TILES_N];

            // Load A fragments for this warp's row(s)
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                int fragRow = warpM * WARP_TILES_M * WMMA_M + i * WMMA_M;
                load_matrix_sync(a_frag[i], &As[buffer][kk][fragRow], BM + 8);
            }

            // Load B fragments for this warp's column(s)
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                int fragCol = warpN * WARP_TILES_N * WMMA_N + j * WMMA_N;
                load_matrix_sync(b_frag[j], &Bs[buffer][kk][fragCol], BN + 8);
            }

            // Matrix multiply-accumulate
            #pragma unroll
            for (int i = 0; i < WARP_TILES_M; i++) {
                #pragma unroll
                for (int j = 0; j < WARP_TILES_N; j++) {
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
        buffer = nextBuffer;
    }

    // Scale by alpha
    if (alpha != 1.0f) {
        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; j++) {
                #pragma unroll
                for (int t = 0; t < acc[i][j].num_elements; t++) {
                    acc[i][j].x[t] = __hmul(acc[i][j].x[t], __float2half(alpha));
                }
            }
        }
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; i++) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; j++) {
            int outM = by * BM + warpM * WARP_TILES_M * WMMA_M + i * WMMA_M;
            int outN = bx * BN + warpN * WARP_TILES_N * WMMA_N + j * WMMA_N;

            if (outM < M && outN < N) {
                // Handle beta
                if (beta != 0.0f) {
                    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
                    load_matrix_sync(c_frag, &C[outM * N + outN], N, mem_row_major);

                    #pragma unroll
                    for (int t = 0; t < c_frag.num_elements; t++) {
                        acc[i][j].x[t] = __hadd(acc[i][j].x[t],
                                                __hmul(c_frag.x[t], __float2half(beta)));
                    }
                }

                store_matrix_sync(&C[outM * N + outN], acc[i][j], N, mem_row_major);
            }
        }
    }
}

// =============================================================================
// Simpler kernel without tensor cores (for smaller matrices / fallback)
// =============================================================================

constexpr int TILE_SIZE = 32;

__global__ void gemm_fp16_naive_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && aCol < K) ?
                     A[row * K + aCol] : __float2half(0.0f);
        Bs[ty][tx] = (bRow < K && col < N) ?
                     B[bRow * N + col] : __float2half(0.0f);

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = alpha * sum;
        if (beta != 0.0f) {
            result += beta * __half2float(C[row * N + col]);
        }
        C[row * N + col] = __float2half(result);
    }
}

// =============================================================================
// Host wrapper
// =============================================================================

void gemm_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Use WMMA kernel for larger matrices
    if (M >= 128 && N >= 128 && K >= 32) {
        dim3 block(NUM_THREADS);
        dim3 grid(cdiv(N, BN), cdiv(M, BM));

        gemm_fp16_wmma_kernel<<<grid, block, 0, stream>>>(
            A, B, C, M, N, K, alpha, beta
        );
    } else {
        // Fallback to naive kernel for small matrices
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE));

        gemm_fp16_naive_kernel<<<grid, block, 0, stream>>>(
            A, B, C, M, N, K, alpha, beta
        );
    }

    CLOVE_CHECK_LAST();
}

// =============================================================================
// Batched GEMM
// =============================================================================

__global__ void gemm_fp16_batched_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int batch, int M, int N, int K,
    float alpha, float beta
) {
    int b = blockIdx.z;

    const half* Ab = A + b * M * K;
    const half* Bb = B + b * K * N;
    half* Cb = C + b * M * N;

    // Same as naive kernel but with batch offset
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && aCol < K) ?
                     Ab[row * K + aCol] : __float2half(0.0f);
        Bs[ty][tx] = (bRow < K && col < N) ?
                     Bb[bRow * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float result = alpha * sum;
        if (beta != 0.0f) {
            result += beta * __half2float(Cb[row * N + col]);
        }
        Cb[row * N + col] = __float2half(result);
    }
}

void gemm_fp16_batched(
    const half* A,
    const half* B,
    half* C,
    int batch, int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(cdiv(N, TILE_SIZE), cdiv(M, TILE_SIZE), batch);

    gemm_fp16_batched_kernel<<<grid, block, 0, stream>>>(
        A, B, C, batch, M, N, K, alpha, beta
    );

    CLOVE_CHECK_LAST();
}

}  // namespace clove
