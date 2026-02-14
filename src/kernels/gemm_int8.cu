#include <clove/kernels/gemm.cuh>

namespace clove {

// =============================================================================
// INT8 GEMM Kernel with Dequantization
// =============================================================================

constexpr int INT8_BM = 64;
constexpr int INT8_BN = 64;
constexpr int INT8_BK = 32;
constexpr int INT8_TM = 8;
constexpr int INT8_TN = 8;
constexpr int INT8_THREADS = 256;

__global__ void gemm_int8_kernel(
    const int8_t* __restrict__ A_q,
    const int8_t* __restrict__ B_q,
    half* __restrict__ C,
    int M, int N, int K,
    float A_scale, int8_t A_zero,
    float B_scale, int8_t B_zero
) {
    __shared__ int8_t As[INT8_BK][INT8_BM + 4];  // +4 to avoid bank conflicts
    __shared__ int8_t Bs[INT8_BK][INT8_BN + 4];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;

    // Each thread computes TM x TN output elements
    const int rowStart = by * INT8_BM + ty * INT8_TM;
    const int colStart = bx * INT8_BN + tx * INT8_TN;

    // INT32 accumulators for precision
    int acc[INT8_TM][INT8_TN] = {0};

    // Pre-compute combined scale
    float combined_scale = A_scale * B_scale;

    // Loop over K dimension
    for (int k = 0; k < K; k += INT8_BK) {
        // Cooperative load A tile
        for (int i = threadIdx.x; i < INT8_BM * INT8_BK; i += INT8_THREADS) {
            int loadM = i % INT8_BM;
            int loadK = i / INT8_BM;
            int globalM = by * INT8_BM + loadM;
            int globalK = k + loadK;

            int8_t val = (globalM < M && globalK < K) ?
                         A_q[globalM * K + globalK] : 0;
            As[loadK][loadM] = val;
        }

        // Cooperative load B tile
        for (int i = threadIdx.x; i < INT8_BK * INT8_BN; i += INT8_THREADS) {
            int loadK = i / INT8_BN;
            int loadN = i % INT8_BN;
            int globalK = k + loadK;
            int globalN = bx * INT8_BN + loadN;

            int8_t val = (globalK < K && globalN < N) ?
                         B_q[globalK * N + globalN] : 0;
            Bs[loadK][loadN] = val;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < INT8_BK; kk++) {
            // Load A values for this thread
            int8_t a_vals[INT8_TM];
            #pragma unroll
            for (int i = 0; i < INT8_TM; i++) {
                a_vals[i] = As[kk][ty * INT8_TM + i];
            }

            // Load B values for this thread
            int8_t b_vals[INT8_TN];
            #pragma unroll
            for (int j = 0; j < INT8_TN; j++) {
                b_vals[j] = Bs[kk][tx * INT8_TN + j];
            }

            // Accumulate (subtract zero points)
            #pragma unroll
            for (int i = 0; i < INT8_TM; i++) {
                int a_dq = static_cast<int>(a_vals[i]) - static_cast<int>(A_zero);
                #pragma unroll
                for (int j = 0; j < INT8_TN; j++) {
                    int b_dq = static_cast<int>(b_vals[j]) - static_cast<int>(B_zero);
                    acc[i][j] += a_dq * b_dq;
                }
            }
        }

        __syncthreads();
    }

    // Write results with scaling
    #pragma unroll
    for (int i = 0; i < INT8_TM; i++) {
        int outRow = rowStart + i;
        if (outRow >= M) continue;

        #pragma unroll
        for (int j = 0; j < INT8_TN; j++) {
            int outCol = colStart + j;
            if (outCol >= N) continue;

            float result = static_cast<float>(acc[i][j]) * combined_scale;
            C[outRow * N + outCol] = __float2half(result);
        }
    }
}

void gemm_int8(
    const int8_t* A_q,
    const int8_t* B_q,
    half* C,
    int M, int N, int K,
    float A_scale, int8_t A_zero,
    float B_scale, int8_t B_zero,
    cudaStream_t stream
) {
    dim3 block(INT8_THREADS);
    dim3 grid(cdiv(N, INT8_BN), cdiv(M, INT8_BM));

    gemm_int8_kernel<<<grid, block, 0, stream>>>(
        A_q, B_q, C, M, N, K,
        A_scale, A_zero, B_scale, B_zero
    );

    CLOVE_CHECK_LAST();
}

}  // namespace clove
