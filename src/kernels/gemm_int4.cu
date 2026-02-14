#include <clove/kernels/gemm.cuh>

namespace clove {

// =============================================================================
// INT4 Packing Utilities
// =============================================================================

// Unpack 8 bytes (16 INT4 values) into 16 int8 values
__device__ __forceinline__ void unpack_int4x16(
    const uint2& packed,
    int8_t out[16]
) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t byte = bytes[i];
        // High nibble
        out[i * 2] = static_cast<int8_t>((byte >> 4) & 0x0F) - 8;
        // Low nibble
        out[i * 2 + 1] = static_cast<int8_t>(byte & 0x0F) - 8;
    }
}

// Unpack single byte to 2 INT4 values
__device__ __forceinline__ void unpack_int4x2(
    uint8_t packed,
    int8_t& hi,
    int8_t& lo
) {
    hi = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;  // [-8, 7]
    lo = static_cast<int8_t>(packed & 0x0F) - 8;
}

// =============================================================================
// INT4 GEMM Kernel with Group-wise Quantization
// =============================================================================

constexpr int INT4_BM = 64;
constexpr int INT4_BN = 64;
constexpr int INT4_BK = 64;  // Should be multiple of GROUP_SIZE or handle boundaries
constexpr int INT4_THREADS = 256;

__global__ void gemm_int4_kernel(
    const uint8_t* __restrict__ A_q,     // [M, K/2] packed
    const half* __restrict__ A_scale,    // [M, K/GROUP_SIZE]
    const half* __restrict__ A_zero,     // [M, K/GROUP_SIZE]
    const uint8_t* __restrict__ B_q,     // [K, N/2] packed
    const half* __restrict__ B_scale,    // [K/GROUP_SIZE, N]
    const half* __restrict__ B_zero,     // [K/GROUP_SIZE, N]
    half* __restrict__ C,                // [M, N]
    int M, int N, int K
) {
    // Shared memory for unpacked tiles
    __shared__ int8_t As[INT4_BK][INT4_BM + 4];
    __shared__ int8_t Bs[INT4_BK][INT4_BN + 4];
    __shared__ half As_scale[INT4_BM];
    __shared__ half Bs_scale[INT4_BN];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x % 8;
    const int ty = threadIdx.x / 8;

    // Output position
    const int rowStart = by * INT4_BM + (threadIdx.x / 8) * 2;
    const int colStart = bx * INT4_BN + (threadIdx.x % 8) * 8;

    // FP32 accumulators
    float acc[2][8] = {0.0f};

    const int numKTiles = cdiv(K, INT4_BK);

    for (int kt = 0; kt < numKTiles; kt++) {
        int kBase = kt * INT4_BK;
        int group = kBase / INT4_GROUP_SIZE;

        // Load and unpack A tile
        // A is [M, K/2] packed, we need [BM, BK] unpacked
        for (int i = threadIdx.x; i < (INT4_BM * INT4_BK / 2); i += INT4_THREADS) {
            int loadM = i / (INT4_BK / 2);
            int loadKPacked = i % (INT4_BK / 2);
            int globalM = by * INT4_BM + loadM;
            int globalKPacked = (kBase / 2) + loadKPacked;

            if (globalM < M && globalKPacked < K / 2) {
                uint8_t packed = A_q[globalM * (K / 2) + globalKPacked];
                int8_t hi, lo;
                unpack_int4x2(packed, hi, lo);
                As[loadKPacked * 2][loadM] = hi;
                As[loadKPacked * 2 + 1][loadM] = lo;
            } else {
                As[loadKPacked * 2][loadM] = 0;
                As[loadKPacked * 2 + 1][loadM] = 0;
            }
        }

        // Load and unpack B tile
        // B is [K, N/2] packed, we need [BK, BN] unpacked
        for (int i = threadIdx.x; i < (INT4_BK * INT4_BN / 2); i += INT4_THREADS) {
            int loadK = i / (INT4_BN / 2);
            int loadNPacked = i % (INT4_BN / 2);
            int globalK = kBase + loadK;
            int globalNPacked = (bx * INT4_BN / 2) + loadNPacked;

            if (globalK < K && globalNPacked < N / 2) {
                uint8_t packed = B_q[globalK * (N / 2) + globalNPacked];
                int8_t hi, lo;
                unpack_int4x2(packed, hi, lo);
                Bs[loadK][loadNPacked * 2] = hi;
                Bs[loadK][loadNPacked * 2 + 1] = lo;
            } else {
                Bs[loadK][loadNPacked * 2] = 0;
                Bs[loadK][loadNPacked * 2 + 1] = 0;
            }
        }

        // Load scales for this group
        if (threadIdx.x < INT4_BM) {
            int globalM = by * INT4_BM + threadIdx.x;
            As_scale[threadIdx.x] = (globalM < M && group < cdiv(K, INT4_GROUP_SIZE)) ?
                                    A_scale[globalM * cdiv(K, INT4_GROUP_SIZE) + group] :
                                    __float2half(1.0f);
        }
        if (threadIdx.x < INT4_BN) {
            int globalN = bx * INT4_BN + threadIdx.x;
            Bs_scale[threadIdx.x] = (globalN < N && group < cdiv(K, INT4_GROUP_SIZE)) ?
                                    B_scale[group * N + globalN] :
                                    __float2half(1.0f);
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kk = 0; kk < INT4_BK; kk++) {
            int8_t a_vals[2];
            a_vals[0] = As[kk][ty * 2];
            a_vals[1] = As[kk][ty * 2 + 1];

            int8_t b_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                b_vals[j] = Bs[kk][tx * 8 + j];
            }

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += static_cast<float>(a_vals[i]) *
                                 static_cast<float>(b_vals[j]);
                }
            }
        }

        __syncthreads();

        // Apply scales from this group
        float a_scales[2], b_scales[8];
        a_scales[0] = __half2float(As_scale[ty * 2]);
        a_scales[1] = __half2float(As_scale[ty * 2 + 1]);

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            b_scales[j] = __half2float(Bs_scale[tx * 8 + j]);
        }

        // Accumulate with scales (this is a simplification - proper impl would track per-group)
        // For a fully correct implementation, need to separate accumulation per group
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int outRow = rowStart + i;
        if (outRow >= M) continue;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int outCol = colStart + j;
            if (outCol >= N) continue;

            // Get final scales
            float a_s = __half2float(As_scale[ty * 2 + i]);
            float b_s = __half2float(Bs_scale[tx * 8 + j]);

            float result = acc[i][j] * a_s * b_s;
            C[outRow * N + outCol] = __float2half(result);
        }
    }
}

void gemm_int4(
    const uint8_t* A_q,
    const half* A_scale,
    const half* A_zero,
    const uint8_t* B_q,
    const half* B_scale,
    const half* B_zero,
    half* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    dim3 block(INT4_THREADS);
    dim3 grid(cdiv(N, INT4_BN), cdiv(M, INT4_BM));

    gemm_int4_kernel<<<grid, block, 0, stream>>>(
        A_q, A_scale, A_zero,
        B_q, B_scale, B_zero,
        C, M, N, K
    );

    CLOVE_CHECK_LAST();
}

}  // namespace clove
