#include <clove/common.cuh>
#include <cfloat>

namespace clove {

// =============================================================================
// Online Softmax (Single Pass)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_online_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch,
    int dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch) return;

    const half* x = input + row * dim;
    half* y = output + row * dim;

    // Each thread processes multiple elements
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    // First pass: find max and compute sum using online algorithm
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);

        if (val > local_max) {
            // Rescale existing sum
            local_sum *= expf(local_max - val);
            local_max = val;
        }
        local_sum += expf(val - local_max);
    }

    // Reduce across threads
    // Need to handle the online algorithm in reduction carefully
    __shared__ float s_max[32];
    __shared__ float s_sum[32];

    int lane = tid % 32;
    int wid = tid / 32;

    // Warp reduce for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, local_sum, offset);

        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum = local_sum + other_sum * expf(other_max - local_max);
        }
    }

    if (lane == 0) {
        s_max[wid] = local_max;
        s_sum[wid] = local_sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 32) {
        local_max = (tid < BLOCK_SIZE / 32) ? s_max[tid] : -FLT_MAX;
        local_sum = (tid < BLOCK_SIZE / 32) ? s_sum[tid] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, local_sum, offset);

            if (other_max > local_max) {
                local_sum = local_sum * expf(local_max - other_max) + other_sum;
                local_max = other_max;
            } else {
                local_sum = local_sum + other_sum * expf(other_max - local_max);
            }
        }
    }

    __shared__ float final_max, final_sum;
    if (tid == 0) {
        final_max = local_max;
        final_sum = local_sum;
    }
    __syncthreads();

    // Second pass: compute softmax
    float inv_sum = 1.0f / final_sum;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);
        y[i] = __float2half(expf(val - final_max) * inv_sum);
    }
}

void softmax(
    const half* input,
    half* output,
    int batch,
    int dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    softmax_online_kernel<BLOCK_SIZE><<<batch, BLOCK_SIZE, 0, stream>>>(
        input, output, batch, dim
    );
    CLOVE_CHECK_LAST();
}

// =============================================================================
// Softmax with Temperature
// =============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_temperature_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch,
    int dim,
    float temperature
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch) return;

    const half* x = input + row * dim;
    half* y = output + row * dim;

    float inv_temp = 1.0f / temperature;

    // Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]) * inv_temp;
        local_max = fmaxf(local_max, val);
    }
    local_max = block_reduce_max<BLOCK_SIZE>(local_max);

    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;

    // Compute exp sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]) * inv_temp;
        local_sum += expf(val - local_max);
    }
    local_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    __shared__ float s_sum;
    if (tid == 0) s_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Compute output
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]) * inv_temp;
        y[i] = __float2half(expf(val - local_max) * inv_sum);
    }
}

void softmax_temperature(
    const half* input,
    half* output,
    int batch,
    int dim,
    float temperature,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    softmax_temperature_kernel<BLOCK_SIZE><<<batch, BLOCK_SIZE, 0, stream>>>(
        input, output, batch, dim, temperature
    );
    CLOVE_CHECK_LAST();
}

// =============================================================================
// Log Softmax
// =============================================================================

template<int BLOCK_SIZE>
__global__ void log_softmax_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int batch,
    int dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= batch) return;

    const half* x = input + row * dim;
    half* y = output + row * dim;

    // Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, __half2float(x[i]));
    }
    local_max = block_reduce_max<BLOCK_SIZE>(local_max);

    __shared__ float s_max;
    if (tid == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;

    // Compute log(sum(exp))
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        local_sum += expf(__half2float(x[i]) - local_max);
    }
    local_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    __shared__ float s_log_sum;
    if (tid == 0) s_log_sum = logf(local_sum);
    __syncthreads();
    float log_sum = s_log_sum;

    // Compute output: x - max - log(sum)
    for (int i = tid; i < dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);
        y[i] = __float2half(val - local_max - log_sum);
    }
}

void log_softmax(
    const half* input,
    half* output,
    int batch,
    int dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    log_softmax_kernel<BLOCK_SIZE><<<batch, BLOCK_SIZE, 0, stream>>>(
        input, output, batch, dim
    );
    CLOVE_CHECK_LAST();
}

}  // namespace clove
