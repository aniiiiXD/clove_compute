#include <clove/kernels/layernorm.cuh>

namespace clove {

// =============================================================================
// RMS LayerNorm Kernel
// =============================================================================

template<int BLOCK_SIZE>
__global__ void rms_norm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const half* x = input + row * hidden_dim;
    half* y = output + row * hidden_dim;

    // Compute sum of squares using float for precision
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Block reduce
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);

    // Broadcast result
    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();

    float rms = s_rms;

    // Normalize and apply weight
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half(val * rms * w);
    }
}

void rms_norm(
    const half* input,
    const half* weight,
    half* output,
    int batch_seq,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    rms_norm_kernel<BLOCK_SIZE><<<batch_seq, BLOCK_SIZE, 0, stream>>>(
        input, weight, output, hidden_dim, eps
    );
    CLOVE_CHECK_LAST();
}

// =============================================================================
// Fused RMS Norm + Residual
// =============================================================================

template<int BLOCK_SIZE>
__global__ void rms_norm_residual_kernel(
    const half* __restrict__ input,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    half* __restrict__ output,
    half* __restrict__ residual_out,
    int hidden_dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const half* x = input + row * hidden_dim;
    const half* r = residual + row * hidden_dim;
    half* y = output + row * hidden_dim;
    half* r_out = residual_out ? residual_out + row * hidden_dim : nullptr;

    // First pass: add residual and compute sum of squares
    extern __shared__ float shared[];
    float* x_float = shared;  // [hidden_dim]

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]) + __half2float(r[i]);
        x_float[i] = val;
        sum_sq += val * val;

        if (r_out) {
            r_out[i] = __float2half(val);
        }
    }
    __syncthreads();

    // Block reduce
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);

    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();

    float rms = s_rms;

    // Normalize and apply weight
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = x_float[i];
        float w = __half2float(weight[i]);
        y[i] = __float2half(val * rms * w);
    }
}

void rms_norm_residual(
    const half* input,
    const half* residual,
    const half* weight,
    half* output,
    half* residual_out,
    int batch_seq,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    size_t smem = hidden_dim * sizeof(float);

    rms_norm_residual_kernel<BLOCK_SIZE><<<batch_seq, BLOCK_SIZE, smem, stream>>>(
        input, residual, weight, output, residual_out, hidden_dim, eps
    );
    CLOVE_CHECK_LAST();
}

// =============================================================================
// Standard LayerNorm
// =============================================================================

template<int BLOCK_SIZE>
__global__ void layer_norm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const half* x = input + row * hidden_dim;
    half* y = output + row * hidden_dim;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        sum += __half2float(x[i]);
    }
    sum = block_reduce_sum<BLOCK_SIZE>(sum);

    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / hidden_dim;
    }
    __syncthreads();

    float mean = s_mean;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float diff = __half2float(x[i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum<BLOCK_SIZE>(var_sum);

    __shared__ float s_inv_std;
    if (tid == 0) {
        s_inv_std = rsqrtf(var_sum / hidden_dim + eps);
    }
    __syncthreads();

    float inv_std = s_inv_std;

    // Normalize, scale, and shift
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = __half2float(x[i]);
        float normalized = (val - mean) * inv_std;
        float scaled = normalized * __half2float(weight[i]) + __half2float(bias[i]);
        y[i] = __float2half(scaled);
    }
}

void layer_norm(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_seq,
    int hidden_dim,
    float eps,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    layer_norm_kernel<BLOCK_SIZE><<<batch_seq, BLOCK_SIZE, 0, stream>>>(
        input, weight, bias, output, hidden_dim, eps
    );
    CLOVE_CHECK_LAST();
}

}  // namespace clove
