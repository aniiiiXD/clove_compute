#include <clove/kernels/attention.cuh>

// RoPE implementation is in attention.cu
// This file can contain additional RoPE utilities if needed

namespace clove {

// Precompute RoPE frequencies for caching
__global__ void compute_rope_freqs_kernel(
    float* freqs,      // [max_seq_len, head_dim/2, 2] (cos, sin)
    int max_seq_len,
    int head_dim,
    float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * (head_dim / 2);

    if (idx >= total) return;

    int pos = idx / (head_dim / 2);
    int dim = idx % (head_dim / 2);

    float freq = 1.0f / powf(theta, static_cast<float>(2 * dim) / head_dim);
    float angle = pos * freq;

    freqs[idx * 2] = cosf(angle);
    freqs[idx * 2 + 1] = sinf(angle);
}

// Apply RoPE using precomputed frequencies
__global__ void apply_rope_cached_kernel(
    half* x,
    const float* freqs,  // [max_seq_len, head_dim/2, 2]
    const int* positions,
    int batch,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = batch * seq_len * num_heads * (head_dim / 2);

    if (idx >= total_pairs) return;

    int pair_idx = idx;
    int dim_pair = pair_idx % (head_dim / 2);
    pair_idx /= (head_dim / 2);
    int head = pair_idx % num_heads;
    pair_idx /= num_heads;
    int seq = pair_idx % seq_len;
    int b = pair_idx / seq_len;

    int pos = positions[b * seq_len + seq];

    // Get precomputed cos/sin
    int freq_idx = pos * (head_dim / 2) + dim_pair;
    float cos_val = freqs[freq_idx * 2];
    float sin_val = freqs[freq_idx * 2 + 1];

    // Apply rotation
    int base_idx = ((b * seq_len + seq) * num_heads + head) * head_dim + dim_pair * 2;
    float x0 = __half2float(x[base_idx]);
    float x1 = __half2float(x[base_idx + 1]);

    x[base_idx] = __float2half(x0 * cos_val - x1 * sin_val);
    x[base_idx + 1] = __float2half(x0 * sin_val + x1 * cos_val);
}

}  // namespace clove
