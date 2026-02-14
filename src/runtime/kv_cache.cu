#include <clove/runtime/tensor.h>
#include <clove/common.cuh>

namespace clove {

// =============================================================================
// KV Cache Implementation
// =============================================================================

class KVCache {
public:
    KVCache(
        int batch_size,
        int num_layers,
        int num_kv_heads,
        int head_dim,
        int max_seq_len
    ) : batch_size_(batch_size),
        num_layers_(num_layers),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        max_seq_len_(max_seq_len) {

        // Allocate contiguous cache
        // Shape: [num_layers, 2, batch, num_kv_heads, max_seq, head_dim]
        // 2 = K and V
        size_t cache_size = num_layers * 2 * batch_size * num_kv_heads *
                           max_seq_len * head_dim * sizeof(half);

        CLOVE_CHECK_CUDA(cudaMalloc(&cache_, cache_size));
        CLOVE_CHECK_CUDA(cudaMemset(cache_, 0, cache_size));

        positions_.resize(batch_size, 0);
    }

    ~KVCache() {
        if (cache_) {
            cudaFree(cache_);
        }
    }

    // Get K cache for a layer: [batch, num_kv_heads, max_seq, head_dim]
    half* k_cache(int layer) {
        size_t layer_size = 2 * batch_size_ * num_kv_heads_ *
                           max_seq_len_ * head_dim_;
        size_t k_offset = layer * layer_size;
        return reinterpret_cast<half*>(cache_) + k_offset;
    }

    // Get V cache for a layer: [batch, num_kv_heads, max_seq, head_dim]
    half* v_cache(int layer) {
        size_t layer_size = 2 * batch_size_ * num_kv_heads_ *
                           max_seq_len_ * head_dim_;
        size_t v_offset = layer * layer_size +
                         batch_size_ * num_kv_heads_ * max_seq_len_ * head_dim_;
        return reinterpret_cast<half*>(cache_) + v_offset;
    }

    int current_length(int batch_idx) const {
        return positions_[batch_idx];
    }

    void advance_position(int batch_idx, int tokens = 1) {
        positions_[batch_idx] += tokens;
    }

    void reset(int batch_idx = -1) {
        if (batch_idx < 0) {
            std::fill(positions_.begin(), positions_.end(), 0);
        } else {
            positions_[batch_idx] = 0;
        }
    }

    size_t memory_usage() const {
        return num_layers_ * 2 * batch_size_ * num_kv_heads_ *
               max_seq_len_ * head_dim_ * sizeof(half);
    }

private:
    void* cache_ = nullptr;
    int batch_size_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int max_seq_len_;
    std::vector<int> positions_;
};

// =============================================================================
// KV Cache Update Kernel
// =============================================================================

__global__ void update_kv_cache_kernel(
    const half* __restrict__ new_k,   // [batch, num_heads, new_tokens, head_dim]
    const half* __restrict__ new_v,   // [batch, num_heads, new_tokens, head_dim]
    half* __restrict__ k_cache,       // [batch, num_heads, max_seq, head_dim]
    half* __restrict__ v_cache,       // [batch, num_heads, max_seq, head_dim]
    const int* __restrict__ positions,// [batch] - current position for each seq
    int batch,
    int num_heads,
    int new_tokens,
    int head_dim,
    int max_seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * new_tokens * head_dim;

    if (idx >= total) return;

    // Decode indices
    int d = idx % head_dim;
    int t = (idx / head_dim) % new_tokens;
    int h = (idx / head_dim / new_tokens) % num_heads;
    int b = idx / head_dim / new_tokens / num_heads;

    int pos = positions[b] + t;
    if (pos >= max_seq_len) return;

    // Source index in new_k/new_v
    int src_idx = ((b * num_heads + h) * new_tokens + t) * head_dim + d;

    // Destination index in cache
    int dst_idx = ((b * num_heads + h) * max_seq_len + pos) * head_dim + d;

    k_cache[dst_idx] = new_k[src_idx];
    v_cache[dst_idx] = new_v[src_idx];
}

void update_kv_cache(
    const half* new_k,
    const half* new_v,
    half* k_cache,
    half* v_cache,
    const int* positions,
    int batch,
    int num_heads,
    int new_tokens,
    int head_dim,
    int max_seq_len,
    cudaStream_t stream
) {
    int total = batch * num_heads * new_tokens * head_dim;
    int block = 256;
    int grid = cdiv(total, block);

    update_kv_cache_kernel<<<grid, block, 0, stream>>>(
        new_k, new_v, k_cache, v_cache, positions,
        batch, num_heads, new_tokens, head_dim, max_seq_len
    );
    CLOVE_CHECK_LAST();
}

}  // namespace clove
