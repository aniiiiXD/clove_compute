#pragma once

#include <clove/common.cuh>
#include <cuda_fp16.h>

namespace clove {

// =============================================================================
// Attention Configuration
// =============================================================================

struct AttentionConfig {
    static constexpr int Br = 64;     // Query block size
    static constexpr int Bc = 64;     // Key block size
    static constexpr int D = 128;     // Head dimension (common in modern LLMs)

    static constexpr float ROPE_THETA = 10000.0f;
};

// =============================================================================
// RoPE (Rotary Position Embedding)
// =============================================================================

// Apply RoPE to Q or K tensor in-place
// x: [batch, seq, num_heads, head_dim]
void apply_rope(
    half* x,
    const int* positions,  // [batch, seq] - position indices
    int batch,
    int seq_len,
    int num_heads,
    int head_dim,
    float theta = 10000.0f,
    cudaStream_t stream = 0
);

// =============================================================================
// Flash Attention (Fused with RoPE)
// =============================================================================

// Standard multi-head attention
// Q, K, V: [batch, num_heads, seq_len, head_dim]
// Output:  [batch, num_heads, seq_len, head_dim]
void flash_attention(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,         // typically 1/sqrt(head_dim)
    bool causal,
    cudaStream_t stream = 0
);

// Attention with fused RoPE (applies RoPE to Q,K before attention)
// Q, K: [batch, seq_len, num_heads, head_dim] (pre-RoPE)
// V:    [batch, seq_len, num_heads, head_dim]
// Output: [batch, seq_len, num_heads, head_dim]
void flash_attention_rope(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    const int* positions,
    int batch,
    int num_heads,
    int num_kv_heads,    // For GQA/MQA
    int seq_len,
    int head_dim,
    float scale,
    float rope_theta,
    bool causal,
    cudaStream_t stream = 0
);

// =============================================================================
// Paged Attention (for KV Cache)
// =============================================================================

// Attention with paged KV cache (vLLM-style)
// For decode phase with variable-length sequences
void paged_attention(
    const half* Q,           // [batch, 1, num_heads, head_dim]
    const half* K_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    const half* V_cache,     // [num_blocks, block_size, num_kv_heads, head_dim]
    half* O,                 // [batch, 1, num_heads, head_dim]
    const int* block_tables, // [batch, max_blocks_per_seq]
    const int* seq_lens,     // [batch]
    int batch,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_seq_len,
    float scale,
    cudaStream_t stream = 0
);

}  // namespace clove
