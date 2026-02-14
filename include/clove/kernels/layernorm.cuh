#pragma once

#include <clove/common.cuh>

namespace clove {

// =============================================================================
// RMS LayerNorm (used in Llama)
// =============================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
void rms_norm(
    const half* input,
    const half* weight,
    half* output,
    int batch_seq,      // batch * seq_len
    int hidden_dim,
    float eps,
    cudaStream_t stream = 0
);

// Fused: RMSNorm(x + residual)
void rms_norm_residual(
    const half* input,
    const half* residual,
    const half* weight,
    half* output,
    half* residual_out,  // optional: store input + residual
    int batch_seq,
    int hidden_dim,
    float eps,
    cudaStream_t stream = 0
);

// =============================================================================
// Standard LayerNorm
// =============================================================================

// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
void layer_norm(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int batch_seq,
    int hidden_dim,
    float eps,
    cudaStream_t stream = 0
);

}  // namespace clove
