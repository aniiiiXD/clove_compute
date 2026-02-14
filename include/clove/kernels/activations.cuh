#pragma once

#include <clove/common.cuh>

namespace clove {

// =============================================================================
// SiLU (Swish)
// =============================================================================

// SiLU: x * sigmoid(x)
void silu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream = 0
);

// =============================================================================
// SwiGLU (Llama-style MLP)
// =============================================================================

// SwiGLU: silu(gate) * up
// gate and up come from parallel projections of input
void swiglu(
    const half* gate,    // [batch, seq, intermediate]
    const half* up,      // [batch, seq, intermediate]
    half* output,        // [batch, seq, intermediate]
    int size,
    cudaStream_t stream = 0
);

// Fused: compute gate and up projections, then SwiGLU
// input @ W_gate -> gate
// input @ W_up -> up
// output = silu(gate) * up
void swiglu_fused(
    const half* input,       // [batch, seq, hidden]
    const half* W_gate,      // [hidden, intermediate]
    const half* W_up,        // [hidden, intermediate]
    half* output,            // [batch, seq, intermediate]
    int batch_seq,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream = 0
);

// =============================================================================
// GeGLU
// =============================================================================

// GeGLU: gelu(gate) * up
void geglu(
    const half* gate,
    const half* up,
    half* output,
    int size,
    cudaStream_t stream = 0
);

// =============================================================================
// GELU
// =============================================================================

// Standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream = 0
);

// Fast GELU approximation (tanh-based)
void gelu_fast(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream = 0
);

// =============================================================================
// ReLU variants
// =============================================================================

void relu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream = 0
);

}  // namespace clove
