#include <clove/runtime/tensor.h>
#include <clove/kernels/gemm.cuh>
#include <clove/kernels/attention.cuh>
#include <clove/kernels/layernorm.cuh>
#include <clove/kernels/activations.cuh>
#include <clove/common.cuh>
#include <vector>

namespace clove {

// =============================================================================
// Llama Model Configuration
// =============================================================================

struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_dim = 4096;
    int intermediate_dim = 11008;
    int num_layers = 32;
    int num_heads = 32;
    int num_kv_heads = 32;  // For GQA, can be < num_heads
    int head_dim = 128;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    int max_seq_len = 4096;

    static LlamaConfig llama_1b() {
        return {
            .vocab_size = 32000,
            .hidden_dim = 2048,
            .intermediate_dim = 5632,
            .num_layers = 22,
            .num_heads = 16,
            .num_kv_heads = 16,
            .head_dim = 128,
            .rope_theta = 10000.0f,
            .rms_norm_eps = 1e-5f,
            .max_seq_len = 4096
        };
    }

    static LlamaConfig llama_3b() {
        return {
            .vocab_size = 32000,
            .hidden_dim = 3200,
            .intermediate_dim = 8640,
            .num_layers = 26,
            .num_heads = 32,
            .num_kv_heads = 32,
            .head_dim = 100,
            .rope_theta = 10000.0f,
            .rms_norm_eps = 1e-5f,
            .max_seq_len = 4096
        };
    }
};

// =============================================================================
// Layer Weights
// =============================================================================

struct LayerWeights {
    Tensor input_norm;    // [hidden]
    Tensor q_proj;        // [hidden, hidden]
    Tensor k_proj;        // [hidden, kv_dim]
    Tensor v_proj;        // [hidden, kv_dim]
    Tensor o_proj;        // [hidden, hidden]
    Tensor post_attn_norm;// [hidden]
    Tensor gate_proj;     // [hidden, intermediate]
    Tensor up_proj;       // [hidden, intermediate]
    Tensor down_proj;     // [intermediate, hidden]
};

struct ModelWeights {
    Tensor embed_tokens;  // [vocab, hidden]
    std::vector<LayerWeights> layers;
    Tensor norm;          // [hidden]
    Tensor lm_head;       // [hidden, vocab]
};

// =============================================================================
// Inference Context
// =============================================================================

class LlamaInference {
public:
    LlamaInference(const LlamaConfig& config)
        : config_(config) {
        // Allocate intermediate buffers
        int max_batch = 32;
        int max_seq = config.max_seq_len;
        int hidden = config.hidden_dim;
        int intermediate = config.intermediate_dim;

        hidden_states_ = Tensor::allocate(
            {max_batch, max_seq, hidden}, DType::FP16);
        attn_out_ = Tensor::allocate(
            {max_batch, max_seq, hidden}, DType::FP16);
        q_ = Tensor::allocate(
            {max_batch, max_seq, config.num_heads, config.head_dim}, DType::FP16);
        k_ = Tensor::allocate(
            {max_batch, max_seq, config.num_kv_heads, config.head_dim}, DType::FP16);
        v_ = Tensor::allocate(
            {max_batch, max_seq, config.num_kv_heads, config.head_dim}, DType::FP16);
        gate_ = Tensor::allocate(
            {max_batch, max_seq, intermediate}, DType::FP16);
        up_ = Tensor::allocate(
            {max_batch, max_seq, intermediate}, DType::FP16);
        mlp_out_ = Tensor::allocate(
            {max_batch, max_seq, hidden}, DType::FP16);

        CLOVE_CHECK_CUDA(cudaStreamCreate(&stream_));
    }

    ~LlamaInference() {
        cudaStreamDestroy(stream_);
    }

    // Forward pass for a single layer
    void transformer_layer(
        Tensor& hidden,           // [batch, seq, hidden]
        const Tensor& positions,  // [batch, seq]
        const LayerWeights& weights,
        int batch,
        int seq_len
    ) {
        int hidden_dim = config_.hidden_dim;
        int num_heads = config_.num_heads;
        int num_kv_heads = config_.num_kv_heads;
        int head_dim = config_.head_dim;
        int intermediate = config_.intermediate_dim;

        // 1. Input RMSNorm
        rms_norm(hidden.data_half(), weights.input_norm.data_half(),
                 attn_out_.data_half(),
                 batch * seq_len, hidden_dim, config_.rms_norm_eps, stream_);

        // 2. QKV Projections
        gemm_fp16(attn_out_.data_half(), weights.q_proj.data_half(),
                  q_.data_half(),
                  batch * seq_len, num_heads * head_dim, hidden_dim,
                  1.0f, 0.0f, stream_);

        gemm_fp16(attn_out_.data_half(), weights.k_proj.data_half(),
                  k_.data_half(),
                  batch * seq_len, num_kv_heads * head_dim, hidden_dim,
                  1.0f, 0.0f, stream_);

        gemm_fp16(attn_out_.data_half(), weights.v_proj.data_half(),
                  v_.data_half(),
                  batch * seq_len, num_kv_heads * head_dim, hidden_dim,
                  1.0f, 0.0f, stream_);

        // 3. Apply RoPE and Attention
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        flash_attention_rope(
            q_.data_half(), k_.data_half(), v_.data_half(),
            attn_out_.data_half(),
            positions.data_ptr<int>(),
            batch, num_heads, num_kv_heads, seq_len, head_dim,
            scale, config_.rope_theta, true, stream_
        );

        // 4. Output projection + residual
        gemm_fp16(attn_out_.data_half(), weights.o_proj.data_half(),
                  hidden.data_half(),
                  batch * seq_len, hidden_dim, hidden_dim,
                  1.0f, 1.0f, stream_);  // beta=1 for residual

        // 5. Post-attention RMSNorm
        rms_norm(hidden.data_half(), weights.post_attn_norm.data_half(),
                 attn_out_.data_half(),
                 batch * seq_len, hidden_dim, config_.rms_norm_eps, stream_);

        // 6. MLP: gate and up projections
        gemm_fp16(attn_out_.data_half(), weights.gate_proj.data_half(),
                  gate_.data_half(),
                  batch * seq_len, intermediate, hidden_dim,
                  1.0f, 0.0f, stream_);

        gemm_fp16(attn_out_.data_half(), weights.up_proj.data_half(),
                  up_.data_half(),
                  batch * seq_len, intermediate, hidden_dim,
                  1.0f, 0.0f, stream_);

        // 7. SwiGLU activation
        swiglu(gate_.data_half(), up_.data_half(), mlp_out_.data_half(),
               batch * seq_len * intermediate, stream_);

        // 8. Down projection + residual
        gemm_fp16(mlp_out_.data_half(), weights.down_proj.data_half(),
                  hidden.data_half(),
                  batch * seq_len, hidden_dim, intermediate,
                  1.0f, 1.0f, stream_);  // beta=1 for residual
    }

private:
    LlamaConfig config_;
    cudaStream_t stream_;

    // Intermediate buffers
    Tensor hidden_states_;
    Tensor attn_out_;
    Tensor q_, k_, v_;
    Tensor gate_, up_, mlp_out_;
};

}  // namespace clove
