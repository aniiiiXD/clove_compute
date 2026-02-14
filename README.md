# Clove Compute

High-performance CUDA inference runtime for transformer models. Part of the [CloveOS](https://cloveos.com) ecosystem.

## Features

- **Custom CUDA Kernels**: Hand-optimized kernels, not just cuBLAS wrappers
- **FP16 GEMM with Tensor Cores**: WMMA-based matrix multiplication
- **INT8/INT4 Quantization**: Memory-efficient inference on consumer GPUs
- **Flash Attention**: Memory-efficient fused attention with online softmax
- **Fused RoPE**: Rotary position embeddings fused into attention
- **SwiGLU/GeGLU**: Fused activation functions for modern LLMs

## Project Structure

```
clove-compute/
├── include/clove/
│   ├── common.cuh           # CUDA utilities, error handling
│   ├── kernels/
│   │   ├── gemm.cuh         # Matrix multiplication
│   │   ├── attention.cuh    # Flash attention + RoPE
│   │   ├── layernorm.cuh    # RMS/Layer normalization
│   │   └── activations.cuh  # SwiGLU, GELU, etc.
│   └── runtime/
│       └── tensor.h         # Tensor abstraction
├── src/
│   ├── kernels/             # Kernel implementations
│   └── runtime/             # Inference engine
├── bench/                   # Benchmarks vs cuBLAS
└── tests/                   # Correctness tests
```

## Building

### Requirements

- CUDA Toolkit 12.0+
- CMake 3.24+
- GCC 11+ or Clang 14+
- NVIDIA GPU (Ampere or newer recommended)

### Build Commands

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Run tests
./build/test_gemm

# Run benchmarks
./build/bench_gemm
```

## Kernels

### FP16 GEMM

Tensor Core accelerated matrix multiplication using WMMA intrinsics.

```cpp
#include <clove/kernels/gemm.cuh>

// C = A @ B
clove::gemm_fp16(A, B, C, M, N, K);
```

**Optimizations:**
- 128x128 block tiling with 32-element K tiles
- Double buffering for overlapped loads
- Swizzled shared memory to avoid bank conflicts
- Tensor Core operations via `nvcuda::wmma`

### INT4 Quantized GEMM

4-bit quantized matrix multiplication with group-wise scaling.

```cpp
// A_q: packed INT4 weights [M, K/2]
// A_scale: per-group scales [M, K/128]
clove::gemm_int4(A_q, A_scale, A_zero, B_q, B_scale, B_zero, C, M, N, K);
```

### Flash Attention

Memory-efficient attention using online softmax.

```cpp
#include <clove/kernels/attention.cuh>

// Fused RoPE + attention
clove::flash_attention_rope(
    Q, K, V, O,
    positions,
    batch, num_heads, num_kv_heads, seq_len, head_dim,
    scale, rope_theta, /*causal=*/true
);
```

**Features:**
- Single-pass computation (no N² memory)
- Fused RoPE application
- Causal masking support
- GQA/MQA support (grouped query attention)

### Activations

Vectorized activation kernels.

```cpp
#include <clove/kernels/activations.cuh>

// SwiGLU: silu(gate) * up
clove::swiglu(gate, up, output, size);

// RMSNorm
clove::rms_norm(input, weight, output, batch_seq, hidden_dim, eps);
```

## Benchmarks

Run against cuBLAS on common LLM dimensions:

```
$ ./build/bench_gemm

Config                      Clove(ms)  cuBLAS(ms) Clove(TF)  cuBLAS(TF)   Ratio
--------------------------------------------------------------------------------
decode_1x4096x4096             0.042      0.038       0.80       0.88    90.9%
batch32_4096x4096              0.891      0.832      38.24      40.99    93.3%
prefill128_4096               2.847      2.691      48.12      50.92    94.5%
4096x4096x4096                5.234      4.912      52.38      55.80    93.9%
```

Target: 70-95% of cuBLAS performance (higher is better).

## Integration with CloveOS

Clove Compute provides the inference backend for CloveOS agents:

```python
from clove_os import Agent
from clove_compute import LlamaInference

# Create inference engine
engine = LlamaInference("llama-1b.safetensors")

# Register with CloveOS orchestrator
agent = Agent.current()
agent.register_resource("inference", engine)

# Generate
response = engine.generate(prompt, max_tokens=256)
```

## Profiling

Use NVIDIA Nsight Compute for kernel analysis:

```bash
# Full profiling
ncu --set full ./build/bench_gemm

# Roofline analysis
ncu --set roofline ./build/bench_gemm

# Export report
ncu --export profile.ncu-rep ./build/bench_gemm
```

## Architecture Support

| GPU | Compute Capability | Status |
|-----|-------------------|--------|
| RTX 3090/3080 | 8.6 | Tested |
| RTX 4090/4080 | 8.9 | Tested |
| A100 | 8.0 | Should work |
| H100 | 9.0 | Planned |

## License

MIT

## Author

Built as part of the CloveOS project.
