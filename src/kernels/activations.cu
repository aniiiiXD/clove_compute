#include <clove/kernels/activations.cuh>

namespace clove {

// =============================================================================
// Activation Functions (Device)
// =============================================================================

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_f(float x) {
    // Exact GELU using erf
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));  // 1/sqrt(2)
}

__device__ __forceinline__ float gelu_fast_f(float x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// =============================================================================
// SiLU Kernel
// =============================================================================

__global__ void silu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 8 elements per thread using float4 (vectorized)
    int vec_idx = idx * 8;
    if (vec_idx + 7 < size) {
        float4 in1 = reinterpret_cast<const float4*>(input)[idx * 2];
        float4 in2 = reinterpret_cast<const float4*>(input)[idx * 2 + 1];

        half* h1 = reinterpret_cast<half*>(&in1);
        half* h2 = reinterpret_cast<half*>(&in2);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            h1[i] = __float2half(silu_f(__half2float(h1[i])));
            h2[i] = __float2half(silu_f(__half2float(h2[i])));
        }

        reinterpret_cast<float4*>(output)[idx * 2] = in1;
        reinterpret_cast<float4*>(output)[idx * 2 + 1] = in2;
    } else {
        // Handle remainder
        for (int i = vec_idx; i < size; i++) {
            output[i] = __float2half(silu_f(__half2float(input[i])));
        }
    }
}

void silu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads * 8);
    silu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    CLOVE_CHECK_LAST();
}

// =============================================================================
// SwiGLU Kernel
// =============================================================================

__global__ void swiglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized processing
    int vec_idx = idx * 8;
    if (vec_idx + 7 < size) {
        float4 g1 = reinterpret_cast<const float4*>(gate)[idx * 2];
        float4 g2 = reinterpret_cast<const float4*>(gate)[idx * 2 + 1];
        float4 u1 = reinterpret_cast<const float4*>(up)[idx * 2];
        float4 u2 = reinterpret_cast<const float4*>(up)[idx * 2 + 1];

        half* hg1 = reinterpret_cast<half*>(&g1);
        half* hg2 = reinterpret_cast<half*>(&g2);
        half* hu1 = reinterpret_cast<half*>(&u1);
        half* hu2 = reinterpret_cast<half*>(&u2);

        float4 out1, out2;
        half* ho1 = reinterpret_cast<half*>(&out1);
        half* ho2 = reinterpret_cast<half*>(&out2);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float g_val = silu_f(__half2float(hg1[i]));
            float u_val = __half2float(hu1[i]);
            ho1[i] = __float2half(g_val * u_val);

            g_val = silu_f(__half2float(hg2[i]));
            u_val = __half2float(hu2[i]);
            ho2[i] = __float2half(g_val * u_val);
        }

        reinterpret_cast<float4*>(output)[idx * 2] = out1;
        reinterpret_cast<float4*>(output)[idx * 2 + 1] = out2;
    } else {
        for (int i = vec_idx; i < size; i++) {
            float g = silu_f(__half2float(gate[i]));
            float u = __half2float(up[i]);
            output[i] = __float2half(g * u);
        }
    }
}

void swiglu(
    const half* gate,
    const half* up,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads * 8);
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, size);
    CLOVE_CHECK_LAST();
}

// =============================================================================
// GeGLU Kernel
// =============================================================================

__global__ void geglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gelu_f(__half2float(gate[idx]));
        float u = __half2float(up[idx]);
        output[idx] = __float2half(g * u);
    }
}

void geglu(
    const half* gate,
    const half* up,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads);
    geglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, size);
    CLOVE_CHECK_LAST();
}

// =============================================================================
// GELU Kernels
// =============================================================================

__global__ void gelu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(gelu_f(__half2float(input[idx])));
    }
}

void gelu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads);
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    CLOVE_CHECK_LAST();
}

__global__ void gelu_fast_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(gelu_fast_f(__half2float(input[idx])));
    }
}

void gelu_fast(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads);
    gelu_fast_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    CLOVE_CHECK_LAST();
}

// =============================================================================
// ReLU Kernel
// =============================================================================

__global__ void relu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(fmaxf(val, 0.0f));
    }
}

void relu(
    const half* input,
    half* output,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = cdiv(size, threads);
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    CLOVE_CHECK_LAST();
}

}  // namespace clove
