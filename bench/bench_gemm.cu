#include <clove/kernels/gemm.cuh>
#include <clove/runtime/tensor.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace clove;

// =============================================================================
// Benchmark Configuration
// =============================================================================

struct BenchConfig {
    int M, N, K;
    const char* name;
};

// Common matrix sizes in LLM inference
std::vector<BenchConfig> bench_configs = {
    // Llama-2 7B style
    {1, 4096, 4096, "decode_1x4096x4096"},
    {32, 4096, 4096, "batch32_4096x4096"},
    {128, 4096, 4096, "prefill128_4096"},
    {512, 4096, 4096, "prefill512_4096"},

    // MLP dimensions (intermediate = 4x hidden typically)
    {1, 11008, 4096, "decode_mlp_up"},
    {1, 4096, 11008, "decode_mlp_down"},
    {128, 11008, 4096, "prefill_mlp_up"},
    {128, 4096, 11008, "prefill_mlp_down"},

    // Square matrices for comparison
    {1024, 1024, 1024, "1024x1024x1024"},
    {2048, 2048, 2048, "2048x2048x2048"},
    {4096, 4096, 4096, "4096x4096x4096"},
};

// =============================================================================
// Timer Utility
// =============================================================================

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// =============================================================================
// cuBLAS Wrapper
// =============================================================================

class CublasGemm {
public:
    CublasGemm() {
        cublasCreate(&handle_);
        cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }

    ~CublasGemm() {
        cublasDestroy(handle_);
    }

    void gemm(const half* A, const half* B, half* C,
              int M, int N, int K, cudaStream_t stream = 0) {
        cublasSetStream(handle_, stream);

        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);

        // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
        // But since our matrices are row-major, we swap A and B
        cublasGemmEx(handle_,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     B, CUDA_R_16F, N,
                     A, CUDA_R_16F, K,
                     &beta,
                     C, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

private:
    cublasHandle_t handle_;
};

// =============================================================================
// Benchmark Function
// =============================================================================

struct BenchResult {
    double clove_ms;
    double cublas_ms;
    double clove_tflops;
    double cublas_tflops;
    double ratio;  // clove / cublas
};

BenchResult benchmark_gemm(int M, int N, int K, int warmup = 10, int iters = 100) {
    // Allocate matrices
    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_clove = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_cublas = Tensor::zeros({M, N}, DType::FP16);

    cudaDeviceSynchronize();

    CublasGemm cublas;
    CudaTimer timer;

    // Warmup - Clove
    for (int i = 0; i < warmup; i++) {
        gemm_fp16(A.data_half(), B.data_half(), C_clove.data_half(),
                  M, N, K, 1.0f, 0.0f, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark - Clove
    timer.start();
    for (int i = 0; i < iters; i++) {
        gemm_fp16(A.data_half(), B.data_half(), C_clove.data_half(),
                  M, N, K, 1.0f, 0.0f, 0);
    }
    float clove_time = timer.stop();
    double clove_ms = clove_time / iters;

    // Warmup - cuBLAS
    for (int i = 0; i < warmup; i++) {
        cublas.gemm(A.data_half(), B.data_half(), C_cublas.data_half(), M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark - cuBLAS
    timer.start();
    for (int i = 0; i < iters; i++) {
        cublas.gemm(A.data_half(), B.data_half(), C_cublas.data_half(), M, N, K);
    }
    float cublas_time = timer.stop();
    double cublas_ms = cublas_time / iters;

    // Compute TFLOPS
    double flops = 2.0 * M * N * K;
    double clove_tflops = flops / (clove_ms * 1e9);
    double cublas_tflops = flops / (cublas_ms * 1e9);
    double ratio = clove_tflops / cublas_tflops;

    return {clove_ms, cublas_ms, clove_tflops, cublas_tflops, ratio};
}

// =============================================================================
// Correctness Check
// =============================================================================

bool verify_correctness(int M, int N, int K) {
    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_clove = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_cublas = Tensor::zeros({M, N}, DType::FP16);

    // Compute with both
    gemm_fp16(A.data_half(), B.data_half(), C_clove.data_half(), M, N, K);

    CublasGemm cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_cublas.data_half(), M, N, K);

    cudaDeviceSynchronize();

    // Copy to host and compare
    std::vector<half> h_clove(M * N);
    std::vector<half> h_cublas(M * N);

    C_clove.copy_to_host(h_clove.data());
    C_cublas.copy_to_host(h_cublas.data());

    cudaDeviceSynchronize();

    // Check relative error
    float max_error = 0.0f;
    float total_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < M * N; i++) {
        float v1 = __half2float(h_clove[i]);
        float v2 = __half2float(h_cublas[i]);
        float abs_err = fabsf(v1 - v2);
        float rel_err = abs_err / (fabsf(v2) + 1e-6f);

        max_error = fmaxf(max_error, rel_err);
        total_error += rel_err;

        if (rel_err > 0.05f) {  // 5% threshold for FP16
            error_count++;
        }
    }

    float avg_error = total_error / (M * N);

    printf("  Correctness: max_rel_err=%.4f, avg_rel_err=%.6f, errors=%d/%d\n",
           max_error, avg_error, error_count, M * N);

    return max_error < 0.1f && error_count < (M * N) / 100;  // Allow 1% errors
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=============================================================\n");
    printf("Clove Compute GEMM Benchmark\n");
    printf("=============================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Peak FP16 TFLOPS: ~%.1f (theoretical)\n",
           prop.clockRate * 1e-6 * prop.multiProcessorCount * 256 * 2 / 1000.0);
    printf("=============================================================\n\n");

    // Verify correctness first
    printf("Verifying correctness...\n");
    bool correct = verify_correctness(256, 256, 256);
    printf("Small matrix (256x256x256): %s\n\n", correct ? "PASS" : "FAIL");

    correct = verify_correctness(1024, 1024, 1024);
    printf("Medium matrix (1024x1024x1024): %s\n\n", correct ? "PASS" : "FAIL");

    // Run benchmarks
    printf("Running benchmarks (warmup=10, iters=100)...\n\n");
    printf("%-25s %10s %10s %10s %10s %8s\n",
           "Config", "Clove(ms)", "cuBLAS(ms)", "Clove(TF)", "cuBLAS(TF)", "Ratio");
    printf("--------------------------------------------------------------------------------\n");

    for (const auto& config : bench_configs) {
        BenchResult result = benchmark_gemm(config.M, config.N, config.K);

        printf("%-25s %10.3f %10.3f %10.2f %10.2f %7.1f%%\n",
               config.name,
               result.clove_ms,
               result.cublas_ms,
               result.clove_tflops,
               result.cublas_tflops,
               result.ratio * 100);
    }

    printf("\n=============================================================\n");
    printf("Note: Ratio > 100%% means Clove is faster than cuBLAS\n");
    printf("=============================================================\n");

    return 0;
}
