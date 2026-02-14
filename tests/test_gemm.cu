#include <clove/kernels/gemm.cuh>
#include <clove/runtime/tensor.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace clove;

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST_ASSERT(cond, msg)                          \
    do {                                                 \
        if (!(cond)) {                                  \
            printf("FAIL: %s\n", msg);                  \
            return false;                               \
        }                                                \
    } while (0)

bool compare_tensors(
    const half* a, const half* b,
    int size,
    float rtol = 0.05f,  // 5% relative tolerance for FP16
    float atol = 1e-3f
) {
    std::vector<half> ha(size), hb(size);

    cudaMemcpy(ha.data(), a, size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, size * sizeof(half), cudaMemcpyDeviceToHost);

    int errors = 0;
    float max_rel_err = 0.0f;

    for (int i = 0; i < size; i++) {
        float va = __half2float(ha[i]);
        float vb = __half2float(hb[i]);
        float abs_err = fabsf(va - vb);
        float rel_err = abs_err / (fabsf(vb) + atol);

        max_rel_err = fmaxf(max_rel_err, rel_err);

        if (rel_err > rtol && abs_err > atol) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at %d: got %.4f, expected %.4f (rel_err=%.4f)\n",
                       i, va, vb, rel_err);
            }
        }
    }

    if (errors > 0) {
        printf("  Total errors: %d/%d, max_rel_err: %.4f\n", errors, size, max_rel_err);
    }

    return errors == 0;
}

// =============================================================================
// cuBLAS Reference
// =============================================================================

class CublasRef {
public:
    CublasRef() {
        cublasCreate(&handle_);
        cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }

    ~CublasRef() {
        cublasDestroy(handle_);
    }

    void gemm(const half* A, const half* B, half* C, int M, int N, int K) {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);

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
// Tests
// =============================================================================

bool test_gemm_small() {
    printf("test_gemm_small: ");

    const int M = 64, N = 64, K = 64;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_ours = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_ref = Tensor::zeros({M, N}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(), M, N, K);

    CublasRef cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_ref.data_half(), M, N, K);

    cudaDeviceSynchronize();

    TEST_ASSERT(compare_tensors(C_ours.data_half(), C_ref.data_half(), M * N),
                "Results don't match cuBLAS");

    printf("PASS\n");
    return true;
}

bool test_gemm_medium() {
    printf("test_gemm_medium: ");

    const int M = 256, N = 256, K = 256;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_ours = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_ref = Tensor::zeros({M, N}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(), M, N, K);

    CublasRef cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_ref.data_half(), M, N, K);

    cudaDeviceSynchronize();

    TEST_ASSERT(compare_tensors(C_ours.data_half(), C_ref.data_half(), M * N),
                "Results don't match cuBLAS");

    printf("PASS\n");
    return true;
}

bool test_gemm_large() {
    printf("test_gemm_large: ");

    const int M = 1024, N = 1024, K = 1024;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_ours = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_ref = Tensor::zeros({M, N}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(), M, N, K);

    CublasRef cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_ref.data_half(), M, N, K);

    cudaDeviceSynchronize();

    TEST_ASSERT(compare_tensors(C_ours.data_half(), C_ref.data_half(), M * N),
                "Results don't match cuBLAS");

    printf("PASS\n");
    return true;
}

bool test_gemm_nonsquare() {
    printf("test_gemm_nonsquare: ");

    // Llama-like dimensions
    const int M = 128, N = 4096, K = 4096;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_ours = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_ref = Tensor::zeros({M, N}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(), M, N, K);

    CublasRef cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_ref.data_half(), M, N, K);

    cudaDeviceSynchronize();

    TEST_ASSERT(compare_tensors(C_ours.data_half(), C_ref.data_half(), M * N),
                "Results don't match cuBLAS");

    printf("PASS\n");
    return true;
}

bool test_gemm_decode() {
    printf("test_gemm_decode (M=1): ");

    // Single-token decode
    const int M = 1, N = 4096, K = 4096;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_ours = Tensor::zeros({M, N}, DType::FP16);
    Tensor C_ref = Tensor::zeros({M, N}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(), M, N, K);

    CublasRef cublas;
    cublas.gemm(A.data_half(), B.data_half(), C_ref.data_half(), M, N, K);

    cudaDeviceSynchronize();

    TEST_ASSERT(compare_tensors(C_ours.data_half(), C_ref.data_half(), M * N),
                "Results don't match cuBLAS");

    printf("PASS\n");
    return true;
}

bool test_gemm_alpha_beta() {
    printf("test_gemm_alpha_beta: ");

    const int M = 128, N = 128, K = 128;
    const float alpha = 0.5f;
    const float beta = 0.25f;

    Tensor A = rand_tensor({M, K}, DType::FP16);
    Tensor B = rand_tensor({K, N}, DType::FP16);
    Tensor C_init = rand_tensor({M, N}, DType::FP16);
    Tensor C_ours = C_init.clone();

    gemm_fp16(A.data_half(), B.data_half(), C_ours.data_half(),
              M, N, K, alpha, beta);

    cudaDeviceSynchronize();

    // Manual verification would go here
    // For now, just check it doesn't crash

    printf("PASS (smoke test)\n");
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=============================================================\n");
    printf("Clove Compute GEMM Tests\n");
    printf("=============================================================\n\n");

    int passed = 0;
    int total = 0;

    #define RUN_TEST(test)  \
        total++;            \
        if (test()) passed++;

    RUN_TEST(test_gemm_small);
    RUN_TEST(test_gemm_medium);
    RUN_TEST(test_gemm_large);
    RUN_TEST(test_gemm_nonsquare);
    RUN_TEST(test_gemm_decode);
    RUN_TEST(test_gemm_alpha_beta);

    printf("\n=============================================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("=============================================================\n");

    return (passed == total) ? 0 : 1;
}
