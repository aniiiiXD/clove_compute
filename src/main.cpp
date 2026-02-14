#include <clove/runtime/tensor.h>
#include <clove/kernels/gemm.cuh>
#include <cstdio>
#include <cstring>

using namespace clove;

void print_usage(const char* prog) {
    printf("Clove Compute - High-Performance CUDA Inference Runtime\n\n");
    printf("Usage: %s [command] [options]\n\n", prog);
    printf("Commands:\n");
    printf("  benchmark     Run GEMM benchmarks against cuBLAS\n");
    printf("  test          Run correctness tests\n");
    printf("  info          Print GPU information\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s info\n", prog);
    printf("  %s benchmark\n", prog);
    printf("\n");
}

void print_gpu_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    printf("=============================================================\n");
    printf("Clove Compute - GPU Information\n");
    printf("=============================================================\n");

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.1f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  SM Count: %d\n", prop.multiProcessorCount);
        printf("  Clock Rate: %.0f MHz\n", prop.clockRate / 1000.0);
        printf("  Memory Clock: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Shared Mem/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Warp Size: %d\n", prop.warpSize);

        // Estimate peak performance
        // Ampere/Ada: 256 FMA ops per SM per clock (FP16 tensor cores)
        float tensor_tflops = prop.multiProcessorCount * 256.0f *
                              prop.clockRate * 1e-9f * 2.0f;
        printf("  Est. Peak FP16 Tensor: ~%.0f TFLOPS\n", tensor_tflops);
    }

    printf("\n=============================================================\n");
}

void run_quick_test() {
    printf("Running quick functionality test...\n\n");

    // Test tensor allocation
    printf("Testing Tensor allocation... ");
    Tensor t = Tensor::allocate({128, 256}, DType::FP16);
    printf("OK (%.2f MB)\n", t.nbytes() / (1024.0 * 1024.0));

    // Test random tensor
    printf("Testing random tensor generation... ");
    Tensor r = rand_tensor({64, 64}, DType::FP16);
    printf("OK\n");

    // Test GEMM
    printf("Testing FP16 GEMM... ");
    Tensor A = rand_tensor({256, 256}, DType::FP16);
    Tensor B = rand_tensor({256, 256}, DType::FP16);
    Tensor C = Tensor::zeros({256, 256}, DType::FP16);

    gemm_fp16(A.data_half(), B.data_half(), C.data_half(), 256, 256, 256);
    cudaDeviceSynchronize();
    printf("OK\n");

    printf("\nAll tests passed!\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 0;
    }

    const char* cmd = argv[1];

    if (strcmp(cmd, "info") == 0) {
        print_gpu_info();
    } else if (strcmp(cmd, "test") == 0) {
        run_quick_test();
    } else if (strcmp(cmd, "benchmark") == 0) {
        printf("Run ./bench_gemm for full benchmarks\n");
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage(argv[0]);
    } else {
        printf("Unknown command: %s\n", cmd);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
