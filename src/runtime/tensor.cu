#include <clove/runtime/tensor.h>
#include <curand_kernel.h>
#include <random>
#include <cstring>

namespace clove {

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data)
    , shape(std::move(other.shape))
    , strides(std::move(other.strides))
    , dtype(other.dtype)
    , owns_memory(other.owns_memory)
    , is_device(other.is_device) {
    other.data = nullptr;
    other.owns_memory = false;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        dtype = other.dtype;
        owns_memory = other.owns_memory;
        is_device = other.is_device;
        other.data = nullptr;
        other.owns_memory = false;
    }
    return *this;
}

// Destructor
Tensor::~Tensor() {
    free();
}

// Compute strides from shape (row-major)
void Tensor::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;

    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Number of elements
int64_t Tensor::numel() const {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// Number of bytes
int64_t Tensor::nbytes() const {
    if (dtype == DType::INT4) {
        return (numel() + 1) / 2;  // 2 values per byte
    }
    return numel() * dtype_size(dtype);
}

// Allocate tensor
Tensor Tensor::allocate(std::vector<int64_t> shape, DType dtype, bool device) {
    Tensor t;
    t.shape = std::move(shape);
    t.dtype = dtype;
    t.is_device = device;
    t.owns_memory = true;
    t.compute_strides();

    size_t bytes = t.nbytes();
    if (bytes > 0) {
        if (device) {
            CLOVE_CHECK_CUDA(cudaMalloc(&t.data, bytes));
        } else {
            t.data = malloc(bytes);
        }
    }
    return t;
}

// Wrap existing pointer
Tensor Tensor::from_ptr(void* ptr, std::vector<int64_t> shape, DType dtype, bool device) {
    Tensor t;
    t.data = ptr;
    t.shape = std::move(shape);
    t.dtype = dtype;
    t.is_device = device;
    t.owns_memory = false;
    t.compute_strides();
    return t;
}

// Allocate and zero-initialize
Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, bool device) {
    Tensor t = allocate(shape, dtype, device);
    if (t.nbytes() > 0) {
        if (device) {
            CLOVE_CHECK_CUDA(cudaMemset(t.data, 0, t.nbytes()));
        } else {
            memset(t.data, 0, t.nbytes());
        }
    }
    return t;
}

// Free memory
void Tensor::free() {
    if (data && owns_memory) {
        if (is_device) {
            cudaFree(data);
        } else {
            ::free(data);
        }
    }
    data = nullptr;
    owns_memory = false;
}

// Clone tensor (deep copy)
Tensor Tensor::clone() const {
    Tensor t = allocate(shape, dtype, is_device);
    if (nbytes() > 0) {
        if (is_device) {
            CLOVE_CHECK_CUDA(cudaMemcpy(t.data, data, nbytes(), cudaMemcpyDeviceToDevice));
        } else {
            memcpy(t.data, data, nbytes());
        }
    }
    return t;
}

// Copy from another tensor
void Tensor::copy_from(const Tensor& other, cudaStream_t stream) {
    assert(numel() == other.numel() && dtype == other.dtype);

    cudaMemcpyKind kind;
    if (is_device && other.is_device) {
        kind = cudaMemcpyDeviceToDevice;
    } else if (is_device && !other.is_device) {
        kind = cudaMemcpyHostToDevice;
    } else if (!is_device && other.is_device) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyHostToHost;
    }

    CLOVE_CHECK_CUDA(cudaMemcpyAsync(data, other.data, nbytes(), kind, stream));
}

// Copy to host buffer
void Tensor::copy_to_host(void* dst, cudaStream_t stream) const {
    if (is_device) {
        CLOVE_CHECK_CUDA(cudaMemcpyAsync(dst, data, nbytes(), cudaMemcpyDeviceToHost, stream));
    } else {
        memcpy(dst, data, nbytes());
    }
}

// Copy from host buffer
void Tensor::copy_from_host(const void* src, cudaStream_t stream) {
    if (is_device) {
        CLOVE_CHECK_CUDA(cudaMemcpyAsync(data, src, nbytes(), cudaMemcpyHostToDevice, stream));
    } else {
        memcpy(data, src, nbytes());
    }
}

// Transfer to device
void Tensor::to_device(cudaStream_t stream) {
    if (is_device) return;

    void* device_data;
    CLOVE_CHECK_CUDA(cudaMalloc(&device_data, nbytes()));
    CLOVE_CHECK_CUDA(cudaMemcpyAsync(device_data, data, nbytes(), cudaMemcpyHostToDevice, stream));

    if (owns_memory) {
        ::free(data);
    }
    data = device_data;
    is_device = true;
    owns_memory = true;
}

// Transfer to host
void Tensor::to_host(cudaStream_t stream) {
    if (!is_device) return;

    void* host_data = malloc(nbytes());
    CLOVE_CHECK_CUDA(cudaMemcpyAsync(host_data, data, nbytes(), cudaMemcpyDeviceToHost, stream));
    CLOVE_CHECK_CUDA(cudaStreamSynchronize(stream));

    if (owns_memory) {
        cudaFree(data);
    }
    data = host_data;
    is_device = false;
    owns_memory = true;
}

// View with different shape (must have same numel)
Tensor Tensor::view(std::vector<int64_t> new_shape) const {
    // Handle -1 dimension
    int64_t total = numel();
    int neg_idx = -1;
    int64_t known = 1;
    for (int i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            assert(neg_idx == -1 && "Only one -1 allowed in shape");
            neg_idx = i;
        } else {
            known *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        new_shape[neg_idx] = total / known;
    }

    assert(numel() == [&](){
        int64_t n = 1;
        for (auto s : new_shape) n *= s;
        return n;
    }());

    Tensor t;
    t.data = data;
    t.shape = std::move(new_shape);
    t.dtype = dtype;
    t.is_device = is_device;
    t.owns_memory = false;
    t.compute_strides();
    return t;
}

// Slice along dimension
Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    if (dim < 0) dim += ndim();
    assert(dim >= 0 && dim < ndim());
    assert(start >= 0 && end <= shape[dim] && start < end);

    Tensor t;
    t.shape = shape;
    t.shape[dim] = end - start;
    t.dtype = dtype;
    t.is_device = is_device;
    t.owns_memory = false;
    t.compute_strides();

    // Calculate offset
    size_t offset = start * strides[dim] * dtype_size(dtype);
    t.data = static_cast<char*>(data) + offset;

    return t;
}

// Remove dimension of size 1
Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += ndim();
    assert(shape[dim] == 1);

    std::vector<int64_t> new_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i != dim) new_shape.push_back(shape[i]);
    }
    return view(new_shape);
}

// Add dimension of size 1
Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;

    std::vector<int64_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    return view(new_shape);
}

// Debug print
void Tensor::print_info(const char* name) const {
    printf("%s: [", name ? name : "Tensor");
    for (int i = 0; i < ndim(); ++i) {
        printf("%ld%s", shape[i], i < ndim() - 1 ? ", " : "");
    }
    printf("] dtype=");
    switch (dtype) {
        case DType::FP32: printf("fp32"); break;
        case DType::FP16: printf("fp16"); break;
        case DType::BF16: printf("bf16"); break;
        case DType::INT8: printf("int8"); break;
        case DType::INT4: printf("int4"); break;
    }
    printf(" %s %.2f MB\n",
           is_device ? "GPU" : "CPU",
           nbytes() / (1024.0 * 1024.0));
}

// Random tensor generation kernel
__global__ void rand_fp16_kernel(half* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float val = curand_uniform(&state) * 2.0f - 1.0f;  // [-1, 1]
        data[idx] = __float2half(val * 0.1f);  // Scale down for stability
    }
}

__global__ void rand_fp32_kernel(float* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float val = curand_uniform(&state) * 2.0f - 1.0f;
        data[idx] = val * 0.1f;
    }
}

Tensor rand_tensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream) {
    Tensor t = Tensor::allocate(shape, dtype, true);

    std::random_device rd;
    unsigned long long seed = rd();

    int n = static_cast<int>(t.numel());
    int block = 256;
    int grid = cdiv(n, block);

    switch (dtype) {
        case DType::FP16:
            rand_fp16_kernel<<<grid, block, 0, stream>>>(t.data_half(), n, seed);
            break;
        case DType::FP32:
            rand_fp32_kernel<<<grid, block, 0, stream>>>(t.data_float(), n, seed);
            break;
        default:
            assert(false && "rand_tensor: unsupported dtype");
    }
    CLOVE_CHECK_LAST();

    return t;
}

}  // namespace clove
