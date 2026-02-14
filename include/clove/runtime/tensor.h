#pragma once

#include <clove/common.cuh>
#include <vector>
#include <memory>
#include <cassert>

namespace clove {

class Tensor {
public:
    void* data = nullptr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DType dtype = DType::FP16;
    bool owns_memory = false;
    bool is_device = true;

    // Default constructor
    Tensor() = default;

    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // No copying (explicit clone required)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Destructor
    ~Tensor();

    // Factory methods
    static Tensor allocate(std::vector<int64_t> shape, DType dtype, bool device = true);
    static Tensor from_ptr(void* ptr, std::vector<int64_t> shape, DType dtype, bool device = true);
    static Tensor zeros(std::vector<int64_t> shape, DType dtype, bool device = true);

    // Size calculations
    int64_t numel() const;
    int64_t nbytes() const;
    int ndim() const { return static_cast<int>(shape.size()); }

    // Accessors
    int64_t dim(int i) const {
        if (i < 0) i += ndim();
        return shape[i];
    }

    // Memory operations
    void free();
    Tensor clone() const;
    void copy_from(const Tensor& other, cudaStream_t stream = 0);
    void to_device(cudaStream_t stream = 0);
    void to_host(cudaStream_t stream = 0);

    // Copy to external buffer
    void copy_to_host(void* dst, cudaStream_t stream = 0) const;
    void copy_from_host(const void* src, cudaStream_t stream = 0);

    // View operations (no copy, no ownership transfer)
    Tensor view(std::vector<int64_t> new_shape) const;
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor squeeze(int dim) const;
    Tensor unsqueeze(int dim) const;

    // Typed accessors
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data); }

    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data); }

    half* data_half() { return data_ptr<half>(); }
    float* data_float() { return data_ptr<float>(); }
    int8_t* data_int8() { return data_ptr<int8_t>(); }
    uint8_t* data_uint8() { return data_ptr<uint8_t>(); }

    // Debug
    void print_info(const char* name = nullptr) const;

private:
    void compute_strides();
};

// Utility: create tensor filled with random values (for testing)
Tensor rand_tensor(std::vector<int64_t> shape, DType dtype, cudaStream_t stream = 0);

}  // namespace clove
