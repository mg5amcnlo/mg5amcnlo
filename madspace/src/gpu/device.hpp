#pragma once

#include "gpu_abstraction.cuh"
#include "madspace/driver/tensor.hpp"

#include <format>

namespace madspace {
namespace gpu {

inline void check_error(gpublasStatus_t status) {
    if (status != GPUBLAS_STATUS_SUCCESS) {
        const char* error_str = gpublasGetStatusString(status);
        throw std::runtime_error(std::format("BLAS error: {}", error_str));
    }
}

inline void check_error(gpurandStatus_t status) {
    if (status != GPURAND_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::format("RAND error: error code {}", static_cast<int>(status))
        );
    }
}

inline void check_error(gpuError_t error) {
    if (error != gpuSuccess) {
        const char* error_str = gpuGetErrorString(error);
        throw std::runtime_error(std::format("GPU error: {}", error_str));
    }
}

inline void check_error() { check_error(gpuGetLastError()); }

class GpuDevice : public Device {
public:
#ifdef __CUDACC__
    static constexpr DeviceType gpu_device_type = DeviceType::cuda;
#else
    static constexpr DeviceType gpu_device_type = DeviceType::hip;
#endif
    virtual std::pair<void*, Tensor>
    allocate(std::size_t size, AllocHint hint) const override;
    void free(void* ptr) const override;
    void memcpy(void* to, void* from, std::size_t size) const override;

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;
    void tensor_cpu(const Tensor& source, Tensor& target) const override;
    DevicePtr device_ptr() const override { return this; }
    DeviceType device_type() const override { return gpu_device_type; }
    void activate() const override { check_error(gpuSetDevice(_index)); }
    void adam_step(
        const Tensor& gradient,
        Tensor& parameter,
        Tensor& exp_avg,
        Tensor& exp_avg_sq,
        double step_size,
        double beta1,
        double beta2,
        double eps,
        double bias_corr2_sqrt
    ) const override;

    GpuDevice(const GpuDevice&) = delete;
    GpuDevice& operator=(GpuDevice&) = delete;

    static const GpuDevice& instance(int index) {
        static std::vector<GpuDevice*> devices = [] {
            int device_count;
            check_error(gpuGetDeviceCount(&device_count));
            std::vector<GpuDevice*> ret;
            ret.reserve(device_count);
            for (int i = 0; i < device_count; ++i) {
                ret.push_back(new GpuDevice(i));
            }
            return ret;
        }();
        return *devices.at(index);
    }

private:
    GpuDevice(int index) : _index(index) {}

    int _index;
};

class MemPool {
public:
    MemPool(
        const GpuDevice& device,
        const std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>>&
            cached_sizes_and_tensors,
        gpuStream_t stream
    );
    ~MemPool();
    std::vector<std::pair<std::size_t, Tensor>> reset(gpuStream_t stream);
    std::pair<void*, Tensor> allocate(
        std::size_t pool_index,
        std::size_t size,
        gpuStream_t stream,
        std::size_t stream_index,
        bool zero_init
    );
    bool free(void* ptr, std::size_t stream_index);
    std::vector<std::pair<std::size_t, std::size_t>> total_sizes() const;

private:
    struct PoolItem {
        Tensor parent_tensor;
        std::size_t size = 0;
        std::size_t capacity = 0;
        std::size_t needed_size = 0;
        std::vector<std::unordered_multimap<std::size_t, std::pair<void*, Tensor>>>
            free_pointers;
    };
    struct AllocItem {
        std::size_t pool_index;
        std::size_t size;
        Tensor parent_tensor;
    };
    std::vector<PoolItem> _pools;
    std::unordered_map<void*, AllocItem> _allocs;
    const GpuDevice& _device;
};

class AsyncGpuDevice {
public:
    AsyncGpuDevice(
        const GpuDevice& device,
        gpuStream_t stream,
        std::size_t stream_index = 0,
        MemPool* mem_pool = nullptr
    ) :
        _device(device),
        _stream(stream),
        _stream_index(stream_index),
        _mem_pool(mem_pool) {}

    std::pair<void*, Tensor> allocate(std::size_t size, AllocHint hint) const;
    void free(void* ptr) const;
    void memcpy(void* to, void* from, std::size_t size) const;

    void tensor_copy(const Tensor& source, Tensor& target) const;
    void tensor_zero(Tensor& tensor) const;
    void tensor_add(const Tensor& source, Tensor& target) const;
    void tensor_cpu(const Tensor& source, Tensor& target) const;
    DevicePtr device_ptr() const { return &_device; }
    void sync_barrier() const {};
    gpuStream_t stream() const { return _stream; }

private:
    const GpuDevice& _device;
    gpuStream_t _stream;
    std::size_t _stream_index;
    MemPool* _mem_pool;
};

extern "C" int device_count();
extern "C" DevicePtr get_device(int index);

} // namespace gpu
} // namespace madspace
