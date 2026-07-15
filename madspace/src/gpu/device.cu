#include "../kernels/kernels.hpp"
#include "device.hpp"
#include "tensor.cuh"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

std::pair<void*, Tensor> GpuDevice::allocate(std::size_t size, AllocHint hint) const {
    activate();
    void* ptr;
    check_error(gpuMalloc(&ptr, size));
    if (needs_zero_init(hint)) {
        check_error(gpuMemset(ptr, 0, size));
    }
    return {ptr, Tensor()};
}

void GpuDevice::free(void* ptr) const {
    activate();
    check_error(gpuFree(ptr));
}

void GpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    activate();
    check_error(gpuMemcpy(to, from, size, gpuMemcpyDefault));
}

void GpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_copy(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_zero(Tensor& tensor) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_zero(tensor);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    activate();
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_add(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    activate();
    check_error(
        gpuMemcpy(target.data(), source.data(), source.byte_size(), gpuMemcpyDefault)
    );
}

void GpuDevice::adam_step(
    const Tensor& gradient,
    Tensor& parameter,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double step_size,
    double beta1,
    double beta2,
    double eps,
    double bias_corr2_sqrt
) const {
    activate();
    AsyncGpuDevice device(*this, gpuStreamPerThread, 0);
    tensor_foreach_dynamic<kernel_adam_step<GpuTypes>, 1, 3>(
        {&gradient},
        {&parameter, &exp_avg, &exp_avg_sq},
        gradient.size(0),
        device,
        step_size,
        beta1,
        beta2,
        eps,
        bias_corr2_sqrt
    );
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

MemPool::MemPool(
    const GpuDevice& device,
    const std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>>&
        cached_sizes_and_tensors,
    gpuStream_t stream
) :
    _device(device) {
    std::size_t pool_count = 0;
    for (auto& [pool_index, size, parent_tensor, zero_init] :
         cached_sizes_and_tensors) {
        if (pool_index >= pool_count) {
            pool_count = pool_index + 1;
        }
    }
    _pools.resize(pool_count);

    AsyncGpuDevice async_device(device, stream);
    for (auto& [pool_index, size, parent_tensor, zero_init] :
         cached_sizes_and_tensors) {
        auto& pool = _pools.at(pool_index);
        if (parent_tensor) {
            pool.parent_tensor = parent_tensor;
            pool.capacity = parent_tensor.byte_size();
            pool.needed_size = parent_tensor.byte_size();
        } else {
            std::size_t word_count = (size + 7) / 8;
            pool.parent_tensor = Tensor(DataType::dt_float, {word_count}, async_device);
            pool.capacity = word_count * 8;
            pool.needed_size = word_count * 8;
        }
        if (zero_init) {
            check_error(gpuMemsetAsync(
                pool.parent_tensor.data(), 0, pool.parent_tensor.byte_size(), stream
            ));
        }
    }
}

MemPool::~MemPool() {
    for (PoolItem& pool : _pools) {
        for (auto& stream_free_pointers : pool.free_pointers) {
            for (auto& [size, item] : stream_free_pointers) {
                auto& [ptr, parent] = item;
                if (!parent) {
                    check_error(gpuFree(ptr));
                }
            }
        }
    }
}

std::vector<std::pair<std::size_t, Tensor>> MemPool::reset(gpuStream_t stream) {
    std::vector<std::pair<std::size_t, Tensor>> parent_tensors;
    for (std::size_t pool_index = 0; PoolItem& pool : _pools) {
        for (auto& stream_free_pointers : pool.free_pointers) {
            for (auto& [size, item] : stream_free_pointers) {
                auto& [ptr, parent] = item;
                if (!parent) {
                    check_error(gpuFreeAsync(ptr, stream));
                }
            }
        }
        if (pool.parent_tensor) {
            parent_tensors.push_back({pool_index, pool.parent_tensor});
        }
        ++pool_index;
    }
    _pools.clear();
    return parent_tensors;
}

std::pair<void*, Tensor> MemPool::allocate(
    std::size_t pool_index,
    std::size_t size,
    gpuStream_t stream,
    std::size_t stream_index,
    bool zero_init
) {
    if (pool_index >= _pools.size()) {
        _pools.resize(pool_index + 1);
    }
    PoolItem& pool = _pools.at(pool_index);
    if (stream_index >= pool.free_pointers.size()) {
        pool.free_pointers.resize(stream_index + 1);
    }
    auto& free_pointers = pool.free_pointers.at(stream_index);
    if (auto search = free_pointers.find(size); search != free_pointers.end()) {
        std::pair<void*, Tensor> ret = search->second;
        _allocs[ret.first] = {
            .pool_index = pool_index,
            .size = size,
            .parent_tensor = ret.second,
        };
        free_pointers.erase(search);
        if (zero_init) {
            check_error(gpuMemsetAsync(ret.first, 0, size, stream));
        }
        return ret;
    } else if (pool.parent_tensor && pool.capacity - pool.size >= size) {
        void* ptr = &static_cast<uint8_t*>(pool.parent_tensor.data())[pool.size];
        pool.size = (pool.size + size + 7) / 8 * 8;
        _allocs[ptr] = {
            .pool_index = pool_index,
            .size = size,
            .parent_tensor = pool.parent_tensor,
        };
        return {ptr, pool.parent_tensor};
    } else {
        void* ptr;
        check_error(gpuMallocAsync(&ptr, size, stream));
        if (zero_init) {
            check_error(gpuMemsetAsync(ptr, 0, size, stream));
        }
        _allocs[ptr] = {
            .pool_index = pool_index,
            .size = size,
            .parent_tensor = Tensor(),
        };
        pool.needed_size += (size + 7) / 8 * 8;
        return {ptr, Tensor()};
    }
}

bool MemPool::free(void* ptr, std::size_t stream_index) {
    auto search = _allocs.find(ptr);
    if (search == _allocs.end()) {
        return false;
    }
    auto& alloc = search->second;
    auto& pool = _pools.at(alloc.pool_index);
    if (stream_index >= pool.free_pointers.size()) {
        pool.free_pointers.resize(stream_index + 1);
    }
    auto& free_pointers = pool.free_pointers.at(stream_index);
    free_pointers.emplace(
        alloc.size, std::pair<void*, Tensor>{ptr, alloc.parent_tensor}
    );
    _allocs.erase(search);
    return true;
}

std::vector<std::pair<std::size_t, std::size_t>> MemPool::total_sizes() const {
    std::vector<std::pair<std::size_t, std::size_t>> ret;
    ret.reserve(_pools.size());
    for (std::size_t index = 0; auto& pool : _pools) {
        if (pool.needed_size > 0) {
            ret.push_back({index, pool.needed_size});
        }
        ++index;
    }
    return ret;
}

std::pair<void*, Tensor>
AsyncGpuDevice::allocate(std::size_t size, AllocHint hint) const {
    if (_mem_pool && hint != AllocHint::normal && size <= 4 * 1024 * 1024) {
        return _mem_pool->allocate(
            static_cast<std::size_t>(hint) - 1,
            size,
            _stream,
            _stream_index,
            needs_zero_init(hint)
        );
    } else {
        void* ptr;
        check_error(gpuMallocAsync(&ptr, size, _stream));
        if (needs_zero_init(hint)) {
            check_error(gpuMemsetAsync(ptr, 0, size, _stream));
        }
        return {ptr, Tensor()};
    }
}

void AsyncGpuDevice::free(void* ptr) const {
    if (!_mem_pool || !_mem_pool->free(ptr, _stream_index)) {
        check_error(gpuFreeAsync(ptr, _stream));
    }
}

void AsyncGpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(gpuMemcpyAsync(to, from, size, gpuMemcpyDefault, _stream));
}

void AsyncGpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    if (source.dtype() == DataType::dt_float && target.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_copy<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else if (source.dtype() == DataType::dt_int &&
               target.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_copy_int<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in copy");
    }
}

void AsyncGpuDevice::tensor_zero(Tensor& tensor) const {
    if (tensor.dtype() == DataType::dt_float) {
        if (tensor.is_contiguous()) {
            check_error(gpuMemsetAsync(tensor.data(), 0, tensor.byte_size(), _stream));
        } else {
            tensor_foreach_dynamic<kernel_zero<GpuTypes>, 1, 1>(
                {&tensor}, {&tensor}, tensor.size(0), *this
            );
        }
    } else if (tensor.dtype() == DataType::dt_int) {
        if (tensor.is_contiguous()) {
            check_error(gpuMemsetAsync(tensor.data(), 0, tensor.byte_size(), _stream));
        } else {
            tensor_foreach_dynamic<kernel_zero_int<GpuTypes>, 1, 1>(
                {&tensor}, {&tensor}, tensor.size(0), *this
            );
        }
    } else {
        throw std::runtime_error("invalid dtype in zero");
    }
}

void AsyncGpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<GpuTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), *this
    );
}

void AsyncGpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(gpuMemcpyAsync(
        target.data(), source.data(), source.byte_size(), gpuMemcpyDefault, _stream
    ));
}

extern "C" int device_count() {
    int device_count;
    check_error(gpuGetDeviceCount(&device_count));
    return device_count;
}

extern "C" DevicePtr get_device(int index) { return &GpuDevice::instance(index); }
