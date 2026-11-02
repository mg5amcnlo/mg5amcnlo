#include "../kernels/kernels.hpp"
#include "device.hpp"
#include "tensor.cuh"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

namespace {
// Wait for work still queued on the stream a tensor is ordered on.
void wait_for_tensor(const Tensor& tensor) {
    if (auto stream = tensor.stream()) {
        check_error(gpuStreamSynchronize(reinterpret_cast<gpuStream_t>(*stream)));
    }
}

// calls func with the kernel policy of a float dtype
template <typename Func>
void dispatch_float(DataType dtype, const char* op, Func&& func) {
    switch (dtype) {
    case DataType::dt_float64:
        func.template operator()<GpuTypes>();
        break;
    case DataType::dt_float32:
        func.template operator()<GpuTypes32>();
        break;
    default:
        throw std::runtime_error(std::string("invalid dtype in ") + op);
    }
}

// sums a contiguous (batch_size, n) source over the batch dimension
template <typename F>
__global__ void
kernel_batch_sum(std::size_t batch_size, std::size_t n, const F* source, F* output) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    F sum = 0;
    for (std::size_t j = 0; j < batch_size; ++j) {
        sum += source[j * n + i];
    }
    output[i] = sum;
}

template <typename F>
void launch_batch_sum(const Tensor& source, Tensor& sum, gpuStream_t stream) {
    std::size_t n = sum.shape().product();
    launch_kernel(
        kernel_batch_sum<F>,
        n,
        stream,
        source.size(0),
        n,
        static_cast<const F*>(source.data()),
        static_cast<F*>(sum.data())
    );
}
} // namespace

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

void GpuDevice::free_on_stream(void* ptr, void* stream) const {
    activate();
    check_error(gpuFreeAsync(ptr, static_cast<gpuStream_t>(stream)));
}

void* GpuDevice::default_free_stream() const {
    return static_cast<void*>(gpuStreamPerThread);
}

void GpuDevice::order_streams(void* from, void* to) const {
    activate();
    static thread_local std::vector<gpuEvent_t> events;
    if (events.size() <= static_cast<std::size_t>(_index)) {
        events.resize(_index + 1);
    }
    gpuEvent_t& event = events.at(_index);
    if (!event) {
        check_error(gpuEventCreate(&event));
    }
    check_error(gpuEventRecord(event, static_cast<gpuStream_t>(from)));
    check_error(gpuStreamWaitEvent(static_cast<gpuStream_t>(to), event));
}

void GpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    activate();
    check_error(gpuMemcpy(to, from, size, gpuMemcpyDefault));
}

void GpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    activate();
    wait_for_tensor(source);
    wait_for_tensor(target);
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_copy(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_zero(Tensor& tensor) const {
    activate();
    wait_for_tensor(tensor);
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_zero(tensor);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    activate();
    wait_for_tensor(source);
    wait_for_tensor(target);
    AsyncGpuDevice(*this, gpuStreamPerThread, 0).tensor_add(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    activate();
    wait_for_tensor(source);
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
    double bias_corr2_sqrt,
    double weight_decay
) const {
    activate();
    wait_for_tensor(gradient);
    wait_for_tensor(parameter);
    wait_for_tensor(exp_avg);
    wait_for_tensor(exp_avg_sq);
    AsyncGpuDevice device(*this, gpuStreamPerThread, 0);
    dispatch_float(parameter.dtype(), "adam_step", [&]<typename T>() {
        using F = typename T::FVal;
        tensor_foreach_dynamic<kernel_adam_step<T>, 1, 3>(
            {&gradient},
            {&parameter, &exp_avg, &exp_avg_sq},
            gradient.size(0),
            device,
            static_cast<F>(step_size),
            static_cast<F>(beta1),
            static_cast<F>(beta2),
            static_cast<F>(eps),
            static_cast<F>(bias_corr2_sqrt),
            static_cast<F>(weight_decay)
        );
    });
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

MemPool::MemPool(
    const GpuDevice& device,
    const std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>>&
        cached_sizes_and_tensors,
    gpuStream_t stream
) :
    _device(device), _stream(stream) {
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
        std::size_t word_count = (size + 7) / 8;
        if (parent_tensor) {
            pool.parent_tensor = parent_tensor;
            pool.capacity = parent_tensor.byte_size();
        } else {
            pool.parent_tensor =
                Tensor(DataType::dt_float64, {word_count}, async_device);
            pool.capacity = word_count * 8;
        }
        pool.needed_size = word_count * 8;
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
                    _device.free_on_stream(ptr, _stream);
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
    // if (_mem_pool && hint != AllocHint::normal && size <= 4 * 1024 * 1024) {
    if (_mem_pool && hint != AllocHint::normal && hint != AllocHint::output &&
        size <= 4 * 1024 * 1024) {
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
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in copy");
    }
    if (source.dtype() == DataType::dt_int32) {
        tensor_foreach_dynamic<kernel_copy_int<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
        return;
    }
    dispatch_float(source.dtype(), "copy", [&]<typename T>() {
        tensor_foreach_dynamic<kernel_copy<T>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    });
}

void AsyncGpuDevice::tensor_zero(Tensor& tensor) const {
    if (tensor.dtype() == DataType::batch_sizes) {
        throw std::runtime_error("invalid dtype in zero");
    }
    if (tensor.is_contiguous()) {
        check_error(gpuMemsetAsync(tensor.data(), 0, tensor.byte_size(), _stream));
        return;
    }
    if (tensor.dtype() == DataType::dt_int32) {
        tensor_foreach_dynamic<kernel_zero_int<GpuTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
        return;
    }
    dispatch_float(tensor.dtype(), "zero", [&]<typename T>() {
        tensor_foreach_dynamic<kernel_zero<T>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    });
}

void AsyncGpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in add");
    }
    if (target.size(0) == 1 && source.size(0) > 1) {
        // a broadcast target (e.g. the gradient of a global) accumulates the
        // whole source batch, as on the CPU
        auto source_contig = source.contiguous(*this, AllocHint::temporary);
        Tensor sum(target.dtype(), target.shape(), *this, AllocHint::temporary);
        dispatch_float(source.dtype(), "add", [&]<typename T>() {
            launch_batch_sum<typename T::FVal>(source_contig, sum, _stream);
        });
        tensor_add(sum, target);
        sum.reset(*this);
        source_contig.reset(*this);
        return;
    }
    dispatch_float(source.dtype(), "add", [&]<typename T>() {
        tensor_foreach_dynamic<kernel_add_inplace<T>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    });
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
