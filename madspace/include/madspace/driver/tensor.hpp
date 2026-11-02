#pragma once

#include "madspace/compgraphs/type.hpp"
#include "madspace/util.hpp"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace madspace {

using SizeVec = std::vector<std::size_t>;

/**
 * Fixed-capacity vector of up to @ref max_size dimension sizes.
 *
 * The shape and stride type used throughout @ref Tensor; a small
 * stack-allocated alternative to `std::vector<std::size_t>` since every
 * tensor has at most @ref max_size dimensions.
 */
class Sizes {
public:
    /// Maximum number of dimensions a `Sizes` can hold.
    static constexpr std::size_t max_size = 4;

    /// Empty (0-dimensional) sizes.
    Sizes() : _size(0) {};
    /// `size` dimensions, all zero.
    explicit Sizes(std::size_t size) : _size(size) {
        if (size > max_size) {
            throw std::out_of_range("maximum dimension exceeded");
        }
        std::fill(begin(), end(), 0);
    };
    /// `size` dimensions, each set to `value`.
    Sizes(std::size_t size, std::size_t value) : _size(size) {
        if (size > max_size) {
            throw std::out_of_range("maximum dimension exceeded");
        }
        std::fill(begin(), end(), value);
    };
    /// One dimension per entry of `values`.
    Sizes(std::initializer_list<std::size_t> values) : _size(values.size()) {
        if (values.size() > max_size) {
            throw std::out_of_range("maximum dimension exceeded");
        }
        std::copy(values.begin(), values.end(), begin());
    }
    /// One dimension per entry of `values`.
    Sizes(const SizeVec& values) : _size(values.size()) {
        if (values.size() > max_size) {
            throw std::out_of_range("maximum dimension exceeded");
        }
        std::copy(values.begin(), values.end(), begin());
    }
    /// The dimension size at `index`.
    std::size_t& operator[](std::size_t index) { return _values[index]; }
    /// The dimension size at `index`.
    const std::size_t& operator[](std::size_t index) const { return _values[index]; }
    /// Number of dimensions.
    std::size_t size() const { return _size; }
    /// Iterator over the dimension sizes.
    std::size_t* begin() { return &_values[0]; }
    /// Iterator past the last dimension size.
    std::size_t* end() { return &_values[_size]; }
    /// Iterator over the dimension sizes.
    const std::size_t* begin() const { return &_values[0]; }
    /// Iterator past the last dimension size.
    const std::size_t* end() const { return &_values[_size]; }
    /// Append one more dimension.
    void push_back(std::size_t item) {
        if (_size >= max_size) {
            throw std::out_of_range("maximum dimension exceeded");
        }
        _values[_size] = item;
        ++_size;
    }
    /// Raw pointer to the dimension sizes.
    std::size_t* data() { return &_values[0]; }
    /// Raw pointer to the dimension sizes.
    const std::size_t* data() const { return &_values[0]; }
    /// Size of the last dimension.
    std::size_t& back() { return _values[_size - 1]; }
    /// Size of the last dimension.
    const std::size_t& back() const { return _values[_size - 1]; }
    /// Product of all dimension sizes, i.e. the total element count.
    std::size_t product() const {
        std::size_t size = 1;
        for (std::size_t dim_size : *this) {
            size *= dim_size;
        }
        return size;
    }

private:
    // zero beyond size(): a view with more dimensions than its tensor (see
    // Tensor::view) must not read stale strides and shapes
    std::size_t _values[max_size] = {};
    std::size_t _size;
};

inline bool operator==(const Sizes& a, const Sizes& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}
inline bool operator!=(const Sizes& a, const Sizes& b) { return !(a == b); }

/// A @ref Tensor's raw data pointer, stride and shape, passed by value to a
/// compute kernel (CPU or GPU).
template <ScalarType T, int _dim>
struct PackedTensorView {
    using DType = T;
    static const int dim = _dim;
    T* data;
    Sizes stride;
    Sizes shape;
    /// Whether the data is a literal stored padded to the CPU SIMD vector size
    /// (see @ref Tensor::is_simd_padded).
    bool simd_padded = false;
};

/**
 * Typed, dimension-checked view onto a @ref Tensor's data.
 *
 * Indexing with `operator[]` or `get` peels off leading dimensions until
 * `_dim` reaches 0, at which point the view converts to and from a single
 * element of type `T`.
 */
template <ScalarType T, int _dim>
class TensorView {
public:
    using DType = T;
    static const int dim = _dim;
    static constexpr bool is_scalar_view = false;

    TensorView(T* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    TensorView(PackedTensorView<T, _dim>& packed_view) :
        _data(packed_view.data),
        _stride(packed_view.stride.data()),
        _shape(packed_view.shape.data()) {}

    TensorView(T& value) : _data(&value), _stride(nullptr), _shape(nullptr) {}

    const TensorView<T, _dim - 1> operator[](std::size_t index) const
        requires(_dim != 0)
    {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    TensorView<T, _dim - 1> operator[](std::size_t index)
        requires(_dim != 0)
    {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    template <typename... I>
    const TensorView<T, _dim - sizeof...(I)> get(I... index) const
        requires(_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    template <typename... I>
    TensorView<T, _dim - sizeof...(I)> get(I... index)
        requires(_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    operator T() const
        requires(_dim == 0)
    {
        return *_data;
    }

    T operator=(T value)
        requires(_dim == 0)
    {
        *_data = value;
        return value;
    }

    T operator+=(T value)
        requires(_dim == 0)
    {
        *_data += value;
        return value;
    }

    TensorView<T, _dim>& operator=(TensorView<T, _dim>& value) = delete;
    std::size_t size(std::size_t index = 0) const { return _shape[index]; }
    T* data() const { return _data; }
    std::size_t* stride() const { return _stride; }
    std::size_t* shape() const { return _shape; }
    T gather(me_int_t index) const
        requires(_dim == 1)
    {
        return (*this)[index];
    }
    void scatter_add(me_int_t index, T value)
        requires(_dim == 1)
    {
        (*this)[index] += value;
    }

private:
    T* _data;
    std::size_t* _stride;
    std::size_t* _shape;
};

class Tensor;
using TensorVec = std::vector<Tensor>;

/// Kind of compute device a @ref Device represents.
enum class DeviceType { cpu, cuda, hip };

/// Hint passed to @ref Device::allocate describing how a tensor will be used,
/// so the allocator can reuse scratch buffers or skip a zero-fill.
enum class AllocHint {
    normal,
    output,
    local,
    /// Short-lived scratch storage, recycled from a pool.
    temporary,
    input_grad,
    local_grad,
    global_grad,
};

/// Whether newly allocated storage for `hint` must be zero-initialized.
inline bool needs_zero_init(AllocHint hint) {
    return hint == AllocHint::input_grad || hint == AllocHint::local_grad ||
        hint == AllocHint::global_grad;
}

/**
 * Compute backend a @ref Tensor's storage lives on.
 *
 * One instance per physical device (CPU, or a specific CUDA or HIP GPU);
 * obtained through `cpu_device()`, `cuda_device()` or `hip_device()`
 * rather than constructed directly. Every device-dependent tensor operation
 * (allocation, copy, the elementary kernels used to train the network
 * globals) is dispatched through this interface.
 */
class Device {
public:
    static constexpr bool stream_ordered_alloc = false;

    virtual ~Device() = default;
    /// Allocate `size` bytes for `hint`; the returned @ref Tensor, if not
    /// empty, is the pool allocation this storage was carved out of.
    virtual std::pair<void*, Tensor>
    allocate(std::size_t size, AllocHint hint) const = 0;
    /// Free a pointer returned by @ref allocate.
    virtual void free(void* ptr) const = 0;
    /// Free a pointer returned by @ref allocate after the work queued on `stream`.
    virtual void free_on_stream(void* ptr, void* stream) const { free(ptr); }
    /// Stream to free stream-ordered storage on if it has no known stream.
    virtual void* default_free_stream() const { return nullptr; }
    /// Make work queued on `to` from now on wait for the work queued on `from`.
    virtual void order_streams(void* from, void* to) const {}
    /// Copy `size` bytes from `from` to `to`, both on this device.
    virtual void memcpy(void* to, void* from, std::size_t size) const = 0;
    /// Copy the elements of `source` into `target`.
    virtual void tensor_copy(const Tensor& source, Tensor& target) const = 0;
    /// Set every element of `tensor` to zero.
    virtual void tensor_zero(Tensor& tensor) const = 0;
    /// Add `source` into `target` element-wise.
    virtual void tensor_add(const Tensor& source, Tensor& target) const = 0;
    /// Copy `source` into `target`, which lives on the CPU.
    virtual void tensor_cpu(const Tensor& source, Tensor& target) const = 0;
    /// This device, as the pointer type used elsewhere in the API.
    virtual const Device* device_ptr() const = 0;
    /// Block until every operation queued on this device has completed.
    virtual void sync_barrier() const {}
    /// The kind of device this is.
    virtual DeviceType device_type() const = 0;
    /// Index of the device among those of its type, 0 for the CPU.
    virtual int device_index() const { return 0; }
    /// Make this device current for the calling thread.
    virtual void activate() const = 0;
    /// One Adam optimizer update step; see @ref AdamOptimizer.
    virtual void adam_step(
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
    ) const = 0;
};

/// Non-owning handle to a @ref Device.
using DevicePtr = const Device*;
/// The CPU device.
DevicePtr cpu_device();
/// CUDA device `index`.
DevicePtr cuda_device(std::size_t index);
/// HIP device `index`.
DevicePtr hip_device(std::size_t index);

/**
 * Reference-counted, device-aware N-dimensional array.
 *
 * The runtime's tensor type: a typed, strided view onto a block of memory on
 * a specific @ref Device (CPU, CUDA, or HIP), shared through reference
 * counting so copying a `Tensor` is cheap. Compute-graph values, matrix
 * element results, trainable globals and event data all flow through this
 * type. From Python a `Tensor` is only ever obtained from another madspace
 * call, never constructed directly; it exports itself through the
 * `__dlpack__` / `__dlpack_device__` protocol so it converts to a NumPy array
 * or a PyTorch tensor without copying.
 */
class Tensor {
public:
    /// Empty tensor holding no storage.
    Tensor() : impl(nullptr) {}

    /// Shares `other`'s storage.
    Tensor(const Tensor& other) : impl(other.impl) {
        if (impl != nullptr) {
            impl->incref();
        }
    }

    /// Takes ownership of `other`'s storage, leaving it empty.
    Tensor(Tensor&& other) noexcept : impl(other.impl) { other.impl = nullptr; }

    /// Allocates a new tensor of `dtype` and `shape` on the CPU.
    Tensor(DataType dtype, const Sizes& shape, AllocHint hint = AllocHint::normal) :
        Tensor(dtype, shape, cpu_device(), hint) {}

    /// Allocates a new tensor of `dtype` and `shape` on `device`.
    Tensor(
        DataType dtype,
        const Sizes& shape,
        DevicePtr device,
        AllocHint hint = AllocHint::normal
    ) :
        impl(new TensorImpl{dtype, shape, device}) {
        auto size = init_stride();
        allocate(size, *device, hint);
    }

    /// Same as above, statically dispatched to a compile-time device type.
    template <typename D>
    Tensor(
        DataType dtype,
        const Sizes& shape,
        const D& device,
        AllocHint hint = AllocHint::normal
    ) :
        impl(new TensorImpl{dtype, shape, device.device_ptr()}) {
        auto size = init_stride();
        allocate(size, device, hint);
    }

    /// Wraps externally-owned CPU memory; `external_reset` runs when the last
    /// reference is dropped, instead of freeing `data`.
    Tensor(
        DataType dtype,
        const Sizes& shape,
        void* data,
        std::function<void()> external_reset
    ) :
        Tensor(dtype, shape, cpu_device(), data, external_reset) {}

    /// Wraps externally-owned memory on `device`; see the CPU overload above.
    Tensor(
        DataType dtype,
        const Sizes& shape,
        DevicePtr device,
        void* data,
        std::function<void()> external_reset
    ) :
        impl(new TensorImpl{dtype, shape, device, data, false, external_reset}) {
        init_stride();
    }

    /// Wraps externally-owned memory with an explicit `stride`, for a
    /// non-contiguous view onto existing data, ordered on `stream` if given.
    Tensor(
        DataType dtype,
        const Sizes& shape,
        const Sizes& stride,
        DevicePtr device,
        void* data,
        std::function<void()> external_reset,
        std::optional<std::uintptr_t> stream = std::nullopt
    ) :
        impl(new TensorImpl{
            dtype, shape, device, data, false, external_reset, nullptr, 1, stride
        }) {
        impl->stream = stream;
        std::size_t stride_prod = 1;
        bool first = true, contiguous = true;
        impl->contiguous_dims = 0;
        for (auto [size_i, stride_i] : zip(shape, stride)) {
            // only the leading run of contiguous dimensions counts
            if (contiguous && (stride_i == stride_prod || size_i == 1)) {
                ++impl->contiguous_dims;
            } else {
                contiguous = false;
            }
            if (first && size_i == 1) {
                impl->stride[0] = 0;
            }
            stride_prod *= size_i;
            first = false;
        }
    }

    /// A `DataType::batch_sizes` tensor holding literal per-channel batch sizes.
    Tensor(const SizeVec& batch_sizes) :
        impl(new TensorImpl{
            DataType::batch_sizes,
            {},
            cpu_device(),
            nullptr,
            true,
            std::nullopt,
            nullptr,
            1,
            {},
            0,
            batch_sizes
        }) {}

    /// Single-value tensor holding `value`, allocated on `device`.
    template <ScalarType T>
    Tensor(T value, DevicePtr device, AllocHint hint = AllocHint::normal) :
        impl(new TensorImpl{dtype_of<T>(), {1}, device}) {
        auto size = init_stride();
        allocate(size, *device, hint);
        device->memcpy(impl->data, &value, sizeof(value));
        if (std::is_same_v<T, me_int_t> && value >= 0) {
            impl->batch_sizes.push_back(value);
        }
    }

    template <ScalarType T, typename D>
    Tensor(T value, const D& device, AllocHint hint = AllocHint::normal) :
        impl(new TensorImpl{dtype_of<T>(), {1}, device.device_ptr()}) {
        auto size = init_stride();
        allocate(size, device, hint);
        device.memcpy(impl->data, &value, sizeof(value));
        if (std::is_same_v<T, me_int_t> && value >= 0) {
            impl->batch_sizes.push_back(value);
        }
    }

    /// Tensor built from a literal `TensorValue` (nested int/float data).
    Tensor(TensorValue value, DevicePtr device, AllocHint hint = AllocHint::normal) :
        impl(new TensorImpl{
            std::visit(
                Overloaded{
                    [](std::vector<me_int_t>) { return DataType::dt_int32; },
                    [](std::vector<double>) { return DataType::dt_float64; },
                },
                std::get<1>(value)
            ),
            [&] {
                auto& val_shape = std::get<0>(value);
                Sizes full_shape(val_shape.size() + 1);
                full_shape[0] = 1;
                std::copy(val_shape.begin(), val_shape.end(), full_shape.begin() + 1);
                return full_shape;
            }(),
            device
        }) {
        auto size = init_stride();
        allocate(size, *device, hint);
        std::visit(
            [&](auto& vec) { device->memcpy(impl->data, vec.data(), size); },
            std::get<1>(value)
        );
    }

    /// Releases this reference; frees the storage once the last one drops.
    ~Tensor() { reset(); }

    /// Shares the assigned tensor's storage, releasing the previous one.
    Tensor& operator=(const Tensor& other) {
        reset();
        impl = other.impl;
        if (impl != nullptr) {
            impl->incref();
        }
        return *this;
    }

    /// Takes ownership of the assigned tensor's storage.
    Tensor& operator=(Tensor&& other) noexcept {
        reset();
        impl = other.impl;
        other.impl = nullptr;
        return *this;
    }

    /// Whether this tensor holds any storage.
    operator bool() const { return impl != nullptr; }

    template <class T, int dim>
    /// Typed, dimension-checked view onto the data for direct element access.
    TensorView<T, dim> view() {
        check_impl();
        check_view_dim(dim, impl->shape.size());
        check_view_dtype<T>();
        T* data = static_cast<T*>(impl->data);
        return TensorView<T, dim>(data, impl->stride.data(), impl->shape.data());
    }

    template <class T, int dim>
    /// Typed, dimension-checked view onto the data for direct element access.
    const TensorView<T, dim> view() const {
        check_impl();
        check_view_dim(dim, impl->shape.size());
        check_view_dtype<T>();
        T* data = static_cast<T*>(impl->data);
        return TensorView<T, dim>(data, impl->stride.data(), impl->shape.data());
    }

    template <class T, int dim>
    /// Like @ref view, with the leading `flatten_count` dimensions merged into one.
    PackedTensorView<T, dim> flat_view(std::size_t flatten_count) const {
        check_impl();
        check_view_dtype<T>();
        T* data = static_cast<T*>(impl->data);
        if (flatten_count <= 1) {
            check_view_dim(dim, impl->shape.size());
            return {data, impl->stride, impl->shape, impl->simd_padded};
        }
        if (flatten_count > impl->contiguous_dims) {
            throw std::invalid_argument("can only flatten contiguous dimensions");
        }
        check_view_dim(dim, impl->shape.size() - flatten_count + 1);
        Sizes stride{1}, shape{1};
        std::size_t i = 0;
        for (; i < flatten_count; ++i) {
            shape[0] *= impl->shape[i];
        }
        for (; i < impl->shape.size(); ++i) {
            shape.push_back(impl->shape[i]);
            stride.push_back(impl->stride[i]);
        }
        return {data, stride, shape};
    }

    /// Raw pointer to the underlying storage.
    void* data() {
        check_impl();
        return impl->data;
    }
    /// Raw pointer to the underlying storage.
    void* data() const {
        check_impl();
        return impl->data;
    }
    /// Size of each dimension.
    const Sizes& shape() const {
        check_impl();
        return impl->shape;
    }
    /// Element stride of each dimension.
    const Sizes& stride() const {
        check_impl();
        return impl->stride;
    }
    /// Size of dimension `i`.
    std::size_t size(std::size_t i) const {
        check_impl();
        return impl->shape[i];
    }
    /// Element type.
    DataType dtype() const {
        check_impl();
        return impl->dtype;
    }
    /// Per-channel batch sizes, for a `DataType::batch_sizes` tensor.
    const SizeVec& batch_sizes() const {
        check_impl();
        return impl->batch_sizes;
    }
    /// The device this tensor's storage lives on.
    DevicePtr device() const {
        check_impl();
        return impl->device;
    }
    /// The GPU stream the storage is ordered on, if known.
    std::optional<std::uintptr_t> stream() const {
        return impl == nullptr ? std::nullopt : storage()->stream;
    }
    /// Set the stream returned by @ref stream, if the storage is stream-ordered.
    void set_stream(std::optional<std::uintptr_t> stream) {
        if (impl != nullptr && storage()->stream_ordered) {
            storage()->stream = stream;
        }
    }
    /// The single integer value of a scalar `DataType::batch_sizes` tensor.
    std::size_t index_value() const {
        check_impl();
        if (impl->batch_sizes.size() > 0) {
            return impl->batch_sizes[0];
        }
        auto cpu_tensor = cpu();
        return cpu_tensor.view<me_int_t, 1>()[0];
    }

    /// Size in bytes of one element.
    std::size_t dtype_size() const {
        check_impl();
        switch (impl->dtype) {
        case DataType::dt_int32:
            return sizeof(me_int_t);
        case DataType::dt_float32:
            return sizeof(float);
        case DataType::dt_float64:
            return sizeof(double);
        case DataType::batch_sizes:
            return 0;
        default:
            throw std::logic_error("invalid data type");
        }
    }

    /// Total size in bytes of the storage.
    std::size_t byte_size() const { return dtype_size() * shape().product(); }

    /// Releases this reference; the tensor is empty afterwards.
    void reset() { reset_on_stream(stream()); }

    template <typename D>
    /// Releases this reference on `device`; the tensor is empty afterwards.
    void reset(const D& device) {
        if (impl == nullptr) {
            return;
        }
        impl->reset(device);
        impl = nullptr;
    }

    /// Releases this reference, freeing stream-ordered storage on `stream` or, if
    /// unknown, on the device's default free stream.
    void reset_on_stream(std::optional<std::uintptr_t> stream) {
        if (impl == nullptr) {
            return;
        }
        impl->reset_on_stream(stream);
        impl = nullptr;
    }

    /// A single index along `axis`, dropping that dimension.
    Tensor select(std::size_t axis, std::size_t index) const;
    /// A contiguous range `[start, stop)` along `axis`.
    Tensor slice(std::size_t axis, std::size_t start, std::size_t stop) const;
    /// Splits `axis` into consecutive chunks of the given `sizes`.
    std::vector<Tensor> split(std::size_t axis, const SizeVec& sizes) const;
    /// Splits `axis` into one tensor per index, dropping that dimension.
    std::vector<Tensor> unstack(std::size_t axis) const;
    /// Inserts a size-1 dimension at `axis`.
    Tensor unsqueeze(std::size_t axis) const;
    /// Broadcasts size-1 dimensions to `shape`, without copying.
    Tensor expand(const Sizes& shape) const;
    /// A view with a different `shape` over the same elements.
    Tensor reshape(const Sizes& shape) const;
    /// Splits dimension `axis` into two, the second of size `factor`.
    Tensor factor_dim(std::size_t axis, std::size_t factor);
    /// Splits along the batch dimension and reshapes each piece.
    std::vector<Tensor> split_and_reshape(const std::vector<Sizes>& shapes) const;

    template <typename D>
    /// A copy of this tensor on the CPU, allocated through `device`, or
    /// `*this` if it is already there.
    Tensor cpu(const D& device) const {
        check_impl();
        if (impl->device == cpu_device()) {
            return *this;
        } else {
            Tensor tensor(impl->dtype, impl->shape);
            device.tensor_cpu(contiguous(device), tensor);
            return tensor;
        }
    }
    /// A copy of this tensor on the CPU, or `*this` if it is already there.
    Tensor cpu() const { return cpu(*impl->device); }
    /// A contiguous CPU copy with the element type converted to `dtype`, or the
    /// CPU tensor itself if it already has that type.
    Tensor cast(DataType dtype) const;

    template <typename D>
    /// Sets every element to zero.
    void zero(const D& device) {
        check_impl();
        device.tensor_zero(*this);
    }
    /// Sets every element to zero.
    void zero() { zero(*impl->device); }

    template <typename D>
    /// Copies `source`'s data into this tensor's storage.
    void copy_from(const Tensor& source, const D& device) {
        check_impl();
        if (source.device() == this->device()) {
            device.tensor_copy(source, *this);
        } else if (is_contiguous()) {
            auto contig_source = source.contiguous();
            device.memcpy(data(), contig_source.data(), byte_size());
        } else {
            throw std::runtime_error(
                "tensor must be contiguous for copy across devices"
            );
        }
    }
    /// Copies `source`'s data into this tensor's storage.
    void copy_from(const Tensor& source) { copy_from(source, *impl->device); }

    template <typename D>
    /// Adds `source` into this tensor element-wise.
    void add(const Tensor& source, const D& device) {
        check_impl();
        device.tensor_add(source, *this);
    }
    /// Adds `source` into this tensor element-wise.
    void add(const Tensor& source) { add(source, *impl->device); }

    template <typename D>
    /// An independent copy of this tensor, allocated through `device`.
    Tensor copy(const D& device, AllocHint hint = AllocHint::normal) const {
        check_impl();
        Tensor tensor(impl->dtype, impl->shape, device, hint);
        device.tensor_copy(*this, tensor);
        return tensor;
    }
    /// An independent copy of this tensor.
    Tensor copy(AllocHint hint = AllocHint::normal) const {
        return copy(*impl->device, hint);
    }

    /// Whether the elements are stored without gaps in column-major (Fortran)
    /// order, i.e. the first index varies fastest.
    bool is_contiguous() const { return impl->contiguous_dims == impl->shape.size(); }

    /// Number of leading dimensions that are contiguous.
    std::size_t contiguous_dims() const { return impl->contiguous_dims; }

    template <typename D>
    /// A contiguous copy allocated through `device`, or `*this` if already
    /// contiguous.
    Tensor contiguous(const D& device, AllocHint hint = AllocHint::normal) const {
        check_impl();
        return is_contiguous() ? *this : copy(device, hint);
    }

    /// A contiguous copy, or `*this` if already contiguous.
    Tensor contiguous(AllocHint hint = AllocHint::normal) const {
        return contiguous(*impl->device, hint);
    }

    template <typename D>
    /// Like @ref contiguous, broadcasting a leading size-1 dimension to
    /// `batch_size`.
    Tensor contiguous(
        std::size_t batch_size, const D& device, AllocHint hint = AllocHint::normal
    ) const {
        check_impl();
        if (size(0) == batch_size) {
            return contiguous(device, hint);
        } else if (size(0) == 1) {
            auto shape = impl->shape;
            shape[0] = batch_size;
            Tensor tensor(impl->dtype, shape, impl->device, hint);
            device.tensor_copy(*this, tensor);
            return tensor;
        } else {
            throw std::runtime_error("invalid batch size");
        }
    }

    /// Like @ref contiguous, broadcasting a leading size-1 dimension to
    /// `batch_size`.
    Tensor contiguous(std::size_t batch_size) const {
        return contiguous(batch_size, *impl->device);
    }

    /// Whether this is the only reference to its storage.
    bool is_only_reference() const {
        check_impl();
        return impl->ref_count.load() == 1;
    }

    /// Maximum number of stored elements (value elements times lanes) of a
    /// literal for which @ref simd_padded_literal pads the storage.
    static constexpr std::size_t max_simd_padded_size = 512;

    /**
     * Literal constant for the CPU SIMD kernels.
     *
     * The tensor has batch size one and batch stride zero like any literal,
     * but each element is stored `lanes` times in consecutive memory, one
     * copy per SIMD vector lane, so that the vectorized kernels can read it
     * with contiguous vector loads. Scalar code sees an ordinary broadcast
     * tensor. Literals with more than @ref max_simd_padded_size stored
     * elements, or `lanes` of one, are stored without padding.
     */
    static Tensor
    simd_padded_literal(const LiteralValue& value, std::size_t lanes, DevicePtr device);

    /// Whether this is a literal stored padded to the SIMD vector size, see
    /// @ref simd_padded_literal. Views and copies of it are not padded.
    bool is_simd_padded() const {
        check_impl();
        return impl->simd_padded;
    }

private:
    struct TensorImpl {
        DataType dtype;
        Sizes shape;
        DevicePtr device;
        void* data;
        bool owns_data = true;
        std::optional<std::function<void()>> external_reset = std::nullopt;
        TensorImpl* data_owner;
        std::atomic<int> ref_count = 1;
        Sizes stride;
        std::size_t contiguous_dims;
        SizeVec batch_sizes;
        bool stream_ordered = false;
        std::optional<std::uintptr_t> stream;
        bool simd_padded = false;

        template <typename D>
        void reset(const D& device) {
            if (ref_count.fetch_sub(1, std::memory_order_acq_rel) != 1) {
                return;
            }
            if (owns_data && data != nullptr) {
                device.free(data);
                --Tensor::tensor_count;
            } else if (data_owner != nullptr) {
                data_owner->reset(device);
            } else if (external_reset) {
                (*external_reset)();
            }
            delete this;
        }

        void reset_on_stream(std::optional<std::uintptr_t> stream) {
            if (ref_count.fetch_sub(1, std::memory_order_acq_rel) != 1) {
                return;
            }
            if (owns_data && data != nullptr) {
                if (stream_ordered) {
                    device->free_on_stream(
                        data,
                        stream ? reinterpret_cast<void*>(*stream)
                               : device->default_free_stream()
                    );
                } else {
                    device->free(data);
                }
                --Tensor::tensor_count;
            } else if (data_owner != nullptr) {
                data_owner->reset_on_stream(stream);
            } else if (external_reset) {
                (*external_reset)();
            }
            delete this;
        }

        void incref() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    };

    TensorImpl* storage() const {
        TensorImpl* item = impl;
        while (item->data_owner != nullptr) {
            item = item->data_owner;
        }
        return item;
    }

    Tensor(TensorImpl* _impl) : impl(_impl) {
        if (impl->data_owner != nullptr) {
            impl->data_owner->incref();
        }
    }
    std::size_t init_stride();

    void check_impl() const {
        if (impl == nullptr) {
            throw std::runtime_error("empty tensor");
        }
    }

    // a view of more dimensions than the tensor has reads strides and shapes that were
    // never set: out-of-bounds accesses, silent on one platform, a memory fault on
    // another (e.g. a kernel taking FIn<T, 1> for a tensor of declared rank 0)
    static void check_view_dim(std::size_t view_dim, std::size_t tensor_dim) {
        if (view_dim > tensor_dim) {
            throw std::invalid_argument(
                "view of dimension " + std::to_string(view_dim) +
                " of a tensor of dimension " + std::to_string(tensor_dim)
            );
        }
    }

    // reading a tensor through the wrong element type gives garbage without an error
    template <typename T>
    void check_view_dtype() const {
        if constexpr (ScalarType<T>) {
            if (impl->dtype != dtype_of<T>() && impl->dtype != DataType::batch_sizes) {
                throw std::invalid_argument("view element type does not match dtype");
            }
        }
    }

    template <typename D>
    void allocate(std::size_t size, const D& device, AllocHint hint) {
        auto [data, parent] = device.allocate(size, hint);
        impl->data = data;
        if (parent) {
            parent.impl->incref();
            impl->owns_data = false;
            impl->data_owner = parent.impl;
        } else if (data != nullptr) {
            ++tensor_count;
            if constexpr (D::stream_ordered_alloc) {
                impl->stream_ordered = true;
                impl->stream = reinterpret_cast<std::uintptr_t>(device.stream());
            }
        }
    }

    TensorImpl* impl;

public:
    /// Number of live tensors that currently own their storage.
    static inline std::size_t tensor_count = 0;
};

} // namespace madspace
