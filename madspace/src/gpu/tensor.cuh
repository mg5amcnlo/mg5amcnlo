#pragma once

#include "device.hpp"
#include "madspace/compgraphs/type.hpp"
#include "madspace/driver/tensor.hpp"

#include <algorithm>

namespace madspace {
namespace gpu {

constexpr std::size_t THREADS_MULTIPLE = 32;
constexpr std::size_t MAX_THREADS_PER_BLOCK = 256;

template <ScalarType T, int _dim, bool packed = false>
class GpuTensorView {
public:
    using DType = T;
    using SizeType = std::conditional_t<packed, std::size_t[_dim], std::size_t*>;
    static const int dim = _dim;

    GpuTensorView(const TensorView<T, _dim>& view)
        requires(packed)
        : _data(view.data()) {
        if constexpr (_dim > 0) {
            std::copy(view.shape(), view.shape() + _dim, std::begin(this->_shape));
            std::copy(view.stride(), view.stride() + _dim, std::begin(this->_stride));
        }
    }

    GpuTensorView(const GpuTensorView<T, _dim, packed>& view) = default;

    __host__ __device__ GpuTensorView(const GpuTensorView<T, _dim, true>& view)
        requires(!packed)
        : _data(view._data), _stride(&view._stride), _shape(&view._shape) {}

    GpuTensorView<T, _dim, packed>& operator=(GpuTensorView<T, _dim, packed>& value)
        requires(packed)
    = default;

    __device__ GpuTensorView(T* data, std::size_t* stride, std::size_t* shape)
        requires(!packed)
        : _data(data), _stride(stride), _shape(shape) {}

    __device__ GpuTensorView(T& value)
        requires(!packed)
        : _data(&value), _stride(nullptr), _shape(nullptr) {}

    __device__ GpuTensorView<T, _dim, packed>&
    operator=(GpuTensorView<T, _dim, packed>& value) = delete;

    __device__ const GpuTensorView<T, _dim - 1> operator[](std::size_t index) const
        requires(_dim != 0)
    {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    __device__ GpuTensorView<T, _dim - 1> operator[](std::size_t index)
        requires(_dim != 0)
    {
        return {&_data[index * _stride[0]], &_stride[1], &_shape[1]};
    }

    template <typename... I>
    __device__ const GpuTensorView<T, _dim - sizeof...(I)> get(I... index) const
        requires(_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    template <typename... I>
    __device__ GpuTensorView<T, _dim - sizeof...(I)> get(I... index)
        requires(_dim >= sizeof...(I))
    {
        T* ptr = _data;
        int i = 0;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I)], &_shape[sizeof...(I)]};
    }

    __device__ operator T() const
        requires(_dim == 0)
    {
        return *_data;
    }

    __device__ T operator=(T value)
        requires(_dim == 0)
    {
        *_data = value;
        return value;
    }

    __device__ T operator+=(T value)
        requires(_dim == 0)
    {
        *_data += value;
        return value;
    }

    __device__ T gather(me_int_t index) const
        requires(_dim == 1)
    {
        return (*this)[index];
    }
    __device__ void scatter_add(me_int_t index, T value)
        requires(_dim == 1)
    {
        (*this)[index] += value;
    }

    __host__ __device__ std::size_t size(std::size_t index = 0) const {
        return _shape[index];
    }
    __host__ __device__ T* data() const { return _data; }
    __host__ __device__ std::size_t* stride() const { return _stride; }
    __host__ __device__ std::size_t* shape() const { return _shape; }

private:
    T* _data;
    SizeType _stride;
    SizeType _shape;
};

template <typename T>
class GpuScalarView {
public:
    using DType = T;
    static constexpr bool is_scalar_view = true;

    __host__ __device__ GpuScalarView(T data) : _data(data) {}
    __host__ __device__ const GpuScalarView<T> operator[](std::size_t index) const {
        return _data;
    }
    template <typename... I>
    __host__ __device__ const GpuScalarView<T> get(I... index) const {
        return _data;
    }
    __host__ __device__ operator T() const { return _data; }

private:
    T _data;
};

// return the tuple of packed GpuTensorViews where the type is extracted from
// the signature of F
template <typename F, int dims>
struct get_views;
template <typename... TParam, int dims>
struct get_views<void (*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(TArg&&... args) {
        return std::make_tuple([&]() {
            if constexpr (ScalarType<TArg>) {
                return GpuScalarView(args);
            } else {
                return GpuTensorView<typename TParam::DType, TParam::dim + dims, true>(
                    args->template view<typename TParam::DType, TParam::dim + dims>()
                );
            }
        }()...);
    }
};

template <auto func, int dims, typename... V>
__device__ void recursive_for(V... views) {
    if constexpr (dims == 0) {
        func(views...);
    } else {
        auto& first_view = [](auto& first, auto&... rest) -> auto& {
            return first;
        }(views...);
        for (std::size_t i = 0; i < first_view.size(); ++i) {
            recursive_for<func, dims - 1>(views[i]...);
        }
    }
}

template <auto func, int dims, typename... V>
__global__ void run_kernel(std::size_t batch_size, V... views) {
    auto& first_view = [](auto& first, auto&... rest) -> auto& {
        return first;
    }(views...);
    std::size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if constexpr (dims == 0) {
        func(views...);
    } else if constexpr (dims == 1) {
        if (index >= batch_size) {
            return;
        }
        func(views[index]...);
    } else if constexpr (dims == 2) {
        std::size_t size1 = first_view.size(1);
        if (index >= batch_size * size1) {
            return;
        }
        std::size_t i = index % batch_size;
        std::size_t j = index / batch_size;
        func(views.get(i, j)...);
    } else if constexpr (dims == 3) {
        std::size_t size1 = first_view.size(1);
        std::size_t size2 = first_view.size(2);
        if (index >= batch_size * size1 * size2) {
            return;
        }
        std::size_t i = index % batch_size;
        std::size_t jk = index / batch_size;
        std::size_t j = jk % size1;
        std::size_t k = jk / size1;
        func(views.get(i, j, k)...);
    } else if constexpr (dims == 4) {
        std::size_t size1 = first_view.size(1);
        std::size_t size2 = first_view.size(2);
        std::size_t size3 = first_view.size(3);
        if (index >= batch_size * size1 * size2) {
            return;
        }
        std::size_t i = index % batch_size;
        std::size_t jkl = index / batch_size;
        std::size_t j = jkl % size1;
        std::size_t kl = jkl / size1;
        std::size_t k = kl % size2;
        std::size_t l = kl / size2;
        func(views.get(i, j, k, l)...);
    }
}

template <typename F, typename... Args>
void launch_kernel(
    F kernel, std::size_t total_count, gpuStream_t stream, Args... args
) {
    if (total_count == 0) {
        return;
    }
    std::size_t n_threads = std::min(
        MAX_THREADS_PER_BLOCK,
        ((total_count + THREADS_MULTIPLE - 1) / THREADS_MULTIPLE) * THREADS_MULTIPLE
    );
    std::size_t n_blocks = (total_count + n_threads - 1) / n_threads;
    kernel<<<n_blocks, n_threads, 0, stream>>>(args...);
    check_error();
}

template <auto func, int n_in, int n_out, int dims, ScalarType... S>
void tensor_foreach(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    const AsyncGpuDevice& device,
    S... scalar_args
) {
    if (batch_size == 0) {
        return;
    }
    // get views to the tensors with the correct types based on the signature of func
    auto views = std::apply(
        get_views<decltype(func), dims>(),
        std::tuple_cat(inputs, outputs, std::make_tuple(scalar_args...))
    );
    auto& first_view = std::get<0>(views);

    std::size_t total_count = batch_size;
    for (std::size_t i = 1; i < dims; ++i) {
        total_count *= first_view.size(i);
    }
    if (total_count == 0) {
        return;
    }

    std::size_t n_threads = std::min(
        MAX_THREADS_PER_BLOCK,
        ((total_count + THREADS_MULTIPLE - 1) / THREADS_MULTIPLE) * THREADS_MULTIPLE
    );
    std::size_t n_blocks = (total_count + n_threads - 1) / n_threads;

    std::apply(
        [&](auto&... args) {
            run_kernel<func, dims>
                <<<n_blocks, n_threads, 0, device.stream()>>>(batch_size, args...);
        },
        views
    );
    check_error();
}

template <typename F>
struct first_param;
template <typename... TParam>
struct first_param<void (*)(TParam...)> {
    static constexpr int dim = std::tuple_element_t<0, std::tuple<TParam...>>::dim;
};

template <auto func, int n_in, int n_out, ScalarType... S>
void tensor_foreach_dynamic(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    const AsyncGpuDevice& device,
    S... scalar_args
) {
    switch (std::get<0>(inputs)->shape().size() - first_param<decltype(func)>::dim) {
    case 1:
        tensor_foreach<func, n_in, n_out, 1>(
            inputs, outputs, batch_size, device, scalar_args...
        );
        break;
    case 2:
        tensor_foreach<func, n_in, n_out, 2>(
            inputs, outputs, batch_size, device, scalar_args...
        );
        break;
    case 3:
        tensor_foreach<func, n_in, n_out, 3>(
            inputs, outputs, batch_size, device, scalar_args...
        );
        break;
    case 4:
        tensor_foreach<func, n_in, n_out, 4>(
            inputs, outputs, batch_size, device, scalar_args...
        );
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

} // namespace gpu
} // namespace madspace
