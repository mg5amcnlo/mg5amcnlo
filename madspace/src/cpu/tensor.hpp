#pragma once

#include "device.hpp"
#include "madspace/driver/tensor.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "simd.hpp"

#include <array>
#include <tuple>
#include <utility>

namespace madspace {
namespace cpu {

// View onto a tensor for the vectorized kernels. The V::lanes lanes of a
// vector are consecutive batch entries, so they are always contiguous in memory
// (batch stride one) and are read and written with plain vector loads and stores.
// Moving to the next vector advances the data pointer by vec_step elements:
// V::lanes for batched tensors, zero for literals that are stored padded to
// simd_vec_size_max identical lanes. Tensors that do not have this layout are handled
// by the scalar kernels instead (see tensor_foreach_impl).
template <class V, ScalarType T, int _dim, bool is_batch>
class VectorizedTensorView {
public:
    using VType = V;
    using DType = T;
    static const int dim = _dim;
    static constexpr bool is_single = false;

    VectorizedTensorView(const TensorView<T, _dim>& view, std::size_t vec_step) :
        _data(view.data()),
        _stride(view.stride()),
        _shape(view.shape()),
        _vec_step(vec_step) {}

    VectorizedTensorView(
        T* data, std::size_t* stride, std::size_t* shape, std::size_t vec_step
    ) :
        _data(data), _stride(stride), _shape(shape), _vec_step(vec_step) {}

    VectorizedTensorView(V& value) :
        _data(reinterpret_cast<T*>(&value)),
        _stride(nullptr),
        _shape(nullptr),
        _vec_step(0) {}

    const VectorizedTensorView<V, T, _dim - 1, false>
    operator[](std::size_t index) const
        requires(_dim != 0)
    {
        return {&_data[offset(index)], &_stride[1], &_shape[1], _vec_step};
    }

    VectorizedTensorView<V, T, _dim - 1, false> operator[](std::size_t index)
        requires(_dim != 0)
    {
        return {&_data[offset(index)], &_stride[1], &_shape[1], _vec_step};
    }

    template <typename... I>
    const VectorizedTensorView<V, T, _dim - sizeof...(I) - 1, false>
    get(std::size_t first_index, I... index) const
        requires(_dim >= sizeof...(I) + 1)
    {
        T* ptr = &_data[offset(first_index)];
        int i = 1;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I) + 1], &_shape[sizeof...(I) + 1], _vec_step};
    }

    template <typename... I>
    VectorizedTensorView<V, T, _dim - sizeof...(I) - 1, false>
    get(std::size_t first_index, I... index)
        requires(_dim >= sizeof...(I) + 1)
    {
        T* ptr = &_data[offset(first_index)];
        int i = 1;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I) + 1], &_shape[sizeof...(I) + 1], _vec_step};
    }

    operator V() const
        requires(_dim == 0)
    {
        return vload(VecTag<V>{}, _data);
    }

    V operator=(V value)
        requires(_dim == 0)
    {
        vstore(_data, value);
        return value;
    }

    V operator+=(V value)
        requires(_dim == 0)
    {
        V new_value = vload(VecTag<V>{}, _data) + value;
        vstore(_data, new_value);
        return new_value;
    }

    VectorizedTensorView<V, T, _dim, is_batch>&
    operator=(VectorizedTensorView<V, T, _dim, is_batch>& value) = delete;

    std::size_t size(std::size_t index = 0) const {
        if (is_batch && index == 0) {
            return _shape[0] / V::lanes;
        } else {
            return _shape[index];
        }
    }

    template <typename IVec>
    V gather(IVec indices) const
        requires(_dim == 1)
    {
        return vgather(VecTag<V>{}, _data, 1, _stride[0], indices);
    }

    template <typename IVec>
    void scatter_add(IVec indices, V values)
        requires(_dim == 1)
    {
        V old_values = vgather(VecTag<V>{}, _data, 1, _stride[0], indices);
        vscatter(_data, 1, _stride[0], indices, old_values + values);
    }

private:
    std::size_t offset(std::size_t index) const {
        if constexpr (is_batch) {
            return index * _vec_step;
        } else {
            return index * _stride[0];
        }
    }

    T* _data;
    std::size_t* _stride;
    std::size_t* _shape;
    std::size_t _vec_step;
};

// View onto an input marked as single in the instruction set, i.e. without a
// batch dimension. The batch index is ignored, and every lane gets the same
// value through a broadcast load.
template <class V, ScalarType T, int _dim, bool is_batch>
class SingleVectorizedTensorView {
public:
    using VType = V;
    using DType = T;
    static const int dim = _dim;
    static constexpr bool is_single = true;

    SingleVectorizedTensorView(const TensorView<T, _dim>& view) :
        _data(view.data()), _stride(view.stride()), _shape(view.shape()) {}

    SingleVectorizedTensorView(T* data, std::size_t* stride, std::size_t* shape) :
        _data(data), _stride(stride), _shape(shape) {}

    const SingleVectorizedTensorView<V, T, _dim - 1, false>
    operator[](std::size_t index) const
        requires(_dim != 0)
    {
        return {&_data[offset(index)], &_stride[1], &_shape[1]};
    }

    template <typename... I>
    const SingleVectorizedTensorView<V, T, _dim - sizeof...(I) - 1, false>
    get(std::size_t first_index, I... index) const
        requires(_dim >= sizeof...(I) + 1)
    {
        T* ptr = &_data[offset(first_index)];
        int i = 1;
        ((ptr = &ptr[index * _stride[i++]]), ...);
        return {ptr, &_stride[sizeof...(I) + 1], &_shape[sizeof...(I) + 1]};
    }

    operator V() const
        requires(_dim == 0)
    {
        return vload_broadcast(VecTag<V>{}, _data);
    }

    // the value as a plain scalar, e.g. for loop bounds or indices
    T value() const
        requires(_dim == 0)
    {
        return *_data;
    }

    std::size_t size(std::size_t index = 0) const { return _shape[index]; }

    template <typename IVec>
    V gather(IVec indices) const
        requires(_dim == 1)
    {
        return vgather(VecTag<V>{}, _data, 0, _stride[0], indices);
    }

private:
    std::size_t offset(std::size_t index) const {
        if constexpr (is_batch) {
            return 0;
        } else {
            return index * _stride[0];
        }
    }

    T* _data;
    std::size_t* _stride;
    std::size_t* _shape;
};

// whether a kernel parameter type is a view onto a single (non-batched) input
template <typename T>
inline constexpr bool is_single_param = requires {
    requires std::remove_cvref_t<T>::is_single;
};

template <typename T>
class ScalarView {
public:
    using DType = T;
    static constexpr bool is_scalar_view = true;

    ScalarView(T data) : _data(data) {}
    const ScalarView<T> operator[](std::size_t index) const { return _data; }
    template <typename... I>
    const ScalarView<T> get(I... index) const {
        return _data;
    }
    operator T() const { return _data; }

private:
    T _data;
};

// return the tuple of flattened PackedTensorViews where the type is extracted
// from the signature of F
template <typename F, int dims>
struct get_flat_views;
template <typename... TParam, int dims>
struct get_flat_views<void (*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(std::size_t flatten_count, TArg&&... args) {
        return std::make_tuple([&]() {
            if constexpr (ScalarType<TArg>) {
                return args;
            } else {
                return args
                    ->template flat_view<typename TParam::DType, TParam::dim + dims>(
                        flatten_count
                    );
            }
        }()...);
    }
};

// return tuple of TensorViews from PackedTensorViews
struct get_views {
    template <typename... TArg>
    auto operator()(TArg&... args) {
        return std::make_tuple([&]() {
            if constexpr (ScalarType<TArg>) {
                return ScalarView(args);
            } else {
                return TensorView<typename TArg::DType, TArg::dim>(args);
            }
        }()...);
    }
};

// return the tuple of VectorizedTensorViews where the type is extracted
// from the signature of F. vec_steps holds the step between vectors for every
// argument (the lane count, or zero for padded literals)
template <typename F, int dims>
struct get_vectorized_views;
template <typename... TParam, int dims>
struct get_vectorized_views<void (*)(TParam...), dims> {
    template <typename... TArg>
    auto operator()(
        const std::array<std::size_t, sizeof...(TArg)>& vec_steps, TArg&... args
    ) {
        return make(vec_steps, std::index_sequence_for<TArg...>(), args...);
    }

private:
    template <typename... TArg, std::size_t... I>
    auto make(
        const std::array<std::size_t, sizeof...(TArg)>& vec_steps,
        std::index_sequence<I...>,
        TArg&... args
    ) {
        return std::make_tuple(make_view<TParam>(vec_steps[I], args)...);
    }

    template <typename P, typename TArg>
    static auto make_view(std::size_t vec_step, TArg& arg) {
        if constexpr (TArg::is_scalar_view) {
            return ScalarView(P(typename TArg::DType(arg)));
        } else if constexpr (is_single_param<P>) {
            return SingleVectorizedTensorView<
                typename P::VType,
                typename P::DType,
                P::dim + dims,
                true>(arg);
        } else {
            return VectorizedTensorView<
                typename P::VType,
                typename P::DType,
                P::dim + dims,
                true>(arg, vec_step);
        }
    }
};

// whether each parameter of F is a single (non-batched) input
template <typename F>
struct single_params;
template <typename... TParam>
struct single_params<void (*)(TParam...)> {
    static constexpr std::array<bool, sizeof...(TParam)> value{
        is_single_param<TParam>...
    };
};

template <typename F>
struct first_param;
template <typename... TParam>
struct first_param<void (*)(TParam...)> {
    static constexpr int dim = std::tuple_element_t<0, std::tuple<TParam...>>::dim;
};

// lane count of a vectorized kernel, taken from the vector type of its first view
template <typename F>
struct vector_lanes;
template <typename... TParam>
struct vector_lanes<void (*)(TParam...)> {
    static constexpr int value = std::remove_cvref_t<
        std::tuple_element_t<0, std::tuple<TParam...>>>::VType::lanes;
};

template <auto func, int dims, typename... V>
inline void nested_for(std::size_t batch_size, std::size_t batch_offset, V... views) {
    std::size_t end_index = batch_offset + batch_size;
    auto& first_view = std::get<0>(std::tie(views...));
    if constexpr (dims == 0) {
        func(views...);
    } else if constexpr (dims == 1) {
        for (std::size_t i = batch_offset; i < end_index; ++i) {
            func(views[i]...);
        }
    } else if constexpr (dims == 2) {
        auto size1 = first_view.size(1);
        for (std::size_t j = 0; j < size1; ++j) {
            for (std::size_t i = batch_offset; i < end_index; ++i) {
                func(views.get(i, j)...);
            }
        }
    } else if constexpr (dims == 3) {
        auto size1 = first_view.size(1);
        auto size2 = first_view.size(2);
        for (std::size_t k = 0; k < size2; ++k) {
            for (std::size_t j = 0; j < size1; ++j) {
                for (std::size_t i = batch_offset; i < end_index; ++i) {
                    func(views.get(i, j, k)...);
                }
            }
        }
    } else if constexpr (dims == 4) {
        auto size1 = first_view.size(1);
        auto size2 = first_view.size(2);
        auto size3 = first_view.size(3);
        for (std::size_t l = 0; l < size3; ++l) {
            for (std::size_t k = 0; k < size2; ++k) {
                for (std::size_t j = 0; j < size1; ++j) {
                    for (std::size_t i = batch_offset; i < end_index; ++i) {
                        func(views.get(i, j, k, l)...);
                    }
                }
            }
        }
    }
}

template <auto func, int dims, typename... V>
inline void nested_for_nobatch(V... views) {
    if constexpr (dims == 0) {
        func(views...);
    } else {
        auto& first_view = std::get<0>(std::tie(views...));
        nested_for<func, dims>(first_view.size(0), 0, views...);
    }
}

template <
    auto scalar_func,
    auto vector_func,
    int n_in,
    int n_out,
    int dims,
    typename D,
    ScalarType... S>
inline void tensor_foreach_impl(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    std::size_t flatten_count,
    const D& device,
    bool single_job,
    S... scalar_args
) {
    // A broadcast *output* (size(0) != batch_size, e.g. a global's gradient
    // accumulator during backward) is shared across the whole batch instead of
    // having one slot per batch element. Splitting the batch across worker threads
    // would then have multiple threads accumulate into the very same memory
    // concurrently with no synchronization, so force a single job in that case
    // (the whole batch handled sequentially by one thread). Broadcast *inputs* are
    // only ever read, never written, in both forward and backward -- concurrent
    // reads of the same memory are safe, so they don't need this.
    // SIMD lanes would also write to the same memory, so use the scalar kernel.
    bool broadcast_output = false;
    for (const Tensor* out : outputs) {
        if (out->size(0) != batch_size) {
            single_job = true;
            broadcast_output = true;
        }
    }

    // get views to the tensors with the correct types based on the signature of
    // scalar_func
    auto flat_views = std::apply(
        get_flat_views<decltype(scalar_func), dims>(),
        std::tuple_cat(
            std::make_tuple(flatten_count),
            inputs,
            outputs,
            std::make_tuple(scalar_args...)
        )
    );

    // The vectorized kernel loads and stores the lanes of a vector with plain
    // contiguous memory accesses. It is only used if every batched argument has
    // batch stride one (or is a literal padded to simd_vec_size_max lanes); single
    // inputs are read with broadcast loads and can have any layout. Otherwise
    // the whole batch is handled by the scalar kernel.
    constexpr std::size_t n_args = n_in + n_out + sizeof...(S);
    std::array<std::size_t, n_args> vec_steps{};
    bool vectorize = !broadcast_output;
    if constexpr (!std::is_same_v<decltype(scalar_func), decltype(vector_func)>) {
        using Single = single_params<decltype(vector_func)>;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    using View = std::tuple_element_t<
                        I,
                        std::remove_reference_t<decltype(flat_views)>>;
                    if constexpr (!ScalarType<View> && !Single::value[I]) {
                        auto& view = std::get<I>(flat_views);
                        if (view.shape[0] == batch_size && view.stride[0] == 1) {
                            vec_steps[I] = vector_lanes<decltype(vector_func)>::value;
                        } else if (view.simd_padded) {
                            vec_steps[I] = 0;
                        } else {
                            vectorize = false;
                        }
                    }
                }(),
                ...
            );
        }(std::make_index_sequence<n_args>());
    }

    device.foreach (
        batch_size,
        [flat_views, vectorize, vec_steps, scalar_args...](
            std::size_t count, std::size_t offset
        ) mutable {
            auto views = std::apply(get_views(), flat_views);
            std::size_t scalar_offset = offset;
            if constexpr (!std::
                              is_same_v<decltype(scalar_func), decltype(vector_func)>) {
                if (vectorize) {
                    auto vectorized_views = std::apply(
                        [&vec_steps](auto&... args) {
                            return get_vectorized_views<decltype(vector_func), dims>()(
                                vec_steps, args...
                            );
                        },
                        views
                    );
                    constexpr std::size_t lanes =
                        vector_lanes<decltype(vector_func)>::value;
                    std::size_t vec_count = count / lanes;
                    std::size_t vec_offset = offset / lanes;
                    std::apply(
                        [vec_count, vec_offset](auto&&... args) {
                            nested_for<vector_func, dims>(
                                vec_count, vec_offset, args...
                            );
                        },
                        vectorized_views
                    );
                    scalar_offset += vec_count * lanes;
                }
            }
            std::size_t scalar_count = offset + count - scalar_offset;
            std::apply(
                [scalar_count, scalar_offset](auto&&... args) {
                    nested_for<scalar_func, dims>(scalar_count, scalar_offset, args...);
                },
                views
            );
        },
        single_job
    );
}

template <
    auto scalar_func,
    auto vector_func,
    int n_in,
    int n_out,
    typename D,
    ScalarType... S>
inline void tensor_foreach_dynamic_impl(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    std::size_t iter_dims,
    const D& device,
    bool single_job,
    S... scalar_args
) {
    std::size_t flatten_count = iter_dims;
    for (auto input : inputs) {
        flatten_count = std::min(flatten_count, input->contiguous_dims());
        if (input->size(0) != batch_size) {
            flatten_count = 0;
        }
    }
    for (auto output : outputs) {
        flatten_count = std::min(flatten_count, output->contiguous_dims());
    }
    if (flatten_count > 1) {
        iter_dims -= flatten_count - 1;
        auto& first_shape = std::get<0>(inputs)->shape();
        for (std::size_t i = 1; i < flatten_count; ++i) {
            batch_size *= first_shape[i];
        }
    }

    switch (iter_dims) {
    case 1:
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 1>(
            inputs,
            outputs,
            batch_size,
            flatten_count,
            device,
            single_job,
            scalar_args...
        );
        break;
    case 2:
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 2>(
            inputs,
            outputs,
            batch_size,
            flatten_count,
            device,
            single_job,
            scalar_args...
        );
        break;
    case 3:
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 3>(
            inputs,
            outputs,
            batch_size,
            flatten_count,
            device,
            single_job,
            scalar_args...
        );
        break;
    case 4:
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, 4>(
            inputs,
            outputs,
            batch_size,
            flatten_count,
            device,
            single_job,
            scalar_args...
        );
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template <
    auto scalar_func,
    auto vector_func,
    int n_in,
    int n_out,
    int dims,
    typename D,
    ScalarType... S>
inline void tensor_foreach(
    std::array<const Tensor*, n_in>& inputs,
    std::array<Tensor*, n_out>& outputs,
    std::size_t batch_size,
    const D& device,
    S... scalar_args
) {
    if (batch_size == 0) {
        return;
    }
    if constexpr (dims > 1) {
        // call the dynamic foreach here as we can potentially be more efficient by
        // flattening contiguous dimensions
        tensor_foreach_dynamic_impl<scalar_func, vector_func, n_in, n_out>(
            inputs, outputs, batch_size, dims, device, false, scalar_args...
        );
    } else {
        tensor_foreach_impl<scalar_func, vector_func, n_in, n_out, dims>(
            inputs, outputs, batch_size, 0, device, false, scalar_args...
        );
    }
}

template <
    auto scalar_func,
    auto vector_func,
    int n_in,
    int n_out,
    typename D,
    ScalarType... S>
inline void tensor_foreach_dynamic(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    const D& device,
    S... scalar_args
) {
    if (batch_size == 0) {
        return;
    }
    tensor_foreach_dynamic_impl<scalar_func, vector_func, n_in, n_out>(
        inputs,
        outputs,
        batch_size,
        std::get<0>(inputs)->shape().size() - first_param<decltype(scalar_func)>::dim,
        device,
        false,
        scalar_args...
    );
}

template <
    auto scalar_func,
    auto vector_func,
    int n_in,
    int n_out,
    typename D,
    ScalarType... S>
inline void tensor_foreach_dynamic_single(
    std::array<const Tensor*, n_in> inputs,
    std::array<Tensor*, n_out> outputs,
    std::size_t batch_size,
    const D& device,
    S... scalar_args
) {
    if (batch_size == 0) {
        return;
    }
    tensor_foreach_dynamic_impl<scalar_func, vector_func, n_in, n_out>(
        inputs,
        outputs,
        batch_size,
        std::get<0>(inputs)->shape().size() - first_param<decltype(scalar_func)>::dim,
        device,
        true,
        scalar_args...
    );
}

} // namespace cpu
} // namespace madspace
