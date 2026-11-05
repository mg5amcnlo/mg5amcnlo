#include "madspace/driver/tensor.hpp"

#include <algorithm>
#include <variant>

using namespace madspace;

Tensor Tensor::select(std::size_t axis, std::size_t index) const {
    check_impl();
    if (axis > impl->shape.size()) {
        throw std::out_of_range("invalid select dimension");
    }
    if (index > impl->shape[axis]) {
        throw std::out_of_range("select: index out of bounds");
    }
    auto new_dim = impl->shape.size() - 1;
    Sizes new_shape(new_dim), new_stride(new_dim);
    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    std::copy(
        impl->shape.begin() + axis + 1, impl->shape.end(), new_shape.begin() + axis
    );
    std::copy(impl->stride.begin(), impl->stride.begin() + axis, new_stride.begin());
    std::copy(
        impl->stride.begin() + axis + 1, impl->stride.end(), new_stride.begin() + axis
    );
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        static_cast<uint8_t*>(impl->data) + index * impl->stride[axis] * dtype_size(),
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        std::min(impl->contiguous_dims, axis)
    });
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t stop) const {
    check_impl();
    if (axis > impl->shape.size()) {
        throw std::out_of_range("invalid slice dimension");
    }
    if (start > stop) {
        throw std::out_of_range("slice: start > stop");
    }
    if (stop > impl->shape[axis]) {
        throw std::out_of_range("slice: stop out of bounds");
    }
    auto new_shape = impl->shape;
    new_shape[axis] = stop - start;
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        static_cast<uint8_t*>(impl->data) + start * impl->stride[axis] * dtype_size(),
        false,
        std::nullopt,
        impl,
        1,
        impl->stride,
        std::min(impl->contiguous_dims, axis + 1)
    });
}

std::vector<Tensor> Tensor::split(std::size_t axis, const SizeVec& sizes) const {
    check_impl();
    std::vector<Tensor> tensors;
    auto split_count = sizes.size();
    tensors.reserve(split_count);
    std::size_t offset = 0;
    for (std::size_t size : sizes) {
        auto next_offset = offset + size;
        tensors.push_back(slice(axis, offset, next_offset));
        offset = next_offset;
    }
    return tensors;
}

std::vector<Tensor> Tensor::unstack(std::size_t axis) const {
    check_impl();
    std::vector<Tensor> tensors;
    auto axis_size = size(axis);
    tensors.reserve(axis_size);
    for (std::size_t i = 0; i < axis_size; ++i) {
        tensors.push_back(select(axis, i));
    }
    return tensors;
}

Tensor Tensor::unsqueeze(std::size_t axis) const {
    check_impl();
    // TODO: check if correct for non-contiguous tensor
    auto new_dim = impl->shape.size() + 1;
    Sizes new_shape(new_dim), new_stride(new_dim);
    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    new_shape[axis] = 1;
    std::copy(
        impl->shape.begin() + axis, impl->shape.end(), new_shape.begin() + axis + 1
    );
    std::copy(impl->stride.begin(), impl->stride.begin() + axis, new_stride.begin());
    new_stride[axis] = axis == 0 ? 1 : impl->stride[axis - 1];
    std::copy(
        impl->stride.begin() + axis, impl->stride.end(), new_stride.begin() + axis + 1
    );
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        impl->contiguous_dims + (axis <= impl->contiguous_dims)
    });
}

Tensor Tensor::expand(const Sizes& shape) const {
    check_impl();
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        Sizes(shape.size(), 0),
        0
    });
}

Tensor Tensor::reshape(const Sizes& new_shape) const {
    check_impl();
    if (!is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous");
    }
    if (new_shape.product() != shape().product()) {
        throw std::runtime_error("Incompatible shapes");
    }
    Tensor ret(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        {},
        0
    });
    ret.init_stride();
    return ret;
}

Tensor Tensor::factor_dim(std::size_t axis, std::size_t factor) {
    check_impl();
    auto new_dim = impl->shape.size() + 1;
    Sizes new_shape(new_dim), new_stride(new_dim);

    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    new_shape[axis] = factor;
    std::copy(
        impl->shape.begin() + axis, impl->shape.end(), new_shape.begin() + axis + 1
    );
    new_shape[axis + 1] /= factor;

    std::copy(
        impl->stride.begin(), impl->stride.begin() + axis + 1, new_stride.begin()
    );
    new_stride[axis + 1] = new_stride[axis] * factor;
    std::copy(
        impl->stride.begin() + axis + 1,
        impl->stride.end(),
        new_stride.begin() + axis + 2
    );

    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        impl->contiguous_dims + (axis <= impl->contiguous_dims)
    });
}

std::vector<Tensor> Tensor::split_and_reshape(const std::vector<Sizes>& shapes) const {
    check_impl();
    if (!is_contiguous() || shape().size() != 1) {
        throw std::runtime_error(
            "split_and_reshape is only available for single-dimensional contiguous "
            "tensors"
        );
    }
    SizeVec size_prods;
    size_prods.reserve(shapes.size());
    for (auto& shape : shapes) {
        size_prods.push_back(shape.product());
    }
    TensorVec split_tensors = split(0, size_prods);
    TensorVec ret;
    ret.reserve(shapes.size());
    for (auto [tensor, shape] : zip(split_tensors, shapes)) {
        ret.push_back(tensor.reshape(shape));
    }
    return ret;
}

std::size_t Tensor::init_stride() {
    std::size_t stride_prod = 1;
    bool first = true;
    for (auto size : impl->shape) {
        if (first && size == 1) {
            impl->stride.push_back(0);
        } else {
            impl->stride.push_back(stride_prod);
        }
        stride_prod *= size;
        first = false;
    }
    impl->contiguous_dims = impl->shape.size();
    return stride_prod * dtype_size();
}

Tensor Tensor::cast(DataType dtype) const {
    Tensor source = cpu().contiguous();
    if (source.dtype() == dtype) {
        return source;
    }
    Tensor target(dtype, source.shape());
    std::size_t count = source.shape().product();
    auto convert = [&]<typename From>(From* from) {
        switch (dtype) {
        case DataType::dt_int32:
            std::copy_n(from, count, static_cast<me_int_t*>(target.data()));
            break;
        case DataType::dt_float32:
            std::copy_n(from, count, static_cast<float*>(target.data()));
            break;
        case DataType::dt_float64:
            std::copy_n(from, count, static_cast<double*>(target.data()));
            break;
        default:
            throw std::invalid_argument("invalid dtype in cast");
        }
    };
    switch (source.dtype()) {
    case DataType::dt_int32:
        convert(static_cast<me_int_t*>(source.data()));
        break;
    case DataType::dt_float32:
        convert(static_cast<float*>(source.data()));
        break;
    case DataType::dt_float64:
        convert(static_cast<double*>(source.data()));
        break;
    default:
        throw std::invalid_argument("invalid dtype in cast");
    }
    return target;
}

Tensor Tensor::simd_padded_literal(
    const LiteralValue& value, std::size_t lanes, DevicePtr device
) {
    auto unpadded = [&]() -> Tensor {
        return std::visit(
            Overloaded{
                [&](const TensorValue& tensor_value) {
                    return Tensor(tensor_value, device);
                },
                [&](std::monostate) -> Tensor {
                    throw std::invalid_argument("empty literal");
                },
                [&](auto scalar) { return Tensor(scalar, device); },
            },
            value
        );
    };
    if (lanes <= 1) {
        return unpadded();
    }

    // shape (with the batch dimension of size one) and flat column-major values
    Sizes shape{1};
    std::variant<std::vector<me_int_t>, std::vector<double>> values;
    if (auto scalar = std::get_if<me_int_t>(&value)) {
        values = std::vector<me_int_t>{*scalar};
    } else if (auto scalar = std::get_if<double>(&value)) {
        values = std::vector<double>{*scalar};
    } else if (auto tensor_value = std::get_if<TensorValue>(&value)) {
        for (int size : std::get<0>(*tensor_value)) {
            shape.push_back(size);
        }
        values = std::get<1>(*tensor_value);
    } else {
        return unpadded();
    }
    std::size_t count = std::visit([](auto& vec) { return vec.size(); }, values);
    if (count * lanes > max_simd_padded_size) {
        return unpadded();
    }

    // element j of lane l is stored at l + lanes * j
    auto [dtype, buffer] = std::visit(
        [&]<typename T>(const std::vector<T>& vec) {
            auto dtype = dtype_of<T>();
            std::vector<T> padded(count * lanes);
            for (std::size_t j = 0; j < count; ++j) {
                std::fill_n(padded.begin() + j * lanes, lanes, vec[j]);
            }
            Tensor buffer(dtype, {count * lanes}, device);
            device->memcpy(buffer.data(), padded.data(), padded.size() * sizeof(T));
            return std::make_pair(dtype, buffer);
        },
        values
    );
    Sizes stride(shape.size());
    std::size_t stride_prod = lanes;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        stride[i] = stride_prod;
        stride_prod *= shape[i];
    }
    Tensor ret(new TensorImpl{
        dtype,
        shape,
        device,
        buffer.data(),
        false,
        std::nullopt,
        buffer.impl,
        1,
        stride,
        1
    });
    ret.impl->simd_padded = true;
    // non-negative integer scalars double as batch sizes, see the scalar constructor
    if (auto scalar = std::get_if<me_int_t>(&value); scalar && *scalar >= 0) {
        ret.impl->batch_sizes.push_back(*scalar);
    }
    return ret;
}
