#include "madspace/driver/tensor.hpp"

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
