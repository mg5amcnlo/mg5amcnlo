#pragma once

#include "definitions.hpp"
#include "madspace/constants.hpp"

namespace madspace {
namespace kernels {

// Kernels

template <typename T>
KERNELSPEC void kernel_copy(FIn<T, 0> in, FOut<T, 0> out) {
    out = in;
}

template <typename T>
KERNELSPEC void kernel_copy_int(IIn<T, 0> in, IOut<T, 0> out) {
    out = in;
}

template <typename T>
KERNELSPEC void kernel_zero(FIn<T, 0> in, FOut<T, 0> out) {
    out = 0.;
}

template <typename T>
KERNELSPEC void kernel_zero_int(IIn<T, 0> in, IOut<T, 0> out) {
    out = 0;
}

template <typename T>
KERNELSPEC void kernel_gather(IIn<T, 0> index, FIn<T, 1> choices, FOut<T, 0> output) {
    output = choices.gather(index);
}

template <typename T>
KERNELSPEC void backward_kernel_gather(
    IIn<T, 0> index,
    FIn<T, 0> output_grad,
    FOut<T, 0> index_grad,
    FOut<T, 1> choices_grad
) {
    choices_grad.scatter_add(index, output_grad);
}

template <typename T>
KERNELSPEC void
kernel_gather_int(IIn<T, 0> index, IIn<T, 1> choices, IOut<T, 0> output) {
    output = choices.gather(index);
}

template <typename T>
KERNELSPEC void
kernel_select_int(IIn<T, 1> input, IIn<T, 1> indices, IOut<T, 1> output) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        output[i] = input.gather(indices[i]);
    }
}

template <typename T>
KERNELSPEC void kernel_select(FIn<T, 1> input, IIn<T, 1> indices, FOut<T, 1> output) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        output[i] = input.gather(indices[i]);
    }
}

template <typename T>
KERNELSPEC void backward_kernel_select(
    IIn<T, 1> indices,
    FIn<T, 1> output_grad,
    FOut<T, 1> input_grad,
    FOut<T, 1> indices_grad
) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        input_grad.scatter_add(indices[i], output_grad[i]);
    }
}

template <typename T>
KERNELSPEC void
kernel_select_vector(FIn<T, 2> input, IIn<T, 1> indices, FOut<T, 2> output) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto output_i = output[i];
        auto input_i = input[single_index(indices[i])];
        for (std::size_t j = 0; j < output_i.size(); ++j) {
            output_i[j] = input_i[j];
        }
    }
}

template <typename T>
KERNELSPEC void kernel_argsort(FIn<T, 1> in, IOut<T, 1> out) {
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = i;
    }
    for (std::size_t i = 1; i < in.size(); ++i) {
        std::size_t index = out[i];
        FVal<T> ref = in[index];
        std::size_t j = i;
        for (; j > 0 && in[out[j - 1]] > ref; --j) {
            out[j] = out[j - 1];
        }
        out[j] = index;
    }
}

} // namespace kernels
} // namespace madspace
