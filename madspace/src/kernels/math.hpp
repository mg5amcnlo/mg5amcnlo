#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_add_inplace(FIn<T, 0> in, FOut<T, 0> out) {
    out += in;
}

template <typename T>
KERNELSPEC void kernel_add(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = in1 + in2;
}

template <typename T>
KERNELSPEC void backward_kernel_add(
    FIn<T, 0> in1,
    FIn<T, 0> in2,
    FIn<T, 0> out_grad,
    FOut<T, 0> in1_grad,
    FOut<T, 0> in2_grad
) {
    in1_grad += out_grad;
    in2_grad += out_grad;
}

template <typename T>
KERNELSPEC void kernel_add_int(IIn<T, 0> in1, IIn<T, 0> in2, IOut<T, 0> out) {
    out = in1 + in2;
}

template <typename T>
KERNELSPEC void kernel_sub(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = in1 - in2;
}

template <typename T>
KERNELSPEC void backward_kernel_sub(
    FIn<T, 0> in1,
    FIn<T, 0> in2,
    FIn<T, 0> out_grad,
    FOut<T, 0> in1_grad,
    FOut<T, 0> in2_grad
) {
    in1_grad += out_grad;
    in2_grad += -out_grad;
}

template <typename T>
KERNELSPEC void kernel_mul(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = in1 * in2;
}

template <typename T>
KERNELSPEC void backward_kernel_mul(
    FIn<T, 0> in1,
    FIn<T, 0> in2,
    FIn<T, 0> out_grad,
    FOut<T, 0> in1_grad,
    FOut<T, 0> in2_grad
) {
    in1_grad += out_grad * in2;
    in2_grad += out_grad * in1;
}

template <typename T>
KERNELSPEC void kernel_div(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = in1 / in2;
}

template <typename T>
KERNELSPEC void backward_kernel_div(
    FIn<T, 0> in1,
    FIn<T, 0> in2,
    FIn<T, 0> out_grad,
    FOut<T, 0> in1_grad,
    FOut<T, 0> in2_grad
) {
    in1_grad += out_grad / in2;
    in2_grad += -out_grad * in1 / (in2 * in2);
}

template <typename T>
KERNELSPEC void kernel_reduce_sum(FIn<T, 1> in, FOut<T, 0> out) {
    FVal<T> sum(0.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        sum = sum + in[i];
    }
    out = sum;
}

template <typename T>
KERNELSPEC void kernel_reduce_sum_vector(FIn<T, 2> in, FOut<T, 1> out) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        FVal<T> sum(0.);
        for (std::size_t j = 0; j < in.size(); ++j) {
            sum = sum + in[j][i];
        }
        out[i] = sum;
    }
}

template <typename T>
KERNELSPEC void kernel_reduce_product(FIn<T, 1> in, FOut<T, 0> out) {
    FVal<T> product(1.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        product = product * in[i];
    }
    out = product;
}

template <typename T>
KERNELSPEC void
backward_kernel_reduce_product(FIn<T, 1> in, FIn<T, 0> out_grad, FOut<T, 1> in_grad) {
    FVal<T> product(1.);
    IVal<T> zero_count(0);
    for (std::size_t i = 0; i < in.size(); ++i) {
        FVal<T> val = in[i];
        auto zero_val = val == 0.;
        product = product * where(zero_val & (zero_count == 0), 1., val);
    }
    auto zero_product = where(zero_count == 0, product, 0.);
    for (std::size_t i = 0; i < in.size(); ++i) {
        FVal<T> val = in[i];
        auto zero_val = val == 0.;
        in_grad[i] += out_grad * where(val == 0., product, zero_product / val);
    }
}

template <typename T>
KERNELSPEC void kernel_sqrt(FIn<T, 0> in, FOut<T, 0> out) {
    out = sqrt(where(in > 0., in, 0.));
}

template <typename T>
KERNELSPEC void
backward_kernel_sqrt(FIn<T, 0> in, FIn<T, 0> out_grad, FOut<T, 0> in_grad) {
    in_grad += where(in > 0., -0.5 * out_grad / sqrt(in), 0.);
}

template <typename T>
KERNELSPEC void kernel_square(FIn<T, 0> in, FOut<T, 0> out) {
    out = in * in;
}

template <typename T>
KERNELSPEC void
backward_kernel_square(FIn<T, 0> in, FIn<T, 0> out_grad, FOut<T, 0> in_grad) {
    in_grad += 2. * in * out_grad;
}

template <typename T>
KERNELSPEC void kernel_min(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = min(in1, in2);
}

template <typename T>
KERNELSPEC void kernel_max(FIn<T, 0> in1, FIn<T, 0> in2, FOut<T, 0> out) {
    out = max(in1, in2);
}

} // namespace kernels
} // namespace madspace
