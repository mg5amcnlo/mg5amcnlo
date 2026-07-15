#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_sample_discrete(
    FIn<T, 0> r, IIn<T, 0> option_count, IOut<T, 0> output, FOut<T, 0> det
) {
    IVal<T> opt_count_i(option_count);
    FVal<T> opt_count_f(opt_count_i);
    IVal<T> option(r * opt_count_f);
    output = option;
    det = opt_count_f;
}

template <typename T>
KERNELSPEC void kernel_sample_discrete_inverse(
    IIn<T, 0> index, IIn<T, 0> option_count, FOut<T, 0> r, FOut<T, 0> det
) {
    IVal<T> opt_count_i(option_count), index_i(index);
    FVal<T> opt_count_f(opt_count_i), index_f(index_i);
    r = (index_f + 0.5) / opt_count_f;
    det = 1. / opt_count_f;
}

template <typename T>
KERNELSPEC void kernel_sample_discrete_probs(
    FIn<T, 0> r, FIn<T, 1> probs, IOut<T, 0> output, FOut<T, 0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    FVal<T> cum_prob(0.), prob_out(0.);
    IVal<T> option(0);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / prob_norm;
        auto mask = r < cum_prob;
        cum_prob = cum_prob + prob;
        option = where(mask, option, IVal<T>(i));
        prob_out = where(mask, prob_out, prob);
    }
    output = option;
    det = 1. / prob_out;
}

template <typename T>
KERNELSPEC void kernel_sample_discrete_probs_inverse(
    IIn<T, 0> index, FIn<T, 1> probs, FOut<T, 0> r, FOut<T, 0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    FVal<T> cum_prob(0.), random(0.), prob_out(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / prob_norm;
        cum_prob = cum_prob + prob;
        auto mask = index == i;
        random = where(mask, cum_prob + 0.5 * prob, random);
        prob_out = where(mask, prob, prob_out);
    }
    r = random;
    det = prob_out;
}

template <typename T>
KERNELSPEC void backward_kernel_sample_discrete_probs_inverse(
    IIn<T, 0> index,
    FIn<T, 1> probs,
    FIn<T, 0> r_grad,
    FIn<T, 0> det_grad,
    IOut<T, 0> index_grad,
    FOut<T, 1> probs_grad
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    FVal<T> det_grad_out(0.);
    auto prob = probs.gather(index) / prob_norm;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs_grad[i] =
            (where(index == i, FVal<T>(1.), 0.) - prob) / prob_norm * det_grad;
    }
}

} // namespace kernels
} // namespace madspace
