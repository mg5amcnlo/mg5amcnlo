#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

// Kernels

template <typename T>
KERNELSPEC void kernel_cut_unphysical(
    FIn<T, 0> w_in, FIn<T, 2> p, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 0> w_out
) {
    FVal<T> w = where(isnan(w_in), 0., w_in);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto p_i = p[i];
        for (std::size_t j = 0; j < 4; ++j) {
            w = where(isnan(p_i[j]), 0., w);
        }
    }
    w_out = where(
        (x1 < 0.) | (x1 > 1.) | isnan(x1) | (x2 < 0.) | (x2 > 1.) | isnan(x2), 0., w
    );
}

template <typename T>
KERNELSPEC void
kernel_cut_one(FIn<T, 0> obs, FSIn<T, 0> min, FSIn<T, 0> max, FOut<T, 0> w) {
    w = where((obs < min) | (obs > max), FVal<T>(0.), 1.);
}

// The window [min, max] is a single pair of scalars applied to every observable of
// the vector.
template <typename T>
KERNELSPEC void
kernel_cut_all(FIn<T, 1> obs, FSIn<T, 0> min, FSIn<T, 0> max, FOut<T, 0> w) {
    FVal<T> cut = 1., min_val = min, max_val = max;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        FVal<T> obs_i = obs[i];
        cut = where((obs_i < min_val) | (obs_i > max_val), 0., cut);
    }
    w = cut;
}

template <typename T>
KERNELSPEC void
kernel_cut_any(FIn<T, 1> obs, FSIn<T, 0> min, FSIn<T, 0> max, FOut<T, 0> w) {
    FVal<T> cut = 0., min_val = min, max_val = max;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        FVal<T> obs_i = obs[i];
        cut = where((obs_i < min_val) | (obs_i > max_val), cut, 1.);
    }
    w = cut;
}

} // namespace kernels
} // namespace madspace
