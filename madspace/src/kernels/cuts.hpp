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
kernel_cut_one(FIn<T, 0> obs, FIn<T, 0> min, FIn<T, 0> max, FOut<T, 0> w) {
    w = where((obs < min) | (obs > max), FVal<T>(0.), 1.);
}

template <typename T>
KERNELSPEC void
kernel_cut_all(FIn<T, 1> obs, FIn<T, 1> min, FIn<T, 1> max, FOut<T, 0> w) {
    FVal<T> cut = 1.;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        cut = where((obs[i] < min[i]) | (obs[i] > max[i]), 0., cut);
    }
    w = cut;
}

template <typename T>
KERNELSPEC void
kernel_cut_any(FIn<T, 1> obs, FIn<T, 1> min, FIn<T, 1> max, FOut<T, 0> w) {
    FVal<T> cut = 0.;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        cut = where((obs[i] < min[i]) | (obs[i] > max[i]), cut, 1.);
    }
    w = cut;
}

} // namespace kernels
} // namespace madspace
