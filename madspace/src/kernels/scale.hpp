#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

// Helper functions

template <typename T>
KERNELSPEC FVal<T> transverse_mass(FIn<T, 2> momenta) {
    FVal<T> mt_sum(0.);
    for (std::size_t i = 2; i < momenta.size(); ++i) {
        auto p = momenta[i];
        mt_sum = mt_sum + sqrt(max(0., p[0] * p[0] - p[3] * p[3]));
    }
    return mt_sum;
}

// Kernels

template <typename T>
KERNELSPEC void kernel_scale_transverse_energy(FIn<T, 2> momenta, FOut<T, 0> scale) {
    FVal<T> et_sum(0.);
    for (std::size_t i = 2; i < momenta.size(); ++i) {
        auto p = momenta[i];
        auto pt2 = p[1] * p[1] + p[2] * p[2];
        auto p2 = pt2 + p[3] * p[3];
        et_sum = et_sum + p[0] * sqrt(pt2 / p2);
    }
    scale = et_sum;
}

template <typename T>
KERNELSPEC void kernel_scale_transverse_mass(FIn<T, 2> momenta, FOut<T, 0> scale) {
    auto mt = transverse_mass<T>(momenta);
    scale = mt;
}

template <typename T>
KERNELSPEC void kernel_scale_half_transverse_mass(FIn<T, 2> momenta, FOut<T, 0> scale) {
    auto hmt = 0.5 * transverse_mass<T>(momenta);
    scale = hmt;
}

template <typename T>
KERNELSPEC void kernel_scale_partonic_energy(FIn<T, 2> momenta, FOut<T, 0> scale) {
    auto e_tot = momenta[0][0] + momenta[1][0];
    auto pz_tot = momenta[0][3] + momenta[1][3];
    auto epart = sqrt(e_tot * e_tot - pz_tot * pz_tot);
    scale = epart;
}

} // namespace kernels
} // namespace madspace
