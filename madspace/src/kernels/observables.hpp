#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// Helper functions

template <typename T>
KERNELSPEC FVal<T> eta(FIn<T, 1> p) {
    auto px = p[1], py = p[2], pz = p[3];
    auto p_mag = sqrt(px * px + py * py + pz * pz);
    auto eta = 0.5 * log((p_mag + pz) / (p_mag - pz));
    return where(p_mag < EPS, 99.0, eta);
}

template <typename T>
KERNELSPEC FVal<T> delta_phi(FIn<T, 1> p1, FIn<T, 1> p2) {
    auto dphi = atan2(p1[2], p1[1]) - atan2(p2[2], p2[1]);
    dphi = where(dphi >= PI, dphi - 2 * PI, dphi);
    dphi = where(dphi < -PI, dphi + 2 * PI, dphi);
    return dphi;
}

// Kernels

template <typename T>
KERNELSPEC void kernel_obs_sqrt_s(FIn<T, 2> p_ext, FOut<T, 0> obs) {
    FourMom<T> p_tot{
        p_ext[0][0] + p_ext[1][0],
        p_ext[0][1] + p_ext[1][1],
        p_ext[0][2] + p_ext[1][2],
        p_ext[0][3] + p_ext[1][3],
    };
    obs = sqrt(max(lsquare<T>(p_tot), 0.));
}

template <typename T>
KERNELSPEC void kernel_obs_e(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = p[0];
}

template <typename T>
KERNELSPEC void kernel_obs_px(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = p[1];
}

template <typename T>
KERNELSPEC void kernel_obs_py(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = p[2];
}

template <typename T>
KERNELSPEC void kernel_obs_pz(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = p[3];
}

template <typename T>
KERNELSPEC void kernel_obs_mass(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = sqrt(max(lsquare<T>(load_mom<T>(p)), 0.));
}

template <typename T>
KERNELSPEC void kernel_obs_pt(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = sqrt(p[1] * p[1] + p[2] * p[2]);
}

template <typename T>
KERNELSPEC void kernel_obs_p_mag(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
}

template <typename T>
KERNELSPEC void kernel_obs_phi(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = atan2(p[2], p[1]);
}

template <typename T>
KERNELSPEC void kernel_obs_theta(FIn<T, 1> p, FOut<T, 0> obs) {
    auto px = p[1], py = p[2], pz = p[3];
    obs = acos(min(1., max(-1., pz / sqrt(px * px + py * py + pz * pz))));
}

template <typename T>
KERNELSPEC void kernel_obs_y(FIn<T, 1> p, FOut<T, 0> obs) {
    auto e = p[0], px = p[1], py = p[2], pz = p[3];
    obs = 0.5 * log((e + pz) / (e - pz));
}

template <typename T>
KERNELSPEC void kernel_obs_y_abs(FIn<T, 1> p, FOut<T, 0> obs) {
    auto e = p[0], px = p[1], py = p[2], pz = p[3];
    obs = fabs(0.5 * log((e + pz) / (e - pz)));
}

template <typename T>
KERNELSPEC void kernel_obs_eta(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = eta<T>(p);
}

template <typename T>
KERNELSPEC void kernel_obs_eta_abs(FIn<T, 1> p, FOut<T, 0> obs) {
    obs = fabs(eta<T>(p));
}

template <typename T>
KERNELSPEC void kernel_obs_delta_eta(FIn<T, 1> p1, FIn<T, 1> p2, FOut<T, 0> obs) {
    obs = eta<T>(p1) - eta<T>(p2);
}

template <typename T>
KERNELSPEC void kernel_obs_delta_phi(FIn<T, 1> p1, FIn<T, 1> p2, FOut<T, 0> obs) {
    obs = delta_phi<T>(p1, p2);
}

template <typename T>
KERNELSPEC void kernel_obs_delta_r(FIn<T, 1> p1, FIn<T, 1> p2, FOut<T, 0> obs) {
    auto deta = eta<T>(p1) - eta<T>(p2);
    auto dphi = delta_phi<T>(p1, p2);
    obs = sqrt(deta * deta + dphi * dphi);
}

} // namespace kernels
} // namespace madspace
