#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_uniform_invariant(
    FIn<T, 0> r, FIn<T, 0> s_min, FIn<T, 0> s_max, FOut<T, 0> s, FOut<T, 0> gs
) {
    gs = s_max - s_min;
    s = s_min + gs * r;
}

template <typename T>
KERNELSPEC void kernel_uniform_invariant_inverse(
    FIn<T, 0> s, FIn<T, 0> s_min, FIn<T, 0> s_max, FOut<T, 0> r, FOut<T, 0> gs
) {
    gs = 1 / (s_max - s_min);
    r = (s - s_min) * gs;
}

template <typename T>
KERNELSPEC void kernel_breit_wigner_invariant(
    FIn<T, 0> r,
    FSIn<T, 0> mass,
    FSIn<T, 0> width,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> s,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass;
    auto gm = mass * width;
    auto y1 = atan((s_min - m2) / gm);
    auto y2 = atan((s_max - m2) / gm);
    auto dy21 = y2 - y1;
    auto _s = gm * tan(y1 + dy21 * r) + m2;
    auto s_sub_m2 = _s - m2;

    s = _s;
    gs = dy21 * (s_sub_m2 * s_sub_m2 + gm * gm) / gm;
}

template <typename T>
KERNELSPEC void kernel_breit_wigner_invariant_inverse(
    FIn<T, 0> s,
    FSIn<T, 0> mass,
    FSIn<T, 0> width,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> r,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass;
    auto gm = mass * width;
    auto y1 = atan((s_min - m2) / gm);
    auto y2 = atan((s_max - m2) / gm);
    auto dy21 = y2 - y1;
    auto s_sub_m2 = s - m2;

    r = (atan(s_sub_m2 / gm) - y1) / dy21;
    gs = gm / (dy21 * (s_sub_m2 * s_sub_m2 + gm * gm));
}

// Breit-Wigner density with its peak flattened: 1/((s-m^2)^2+(m*w)^2) outside
// the window (m -+ window*w)^2, and inside it the constant average of the two
// edge values. Used for the $-excluded propagators, whose matrix element
// vanishes in that window: the channel keeps a (flat) coverage of it without
// piling its points up at the pole.
template <typename T>
KERNELSPEC void flat_window_segments(
    FVal<T> mass,
    FVal<T> width,
    FVal<T> window,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FVal<T>& m2,
    FVal<T>& gm,
    FVal<T>& a,
    FVal<T>& b,
    FVal<T>& c,
    FVal<T>& y_min,
    FVal<T>& y_b,
    FVal<T>& w1,
    FVal<T>& w2,
    FVal<T>& w_tot
) {
    m2 = mass * mass;
    gm = mass * width;
    auto e_lo = mass - window * width;
    auto lo = where(e_lo > 0., e_lo * e_lo, 0.);
    auto e_hi = mass + window * width;
    auto hi = e_hi * e_hi;
    a = where(lo < s_min, s_min, where(lo > s_max, s_max, lo));
    b = where(hi < s_min, s_min, where(hi > s_max, s_max, hi));
    auto d_lo = lo - m2, d_hi = hi - m2;
    c = 0.5 * (1. / (d_lo * d_lo + gm * gm) + 1. / (d_hi * d_hi + gm * gm));
    y_min = atan((s_min - m2) / gm);
    auto y_a = atan((a - m2) / gm);
    y_b = atan((b - m2) / gm);
    auto y_max = atan((s_max - m2) / gm);
    w1 = (y_a - y_min) / gm;
    w2 = c * (b - a);
    w_tot = w1 + w2 + (y_max - y_b) / gm;
}

template <typename T>
KERNELSPEC void kernel_flat_window_breit_wigner_invariant(
    FIn<T, 0> r,
    FSIn<T, 0> mass,
    FSIn<T, 0> width,
    FSIn<T, 0> window,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> s,
    FOut<T, 0> gs
) {
    FVal<T> m2, gm, a, b, c, y_min, y_b, w1, w2, w_tot;
    flat_window_segments<T>(
        mass, width, window, s_min, s_max, m2, gm, a, b, c, y_min, y_b, w1, w2, w_tot
    );
    auto t = r * w_tot;
    auto s_bw_lo = gm * tan(y_min + t * gm) + m2;
    auto s_flat = a + (t - w1) / c;
    auto s_bw_hi = gm * tan(y_b + (t - w1 - w2) * gm) + m2;
    auto in_lo = t < w1;
    auto in_flat = t < w1 + w2;
    auto _s = where(in_lo, s_bw_lo, where(in_flat, s_flat, s_bw_hi));
    auto d = _s - m2;
    auto f =
        where(in_lo, 1. / (d * d + gm * gm), where(in_flat, c, 1. / (d * d + gm * gm)));
    s = _s;
    gs = w_tot / f;
}

template <typename T>
KERNELSPEC void kernel_flat_window_breit_wigner_invariant_inverse(
    FIn<T, 0> s,
    FSIn<T, 0> mass,
    FSIn<T, 0> width,
    FSIn<T, 0> window,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> r,
    FOut<T, 0> gs
) {
    FVal<T> m2, gm, a, b, c, y_min, y_b, w1, w2, w_tot;
    flat_window_segments<T>(
        mass, width, window, s_min, s_max, m2, gm, a, b, c, y_min, y_b, w1, w2, w_tot
    );
    auto d = s - m2;
    auto y = atan(d / gm);
    auto bw = 1. / (d * d + gm * gm);
    auto in_lo = s < a;
    auto in_flat = s < b;
    auto cdf = where(
        in_lo,
        (y - y_min) / gm,
        where(in_flat, w1 + c * (s - a), w1 + w2 + (y - y_b) / gm)
    );
    auto f = where(in_lo, bw, where(in_flat, c, bw));
    r = cdf / w_tot;
    gs = f / w_tot;
}

template <typename T>
KERNELSPEC void kernel_stable_invariant(
    FIn<T, 0> r,
    FSIn<T, 0> mass,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> s,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass - 1e-2;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;

    s = pow(q_max, r) * pow(q_min, 1 - r) + m2;
    gs = (s - m2) * log(q_max / q_min);
}

template <typename T>
KERNELSPEC void kernel_stable_invariant_inverse(
    FIn<T, 0> s,
    FSIn<T, 0> mass,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> r,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass - 1e-2;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto q = s - m2;
    auto log_q_max_min = log(q_max / q_min);

    r = log(q / q_min) / log_q_max_min;
    gs = 1 / (q * log_q_max_min);
}

template <typename T>
KERNELSPEC void kernel_stable_invariant_nu(
    FIn<T, 0> r,
    FSIn<T, 0> mass,
    FSIn<T, 0> nu,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> s,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass - 1e-2;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto power = 1.0 - nu;
    auto qmaxpow = pow(q_max, power);
    auto qminpow = pow(q_min, power);
    auto _s = pow(r * qmaxpow + (1 - r) * qminpow, 1 / power) + m2;

    s = _s;
    gs = (qmaxpow - qminpow) * pow(_s - m2, nu) / power;
}

template <typename T>
KERNELSPEC void kernel_stable_invariant_nu_inverse(
    FIn<T, 0> s,
    FSIn<T, 0> mass,
    FSIn<T, 0> nu,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FOut<T, 0> r,
    FOut<T, 0> gs
) {
    auto m2 = mass * mass - 1e-2;
    auto q = s - m2;
    auto q_max = s_max - m2;
    auto q_min = s_min - m2;
    auto power = 1.0 - nu;
    auto qpow = pow(q, power);
    auto qmaxpow = pow(q_max, power);
    auto qminpow = pow(q_min, power);
    auto dqpow = qmaxpow - qminpow;

    r = (qpow - qminpow) / dqpow;
    gs = power / (dqpow * pow(q, nu));
}

} // namespace kernels
} // namespace madspace
