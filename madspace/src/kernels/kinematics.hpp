#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template <typename T>
struct FourMom {
    KERNELSPEC FVal<T>& operator[](int i) { return p[i]; }
    KERNELSPEC const FVal<T>& operator[](int i) const { return p[i]; }
    FVal<T> p[4];
};

template <typename F, typename S>
struct Pair {
    F first;
    S second;
};

template <typename A, typename B, typename C>
struct Triplet {
    A first;
    B second;
    C third;
};

template <typename A, typename B, typename C, typename D>
struct Quartuplet {
    A first;
    B second;
    C third;
    D fourth;
};

template <typename A, typename B, typename C, typename D, typename E, typename F>
struct Sextuplet {
    A first;
    B second;
    C third;
    D fourth;
    E fifth;
    F sixth;
};

template <
    typename A,
    typename B,
    typename C,
    typename D,
    typename E,
    typename F,
    typename G,
    typename H,
    typename I,
    typename J>
struct Decuplet {
    A first;
    B second;
    C third;
    D fourth;
    E fifth;
    F sixth;
    G seventh;
    H eighth;
    I ninth;
    J tenth;
};

template <typename T>
KERNELSPEC FourMom<T> load_mom(FIn<T, 1> p) {
    return {p[0], p[1], p[2], p[3]};
}

template <typename T>
KERNELSPEC void store_mom(FOut<T, 1> p_to, FourMom<T> p_from) {
    for (int i = 0; i < 4; ++i) {
        p_to[i] = p_from[i];
    }
}

template <typename T>
KERNELSPEC FVal<T> kaellen(FVal<T> x, FVal<T> y, FVal<T> z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>>
t_inv_min_max(FVal<T> s, FVal<T> ma_2, FVal<T> mb_2, FVal<T> m1_2, FVal<T> m2_2) {
    auto ysqr = kaellen<T>(s, ma_2, mb_2) * kaellen<T>(s, m1_2, m2_2);
    auto yr = where(ysqr > EPS, sqrt(ysqr), EPS);
    auto m_sum = ma_2 + m1_2;
    auto prod = (s + ma_2 - mb_2) * (s + m1_2 - m2_2);
    auto s_eps = s + EPS;
    auto y1 = m_sum - 0.5 * (prod - yr) / s_eps;
    auto y2 = m_sum - 0.5 * (prod + yr) / s_eps;
    auto t_min_tmp = -max(y2, y1);
    auto t_max_tmp = -min(y1, y2);
    auto t_min = max(t_min_tmp, 0.);
    // an empty range gets a width of a few ulps rather than an absolute EPS,
    // which is below one ulp of t for |t| > ~1e4 and left max == min
    auto t_max = where(
        t_max_tmp > t_min, t_max_tmp, t_min + max(FVal<T>(EPS), 1e-14 * fabs(t_min))
    );
    return {t_min, t_max};
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>>
t1_inv_min_max_doublet(FVal<T> s, FVal<T> m1_2, FVal<T> mir_min_2) {
    // |t1| bounds for pa+pb -> p_i + p_ir with massless pa,pb, fixed m_i,
    // and m_ir >= mir_min. Same shape as standard t bounds with
    // ma=mb=0, m2=mir_min.
    return t_inv_min_max<T>(s, 0., 0., m1_2, mir_min_2);
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>>
t2_inv_min_max_doublet(FVal<T> s, FVal<T> m1_2, FVal<T> mir_min_2, FVal<T> t1_abs) {
    // |t2| bounds given fixed s, m_i, mir_min, |t1|.
    // Derivation (abs-value convention): from the signed-t formulas in the
    // Fortran double_t, flip both signs.
    auto denom = m1_2 + t1_abs + EPS;
    auto t2_min_raw = m1_2 * (s - t1_abs - m1_2) / denom;
    auto t2_max_raw = s - t1_abs - m1_2 - mir_min_2;
    auto t2_min = max(t2_min_raw, 0.);
    auto t2_max = where(t2_max_raw > t2_min, t2_max_raw, t2_min + EPS);
    return {t2_min, t2_max};
}

// ETmin floor on the double-t central single particle (mass^2 m1_2) and its recoil
// system, ported from AmpliCol double_t. AmpliCol writes signed-t (lines 605-608);
// we work in |t| = -t, so its tmin=max/tmax=min map to our |t|_min/|t|_max with both
// edges negated.
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> doublet_t1_etmin_clamp(
    FVal<T> t_min,
    FVal<T> t_max,
    FVal<T> s,
    FVal<T> m1_2,
    FVal<T> etmin_i,
    FVal<T> etmin_ir
) {
    auto sqrt_s = sqrt(max(s, EPS));
    auto ei2 = etmin_i * etmin_i;
    auto eir2 = etmin_ir * etmin_ir;
    auto lam = s * s + ei2 * ei2 + eir2 * eir2 - 2.0 * s * ei2 - 2.0 * ei2 * eir2 -
        2.0 * eir2 * s;
    auto pzmax = sqrt(max(lam, 0.)) / (2.0 * sqrt_s);
    auto Eimax = sqrt_s - sqrt(max(eir2 + pzmax * pzmax, 0.));
    auto cut_min = sqrt_s * (Eimax - pzmax) - m1_2;
    auto cut_max = sqrt_s * (Eimax + pzmax) - m1_2;
    auto active = (etmin_i > EPS) | (etmin_ir > EPS);
    auto lo = where(active, max(t_min, cut_min), t_min);
    auto hi = where(active, min(t_max, cut_max), t_max);
    auto ok = hi > lo;
    return {where(ok, lo, t_min), where(ok, hi, t_max)};
}

// ETmin floor on |t2| given the sampled |t1| (= t1_abs). AmpliCol double_t lines
// 625-626 (signed t).
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> doublet_t2_etmin_clamp(
    FVal<T> t_min,
    FVal<T> t_max,
    FVal<T> s,
    FVal<T> m1_2,
    FVal<T> t1_abs,
    FVal<T> etmin_i,
    FVal<T> etmin_ir
) {
    auto ei2 = etmin_i * etmin_i;
    auto eir2 = etmin_ir * etmin_ir;
    auto denom2 = s - t1_abs - m1_2;
    auto cut_min = s * ei2 / max(m1_2 + t1_abs, EPS) - m1_2;
    auto cut_max = s * (1.0 - eir2 / max(denom2, EPS)) - m1_2;
    auto active = (etmin_i > EPS) | (etmin_ir > EPS);
    auto lo = where(active, max(t_min, cut_min), t_min);
    auto hi = where(active & (denom2 > EPS), min(t_max, cut_max), t_max);
    auto ok = hi > lo;
    return {where(ok, lo, t_min), where(ok, hi, t_max)};
}

template <typename T>
KERNELSPEC FVal<T> lsquare(FourMom<T> p) {
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
}

template <typename T>
KERNELSPEC FVal<T> esquare(FourMom<T> p) {
    return p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
}

// Rotation R(q) that takes the z axis onto the direction of q, used to set up
// the scattering frame around an incoming momentum. For q in the upper
// hemisphere it is the minimal rotation about z x q,
//
//         [ 1 - a^2 h   -a b h     a ]
//   R_+ = [ -a b h     1 - b^2 h   b ],   (a, b, c) = q / |q|,  h = 1 / (1 + c),
//         [ -a         -b          c ]
//
// and in the lower one F R_+(F q) with F = diag(-1, 1, -1), the rotation by pi
// about y. Both are smooth across q parallel to the z axis, where they reduce
// to the identity and to F. The frame therefore moves only at rounding level
// when q does, so forward and inverse agree on the azimuth even when they
// build q from differently rounded momenta. (The previous construction switched
// between an azimuth set by atan2(q_y, q_x) and a fixed one on the exact test
// q_t == 0, and turned the azimuth by an arbitrary angle, often pi/2, whenever
// the two sides rounded q_t differently.)
template <typename T>
KERNELSPEC FVal<T> ldot(FourMom<T> p1, FourMom<T> p2) {
    return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] - p1[3] * p2[3];
}

template <typename T>
KERNELSPEC FVal<T> edot(FourMom<T> p1, FourMom<T> p2) {
    return p1[1] * p2[1] + p1[2] * p2[2] + p1[3] * p2[3];
}

template <typename T>
KERNELSPEC FourMom<T> rotate(FourMom<T> p, FourMom<T> q) {
    auto qq = sqrt(max(esquare<T>(q), FVal<T>(1e-300)));
    auto neg = q[3] < 0.;
    auto a = where(neg, -q[1], q[1]) / qq;
    auto b = q[2] / qq;
    auto c = where(neg, -q[3], q[3]) / qq;
    auto h = 1. / (1. + c);
    auto x = (1. - a * a * h) * p[1] - a * b * h * p[2] + a * p[3];
    auto y = -a * b * h * p[1] + (1. - b * b * h) * p[2] + b * p[3];
    auto z = -a * p[1] - b * p[2] + c * p[3];
    return {p[0], where(neg, -x, x), y, where(neg, -z, z)};
}

// Inverse of `rotate`: R(q)^T.
template <typename T>
KERNELSPEC FourMom<T> rotate_inverse(FourMom<T> p, FourMom<T> q) {
    auto qq = sqrt(max(esquare<T>(q), FVal<T>(1e-300)));
    auto neg = q[3] < 0.;
    auto a = where(neg, -q[1], q[1]) / qq;
    auto b = q[2] / qq;
    auto c = where(neg, -q[3], q[3]) / qq;
    auto h = 1. / (1. + c);
    auto px = where(neg, -p[1], p[1]);
    auto pz = where(neg, -p[3], p[3]);
    return {
        p[0],
        (1. - a * a * h) * px - a * b * h * p[2] - a * pz,
        -a * b * h * px + (1. - b * b * h) * p[2] - b * pz,
        a * px + b * p[2] + c * pz
    };
}

// Orthonormal frame (x_hat, y_hat, z_hat), spatial components 1..3, with z_hat
// along q_z and x_hat along the part of q_x perpendicular to it. Shared by
// rotate_two_ref and its inverse so that both build the identical frame.
//
// Both directions are normalised by their own length first, so the frame
// does not depend on the scale of q_x: a soft q_x (a soft recoil momentum in
// the 2->3 block) used to hit an absolute floor on |q_x perp| that left x_hat
// shorter than 1, the rotation shrank p, and the daughters came out far off
// shell. The perpendicular part gets a second Gram-Schmidt pass: when q_x is
// close to z_hat the first one leaves a component along z_hat of relative
// size eps / sin(angle), the frame is not orthonormal, and the rotation
// changes |p|. When q_x has no usable perpendicular part (parallel to q_z to
// ~1e-14, or zero) the azimuth is undefined, and a fixed axis perpendicular
// to z_hat is taken instead.
template <typename T>
KERNELSPEC Triplet<FourMom<T>, FourMom<T>, FourMom<T>>
two_ref_frame(FourMom<T> q_z, FourMom<T> q_x) {
    auto qz_n = sqrt(max(esquare<T>(q_z), FVal<T>(1e-300)));
    auto zx = q_z[1] / qz_n, zy = q_z[2] / qz_n, zz = q_z[3] / qz_n;

    auto qx_n = sqrt(max(esquare<T>(q_x), FVal<T>(1e-300)));
    auto ux = q_x[1] / qx_n, uy = q_x[2] / qx_n, uz = q_x[3] / qx_n;
    auto u_dot_z = ux * zx + uy * zy + uz * zz;
    auto rx0 = ux - u_dot_z * zx;
    auto ry0 = uy - u_dot_z * zy;
    auto rz0 = uz - u_dot_z * zz;
    auto r_dot_z = rx0 * zx + ry0 * zy + rz0 * zz;
    auto rx = rx0 - r_dot_z * zx;
    auto ry = ry0 - r_dot_z * zy;
    auto rz = rz0 - r_dot_z * zz;
    auto rn2 = rx * rx + ry * ry + rz * rz;

    // fallback: the world x (or y, if z_hat is close to x) axis, made
    // perpendicular to z_hat
    auto use_y = fabs(zx) > 0.9;
    auto ex = where(use_y, FVal<T>(0.), FVal<T>(1.));
    auto ey = where(use_y, FVal<T>(1.), FVal<T>(0.));
    auto e_dot_z = ex * zx + ey * zy;
    auto fx = ex - e_dot_z * zx;
    auto fy = ey - e_dot_z * zy;
    auto fz = -e_dot_z * zz;
    auto degenerate = rn2 < 1e-28;
    rx = where(degenerate, fx, rx);
    ry = where(degenerate, fy, ry);
    rz = where(degenerate, fz, rz);
    auto rn = sqrt(max(rx * rx + ry * ry + rz * rz, FVal<T>(1e-300)));
    auto xx = rx / rn, xy = ry / rn, xz = rz / rn;

    // y_hat = z_hat x x_hat
    auto yx = zy * xz - zz * xy;
    auto yy = zz * xx - zx * xz;
    auto yz = zx * xy - zy * xx;
    return {
        FourMom<T>{FVal<T>(0.), xx, xy, xz},
        FourMom<T>{FVal<T>(0.), yx, yy, yz},
        FourMom<T>{FVal<T>(0.), zx, zy, zz}
    };
}

template <typename T>
KERNELSPEC FourMom<T> rotate_two_ref(FourMom<T> p, FourMom<T> q_z, FourMom<T> q_x) {
    // Forward rotation: take p from the canonical frame (e_z along q_z, e_x
    // along the part of q_x perpendicular to it, see two_ref_frame) into the
    // lab frame. Energy unchanged.
    //
    // Used by kernel_two_to_three_particle_scattering: q_z = pa_com,
    // q_x = p3 boosted into the p_12 rest frame. Picks a unique azimuth
    // so that the (q_z, q_x) plane is the phi=0 half-plane.
    auto frame = two_ref_frame<T>(q_z, q_x);
    auto x = frame.first, y = frame.second, z = frame.third;
    // world spatial = p[1]*x_hat + p[2]*y_hat + p[3]*z_hat
    return FourMom<T>{
        p[0],
        p[1] * x[1] + p[2] * y[1] + p[3] * z[1],
        p[1] * x[2] + p[2] * y[2] + p[3] * z[2],
        p[1] * x[3] + p[2] * y[3] + p[3] * z[3],
    };
}

template <typename T>
KERNELSPEC FourMom<T>
rotate_two_ref_inverse(FourMom<T> p, FourMom<T> q_z, FourMom<T> q_x) {
    // Inverse of rotate_two_ref: take a vector p from the lab frame into the
    // canonical frame defined by (q_z, q_x). Energy unchanged.
    auto frame = two_ref_frame<T>(q_z, q_x);
    auto x = frame.first, y = frame.second, z = frame.third;
    // canonical components = world spatial . (x_hat, y_hat, z_hat)
    return FourMom<T>{
        p[0],
        p[1] * x[1] + p[2] * x[2] + p[3] * x[3],
        p[1] * y[1] + p[2] * y[2] + p[3] * y[3],
        p[1] * z[1] + p[2] * z[2] + p[3] * z[3],
    };
}

template <typename T>
KERNELSPEC FourMom<T> boost(FourMom<T> k, FourMom<T> p_boost, FVal<T> sign) {
    // Perform the boost
    // This is in fact a numerically more stable implementation than often used
    auto p2_boost = lsquare<T>(p_boost);
    auto rsq = sqrt(max(EPS2, p2_boost));
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    return FourMom<T>{
        e, k[1] + c1 * p_boost[1], k[2] + c1 * p_boost[2], k[3] + c1 * p_boost[3]
    };
}

template <typename T>
KERNELSPEC FourMom<T> boost_light_cone(
    FourMom<T> k, FVal<T> k_mass2, FourMom<T> p_boost, FVal<T> p_mass2, FVal<T> sign
) {
    // Same boost as `boost`, for a momentum k of known mass^2 k_mass2, written
    // in light-cone components along the boost axis n:
    //   k_+ -> e^{+y} k_+,  k_- -> e^{-y} k_-,  k_perp unchanged,
    //   e^{y} = (p0 + |p|) / m.
    // In `boost` a momentum thrown against a boost with gamma >> 1 comes out of
    // the cancellation of terms of order gamma E and loses a factor gamma^2 of
    // its relative precision, which leaves light-like momenta off shell by up
    // to a few 1e-6 of their energy in the long color-ordered chains. Here the
    // smaller of k_+, k_- is taken from the mass shell,
    // k_+ k_- = k_mass2 + k_perp^2, both boosted components are positive
    // products, and the result is accurate to a few ulps of its own energy,
    // including its mass. p_mass2 is the mass^2 of p_boost, as the caller has
    // already computed it, so that the boost is the one of the rest frame the
    // caller built k in.
    // |p| only sets the direction n; an absolute floor here (EPS2) left n
    // shorter than 1 for a system almost at rest, |p| < 1e-12
    auto p_mag = sqrt(max(esquare<T>(p_boost), FVal<T>(1e-300)));
    auto m_p = sqrt(max(p_mass2, EPS2));
    auto e_plus = (p_boost[0] + p_mag) / m_p;
    auto e_minus = m_p / (p_boost[0] + p_mag);
    auto exp_y = where(sign > 0., e_plus, e_minus);
    auto exp_my = where(sign > 0., e_minus, e_plus);

    auto nx = p_boost[1] / p_mag, ny = p_boost[2] / p_mag, nz = p_boost[3] / p_mag;
    auto k_par = k[1] * nx + k[2] * ny + k[3] * nz;
    auto kx = k[1] - k_par * nx, ky = k[2] - k_par * ny, kz = k[3] - k_par * nz;
    auto trans2 = max(k_mass2 + kx * kx + ky * ky + kz * kz, 0.);
    auto k_plus_direct = k[0] + k_par;
    auto k_minus_direct = k[0] - k_par;
    auto forward = k_par >= 0.;
    auto k_plus =
        where(forward, k_plus_direct, trans2 / max(k_minus_direct, FVal<T>(EPS2)));
    auto k_minus =
        where(forward, trans2 / max(k_plus_direct, FVal<T>(EPS2)), k_minus_direct);

    auto k_plus_new = exp_y * k_plus;
    auto k_minus_new = exp_my * k_minus;
    auto k_par_new = 0.5 * (k_plus_new - k_minus_new);
    return FourMom<T>{
        0.5 * (k_plus_new + k_minus_new),
        kx + k_par_new * nx,
        ky + k_par_new * ny,
        kz + k_par_new * nz
    };
}

// Lab momenta of a two-body system p_tot -> k1 + k2 given back to back in its
// rest frame, with k1 + k2 = p_tot by construction. The softer of the two is
// boosted with boost_light_cone on its own mass shell and the harder one is
// p_tot minus it. The harder one carries at least half the energy, so it keeps
// the relative precision of p_tot; taking the softer one as the difference
// instead (or boosting it with `boost`) leaves it off shell when it is thrown
// backwards against a fast p_tot.
template <typename T>
KERNELSPEC Pair<FourMom<T>, FourMom<T>> boost_two_body(
    FourMom<T> k1_rest,
    FVal<T> k1_mass2,
    FourMom<T> k2_rest,
    FVal<T> k2_mass2,
    FourMom<T> p_tot,
    FVal<T> p_tot_mass2
) {
    auto k1 = boost_light_cone<T>(k1_rest, k1_mass2, p_tot, p_tot_mass2, 1.);
    auto k2 = boost_light_cone<T>(k2_rest, k2_mass2, p_tot, p_tot_mass2, 1.);
    auto k1_soft = k1[0] <= k2[0];
    FourMom<T> k1_out, k2_out;
    for (int i = 0; i < 4; ++i) {
        k1_out[i] = where(k1_soft, k1[i], p_tot[i] - k2[i]);
        k2_out[i] = where(k1_soft, p_tot[i] - k1[i], k2[i]);
    }
    return {k1_out, k2_out};
}

template <typename T>
KERNELSPEC void boost_beam(
    FIn<T, 2> q,
    FSIn<T, 1> masses,
    FVal<T> x1,
    FVal<T> x2,
    FVal<T> sign,
    FOut<T, 2> p_out
) {
    // Boost along z with rapidity y = log(x1 / x2) / 2. The first two momenta
    // are the incoming ones.
    //
    // Written as E cosh y + p_z sinh y, a momentum moving against the boost
    // comes out of the cancellation of terms of order e^{|y|} E and loses up
    // to a factor e^{2|y|} of relative precision; light-like momenta ended up
    // off shell by up to a few 1e-5 of their energy in the lab frame. The boost
    // is done in light-cone components instead, q_+ = E + p_z -> e^{+y} q_+,
    // q_- = E - p_z -> e^{-y} q_-, with the smaller of q_+, q_- taken from the
    // mass shell, q_+ q_- = m^2 + p_T^2, and the known mass of the particle (a
    // negative entry means unknown, and the mass is read off the momentum).
    // Both components then keep their relative precision whatever the
    // direction of the momentum and of the boost, also out of the lab frame.
    // Only momenta on their mass shell to rounding are projected; one that is
    // not (a degenerate point, or partonic momenta that carry larger errors
    // from elsewhere) keeps the linear boost.
    auto exp_rap = sqrt(x1 / x2);
    auto exp_rap_inv = sqrt(x2 / x1);
    auto e_plus = where(sign > 0., exp_rap, exp_rap_inv);
    auto e_minus = where(sign > 0., exp_rap_inv, exp_rap);
    auto cosh_rap = 0.5 * (e_plus + e_minus);
    auto sinh_rap = 0.5 * (e_plus - e_minus);
    // light-cone components of the outgoing momenta, summed directly and from
    // the light-cone components the boost uses
    FVal<T> sum_plus_direct(0.), sum_minus_direct(0.), sum_plus(0.), sum_minus(0.);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto q_i = q[i];
        auto p_out_i = p_out[i];
        FVal<T> mass(masses[i]);
        FVal<T> q0(q_i[0]), q1(q_i[1]), q2(q_i[2]), q3(q_i[3]);
        auto lin_0 = q0 * cosh_rap + q3 * sinh_rap;
        auto lin_3 = q3 * cosh_rap + q0 * sinh_rap;

        auto pt2 = q1 * q1 + q2 * q2;
        auto m2 = where(mass >= 0., mass * mass, max(q0 * q0 - q3 * q3 - pt2, 0.));
        auto trans2 = m2 + pt2;
        auto q_plus_direct = q0 + q3;
        auto q_minus_direct = q0 - q3;
        auto on_shell =
            fabs(q_plus_direct * q_minus_direct - trans2) <= 1e-13 * q0 * q0;
        auto light_cone = (q_plus_direct >= 0.) & (q_minus_direct >= 0.) & on_shell;
        auto q_plus = where(
            light_cone & (q3 < 0.),
            trans2 / max(q_minus_direct, FVal<T>(EPS2)),
            q_plus_direct
        );
        auto q_minus = where(
            light_cone & (q3 >= 0.),
            trans2 / max(q_plus_direct, FVal<T>(EPS2)),
            q_minus_direct
        );
        auto p_plus = e_plus * q_plus;
        auto p_minus = e_minus * q_minus;
        p_out_i[0] = where(light_cone, 0.5 * (p_plus + p_minus), lin_0);
        p_out_i[1] = q1;
        p_out_i[2] = q2;
        p_out_i[3] = where(light_cone, 0.5 * (p_plus - p_minus), lin_3);
        if (i >= 2) {
            sum_plus_direct = sum_plus_direct + q_plus_direct;
            sum_minus_direct = sum_minus_direct + q_minus_direct;
            sum_plus = sum_plus + q_plus;
            sum_minus = sum_minus + q_minus;
        }
    }

    // The projection onto the mass shell moves a momentum by its mass-shell
    // error times e^{|y|}. So that momentum stays conserved as by the linear
    // boost, these shifts are handed to the outgoing momentum with the largest
    // energy: it is set to the boosted total of the outgoing momenta minus all
    // others. It carries at least 1/n of the energy, so this costs it no
    // relative precision. The total is boosted in light-cone components too,
    // each summed directly where that is precise, as in the partonic frame,
    // and from the light-cone components of the momenta where the direct sum
    // cancels, as out of the lab frame. (The linear boost of the total lost
    // e^{2|y|} of relative precision there, 5e-9 of sqrt(s_hat) in the
    // hardest momentum for |y| ~ 7, and so did the round trip of
    // PhaseSpaceMapping.)
    if (q.size() > 3) {
        auto energy = 0.5 * (sum_plus_direct + sum_minus_direct);
        auto total_plus =
            e_plus * where(sum_plus_direct >= 0.5 * energy, sum_plus_direct, sum_plus);
        auto total_minus = e_minus *
            where(sum_minus_direct >= 0.5 * energy, sum_minus_direct, sum_minus);
        auto best_e = FVal<T>(p_out[2][0]);
        auto best_i = IVal<T>(2);
        for (std::size_t i = 3; i < q.size(); ++i) {
            FVal<T> e_i(p_out[i][0]);
            auto larger = e_i > best_e;
            best_i = where(larger, IVal<T>(static_cast<int>(i)), best_i);
            best_e = where(larger, e_i, best_e);
        }
        FVal<T> others_plus(0.), others_minus(0.);
        for (std::size_t i = 2; i < q.size(); ++i) {
            auto is_best = best_i == IVal<T>(static_cast<int>(i));
            FVal<T> e_i(p_out[i][0]), pz_i(p_out[i][3]);
            others_plus = others_plus + where(is_best, FVal<T>(0.), e_i + pz_i);
            others_minus = others_minus + where(is_best, FVal<T>(0.), e_i - pz_i);
        }
        auto best_plus = total_plus - others_plus;
        auto best_minus = total_minus - others_minus;
        for (std::size_t i = 2; i < q.size(); ++i) {
            auto is_best = best_i == IVal<T>(static_cast<int>(i));
            p_out[i][0] =
                where(is_best, 0.5 * (best_plus + best_minus), FVal<T>(p_out[i][0]));
            p_out[i][3] =
                where(is_best, 0.5 * (best_plus - best_minus), FVal<T>(p_out[i][3]));
        }
    }
}

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> p1com_from_tabs_cs(
    FourMom<T> pa_com,
    FVal<T> s_tot,
    FVal<T> cos_phi,
    FVal<T> sin_phi,
    FVal<T> t_abs,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> ma_2,
    FVal<T> mb_2
) {
    // this function is based on the gentcms subroutine in MG5 (genps.f)
    // Note: expects t_abs (positive t-ivariant)
    auto m_tot = sqrt(s_tot);

    auto ed = (m1 - m2) * (m1 + m2) / m_tot;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + s_tot;
    auto pp = 0.5 * where(m1 * m2 == 0., m_tot - fabs(ed), sqrt(max(pp2, EPS)));

    auto pa_com_mag =
        sqrt(pa_com[1] * pa_com[1] + pa_com[2] * pa_com[2] + pa_com[3] * pa_com[3]);
    FourMom<T> p1_com;
    auto e1_com = 0.5 * (m_tot + ed);
    p1_com[0] = max(e1_com, 0.);
    // p1 along pa at |t| = t_lo, against it at t_lo + 4 |pa| pp. t_lo through
    // E - p = m^2 / (E + p), and p1_z, p1_t from the distances to the ends of
    // the range: pp^2 - p1_z^2 cancels near an end, and differently with FMA
    auto plus_a = pa_com[0] + pa_com_mag;
    auto plus_1 = p1_com[0] + pp;
    auto t_lo = (plus_a - plus_1) * (m1 * m1 / plus_1 - ma_2 / plus_a);
    auto pa_2 = 2. * pa_com_mag;
    auto to_lo = (t_abs - t_lo) / pa_2;
    auto to_hi = (t_lo + 2. * pa_2 * pp - t_abs) / pa_2;
    p1_com[3] = 0.5 * (to_hi - to_lo);
    auto pt = sqrt(max(to_lo * to_hi, 0.));
    p1_com[1] = pt * cos_phi;
    p1_com[2] = pt * sin_phi;

    auto det = PI / (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2)));
    return {p1_com, det};
}

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> p1com_from_tabs_phi(
    FourMom<T> pa_com,
    FVal<T> s_tot,
    FVal<T> phi,
    FVal<T> t_abs,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> ma_2,
    FVal<T> mb_2
) {
    return p1com_from_tabs_cs<T>(
        pa_com, s_tot, cos(phi), sin(phi), t_abs, m1, m2, ma_2, mb_2
    );
}

template <typename T>
KERNELSPEC Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> phi_m1_m2_from_p1com(
    FourMom<T> p1_com, FourMom<T> p2, FVal<T> s_tot, FVal<T> ma_2, FVal<T> mb_2
) {
    auto phi = atan2(p1_com[2], p1_com[1]);
    auto det = (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2))) / PI;
    auto m1 = sqrt(max(EPS2, lsquare<T>(p1_com)));
    auto m2 = sqrt(max(EPS2, lsquare<T>(p2)));
    return {phi, m1, m2, det};
}

// Kernels

template <typename T>
KERNELSPEC void kernel_boost_beam(
    FIn<T, 2> p1, FSIn<T, 1> masses, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out
) {
    boost_beam<T>(p1, masses, x1, x2, 1.0, p_out);
}

template <typename T>
KERNELSPEC void kernel_boost_beam_inverse(
    FIn<T, 2> p1, FSIn<T, 1> masses, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out
) {
    boost_beam<T>(p1, masses, x1, x2, -1.0, p_out);
}

template <typename T>
KERNELSPEC void kernel_com_p_in(FIn<T, 0> e_cm, FOut<T, 1> p1, FOut<T, 1> p2) {
    auto p_com = e_cm / 2;
    p1[0] = p_com;
    p1[1] = 0;
    p1[2] = 0;
    p1[3] = p_com;
    p2[0] = p_com;
    p2[1] = 0;
    p2[2] = 0;
    p2[3] = -p_com;
}

template <typename T>
KERNELSPEC void kernel_r_to_x1x2(
    FIn<T, 0> r,
    FIn<T, 0> s_hat,
    FSIn<T, 0> s_lab,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> det
) {
    auto tau = s_hat / s_lab;
    x1 = pow(tau, r);
    x2 = pow(tau, (1 - r));
    det = fabs(log(tau)) / s_lab;
}

template <typename T>
KERNELSPEC void kernel_x1x2_to_r(
    FIn<T, 0> x1, FIn<T, 0> x2, FSIn<T, 0> s_lab, FOut<T, 0> r, FOut<T, 0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

// r_to_x1x2 with the rapidity of the partonic system, y = log(x1 / x2) / 2,
// restricted to |y| <= y_max. r_to_x1x2 has y = (1 - 2 r) L / 2 with
// L = -log(tau), uniform over |y| <= L / 2; here the same line is squeezed to
// |y| <= min(y_max, L / 2), and the Jacobian shrinks by the same factor.
// Where the window does not bind, this is r_to_x1x2 itself.
template <typename T>
KERNELSPEC void kernel_r_to_x1x2_window(
    FIn<T, 0> r,
    FIn<T, 0> s_hat,
    FSIn<T, 0> s_lab,
    FSIn<T, 0> y_max,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> det
) {
    auto tau = s_hat / s_lab;
    auto log_tau_abs = fabs(log(tau));
    auto squeeze = min(2. * y_max / max(log_tau_abs, EPS), 1.);
    auto r_window = where(squeeze < 1., 0.5 + (r - 0.5) * squeeze, FVal<T>(r));
    x1 = pow(tau, r_window);
    x2 = pow(tau, (1 - r_window));
    det = squeeze * log_tau_abs / s_lab;
}

template <typename T>
KERNELSPEC void kernel_x1x2_to_r_window(
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> s_lab,
    FSIn<T, 0> y_max,
    FOut<T, 0> r,
    FOut<T, 0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    auto log_tau_abs = fabs(log_tau);
    auto squeeze = min(2. * y_max / max(log_tau_abs, EPS), 1.);
    auto r_window = log(x1) / log_tau;
    r = where(squeeze < 1., 0.5 + (r_window - 0.5) / squeeze, r_window);
    det = s_lab / (squeeze * log_tau_abs);
}

template <typename T>
KERNELSPEC void kernel_diff_cross_section(
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> pdf1,
    FIn<T, 0> pdf2,
    FIn<T, 0> matrix_element,
    FSIn<T, 0> e_cm2,
    FOut<T, 0> result
) {
    result = INV_GEV2_TO_PB * matrix_element * pdf1 * pdf2 /
        (2. * e_cm2 * x1 * x1 * x2 * x2);
}

template <typename T>
KERNELSPEC void kernel_t_inv_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    // this function is based on the yminmax subroutine in MG5 (genps.f)
    // returns the absolute value of the t invariant
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }

    auto s = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto t_min_max = t_inv_min_max<T>(s, ma_2, mb_2, m1_2, m2_2);
    t_min = t_min_max.first;
    t_max = t_min_max.second;
}

template <typename T>
KERNELSPEC void kernel_t_inv_value_and_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> t_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    // returns the absolute value of the t invariant and its min/max
    FourMom<T> pa1, p_tot;
    for (int i = 0; i < 4; ++i) {
        pa1[i] = pa[i] - p1[i];
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto t_temp = lsquare<T>(pa1);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto t_min_max = t_inv_min_max<T>(s, ma_2, mb_2, m1_2, m2_2);
    t_min = t_min_max.first;
    t_max = t_min_max.second;
    t_abs = -t_temp;
}

// Cut-derived |t| interval from the AmpliCol/gen23 ETmin construction
// (phase_space_gen23.f03, gen23_one_step, lines 698-727).
// Convention chosen to match the kernels' m1/m2 mass labels:
//   particle 1 == recoil system    "ir"  (mass^2 m1_2, ETmin floor etmin_1)
//   particle 2 == single peeled jet "i"  (mass^2 m2_2, ETmin floor etmin_2)
// t is measured w.r.t. beam a: t = (pa - p1)^2 = (pb - p2)^2, so the angle->|t|
// offset carries the recoil mass^2 (m1).
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> t_cut_bounds_etmin(
    FourMom<T> p_tot,
    FourMom<T> pa,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> etmin_1,
    FVal<T> etmin_2
) {
    // squared masses recomputed from momenta (inverse direction) can be slightly
    // negative; clamp them so that the square roots below cannot give NaN
    m1_2 = max(m1_2, 0.);
    m2_2 = max(m2_2, 0.);
    auto pt_tot2 = p_tot[1] * p_tot[1] + p_tot[2] * p_tot[2];
    auto s = lsquare<T>(p_tot);
    auto piir0 = sqrt(max(s, 0.) + pt_tot2);
    auto ppa0 = (p_tot[0] * pa[0] - p_tot[3] * pa[3]) / piir0;
    // peeled jet (2): pt floor^2 = etmin_2^2 - m2^2; recoil (1) floor corrected
    // for the system's own pt.
    auto pt2_floor = etmin_2 * etmin_2 - m2_2;
    auto d = sqrt(pt_tot2) - sqrt(max(pt2_floor, 0.));
    auto et_corr = sqrt(m1_2 + d * d);
    auto eff_1 = max(etmin_1, where(pt_tot2 < pt2_floor, et_corr, sqrt(max(m1_2, 0.))));
    auto eff_2 = max(etmin_2, sqrt(max(m2_2, 0.)));
    auto base = piir0 * piir0 - eff_2 * eff_2 + eff_1 * eff_1;
    auto root = (piir0 - eff_2 - eff_1) * (piir0 + eff_2 - eff_1) *
        (piir0 - eff_2 + eff_1) * (piir0 + eff_2 + eff_1);
    auto rootsq = sqrt(max(root, 0.));
    auto ratio = ppa0 / piir0;
    // signed-t -> |t|: min uses (base - root), max uses (base + root); the
    // angle->|t| offset carries the recoil mass^2 (m1).
    return {ratio * (base - rootsq) - m1_2, ratio * (base + rootsq) - m1_2};
}

// Clamp the kinematic |t| range against the ETmin cut interval. Both kernels
// return the absolute (positive) t invariant. The ETmin bound is a sampling
// optimization, not the cut enforcement: the integrator applies the real cut on
// the final momenta (cf. AmpliCol's explicit ET-check on the generated
// solutions).
template <typename T>
KERNELSPEC void kernel_t_inv_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto t_min_max = t_inv_min_max<T>(s, ma_2, mb_2, m1_2, m2_2);
    auto tmn = t_min_max.first;
    auto tmx = t_min_max.second;

    // particle 1 = recoil (m1, etmin_1), particle 2 = peeled (m2, etmin_2)
    auto cut =
        t_cut_bounds_etmin<T>(p_tot, load_mom<T>(pa), m1_2, m2_2, etmin_1, etmin_2);

    // If the incoming particle pa is not along the beam axis, the ETmin cut is
    // is too tight. In that case, we just use the kinematic bounds.
    // auto tmn_c = max(tmn, cut.first);
    // auto tmx_c = min(tmx, cut.second);
    auto proj_pt2 = pa[1] * pa[1] + pa[2] * pa[2];
    auto proj_along_z = proj_pt2 <= (pa[0] * pa[0]) * 1e-9;
    auto tmn_c = where(proj_along_z, max(tmn, cut.first), tmn);
    auto tmx_c = where(proj_along_z, min(tmx, cut.second), tmx);
    auto ok = tmx_c > tmn_c;
    t_min = where(ok, tmn_c, tmn);
    t_max = where(ok, tmx_c, tmx);
}

template <typename T>
KERNELSPEC FVal<T> logistic(FVal<T> z) {
    return 1. / (1. + exp(-z));
}

// exp with its argument kept where the result is finite, so that a product
// with a vanishing factor stays 0 instead of becoming 0 * inf = NaN
template <typename T>
KERNELSPEC FVal<T> bounded_exp(FVal<T> z) {
    return exp(min(max(z, -700.), 700.));
}

// Whether a rapidity bound handed to the |t| clamps below is in use: negative
// means no bound, and so does one of 100 or more, which no momentum a collider
// event can hold gets near. Dropping it only widens the range, so it stays a
// consequence of the cuts, and it keeps every exponential below finite.
template <typename T>
KERNELSPEC auto rapidity_bound_on(FVal<T> y_max) {
    return (y_max >= 0.) & (y_max < 100.);
}

// Narrows the |t| range of a 2 -> 2 scattering pa pb -> p1 p2 whose incoming
// momenta are the two (massless) beams in the partonic centre-of-mass frame,
// given bounds on the absolute lab rapidities of the outgoing p1 (y_max1, the
// recoil) and p2 (y_max2, the peeled particle); negative means no bound.
// With P = pa + pb and t = (pa - p1)^2 = (pb - p2)^2:
//   w = pb.p2 = (m2^2 + |t|) / 2,   pa.p2 = A - w,  A = P.p2 = (s + m2^2 - m1^2)/2,
// and the rapidity of a momentum q along pa is log(pb.q / pa.q) / 2, so
//   y_a(p2) = log(w / (A - w)) / 2,
//   y_a(p1) = log((s/2 - w) / (s/2 - A + w)) / 2,
// both monotonic in w. The lab rapidity along pa is y_a + y_boost, where
// y_boost = beam_sign * log(x1 / x2) / 2 (beam_sign = +1 if pa is beam 1), and
// |y_lab| <= y_max turns into an interval of w, hence of |t|. If the clamp
// would leave nothing, the input range is returned: the cuts then reject
// every point anyway, and the range stays a valid one to sample.
template <typename T>
KERNELSPEC void kernel_t_rapidity_clamp(
    FIn<T, 0> t_min,
    FIn<T, 0> t_max,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> beam_sign,
    FSIn<T, 0> y_max1,
    FSIn<T, 0> y_max2,
    FOut<T, 0> t_min_out,
    FOut<T, 0> t_max_out
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1, m2_2 = m2 * m2;
    auto a = 0.5 * (s + m2_2 - m1_2);
    auto b = 0.5 * s - a;
    auto y_boost = beam_sign * 0.5 * log(x1 / x2);

    auto on1 = rapidity_bound_on<T>(y_max1), on2 = rapidity_bound_on<T>(y_max2);

    // Each bound on w is turned into one on |t| = 2 w - m2^2 and only then
    // compared with the input range, which an unused or looser bound therefore
    // leaves exactly as it was.
    FVal<T> lo = t_min, hi = t_max;
    // peeled particle: w / (A - w) = exp(2 y_a), increasing in w
    auto w_lo2 = a * logistic<T>(2. * (-y_max2 - y_boost));
    auto w_hi2 = a * logistic<T>(2. * (y_max2 - y_boost));
    lo = where(on2, max(lo, 2. * w_lo2 - m2_2), lo);
    hi = where(on2, min(hi, 2. * w_hi2 - m2_2), hi);
    // recoil: (s/2 - w) / (B + w) = exp(z), z = 2 y_a, gives
    // w = (s/2) logistic(-z) - B logistic(z), decreasing in z
    auto z_hi = 2. * (y_max1 - y_boost), z_lo = 2. * (-y_max1 - y_boost);
    auto w_lo1 = 0.5 * s * logistic<T>(-z_hi) - b * logistic<T>(z_hi);
    auto w_hi1 = 0.5 * s * logistic<T>(-z_lo) - b * logistic<T>(z_lo);
    lo = where(on1, max(lo, 2. * w_lo1 - m2_2), lo);
    hi = where(on1, min(hi, 2. * w_hi1 - m2_2), hi);

    auto ok = hi > lo;
    t_min_out = where(ok, lo, FVal<T>(t_min));
    t_max_out = where(ok, hi, FVal<T>(t_max));
}

// Narrows |t1| of the double-t central block (see t2_rapidity_clamp_doublet) to
// the values that leave some |t2| for the rapidity bounds. At fixed |t1| every
// bound on |t2| is linear in |t1|:
//   p1 (bound y_max1): (m1^2 + |t1|) e^{2 lo1} - m1^2 <= |t2|
//                       <= (m1^2 + |t1|) e^{2 hi1} - m1^2,
//   p2 (bound y_max2): s - (s - m1^2 - |t1|) e^{2 hi2} - m1^2 <= |t2|
//                       <= s - (s - m1^2 - |t1|) e^{2 lo2} - m1^2,
//   kinematics:        0 <= |t2| <= s - |t1| - m1^2 - mir_min^2,
// (lo, hi = -+y_max - y_boost), so each lower <= upper pair is a half-line in
// |t1|. The kinematic lower end used here (0) is below the true one, so the
// interval contains every |t1| that has a passing |t2|.
// The transverse-energy floors add two more: for any momentum q,
// (2 pa.q)(2 pb.q) = s M_T(q)^2, with M_T of a system at least the sum of its
// members' m_T, and pb.q = pa.q e^{2 y_a(q)}. For p1, M_T >= etmin_1 and
// y_a <= hi1 give (m1^2 + |t1|) >= sqrt(s) etmin_1 e^{-hi1}; for p2
// (2 pa.p2 = s - m1^2 - |t1|), M_T >= etmin_2 and y_a(p2) <= hi2 give
// (s - m1^2 - |t1|) >= sqrt(s) etmin_2 e^{-hi2}. An empty result returns the
// input range.
template <typename T>
KERNELSPEC void kernel_t1_rapidity_clamp_doublet(
    FIn<T, 0> t1_min,
    FIn<T, 0> t1_max,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> beam_sign,
    FSIn<T, 0> y_max1,
    FSIn<T, 0> y_max2,
    FOut<T, 0> t1_min_out,
    FOut<T, 0> t1_max_out
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto y_boost = beam_sign * 0.5 * log(x1 / x2);
    auto e1_lo = bounded_exp<T>(2. * (-y_max1 - y_boost));
    auto e1_hi = bounded_exp<T>(2. * (y_max1 - y_boost));
    auto e2_lo = bounded_exp<T>(2. * (-y_max2 - y_boost));
    auto e2_hi = bounded_exp<T>(2. * (y_max2 - y_boost));
    auto on1 = rapidity_bound_on<T>(y_max1), on2 = rapidity_bound_on<T>(y_max2);

    // the bounds on |t2| as alpha + beta |t1|; an inactive one is replaced by
    // the kinematic bound of the same side
    FVal<T> lo_alpha[3] = {
        FVal<T>(0.),
        where(on1, m1_2 * e1_lo - m1_2, FVal<T>(0.)),
        where(on2, s - (s - m1_2) * e2_hi - m1_2, FVal<T>(0.))
    };
    FVal<T> lo_beta[3] = {
        FVal<T>(0.), where(on1, e1_lo, FVal<T>(0.)), where(on2, e2_hi, FVal<T>(0.))
    };
    auto kin_alpha = s - m1_2 - mir_min * mir_min;
    FVal<T> hi_alpha[3] = {
        kin_alpha,
        where(on1, m1_2 * e1_hi - m1_2, kin_alpha),
        where(on2, s - (s - m1_2) * e2_lo - m1_2, kin_alpha)
    };
    FVal<T> hi_beta[3] = {
        FVal<T>(-1.), where(on1, e1_hi, FVal<T>(-1.)), where(on2, e2_lo, FVal<T>(-1.))
    };

    FVal<T> lo = t1_min, hi = t1_max;
    auto sqrt_s = sqrt(max(s, EPS));
    lo = where(
        on1, max(lo, sqrt_s * etmin_1 * bounded_exp<T>(-(y_max1 - y_boost)) - m1_2), lo
    );
    hi = where(
        on2,
        min(hi, s - m1_2 - sqrt_s * etmin_2 * bounded_exp<T>(-(y_max2 - y_boost))),
        hi
    );
    auto feasible = lo <= hi;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // lo_alpha_i + lo_beta_i t <= hi_alpha_j + hi_beta_j t
            auto slope = lo_beta[i] - hi_beta[j];
            auto rhs = hi_alpha[j] - lo_alpha[i];
            auto bound = rhs / where(slope == 0., FVal<T>(1.), slope);
            hi = where(slope > 0., min(hi, bound), hi);
            lo = where(slope < 0., max(lo, bound), lo);
            feasible = feasible & ((slope != 0.) | (rhs >= 0.));
        }
    }
    // with no bound in use, the input range stands as it is
    auto ok = (on1 | on2) & feasible & (hi > lo);
    t1_min_out = where(ok, lo, FVal<T>(t1_min));
    t1_max_out = where(ok, hi, FVal<T>(t1_max));
}

// Narrows |t2| of the double-t central block pa pb -> p1 p2 (pa, pb the massless
// beams, |t1| = -(pa - p1)^2, |t2| = -(pb - p1)^2, see double_t_scattering)
// given |t1| and bounds on the absolute lab rapidities of p1 (y_max1) and p2
// (y_max2); negative means no bound. Here pa.p1 = (m1^2 + |t1|) / 2 and
// pb.p1 = (m1^2 + |t2|) / 2, and for p2 = pa + pb - p1, pa.p2 = s/2 - pa.p1 and
// pb.p2 = s/2 - pb.p1, so both rapidities along pa, log(pb.q / pa.q) / 2, are
// monotonic in |t2| at fixed |t1|. The lab rapidity is that plus y_boost =
// beam_sign * log(x1 / x2) / 2. An empty result returns the input range, as in
// t_rapidity_clamp.
template <typename T>
KERNELSPEC void kernel_t2_rapidity_clamp_doublet(
    FIn<T, 0> t2_min,
    FIn<T, 0> t2_max,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> t1_abs,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> beam_sign,
    FSIn<T, 0> y_max1,
    FSIn<T, 0> y_max2,
    FOut<T, 0> t2_min_out,
    FOut<T, 0> t2_max_out
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto y_boost = beam_sign * 0.5 * log(x1 / x2);
    // pa.p1, and pb.p1 as the variable w
    auto a1 = 0.5 * (m1_2 + t1_abs);
    auto a2 = 0.5 * s - a1;

    auto on1 = rapidity_bound_on<T>(y_max1), on2 = rapidity_bound_on<T>(y_max2);

    // bounds on w = pb.p1, turned into |t2| = 2 w - m1^2 before they meet the
    // input range (see t_rapidity_clamp)
    FVal<T> lo = t2_min, hi = t2_max;
    // p1: w / a1 = exp(2 y_a), increasing in w
    auto w_lo1 = a1 * bounded_exp<T>(2. * (-y_max1 - y_boost));
    auto w_hi1 = a1 * bounded_exp<T>(2. * (y_max1 - y_boost));
    lo = where(on1, max(lo, 2. * w_lo1 - m1_2), lo);
    hi = where(on1, min(hi, 2. * w_hi1 - m1_2), hi);
    // p2: (s/2 - w) / a2 = exp(2 y_a), decreasing in w
    auto w_lo2 = 0.5 * s - a2 * bounded_exp<T>(2. * (y_max2 - y_boost));
    auto w_hi2 = 0.5 * s - a2 * bounded_exp<T>(2. * (-y_max2 - y_boost));
    lo = where(on2, max(lo, 2. * w_lo2 - m1_2), lo);
    hi = where(on2, min(hi, 2. * w_hi2 - m1_2), hi);

    auto ok = hi > lo;
    t2_min_out = where(ok, lo, FVal<T>(t2_min));
    t2_max_out = where(ok, hi, FVal<T>(t2_max));
}

template <typename T>
KERNELSPEC void kernel_t_inv_value_and_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FOut<T, 0> t_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> pa1, p_tot;
    for (int i = 0; i < 4; ++i) {
        pa1[i] = pa[i] - p1[i];
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto t_temp = lsquare<T>(pa1);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto t_min_max = t_inv_min_max<T>(s, ma_2, mb_2, m1_2, m2_2);
    auto tmn = t_min_max.first;
    auto tmx = t_min_max.second;

    // particle 1 = recoil (m1, etmin_1), particle 2 = peeled (m2, etmin_2)
    auto cut =
        t_cut_bounds_etmin<T>(p_tot, load_mom<T>(pa), m1_2, m2_2, etmin_1, etmin_2);
    // auto tmn_c = max(tmn, cut.first);
    // auto tmx_c = min(tmx, cut.second);
    auto proj_pt2 = pa[1] * pa[1] + pa[2] * pa[2];
    auto proj_along_z = proj_pt2 <= (pa[0] * pa[0]) * 1e-9;
    auto tmn_c = where(proj_along_z, max(tmn, cut.first), tmn);
    auto tmx_c = where(proj_along_z, min(tmx, cut.second), tmx);
    auto ok = tmx_c > tmn_c;
    t_min = where(ok, tmn_c, tmn);
    t_max = where(ok, tmx_c, tmx);
    t_abs = -t_temp;
}

template <typename T>
KERNELSPEC void kernel_invariants_from_momenta(
    FIn<T, 2> p_ext, FSIn<T, 2> factors, FOut<T, 1> invariants
) {
    for (std::size_t i = 0; i < invariants.size(); ++i) {
        auto factors_i = factors[i];
        FourMom<T> p_sum{0., 0., 0., 0.};
        for (std::size_t j = 0; j < p_ext.size(); ++j) {
            auto p_j = p_ext[j];
            auto factor_ij = factors_i[j];
            p_sum[0] = p_sum[0] + factor_ij * p_j[0];
            p_sum[1] = p_sum[1] + factor_ij * p_j[1];
            p_sum[2] = p_sum[2] + factor_ij * p_j[2];
            p_sum[3] = p_sum[3] + factor_ij * p_j[3];
        }
        invariants[i] = lsquare<T>(p_sum);
    }
}

template <typename T>
KERNELSPEC void kernel_t1_inv_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> etmin_i,
    FIn<T, 0> etmin_ir,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto bounds = t1_inv_min_max_doublet<T>(s, m1_2, mir_min * mir_min);
    auto c = doublet_t1_etmin_clamp<T>(
        bounds.first, bounds.second, s, m1_2, etmin_i, etmin_ir
    );
    t_min = c.first;
    t_max = c.second;
}

template <typename T>
KERNELSPEC void kernel_t1_inv_value_and_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> etmin_i,
    FIn<T, 0> etmin_ir,
    FOut<T, 0> t_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> pa1, p_tot;
    for (int i = 0; i < 4; ++i) {
        pa1[i] = pa[i] - p1[i];
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto bounds = t1_inv_min_max_doublet<T>(s, m1_2, mir_min * mir_min);
    auto c = doublet_t1_etmin_clamp<T>(
        bounds.first, bounds.second, s, m1_2, etmin_i, etmin_ir
    );
    t_min = c.first;
    t_max = c.second;
    t_abs = -lsquare<T>(pa1);
}

template <typename T>
KERNELSPEC void kernel_t2_inv_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> t1_abs,
    FIn<T, 0> etmin_i,
    FIn<T, 0> etmin_ir,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto bounds = t2_inv_min_max_doublet<T>(s, m1_2, mir_min * mir_min, t1_abs);
    auto c = doublet_t2_etmin_clamp<T>(
        bounds.first, bounds.second, s, m1_2, t1_abs, etmin_i, etmin_ir
    );
    t_min = c.first;
    t_max = c.second;
}

template <typename T>
KERNELSPEC void kernel_t2_inv_value_and_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> t1_abs,
    FIn<T, 0> etmin_i,
    FIn<T, 0> etmin_ir,
    FOut<T, 0> t_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> pb1, p_tot;
    for (int i = 0; i < 4; ++i) {
        pb1[i] = pb[i] - p1[i];
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto m1_2 = m1 * m1;
    auto bounds = t2_inv_min_max_doublet<T>(s, m1_2, mir_min * mir_min, t1_abs);
    auto c = doublet_t2_etmin_clamp<T>(
        bounds.first, bounds.second, s, m1_2, t1_abs, etmin_i, etmin_ir
    );
    t_min = c.first;
    t_max = c.second;
    t_abs = -lsquare<T>(pb1);
}

template <typename T>
KERNELSPEC void
kernel_pt_eta_phi_x(FIn<T, 2> p_ext, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 1> output) {
    output[0] = x1;
    output[1] = x2;
    for (std::size_t i = 2; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto px = p_i[1], py = p_i[2], pz = p_i[3];
        auto pt2 = px * px + py * py + 1e-6;
        output[3 * i - 4] = 0.5 * log(pt2);
        output[3 * i - 3] = atan2(py, px);
        output[3 * i - 2] = atanh(pz / sqrt(pt2 + pz * pz));
    }
}

template <typename T>
KERNELSPEC void
kernel_mirror_momenta(FIn<T, 2> p_ext, IIn<T, 0> mirror, FOut<T, 2> p_out) {
    auto sign = 1. - 2. * FVal<T>(IVal<T>(mirror));
    for (std::size_t i = 0; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto q_i = p_out[i];
        q_i[0] = p_i[0];
        q_i[1] = p_i[1];
        q_i[2] = sign * p_i[2];
        q_i[3] = sign * p_i[3];
    }
}

template <typename T>
KERNELSPEC void
kernel_momenta_to_x1x2(FIn<T, 2> p_ext, FSIn<T, 0> e_cm, FOut<T, 0> x1, FOut<T, 0> x2) {
    x1 = 2. * p_ext[0][0] / e_cm;
    x2 = 2. * p_ext[1][0] / e_cm;
}

// Boost every external momentum into the rest frame of the sum of the momenta
// selected by frame_mask. This is the frame the matrix element is evaluated in
// when the run card asks for one (me_frame); it only makes a difference for an
// amplitude that is not Lorentz invariant, i.e. a polarised one. Same
// definition as madevent's boost_to_frame (Template/LO/SubProcesses/genps.f).
template <typename T>
KERNELSPEC void
kernel_boost_to_frame(FIn<T, 2> p_ext, ISIn<T, 1> frame_mask, FOut<T, 2> p_out) {
    FourMom<T> p_boost{0., 0., 0., 0.};
    std::size_t selected_count = 0, selected_index = 0;
    for (std::size_t i = 0; i < p_ext.size(); ++i) {
        if (single_index(frame_mask[i]) == 0) {
            continue;
        }
        auto p_i = p_ext[i];
        for (int j = 0; j < 4; ++j) {
            p_boost[j] = p_boost[j] + p_i[j];
        }
        ++selected_count;
        selected_index = i;
    }
    if (selected_count == 0) {
        // Nothing selected, so nothing defines a frame: pass the momenta
        // through rather than boosting by a null vector, whose zero invariant
        // mass boost() would floor at EPS2 and turn every energy into garbage.
        // Same choice as the qq == 0 branch of the HELAS boostx madevent uses.
        for (std::size_t i = 0; i < p_ext.size(); ++i) {
            store_mom<T>(p_out[i], load_mom<T>(p_ext[i]));
        }
        return;
    }
    for (std::size_t i = 0; i < p_ext.size(); ++i) {
        auto p_i = boost<T>(load_mom<T>(p_ext[i]), p_boost, -1.);
        // A single particle defining the frame must come out exactly at rest.
        // boost() only gets there up to the rounding of the boost factor,
        // leaving a residual three-momentum of a few 1e-14 whose direction is
        // pure noise, and that is not harmless: vxxxxx branches on pp == 0 and
        // for a massive vector at exactly zero three-momentum takes the frame z
        // axis as quantisation axis; otherwise it builds the polarisation
        // vectors from the momentum direction, i.e. from the rounding noise. So
        // impose the defining property of the frame explicitly. The energy is
        // left untouched: zeroing the three-momentum shifts the invariant mass
        // by O(1e-28) relative, and HELAS takes the mass as a separate
        // argument anyway. With two or more selected particles it is only their
        // sum that is at rest and no single leg sits on the branch point, so
        // this is needed for selected_count == 1 only.
        if (selected_count == 1 && i == selected_index) {
            p_i[1] = 0.;
            p_i[2] = 0.;
            p_i[3] = 0.;
        }
        store_mom<T>(p_out[i], p_i);
    }
}

} // namespace kernels
} // namespace madspace
