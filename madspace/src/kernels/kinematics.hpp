#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template <typename T>
struct FourMom {
    KERNELSPEC FVal<T>& operator[](int i) { return p[i]; }
    KERNELSPEC FVal<T>& operator[](int i) const { return p[i]; }
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
    auto t_max = where(t_max_tmp > t_min, t_max_tmp, t_min + EPS);
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

template <typename T>
KERNELSPEC FVal<T> lsquare(FourMom<T> p) {
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
}

template <typename T>
KERNELSPEC FVal<T> esquare(FourMom<T> p) {
    return p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
}

template <typename T>
KERNELSPEC FourMom<T> rotate(FourMom<T> p, FourMom<T> q) {
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq2 = qt2 + q[3] * q[3];
    auto qt = sqrt(max(qt2, EPS2));
    auto qq = sqrt(max(qq2, EPS2));

    // General rotation (valid when qt2 > 0; numerically safe because qt,qq>=eps)
    FourMom<T> r_gen = {
        p[0],
        q[1] * q[3] / (qq * qt) * p[1] - q[2] / qt * p[2] + q[1] / qq * p[3],
        q[2] * q[3] / (qq * qt) * p[1] + q[1] / qt * p[2] + q[2] / qq * p[3],
        -qt / qq * p[1] + q[3] / qq * p[3]
    };

    // Degenerate case qt2 == 0: choose identity for qz>=0 else
    // (px,,py,pz)->(-px,-py,-pz)
    FourMom<T> r_deg_pos = p;
    FourMom<T> r_deg_neg = {p[0], -p[1], p[2], -p[3]};

    auto mask_deg = (qt2 == 0.);
    auto mask_neg = (q[3] < 0.);

    // pick degenerate result depending on sign(qz)
    FourMom<T> r_deg = {
        where(mask_neg, r_deg_neg[0], r_deg_pos[0]),
        where(mask_neg, r_deg_neg[1], r_deg_pos[1]),
        where(mask_neg, r_deg_neg[2], r_deg_pos[2]),
        where(mask_neg, r_deg_neg[3], r_deg_pos[3]),
    };

    // final: if degenerate use r_deg else r_gen
    return {
        where(mask_deg, r_deg[0], r_gen[0]),
        where(mask_deg, r_deg[1], r_gen[1]),
        where(mask_deg, r_deg[2], r_gen[2]),
        where(mask_deg, r_deg[3], r_gen[3]),
    };
}

template <typename T>
KERNELSPEC FourMom<T> rotate_inverse(FourMom<T> p, FourMom<T> q) {
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq2 = qt2 + q[3] * q[3];
    auto qt = sqrt(max(qt2, EPS2));
    auto qq = sqrt(max(qq2, EPS2));

    // General rotation (valid when qt2 > 0; numerically safe because qt,qq>=eps)
    FourMom<T> r_gen = {
        p[0],
        q[1] * q[3] / (qq * qt) * p[1] + q[2] * q[3] / (qq * qt) * p[2] -
            p[3] * qt / qq,
        -q[2] / qt * p[1] + q[1] / qt * p[2],
        q[1] / qq * p[1] + q[2] / qq * p[2] + q[3] / qq * p[3]
    };

    // Degenerate case qt2 == 0: choose identity for qz>=0 else
    // (px,,py,pz)->(-px,-py,-pz)
    FourMom<T> r_deg_pos = p;
    FourMom<T> r_deg_neg = {p[0], -p[1], p[2], -p[3]};

    auto mask_deg = (qt2 == 0.);
    auto mask_neg = (q[3] < 0.);

    // pick degenerate result depending on sign(qz)
    FourMom<T> r_deg = {
        where(mask_neg, r_deg_neg[0], r_deg_pos[0]),
        where(mask_neg, r_deg_neg[1], r_deg_pos[1]),
        where(mask_neg, r_deg_neg[2], r_deg_pos[2]),
        where(mask_neg, r_deg_neg[3], r_deg_pos[3]),
    };

    // final: if degenerate use r_deg else r_gen
    return {
        where(mask_deg, r_deg[0], r_gen[0]),
        where(mask_deg, r_deg[1], r_gen[1]),
        where(mask_deg, r_deg[2], r_gen[2]),
        where(mask_deg, r_deg[3], r_gen[3]),
    };
}

template <typename T>
KERNELSPEC FourMom<T> rotate_two_ref(FourMom<T> p, FourMom<T> q_z, FourMom<T> q_x) {
    // Forward rotation: take p from a canonical frame
    //   (e_z along q_z, e_x in the plane spanned by q_z and q_x with positive
    //    component along q_x's perpendicular part)
    // into the lab frame. Energy unchanged.
    //
    // Used by kernel_two_to_three_particle_scattering: q_z = pa_com,
    // q_x = p3 boosted into the p_12 rest frame. Picks a unique azimuth
    // so that the (q_z, q_x) plane is the phi=0 half-plane.

    // z_hat = q_z / |q_z|
    auto qz_n2 = q_z[1] * q_z[1] + q_z[2] * q_z[2] + q_z[3] * q_z[3];
    auto qz_n = sqrt(max(qz_n2, EPS2));
    auto zx = q_z[1] / qz_n, zy = q_z[2] / qz_n, zz = q_z[3] / qz_n;

    // x_hat = (q_x perp to z_hat) / |...|
    auto qx_dot_z = q_x[1] * zx + q_x[2] * zy + q_x[3] * zz;
    auto rx = q_x[1] - qx_dot_z * zx;
    auto ry = q_x[2] - qx_dot_z * zy;
    auto rz = q_x[3] - qx_dot_z * zz;
    auto rn2 = rx * rx + ry * ry + rz * rz;
    auto rn = sqrt(max(rn2, EPS2));
    auto xx = rx / rn, xy = ry / rn, xz = rz / rn;

    // y_hat = z_hat x x_hat
    auto yx = zy * xz - zz * xy;
    auto yy = zz * xx - zx * xz;
    auto yz = zx * xy - zy * xx;

    // world spatial = p[1]*x_hat + p[2]*y_hat + p[3]*z_hat
    return FourMom<T>{
        p[0],
        p[1] * xx + p[2] * yx + p[3] * zx,
        p[1] * xy + p[2] * yy + p[3] * zy,
        p[1] * xz + p[2] * yz + p[3] * zz,
    };
}

template <typename T>
KERNELSPEC FourMom<T>
rotate_two_ref_inverse(FourMom<T> p, FourMom<T> q_z, FourMom<T> q_x) {
    // Inverse of rotate_two_ref: take a vector p from the lab frame into the
    // canonical frame defined by (q_z, q_x). Energy unchanged.

    auto qz_n2 = q_z[1] * q_z[1] + q_z[2] * q_z[2] + q_z[3] * q_z[3];
    auto qz_n = sqrt(max(qz_n2, EPS2));
    auto zx = q_z[1] / qz_n, zy = q_z[2] / qz_n, zz = q_z[3] / qz_n;

    auto qx_dot_z = q_x[1] * zx + q_x[2] * zy + q_x[3] * zz;
    auto rx = q_x[1] - qx_dot_z * zx;
    auto ry = q_x[2] - qx_dot_z * zy;
    auto rz = q_x[3] - qx_dot_z * zz;
    auto rn2 = rx * rx + ry * ry + rz * rz;
    auto rn = sqrt(max(rn2, EPS2));
    auto xx = rx / rn, xy = ry / rn, xz = rz / rn;

    auto yx = zy * xz - zz * xy;
    auto yy = zz * xx - zx * xz;
    auto yz = zx * xy - zy * xx;

    // canonical components = world spatial . (x_hat, y_hat, z_hat)
    return FourMom<T>{
        p[0],
        p[1] * xx + p[2] * xy + p[3] * xz,
        p[1] * yx + p[2] * yy + p[3] * yz,
        p[1] * zx + p[2] * zy + p[3] * zz,
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
KERNELSPEC void
boost_beam(FIn<T, 2> q, FVal<T> x1, FVal<T> x2, FVal<T> sign, FOut<T, 2> p_out) {
    auto exp_rap = sqrt(x1 / x2);
    auto exp_rap_inv = 1. / exp_rap;
    auto cosh_rap = 0.5 * (exp_rap + exp_rap_inv);
    auto sinh_rap = 0.5 * (exp_rap - exp_rap_inv);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto q_i = q[i];
        auto p_out_i = p_out[i];
        p_out_i[0] = q_i[0] * cosh_rap + sign * q_i[3] * sinh_rap;
        p_out_i[1] = q_i[1];
        p_out_i[2] = q_i[2];
        p_out_i[3] = q_i[3] * cosh_rap + sign * q_i[0] * sinh_rap;
    }
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
    p1_com[3] =
        -(m1 * m1 + ma_2 + t_abs - 2. * p1_com[0] * pa_com[0]) / (2.0 * pa_com_mag);
    auto pt2 = pp * pp - p1_com[3] * p1_com[3];
    auto pt = sqrt(max(pt2, 0.));
    p1_com[1] = pt * cos(phi);
    p1_com[2] = pt * sin(phi);

    auto det = PI / (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2)));
    return {p1_com, det};
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
KERNELSPEC void
kernel_boost_beam(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, 1.0, p_out);
}

template <typename T>
KERNELSPEC void
kernel_boost_beam_inverse(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, -1.0, p_out);
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
    FIn<T, 0> s_lab,
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
    FIn<T, 0> x1, FIn<T, 0> x2, FIn<T, 0> s_lab, FOut<T, 0> r, FOut<T, 0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

template <typename T>
KERNELSPEC void kernel_diff_cross_section(
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> pdf1,
    FIn<T, 0> pdf2,
    FIn<T, 0> matrix_element,
    FIn<T, 0> e_cm2,
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

template <typename T>
KERNELSPEC void kernel_invariants_from_momenta(
    FIn<T, 2> p_ext, FIn<T, 2> factors, FOut<T, 1> invariants
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
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto bounds = t1_inv_min_max_doublet<T>(s, m1 * m1, mir_min * mir_min);
    t_min = bounds.first;
    t_max = bounds.second;
}

template <typename T>
KERNELSPEC void kernel_t1_inv_value_and_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
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
    auto bounds = t1_inv_min_max_doublet<T>(s, m1 * m1, mir_min * mir_min);
    t_min = bounds.first;
    t_max = bounds.second;
    t_abs = -lsquare<T>(pa1);
}

template <typename T>
KERNELSPEC void kernel_t2_inv_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> t1_abs,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s = lsquare<T>(p_tot);
    auto bounds = t2_inv_min_max_doublet<T>(s, m1 * m1, mir_min * mir_min, t1_abs);
    t_min = bounds.first;
    t_max = bounds.second;
}

template <typename T>
KERNELSPEC void kernel_t2_inv_value_and_min_max_doublet(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p1,
    FIn<T, 0> m1,
    FIn<T, 0> mir_min,
    FIn<T, 0> t1_abs,
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
    auto bounds = t2_inv_min_max_doublet<T>(s, m1 * m1, mir_min * mir_min, t1_abs);
    t_min = bounds.first;
    t_max = bounds.second;
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
kernel_momenta_to_x1x2(FIn<T, 2> p_ext, FIn<T, 0> e_cm, FOut<T, 0> x1, FOut<T, 0> x2) {
    x1 = 2. * p_ext[0][0] / e_cm;
    x2 = 2. * p_ext[1][0] / e_cm;
}

} // namespace kernels
} // namespace madspace
