#pragma once

#include "kinematics.hpp"
#include "lup_det.hpp"

namespace madspace {
namespace kernels {

// Helper functions

template <typename T>
KERNELSPEC Triplet<FourMom<T>, FourMom<T>, FVal<T>> three_body_decay(
    FVal<T> r_e1,
    FVal<T> r_e2,
    FVal<T> r_phi,
    FVal<T> r_cos_theta,
    FVal<T> r_beta,
    FVal<T> m0,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> m3
) {
    // this is based on section G.3 in
    // https://inspirehep.net/literature/1784296

    // define angles and determinants
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto beta = PI * (2. * r_beta - 1.);
    auto det_omega = 8 * PI * PI;

    // Define mass squares
    auto m1sq = m1 * m1;
    auto m2sq = m2 * m2;
    auto m3sq = m3 * m3;

    // define energy E1
    auto E1_max = m0 / 2 + (m1sq - (m2 + m3) * (m2 + m3)) / (2 * m0);
    auto E1 = m1 + (E1_max - m1) * r_e1;
    auto det_E1 = E1_max - m1;

    // get boundaries
    auto Delta = 2 * m0 * (m0 / 2 - E1) + m1sq;
    auto Delta23 = m2sq - m3sq;
    auto dE2 =
        (E1 * E1 - m1sq) * ((Delta + Delta23) * (Delta + Delta23) - 4 * m2sq * Delta);
    auto E2a = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) - sqrt(dE2));
    auto E2b = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) + sqrt(dE2));
    auto E2_min = min(E2a, E2b);
    auto E2_max = max(E2a, E2b);
    auto E2 = E2_min + (E2_max - E2_min) * r_e2;
    auto det_E2 = E2_max - E2_min;

    // calculate abs momentas
    auto pp1s = E1 * E1 - m1sq;
    auto pp1 = where(m1sq == 0, E1, sqrt(max(pp1s, EPS)));
    auto pp2s = E2 * E2 - m2sq;
    auto pp2 = where(m2sq == 0, E2, sqrt(max(pp2s, EPS)));

    // calculate cosalpha
    auto num_alpha_1 = 2 * m0 * (m0 / 2 - E1 - E2);
    auto num_alpha_2 = m1sq + m2sq + 2 * E1 * E2 - m3sq;
    auto denom_alpha = 2 * pp1 * pp2;
    auto cos_alpha = (num_alpha_1 + num_alpha_2) / denom_alpha;

    // build momenta p1
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    FourMom<T> p1{
        max(E1, 0.),
        pp1 * sin_theta * cos(phi),
        pp1 * sin_theta * sin(phi),
        pp1 * cos_theta
    };

    // build momenta p2
    auto sin_alpha = sqrt((1. - cos_alpha) * (1 + cos_alpha));
    FourMom<T> p2{
        max(E2, 0.),
        pp2 *
            (sin_alpha * cos(beta) * cos_theta * cos(phi) +
             cos_alpha * sin_theta * cos(phi) - sin_alpha * sin(beta) * sin(phi)),
        pp2 *
            (sin_alpha * sin(beta) * cos(phi) +
             sin_alpha * cos(beta) * cos_theta * sin(phi) +
             cos_alpha * sin_theta * sin(phi)),
        pp2 * (cos_alpha * cos_theta - sin_alpha * cos(beta) * sin_theta)
    };

    auto det = det_omega * det_E2 * det_E1 / 8;
    return {p1, p2, det};
}

template <typename T>
KERNELSPEC Decuplet<
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>>
three_body_decay_inverse(FourMom<T> p1, FourMom<T> p2, FourMom<T> p3) {
    // this is based on section G.3 in
    // https://inspirehep.net/literature/1784296

    // Define total momentum
    FourMom<T> p0;
    for (int i = 0; i < 4; ++i) {
        p0[i] = p1[i] + p2[i] + p3[i];
    }
    auto m0 = sqrt(max(EPS2, lsquare<T>(p0)));
    auto m1 = sqrt(max(EPS2, lsquare<T>(p1)));
    auto m2 = sqrt(max(EPS2, lsquare<T>(p2)));
    auto m3 = sqrt(max(EPS2, lsquare<T>(p3)));

    // Define mass squares
    auto m1sq = m1 * m1;
    auto m2sq = m2 * m2;
    auto m3sq = m3 * m3;

    // define energy E1
    auto E1_max = m0 / 2 + (m1sq - (m2 + m3) * (m2 + m3)) / (2 * m0);
    auto E1 = p1[0];
    auto r_e1 = (p1[0] - m1) / (E1_max - m1);
    auto det_E1 = E1_max - m1;

    // get boundaries
    auto Delta = 2 * m0 * (m0 / 2 - E1) + m1sq;
    auto Delta23 = m2sq - m3sq;
    auto dE2 =
        (E1 * E1 - m1sq) * ((Delta + Delta23) * (Delta + Delta23) - 4 * m2sq * Delta);
    auto E2a = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) - sqrt(dE2));
    auto E2b = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) + sqrt(dE2));
    auto E2_min = min(E2a, E2b);
    auto E2_max = max(E2a, E2b);
    auto E2 = p2[0];
    auto r_e2 = (p2[0] - E2_min) / (E2_max - E2_min);
    auto det_E2 = E2_max - E2_min;

    // calculate abs momentas
    auto pp1s = E1 * E1 - m1sq;
    auto pp1 = where(m1sq == 0, E1, sqrt(max(pp1s, EPS)));
    auto pp2s = E2 * E2 - m2sq;
    auto pp2 = where(m2sq == 0, E2, sqrt(max(pp2s, EPS)));

    // calculate cosalpha
    auto num_alpha_1 = 2 * m0 * (m0 / 2 - E1 - E2);
    auto num_alpha_2 = m1sq + m2sq + 2 * E1 * E2 - m3sq;
    auto denom_alpha = 2 * pp1 * pp2;
    auto cos_alpha = (num_alpha_1 + num_alpha_2) / denom_alpha;

    // calculate angles and determinants
    auto phi = atan2(p1[2], p1[1]);
    auto cos_theta = p1[3] / pp1;
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));

    auto r_phi = (phi / PI + 1.) / 2.;
    auto r_cos_theta = (cos_theta + 1.) / 2.;

    // calculate beta angle
    auto A = (cos_alpha * cos_theta - p2[3] / pp2) / sin_theta;
    auto B = -p2[1] / pp2 * sin(phi) + p2[2] / pp2 * cos(phi);
    auto beta = atan2(B, A);

    auto r_beta = (beta / PI + 1.) / 2.;
    auto det_omega = 8 * PI * PI;

    auto det = det_omega * det_E1 * det_E2 / 8.;
    return {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, 1. / det};
}

template <typename T>
KERNELSPEC FVal<T> bk_V(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    // Matrix from Byckling-Kajantie eq.(11), 10.1103/PhysRev.187.2008.
    // Asymmetric (a22 = a33 = 0; a12 = a21 but a13 != a31, a23 != a32).
    auto a11 = 2.0 * s12;
    auto a12 = ma_2 + s12 - t2;
    auto a13 = s12 + m1_2 - m2_2;
    auto a22 = 2.0 * ma_2;
    auto a23 = ma_2 + m1_2 + t1_abs;
    auto a31 = m0_2 + s12 - m3_2;
    auto a32 = m0_2 + ma_2 - mb_2;

    // Polynomial fallback (the previous implementation), used by the LU
    // helper if the matrix's leading pivots are below the LU tolerance.
    // Expansion along row 3, using a22 = a33 = 0 and a21 = a12.
    auto poly_det = a31 * (a12 * a23 - a13 * a22) + a32 * (a12 * a13 - a11 * a23);

    auto det = lup_det3_general<T>(
        a11, a12, a13, a12, a22, a23, a31, a32, FVal<T>(0.0), poly_det
    );
    return -det / 8.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_gram4(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12,
    FVal<T> s23
) {
    // 4x4 Gram determinant, see Eq.(B6) in 10.1103/PhysRev.187.2008.
    // Note: expects the absolute value of t1.

    // Upper-triangular entries of the symmetric Gram matrix.
    auto a11 = 2.0 * ma_2;
    auto a12 = ma_2 - t1_abs - m1_2;
    auto a13 = ma_2 + t2 - s12;
    auto a14 = ma_2 + mb_2 - m0_2;
    auto a22 = -2.0 * t1_abs;
    auto a23 = t2 - t1_abs - m2_2;
    auto a24 = mb_2 - t1_abs - s23;
    auto a33 = 2.0 * t2;
    auto a34 = t2 + mb_2 - m3_2;
    auto a44 = 2.0 * mb_2;

    // Polynomial fallback (the previous "hard-coded because easier" expansion),
    // re-organized to keep the inner 2x2 minors as their own subexpressions.
    // Used by the LU helper if any pivot is below tolerance.
    auto poly_det = a14 * a14 * (a23 * a23 - a22 * a33) +
        a13 * a13 * (a24 * a24 - a22 * a44) + a12 * a12 * (a34 * a34 - a33 * a44) -
        a23 * a23 * a11 * a44 - a24 * a24 * a11 * a33 - a34 * a34 * a11 * a22 +
        a11 * a22 * a33 * a44 + 2.0 * a11 * a23 * a24 * a34 +
        2.0 * a12 * a13 * (a23 * a44 - a24 * a34) +
        2.0 * a12 * a14 * (a24 * a33 - a23 * a34) +
        2.0 * a13 * a14 * (a22 * a34 - a23 * a24);

    auto det = lup_det4<T>(a11, a12, a13, a14, a22, a23, a24, a33, a34, a44, poly_det);
    return det / 16.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_sqrt_g3i_g3im1(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    auto a11 = 2 * s12;
    auto a12 = s12 + ma_2 - t2;
    auto a13 = s12 + m0_2 - m3_2;
    auto b13 = s12 + m1_2 - m2_2;
    auto a22 = 2 * ma_2;
    auto a23 = ma_2 + m0_2 - mb_2;
    auto b23 = ma_2 + m1_2 + t1_abs;
    auto a33 = 2 * m0_2;
    auto b33 = 2 * m1_2;

    // Polynomial fallback for each 3x3 Gram (cofactor expansion). Used by the
    // LU helper independently per matrix if its pivots are below tolerance.
    auto poly_g3i = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a13 * a23) +
        a13 * (a12 * a23 - a13 * a22);
    auto poly_g3im1 = a11 * (a22 * b33 - b23 * b23) - a12 * (a12 * b33 - b13 * b23) +
        b13 * (a12 * b23 - b13 * a22);

    auto g3i = lup_det3<T>(a11, a12, a13, a22, a23, a33, poly_g3i);
    auto g3im1 = lup_det3<T>(a11, a12, b13, a22, b23, b33, poly_g3im1);

    return sqrt(g3i * g3im1) / 8.0;
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_min_max(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = max(kaellen<T>(s12, ma_2, t2), EPS);

    auto sa = m0_2 + m1_2 + 8 * (V + sqrtGG) / (lambda);
    auto sb = m0_2 + m1_2 + 8 * (V - sqrtGG) / (lambda);
    auto s_min = min(sa, sb);
    auto s_max = max(sa, sb);
    return {s_min, s_max};
}

template <typename T>
KERNELSPEC FVal<T> get_phi_from_s23(
    IVal<T> phi_choice,
    FVal<T> m0_2,
    FVal<T> s23,
    FVal<T> s12,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2
) {
    // Computes the azimuthal angle phi from s23, using the formulae
    // in appendix B of 10.1103/PhysRev.187.2008
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = kaellen<T>(s12, ma_2, t2);
    auto cos_phi = (lambda * (s23 - m0_2 - m1_2) - 8 * V) / (8 * sqrtGG);
    auto phi = where(phi_choice == 1, -acos(cos_phi), acos(cos_phi));
    return phi;
}

// Kernels

template <typename T>
KERNELSPEC void kernel_three_body_decay_com(
    FIn<T, 0> r_e1,
    FIn<T, 0> r_e2,
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> r_beta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> m3,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 1> p3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay<T>(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3);
    auto p1_tmp = decay_out.first;
    auto p2_tmp = decay_out.second;
    auto det_tmp = decay_out.third;
    store_mom<T>(p1, p1_tmp);
    store_mom<T>(p2, p2_tmp);
    det = det_tmp;
    auto e3 = m0 - p1_tmp[0] - p2_tmp[0];
    p3[0] = max(e3, 0.);
    p3[1] = -p1_tmp[1] - p2_tmp[1];
    p3[2] = -p1_tmp[2] - p2_tmp[2];
    p3[3] = -p1_tmp[3] - p2_tmp[3];
}

template <typename T>
KERNELSPEC void kernel_three_body_decay_com_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FOut<T, 0> r_e1,
    FOut<T, 0> r_e2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> r_beta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> m3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay_inverse<T>(load_mom<T>(p1), load_mom<T>(p2), load_mom<T>(p3));
    r_e1 = decay_out.first;
    r_e2 = decay_out.second;
    r_phi = decay_out.third;
    r_cos_theta = decay_out.fourth;
    r_beta = decay_out.fifth;
    m0 = decay_out.sixth;
    m1 = decay_out.seventh;
    m2 = decay_out.eighth;
    m3 = decay_out.ninth;
    det = decay_out.tenth;
}

template <typename T>
KERNELSPEC void kernel_three_body_decay(
    FIn<T, 0> r_e1,
    FIn<T, 0> r_e2,
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> r_beta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> m3,
    FIn<T, 1> p0,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 1> p3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay<T>(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3);
    auto p1_tmp = decay_out.first;
    auto p2_tmp = decay_out.second;
    auto det_tmp = decay_out.third;
    store_mom<T>(p1, boost<T>(p1_tmp, load_mom<T>(p0), 1.));
    store_mom<T>(p2, boost<T>(p2_tmp, load_mom<T>(p0), 1.));
    det = det_tmp;
    auto e3 = p0[0] - p1[0] - p2[0];
    p3[0] = max(e3, 0.);
    p3[1] = p0[1] - p1[1] - p2[1];
    p3[2] = p0[2] - p1[2] - p2[2];
    p3[3] = p0[3] - p1[3] - p2[3];
}

template <typename T>
KERNELSPEC void kernel_three_body_decay_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FOut<T, 0> r_e1,
    FOut<T, 0> r_e2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> r_beta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> m3,
    FOut<T, 1> p0,
    FOut<T, 0> det
) {
    // Define total momentum
    FourMom<T> ptot;
    for (int i = 0; i < 4; ++i) {
        ptot[i] = p1[i] + p2[i] + p3[i];
    }
    store_mom<T>(p0, ptot);
    auto p1_com = boost<T>(load_mom<T>(p1), ptot, -1.);
    auto p2_com = boost<T>(load_mom<T>(p2), ptot, -1.);
    auto p3_com = boost<T>(load_mom<T>(p3), ptot, -1.);
    auto decay_out = three_body_decay_inverse<T>(p1_com, p2_com, p3_com);
    r_e1 = decay_out.first;
    r_e2 = decay_out.second;
    r_phi = decay_out.third;
    r_cos_theta = decay_out.fourth;
    r_beta = decay_out.fifth;
    m0 = decay_out.sixth;
    m1 = decay_out.seventh;
    m2 = decay_out.eighth;
    m3 = decay_out.ninth;
    det = decay_out.tenth;
}

// Kernels for 2->3 scattering

template <typename T>
KERNELSPEC void kernel_s23_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_tot, p_12, pt2;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
        p_12[i] = pa[i] + pb[i] - p3[i];
        pt2[i] = pb[i] - p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto s12 = lsquare<T>(p_12);
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto t2 = lsquare<T>(pt2);

    auto s23_out = s23_min_max<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    s23_min = s23_out.first;
    s23_max = s23_out.second;
}

template <typename T>
KERNELSPEC void kernel_s23_value_and_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> s_23,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_tot, p_12, pt2, p_23;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
        p_12[i] = p1[i] + p2[i];
        pt2[i] = pb[i] - p3[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto s12 = lsquare<T>(p_12);
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto t2 = lsquare<T>(pt2);

    auto s23_out = s23_min_max<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    s23_min = s23_out.first;
    s23_max = s23_out.second;
    s_23 = lsquare<T>(p_23);
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering(
    IIn<T, 0> phi_index,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> s23,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_12, p_c, p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
        p_12[i] = pa[i] + pb[i] - p3[i];
        p_c[i] = pb[i] - p3[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m0_2 = lsquare<T>(p_tot);
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto phi = get_phi_from_s23<T>(
        phi_index, m0_2, s23, s12, t1_abs, t2, ma_2, mb_2, m1_2, m2_2, m3_2
    );

    auto scatter_out =
        p1com_from_tabs_phi<T>(pa_com, s12, phi, t1_abs, m1, m2, ma_2, t2);
    auto p1_com = scatter_out.first;
    auto gram4 = bk_gram4<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12, s23);
    auto det_2to3 = 1 / (8 * sqrt(max(-gram4, EPS2)));
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto p1_rot = rotate_two_ref<T>(p1_com, pa_com, p3_p12);
    auto p1_lab = boost<T>(p1_rot, p_12, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_12[i] - p1_lab[i];
    }
    det = det_2to3 / 2; // factor 1/2 as acos allows for two choices of phi
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t1_abs,
    FIn<T, 0> s23,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    IOut<T, 0> phi_index,
    FOut<T, 0> det
) {
    FourMom<T> p_12, p_c, p_tot;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
        p_tot[i] = p_12[i] + p3[i];
        p_c[i] = pb[i] - p3[i];
    }
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m0_2 = lsquare<T>(p_tot);
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);

    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto p1_com = boost<T>(load_mom<T>(p1), p_12, -1.);
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto p1_rot = rotate_two_ref_inverse<T>(p1_com, pa_com, p3_p12);

    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto phi = atan2(p1_rot[2], p1_rot[1]);
    phi_index = where(
        phi < 0, IVal<T>(1), IVal<T>(0)
    ); // choose phi index based on the value of phi

    auto gram4 = bk_gram4<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12, s23);
    auto det_2to3 = 8 * sqrt(max(-gram4, EPS2));
    m1 = sqrt(max(EPS2, m1_2));
    m2 = sqrt(max(EPS2, m2_2));
    det = 2 * det_2to3;
}

} // namespace kernels
} // namespace madspace
