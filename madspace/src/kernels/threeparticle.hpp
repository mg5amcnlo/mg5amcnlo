#pragma once

#include "kinematics.hpp"

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

// Width given to an empty s23 range, relative to s23, since an absolute EPS can be
// below one ulp of s23.
template <typename T>
KERNELSPEC FVal<T> s23_min_width(FVal<T> s) {
    return max(FVal<T>(EPS), 1e-10 * fabs(s));
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_min_max(
    FourMom<T> pa,
    FourMom<T> p3,
    FourMom<T> p_12,
    FVal<T> s12,
    FVal<T> t1_abs,
    FVal<T> m1,
    FVal<T> m2
) {
    // Range of s23 = (p_12 + p3 - p1)^2 as p1 turns about pa at fixed s12 and t1.
    // In the p_12 rest frame, with z along pa and p3 in the phi = 0 half-plane
    // (the frame of rotate_two_ref),
    //   s23 = m0^2 + m1^2 - 2 [(sqrt(s12) + E3) E1 - p3_z p1_z]
    //         + 2 p3_x p1_t cos(phi),
    // so the range has width 4 p3_x p1_t. Every factor of the width is read off
    // momenta, which keeps its relative precision when a particle is soft.
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = p_12[i] + p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(pa);
    auto pa_com = boost<T>(pa, p_12, -1.);
    auto p3_com = boost<T>(p3, p_12, -1.);

    // p1 at phi = 0: energy, momentum along pa (p1_z) and across it (p1_t >= 0)
    auto p1_out = p1com_from_tabs_phi<T>(
        pa_com, s12, FVal<T>(0.), t1_abs, m1, m2, ma_2, FVal<T>(0.)
    );
    auto p1_com = p1_out.first;

    // p3 along pa (p3_z) and across it (p3_x >= 0)
    auto pa_mag = sqrt(max(esquare<T>(pa_com), EPS2));
    auto p3_z =
        (p3_com[1] * pa_com[1] + p3_com[2] * pa_com[2] + p3_com[3] * pa_com[3]) /
        pa_mag;
    auto cross_x = p3_com[2] * pa_com[3] - p3_com[3] * pa_com[2];
    auto cross_y = p3_com[3] * pa_com[1] - p3_com[1] * pa_com[3];
    auto cross_z = p3_com[1] * pa_com[2] - p3_com[2] * pa_com[1];
    auto p3_x =
        sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z) / pa_mag;

    auto center = m0_2 + m1 * m1 -
        2. * ((sqrt(max(s12, 0.)) + p3_com[0]) * p1_com[0] - p3_z * p1_com[3]);
    auto half_width = 2. * p3_x * p1_com[1];
    return {center - half_width, center + half_width};
}

// The range as handed to the s23 sampling. It is empty when p1 or p3 is along
// pa (t1 at an end of its range, or a collinear or vanishing p3): the azimuth
// is then undefined and the weight zero. A small width there keeps the
// sampling and its inverse finite; the Jacobian of the scattering kernel keeps
// using the true width and stays zero.
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_sampling_range(Pair<FVal<T>, FVal<T>> range) {
    auto lo = range.first, hi = range.second;
    return {lo, where(hi - lo > s23_min_width<T>(lo), hi, lo + s23_min_width<T>(lo))};
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
    FIn<T, 1> p12,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    auto p_12 = load_mom<T>(p12);
    auto s23_out = s23_sampling_range<T>(s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(p3), p_12, lsquare<T>(p_12), t1_abs, m1, m2
    ));
    s23_min = s23_out.first;
    s23_max = s23_out.second;
}

template <typename T>
KERNELSPEC void kernel_s23_value_and_min_max(
    FIn<T, 1> pa,
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
    FourMom<T> p_12, p_23;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = p1[i] + p2[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m1 = sqrt(max(lsquare<T>(load_mom<T>(p1)), 0.));
    auto m2 = sqrt(max(lsquare<T>(load_mom<T>(p2)), 0.));

    auto s23_out = s23_sampling_range<T>(s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(p3), p_12, lsquare<T>(p_12), t1_abs, m1, m2
    ));
    s23_min = s23_out.first;
    s23_max = s23_out.second;
    s_23 = lsquare<T>(p_23);
}

// AmpliCol block-B s-channel ETmin refinement (gen23 lines 745-762). Tightens
// the sampled s-channel mass s23 = (peeled + im1)^2 using the recoil's ETmin
// floor (upper bound, smax) and the peeled particle's ETmin + dR cut (lower
// bound, smin -- only for a massless peeled particle).
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_etmin_clamp(
    FVal<T> smn,
    FVal<T> smx,
    FourMom<T> piir,
    FourMom<T> pim1,
    FourMom<T> pib,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> etmin_1,
    FVal<T> etmin_2,
    FVal<T> drcut
) {
    auto active = (etmin_2 > EPS) | (etmin_1 > EPS);
    // Block B works in the frame where pz(im1)=0 (z-boost by the IM1 rapidity),
    // with im1 rotated onto +x. mT(im1)=sqrt(E^2-pz^2)=sqrt(m3^2+pt(p3)^2).
    auto Eim1 = pim1[0];
    auto pzim1 = pim1[3];
    auto mT_im1 = sqrt(max(Eim1 * Eim1 - pzim1 * pzim1, EPS));
    // Boost p_12 (=piir, the i+ir system) and pa (=pib) into that frame; im1's
    // own energy there is mT(im1) and its x-momentum is pt(p3).
    auto piir_0 = (piir[0] * Eim1 - piir[3] * pzim1) / mT_im1;
    auto pim1_0 = mT_im1;
    auto pim1_1 = sqrt(pim1[1] * pim1[1] + pim1[2] * pim1[2]);
    auto pib_0 = max((pib[0] * Eim1 - pib[3] * pzim1) / mT_im1, EPS);
    // Block-B recoil ET floor (t-dependent): invm(ir)-invm(ir+ib) = m1_2 + t1_abs.
    auto denom = max(m1_2 + t1_abs, EPS);
    auto etminir_b =
        max(etmin_1, pib_0 * etmin_1 * etmin_1 / denom + denom / (4. * pib_0));
    // smax: the recoil's ETmin caps the s-channel mass from above.
    auto e_eff = piir_0 - etminir_b;
    auto disc = e_eff * e_eff - m2_2;
    auto smax_b = m2_2 + m3_2 + 2. * e_eff * pim1_0 + 2. * sqrt(max(disc, 0.)) * pim1_1;
    auto smax_ok = active & (e_eff > 0.) & (disc > 0.) & (smax_b > smn);
    auto smx_new = where(smax_ok, min(smx, smax_b), smx);
    // smin: a massless peeled particle needs ET -> minimum s-channel mass. Masslessness
    // is judged relative to the energy scale, as m2_2 read off momenta is only ~1e-16
    // E^2.
    auto smin_b = 2. * etmin_2 * (pim1_0 - pim1_1 * cos(drcut));
    auto massless_2 = m2_2 < 1e-10 * piir_0 * piir_0;
    auto smin_ok = active & massless_2 & (smin_b > smn) & (smin_b < smx_new);
    auto smn_new = where(smin_ok, max(smn, smin_b), smn);
    smx_new = where(smx_new > smn_new, smx_new, smn_new + s23_min_width<T>(smn_new));
    return {smn_new, smx_new};
}

// Clamp the s23 range against the lower bound s23_min_cut (minimum mass^2 of the
// (2,3) system implied by the pt / m_inv_min / dR cuts, gen23's invm_min) and
// the ETmin refinement of s23_etmin_clamp.
template <typename T>
KERNELSPEC void kernel_s23_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> p12,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FIn<T, 0> drcut,
    FIn<T, 0> s23_min_cut,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    auto p_12 = load_mom<T>(p12);
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto s23_out = s23_sampling_range<T>(s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(p3), p_12, lsquare<T>(p_12), t1_abs, m1, m2
    ));
    auto smn = s23_out.first;
    auto smx = s23_out.second;

    FVal<T> smin_cut(s23_min_cut);
    smn = where(smin_cut > 0., max(smn, smin_cut), smn);
    smx = where(smx > smn, smx, smn + s23_min_width<T>(smn));

    // Block-B ETmin refinement: piir = p_12 (recoil system), pim1 = p3, pib = pa.
    auto sb = s23_etmin_clamp<T>(
        smn,
        smx,
        p_12,
        load_mom<T>(p3),
        load_mom<T>(pa),
        m1_2,
        m2_2,
        m3_2,
        t1_abs,
        etmin_1,
        etmin_2,
        drcut
    );

    s23_min = sb.first;
    s23_max = sb.second;
}

template <typename T>
KERNELSPEC void kernel_s23_value_and_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FIn<T, 0> drcut,
    FIn<T, 0> s23_min_cut,
    FOut<T, 0> s_23,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    FourMom<T> p_12, p_23;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = p1[i] + p2[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));

    auto s23_out = s23_sampling_range<T>(s23_min_max<T>(
        load_mom<T>(pa),
        load_mom<T>(p3),
        p_12,
        lsquare<T>(p_12),
        t1_abs,
        sqrt(max(m1_2, 0.)),
        sqrt(max(m2_2, 0.))
    ));
    auto smn = s23_out.first;
    auto smx = s23_out.second;

    FVal<T> smin_cut(s23_min_cut);
    smn = where(smin_cut > 0., max(smn, smin_cut), smn);
    smx = where(smx > smn, smx, smn + s23_min_width<T>(smn));

    // Block-B ETmin refinement (must mirror the forward kernel exactly so the
    // sampled s23 range is identical and the round-trip stays invertible).
    auto sb = s23_etmin_clamp<T>(
        smn,
        smx,
        p_12,
        load_mom<T>(p3),
        load_mom<T>(pa),
        m1_2,
        m2_2,
        m3_2,
        t1_abs,
        etmin_1,
        etmin_2,
        drcut
    );

    s23_min = sb.first;
    s23_max = sb.second;
    s_23 = lsquare<T>(p_23);
}

// Arcsine map of the s23 sampling variable.
//
// At fixed s12, t1 and t2, s23 is linear in cos(phi) on its physical range
// [s_phys_min, s_phys_max] and the 2->3 measure is flat in phi. A sampling density
// that stays finite at the edges of that range thus leaves an integrable
// 1/|sin(phi)| in the weight with a log-divergent variance. With
//   u = (s23 - s_phys_min) / (s_phys_max - s_phys_min) = cos^2(phi / 2),
// the map is flat in theta = (pi - phi) / 2 on the part [theta_a, theta_b] of [0, pi/2]
// covered by the sampling range [s_min, s_max], and gives the position x in [0, 1]:
//   x = (sin^2 theta - sin^2 theta_a) / (sin^2 theta_b - sin^2 theta_a),
//   theta = theta_a + (theta_b - theta_a) r.
// dx/dr vanishes like |sin(phi)| at a physical edge, which cancels the 1/|sin(phi)|
// (not at an edge set by a cut). Differences of sin^2 are written as
// sin(B - A) sin(B + A) to keep relative precision. This function returns
// theta_a and theta_b; kernel_s23_arcsine_sample applies the map.
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_arcsine_angles(
    FVal<T> s_min, FVal<T> s_max, FVal<T> s_phys_min, FVal<T> s_phys_max
) {
    auto width = s_phys_max - s_phys_min;
    auto width_safe = where(width > 0., width, FVal<T>(1.));
    // u and 1 - u of both ends, each from its own difference
    auto ua = min(max((s_min - s_phys_min) / width_safe, 0.), 1.);
    auto ua_c = min(max((s_phys_max - s_min) / width_safe, 0.), 1.);
    auto ub = min(max((s_max - s_phys_min) / width_safe, 0.), 1.);
    auto ub_c = min(max((s_phys_max - s_max) / width_safe, 0.), 1.);
    auto theta_a = atan2(sqrt(ua), sqrt(ua_c));
    auto theta_b = atan2(sqrt(ub), sqrt(ub_c));
    // an empty physical range leaves nothing to map: theta_b = theta_a
    return {theta_a, where(width > 0., theta_b, theta_a)};
}

// Position of s23 in its kinematic range, u = (s23 - s_phys_min) / width and
// 1 - u = (s_phys_max - s23) / width, each from its own difference. Used when
// s23 itself is sampled; the arcsine sampling below gives u and 1 - u
// directly and keeps their relative precision at the edges of the range.
template <typename T>
KERNELSPEC void kernel_s23_position(
    FIn<T, 0> s23,
    FIn<T, 0> s_phys_min,
    FIn<T, 0> s_phys_max,
    FOut<T, 0> u,
    FOut<T, 0> u_c
) {
    FVal<T> s23_val(s23), lo(s_phys_min), hi(s_phys_max);
    auto width = hi - lo;
    auto width_safe = where(width > 0., width, FVal<T>(1.));
    u = min(max((s23_val - lo) / width_safe, 0.), 1.);
    u_c = min(max((hi - s23_val) / width_safe, 0.), 1.);
}

// Power-law s23 importance sampling (flat for power 0, logarithmic for power 1,
// 1/(s - m^2)^power otherwise, with m^2 -> m^2 - 1e-2) from x and 1 - x. Returns
// s23 - s_min, s_max - s23 and d s23 / d x. Both distances are computed through
// expm1 and log1p so that they keep relative precision at the range edges.
template <typename T>
KERNELSPEC Triplet<FVal<T>, FVal<T>, FVal<T>> s23_power_offsets(
    FVal<T> x, FVal<T> x_c, FVal<T> s_min, FVal<T> s_max, FVal<T> power, FVal<T> mass
) {
    auto m2 = mass * mass - 1e-2;
    auto qa = max(s_min - m2, EPS2);
    auto qb = max(s_max - m2, EPS2);
    auto log_ratio = log(qb / qa);
    // flat
    auto width = s_max - s_min;
    auto flat_lo = width * x;
    auto flat_hi = width * x_c;
    // logarithmic: q = qa exp(x L) = qb exp(-x_c L)
    auto log_lo = qa * expm1(x * log_ratio);
    auto log_hi = -qb * expm1(-x_c * log_ratio);
    auto log_det = (qa + log_lo) * log_ratio;
    // power law: q^p = qa^p (1 + x R) = qb^p (1 - x_c R / (1 + R)), p = 1 - power
    auto p = 1. - power;
    auto p_safe = where(fabs(p) > 0., p, FVal<T>(1.));
    auto big_r = expm1(p_safe * log_ratio);
    auto pow_lo = qa * expm1(log1p(x * big_r) / p_safe);
    auto pow_hi = -qb * expm1(log1p(-x_c * big_r / (1. + big_r)) / p_safe);
    auto q = qa + pow_lo;
    auto pow_det = pow(qa, p_safe) * big_r * pow(q, power) / p_safe;

    auto is_flat = power == 0.;
    auto is_log = power == 1.;
    auto lo = where(is_flat, flat_lo, where(is_log, log_lo, pow_lo));
    auto hi = where(is_flat, flat_hi, where(is_log, log_hi, pow_hi));
    auto det = where(is_flat, width, where(is_log, log_det, pow_det));
    return {max(lo, 0.), max(hi, 0.), det};
}

// Arcsine map in phi (see s23_arcsine_angles) composed with the s23 importance
// sampling (s23_power_offsets) in one step, r -> x -> s23. It returns u and 1 - u
// of s23 in its physical range, from separate distances to both ends, since going
// through s23 itself would lose relative precision near the edges.
template <typename T>
KERNELSPEC void kernel_s23_arcsine_sample(
    FIn<T, 0> r,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FIn<T, 0> s_phys_min,
    FIn<T, 0> s_phys_max,
    FSIn<T, 0> power,
    FSIn<T, 0> mass,
    FOut<T, 0> u,
    FOut<T, 0> u_c,
    FOut<T, 0> det
) {
    auto angles = s23_arcsine_angles<T>(s_min, s_max, s_phys_min, s_phys_max);
    auto theta_a = angles.first;
    auto theta_b = angles.second;
    auto dtheta = theta_b - theta_a;
    auto den = sin(dtheta) * sin(theta_a + theta_b);
    auto ok = (dtheta > 0.) & (den > 0.);
    auto den_safe = where(ok, den, FVal<T>(1.));

    // x = (sin^2 theta - sin^2 theta_a) / den, 1 - x from the other end
    FVal<T> r_val(r);
    auto x_arc = sin(dtheta * r_val) * sin(2. * theta_a + dtheta * r_val) / den_safe;
    auto x_c_arc = sin(dtheta * (1. - r_val)) *
        sin(2. * theta_b - dtheta * (1. - r_val)) / den_safe;
    // dx/dr = dtheta sin(2 theta) / den, sin(2 theta) = 2 sqrt(v (1 - v))
    auto sin_a = sin(theta_a);
    auto cos_b = cos(theta_b);
    auto v = max(sin_a * sin_a + x_arc * den_safe, 0.);
    auto v_c = max(cos_b * cos_b + x_c_arc * den_safe, 0.);
    auto dx_dr = dtheta * 2. * sqrt(v * v_c) / den_safe;
    auto x = where(ok, min(max(x_arc, 0.), 1.), r_val);
    auto x_c = where(ok, min(max(x_c_arc, 0.), 1.), 1. - r_val);
    dx_dr = where(ok, dx_dr, FVal<T>(1.));

    auto offsets = s23_power_offsets<T>(x, x_c, s_min, s_max, power, mass);
    FVal<T> lo(s_phys_min), hi(s_phys_max), a(s_min), b(s_max);
    auto width = hi - lo;
    auto width_safe = where(width > 0., width, FVal<T>(1.));
    u = min(max(((a - lo) + offsets.first) / width_safe, 0.), 1.);
    u_c = min(max(((hi - b) + offsets.second) / width_safe, 0.), 1.);
    det = dx_dr * offsets.third;
}

template <typename T>
KERNELSPEC void kernel_s23_arcsine_sample_inverse(
    FIn<T, 0> u,
    FIn<T, 0> u_c,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FIn<T, 0> s_phys_min,
    FIn<T, 0> s_phys_max,
    FSIn<T, 0> power,
    FSIn<T, 0> mass,
    FOut<T, 0> r,
    FOut<T, 0> det
) {
    FVal<T> lo(s_phys_min), hi(s_phys_max), a(s_min), b(s_max), pw(power);
    auto width = hi - lo;
    // distances of s23 to both ends of the sampled range
    auto d_lo = max(width * u - (a - lo), 0.);
    auto d_hi = max(width * u_c - (hi - b), 0.);

    // invert the power map from both ends
    auto m2 = FVal<T>(mass) * FVal<T>(mass) - 1e-2;
    auto qa = max(a - m2, EPS2);
    auto qb = max(b - m2, EPS2);
    auto log_ratio = log(qb / qa);
    auto range = b - a;
    auto range_safe = where(range > 0., range, FVal<T>(1.));
    auto flat_x = d_lo / range_safe;
    auto flat_x_c = d_hi / range_safe;
    auto log_safe = where(fabs(log_ratio) > 0., log_ratio, FVal<T>(1.));
    auto log_x = log1p(d_lo / qa) / log_safe;
    auto log_x_c = -log1p(-d_hi / qb) / log_safe;
    auto p = 1. - pw;
    auto p_safe = where(fabs(p) > 0., p, FVal<T>(1.));
    auto big_r = expm1(p_safe * log_ratio);
    auto big_r_safe = where(fabs(big_r) > 0., big_r, FVal<T>(1.));
    auto pow_x = expm1(p_safe * log1p(d_lo / qa)) / big_r_safe;
    auto pow_x_c = -expm1(p_safe * log1p(-d_hi / qb)) * (1. + big_r) / big_r_safe;
    auto is_flat = pw == 0.;
    auto is_log = pw == 1.;
    auto x = min(max(where(is_flat, flat_x, where(is_log, log_x, pow_x)), 0.), 1.);
    auto x_c =
        min(max(where(is_flat, flat_x_c, where(is_log, log_x_c, pow_x_c)), 0.), 1.);
    auto offsets = s23_power_offsets<T>(x, x_c, a, b, pw, FVal<T>(mass));

    // invert the arcsine map from (x, 1 - x)
    auto angles = s23_arcsine_angles<T>(s_min, s_max, s_phys_min, s_phys_max);
    auto theta_a = angles.first;
    auto theta_b = angles.second;
    auto dtheta = theta_b - theta_a;
    auto den = sin(dtheta) * sin(theta_a + theta_b);
    auto ok = (dtheta > 0.) & (den > 0.);
    auto den_safe = where(ok, den, FVal<T>(1.));
    auto dtheta_safe = where(ok, dtheta, FVal<T>(1.));
    auto sin_a = sin(theta_a);
    auto cos_b = cos(theta_b);
    auto v = max(sin_a * sin_a + x * den_safe, 0.);
    auto v_c = max(cos_b * cos_b + x_c * den_safe, 0.);
    auto theta = atan2(sqrt(v), sqrt(v_c));
    auto r_arc = (theta - theta_a) / dtheta_safe;
    auto dr_dx =
        where(v * v_c > 0., den_safe / (dtheta_safe * 2. * sqrt(v * v_c)), FVal<T>(0.));
    r = where(ok, min(max(r_arc, 0.), 1.), x);
    dr_dx = where(ok, dr_dx, FVal<T>(1.));
    det = where(offsets.third > 0., dr_dx / offsets.third, FVal<T>(0.));
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering(
    IIn<T, 0> phi_index,
    FIn<T, 1> pa,
    FIn<T, 1> p12,
    FIn<T, 1> p3,
    FIn<T, 0> u,
    FIn<T, 0> u_c,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    // p_12 is an input rather than pa + pb - p3, which would carry an absolute error
    // of order the beam energy when p_12 is soft.
    auto p_12 = load_mom<T>(p12);
    FourMom<T> p_c;
    for (int i = 0; i < 4; ++i) {
        p_c[i] = p_12[i] - pa[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);

    // On the s23 range, cos(phi) = 2 u - 1 with u = (s23 - s23_min) / (s23_max -
    // s23_min) and -G4 = lambda(s12, ma^2, t2) / 16 * (s23 - s23_min) * (s23_max -
    // s23). Both are evaluated through u and 1 - u (inputs) and a range from momenta
    // (s23_min_max), as forming G4 from the invariants cancels catastrophically for a
    // soft particle.
    auto s23_range =
        s23_min_max<T>(load_mom<T>(pa), load_mom<T>(p3), p_12, s12, t1_abs, m1, m2);
    auto s23_width = s23_range.second - s23_range.first;
    FVal<T> u_val(u), u_c_val(u_c);
    // cos(phi) = u - (1 - u), |sin(phi)| = 2 sqrt(u (1 - u)); dividing by u + (1 - u)
    // keeps cos^2 + sin^2 = 1 to rounding so that the rotation preserves |p1|.
    auto u_sum = u_val + u_c_val;
    auto u_sum_safe = where(u_sum > 0., u_sum, FVal<T>(1.));
    auto cos_phi = (u_val - u_c_val) / u_sum_safe;
    auto sin_phi_abs = 2. * sqrt(max(u_val * u_c_val, 0.)) / u_sum_safe;
    auto sin_phi = where(phi_index == 1, -sin_phi_abs, sin_phi_abs);
    // 8 sqrt(-G4) = sqrt(lambda) * (s23_max - s23_min) * |sin(phi)|
    auto sqrt_neg_gram4_x8 =
        sqrt(max(kaellen<T>(s12, ma_2, t2), 0.)) * s23_width * sin_phi_abs;
    // zero weight at the edges of the range (sin(phi) = 0)
    auto det_2to3 = where(sqrt_neg_gram4_x8 > 0., 1. / sqrt_neg_gram4_x8, FVal<T>(0.));

    auto scatter_out =
        p1com_from_tabs_cs<T>(pa_com, s12, cos_phi, sin_phi, t1_abs, m1, m2, ma_2, t2);
    auto p1_com = scatter_out.first;
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto p1_rot = rotate_two_ref<T>(p1_com, pa_com, p3_p12);
    // p1 and p2 back to back in the p_12 rest frame, boosted as in boost_two_body
    auto m12_rest = sqrt(max(s12, EPS2));
    auto e2_com = 0.5 * (m12_rest - (m1 - m2) * (m1 + m2) / m12_rest);
    FourMom<T> p2_rot{max(e2_com, 0.), -p1_rot[1], -p1_rot[2], -p1_rot[3]};
    auto p_out = boost_two_body<T>(p1_rot, m1 * m1, p2_rot, m2 * m2, p_12, s12);
    store_mom<T>(p1, p_out.first);
    store_mom<T>(p2, p_out.second);
    det = det_2to3 / 2; // factor 1/2 as acos allows for two choices of phi
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FIn<T, 1> pa,
    FIn<T, 1> p12,
    FIn<T, 0> t1_abs,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    IOut<T, 0> phi_index,
    FOut<T, 0> u,
    FOut<T, 0> u_c,
    FOut<T, 0> det
) {
    // p_12 is an input rather than pa + pb - p3, which would carry an absolute error
    // of order the beam energy when p_12 is soft.
    auto p_12 = load_mom<T>(p12);
    FourMom<T> p_c;
    for (int i = 0; i < 4; ++i) {
        p_c[i] = p_12[i] - pa[i];
    }
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
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

    // 8 sqrt(-G4) = sqrt(lambda) * (s23_max - s23_min) * |sin(phi)|, see the
    // forward kernel; |sin(phi)| is read off the momenta directly.
    auto s23_range = s23_min_max<T>(
        load_mom<T>(pa),
        load_mom<T>(p3),
        p_12,
        s12,
        t1_abs,
        sqrt(max(m1_2, 0.)),
        sqrt(max(m2_2, 0.))
    );
    auto pt2_rot = p1_rot[1] * p1_rot[1] + p1_rot[2] * p1_rot[2];
    auto sin_phi_abs =
        where(pt2_rot > 0., fabs(p1_rot[2]) / sqrt(pt2_rot), FVal<T>(0.));
    // u = (1 + cos phi) / 2 and 1 - u read off the momentum, the one close to
    // zero as p_y^2 / (2 p_t (p_t -+ p_x)) rather than as a difference
    auto pt_rot = sqrt(pt2_rot);
    auto px = p1_rot[1], py2 = p1_rot[2] * p1_rot[2];
    auto pt_safe = where(pt2_rot > 0., pt_rot, FVal<T>(1.));
    auto sum_pos = pt_safe + fabs(px);
    auto big = sum_pos / (2. * pt_safe);
    auto small = py2 / (2. * pt_safe * sum_pos);
    auto forward = px >= 0.;
    u = where(pt2_rot > 0., where(forward, big, small), FVal<T>(0.5));
    u_c = where(pt2_rot > 0., where(forward, small, big), FVal<T>(0.5));
    auto det_2to3 = sqrt(max(kaellen<T>(s12, ma_2, t2), 0.)) *
        (s23_range.second - s23_range.first) * sin_phi_abs;
    m1 = sqrt(max(EPS2, m1_2));
    m2 = sqrt(max(EPS2, m2_2));
    det = 2 * det_2to3;
}

// Whether the solution of the 2->3 step with the given sign of sin(phi) passes
// the transverse-energy floors, E^2 - p_z^2 >= etmin^2, of p1 and p2. The
// momenta are formed as in kernel_two_to_three_particle_scattering.
template <typename T>
KERNELSPEC BVal<T> two_to_three_branch_passes(
    FourMom<T> pa_com,
    FourMom<T> p3_p12,
    FourMom<T> p_12,
    FVal<T> s12,
    FVal<T> cos_phi,
    FVal<T> sin_phi,
    FVal<T> t1_abs,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> ma_2,
    FVal<T> t2,
    FVal<T> etmin_1,
    FVal<T> etmin_2
) {
    auto p1_com =
        p1com_from_tabs_cs<T>(pa_com, s12, cos_phi, sin_phi, t1_abs, m1, m2, ma_2, t2)
            .first;
    auto p1 = boost<T>(rotate_two_ref<T>(p1_com, pa_com, p3_p12), p_12, 1.);
    FourMom<T> p2;
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_12[i] - p1[i];
    }
    auto mt2_1 = (p1[0] - p1[3]) * (p1[0] + p1[3]);
    auto mt2_2 = (p2[0] - p2[3]) * (p2[0] + p2[3]);
    return ((etmin_1 <= 0.) | (mt2_1 >= etmin_1 * etmin_1)) &
        ((etmin_2 <= 0.) | (mt2_2 >= etmin_2 * etmin_2));
}

// Choice between the two azimuthal solutions of the 2->3 step as in AmpliCol's
// gen23_one_step: both are formed, and only those passing the
// transverse-energy floors of p1 and p2 are kept. If both pass, index_in picks
// one (factor 1); if one passes it is taken whatever index_in is, and the two
// values of index_in lead to the same point (factor 1/2); if none passes the
// weight is zero. The factor is that of a uniformly sampled index_in.
template <typename T>
KERNELSPEC void kernel_two_to_three_branch_choice(
    IIn<T, 0> index_in,
    FIn<T, 1> pa,
    FIn<T, 1> p12,
    FIn<T, 1> p3,
    FIn<T, 0> u,
    FIn<T, 0> u_c,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    IOut<T, 0> index_out,
    FOut<T, 0> factor,
    FOut<T, 0> inv_factor
) {
    auto p_12 = load_mom<T>(p12);
    auto p_a = load_mom<T>(pa);
    FourMom<T> p_c;
    for (int i = 0; i < 4; ++i) {
        p_c[i] = p_12[i] - p_a[i];
    }
    auto pa_com = boost<T>(p_a, p_12, -1.);
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto ma_2 = lsquare<T>(p_a);
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);
    FVal<T> u_val(u), u_c_val(u_c), e1(etmin_1), e2(etmin_2);
    auto u_sum = u_val + u_c_val;
    auto u_sum_safe = where(u_sum > 0., u_sum, FVal<T>(1.));
    auto cos_phi = (u_val - u_c_val) / u_sum_safe;
    auto sin_phi_abs = 2. * sqrt(max(u_val * u_c_val, 0.)) / u_sum_safe;
    // index 0: sin(phi) >= 0, index 1: sin(phi) < 0 (as the scattering kernel)
    auto pass_0 = two_to_three_branch_passes<T>(
        pa_com,
        p3_p12,
        p_12,
        s12,
        cos_phi,
        sin_phi_abs,
        t1_abs,
        m1,
        m2,
        ma_2,
        t2,
        e1,
        e2
    );
    auto pass_1 = two_to_three_branch_passes<T>(
        pa_com,
        p3_p12,
        p_12,
        s12,
        cos_phi,
        -sin_phi_abs,
        t1_abs,
        m1,
        m2,
        ma_2,
        t2,
        e1,
        e2
    );
    IVal<T> index(index_in);
    index_out = where(
        pass_0 & pass_1,
        index,
        where(pass_0, IVal<T>(0), where(pass_1, IVal<T>(1), index))
    );
    factor = where(
        pass_0 & pass_1, FVal<T>(1.), where(pass_0 | pass_1, FVal<T>(0.5), FVal<T>(0.))
    );
    inv_factor = where(
        pass_0 & pass_1, FVal<T>(1.), where(pass_0 | pass_1, FVal<T>(2.), FVal<T>(0.))
    );
}

} // namespace kernels
} // namespace madspace
