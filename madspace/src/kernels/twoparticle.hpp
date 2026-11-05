#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// Helper functions

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>>
two_body_decay(FVal<T> r_phi, FVal<T> r_cos_theta, FVal<T> m0, FVal<T> m1, FVal<T> m2) {
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto m0_clip = max(m0, EPS);

    // this part is based on the mom2cx subroutine from HELAS
    // used in MG5 (aloha_functions.f)
    auto ed = (m1 - m2) * (m1 + m2) / m0_clip;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + m0 * m0;
    auto pp_open = 0.5 * where(m1 * m2 == 0., m0 - fabs(ed), sqrt(max(pp2, EPS)));
    auto e1_open = 0.5 * (m0 + ed);

    // Below threshold (m0 < m1 + m2) there is no phase space: substitute a massless
    // splitting to keep p1 physical (no NaNs in boost) and set det = 0. pp2 > 0 is
    // the threshold test; pp_open > 0 covers the massless shortcut going negative.
    auto open = (pp2 > 0.) & (pp_open > 0.);
    auto pp = where(open, pp_open, 0.5 * m0_clip);
    auto e1 = where(open, e1_open, 0.5 * m0_clip);

    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    FourMom<T> p1{
        max(e1, 0.),
        pp * sin_theta * cos(phi),
        pp * sin_theta * sin(phi),
        pp * cos_theta
    };

    auto det = where(open, PI * pp_open / m0_clip, FVal<T>(0.));
    return {p1, det};
}

template <typename T>
KERNELSPEC Sextuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>>
two_body_decay_inverse(FourMom<T> p0, FourMom<T> p1_com, FourMom<T> p2) {
    auto m0 = sqrt(max(EPS2, lsquare<T>(p0)));
    auto m1 = sqrt(max(EPS2, lsquare<T>(p1_com)));
    auto m2 = sqrt(max(EPS2, lsquare<T>(p2)));

    auto pp1 = sqrt(max(EPS2, esquare<T>(p1_com)));
    auto cos_theta = p1_com[3] / pp1;
    auto phi = atan2(p1_com[2], p1_com[1]);

    auto ed = (m1 - m2) * (m1 + m2) / m0;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + m0 * m0;
    auto pp = 0.5 * where(m1 * m2 == 0., m0 - fabs(ed), sqrt(max(pp2, EPS)));
    auto det_inv = PI * pp1 / max(m0, EPS);

    auto r_phi = (phi / PI + 1.) / 2.;
    auto r_cos_theta = (cos_theta + 1.) / 2.;
    auto det = 1 / det_inv;
    return {r_phi, r_cos_theta, m0, m1, m2, det};
}

// The range of cos(theta) in a two-body decay generated in the partonic
// centre-of-mass frame that the cuts on the two products leave open. That frame
// differs from the lab by a boost along the beam with rapidity
// y = log(x1 / x2) / 2, so
//   * both products have the transverse momentum p sin(theta), and
//     p sin(theta) >= pt_min bounds |cos(theta)|,
//   * a product's lab rapidity is y + atanh(beta cos(theta)) (beta = p / E,
//     cos(theta) of the second product is minus that of the first), and
//     |rapidity| <= y_max bounds cos(theta) from both sides.
// A negative y_max stands for no bound. For the cuts these come from, y_max is
// a pseudorapidity cut, which bounds the rapidity because |y| <= |eta|. The
// returned interval can be empty (lower end above the upper one).
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> com_cos_theta_window(
    FVal<T> m0,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> x1,
    FVal<T> x2,
    FVal<T> pt_min,
    FVal<T> y_max1,
    FVal<T> y_max2
) {
    // the momentum and energies of two_body_decay
    auto m0_clip = max(m0, EPS);
    auto ed = (m1 - m2) * (m1 + m2) / m0_clip;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + m0 * m0;
    auto pp = 0.5 * where(m1 * m2 == 0., m0 - fabs(ed), sqrt(max(pp2, EPS)));
    pp = max(pp, EPS);
    auto e1 = max(0.5 * (m0 + ed), pp);
    auto e2 = max(0.5 * (m0 - ed), pp);

    auto pt_ratio = pt_min / pp;
    auto cos_pt = sqrt(max(1. - pt_ratio * pt_ratio, 0.));
    FVal<T> cos_lo = -cos_pt, cos_hi = cos_pt;

    auto y = 0.5 * log(x1 / x2);
    auto beta1 = pp / e1, beta2 = pp / e2;
    cos_lo = where(y_max1 >= 0., max(cos_lo, tanh(-y_max1 - y) / beta1), cos_lo);
    cos_hi = where(y_max1 >= 0., min(cos_hi, tanh(y_max1 - y) / beta1), cos_hi);
    cos_lo = where(y_max2 >= 0., max(cos_lo, tanh(y - y_max2) / beta2), cos_lo);
    cos_hi = where(y_max2 >= 0., min(cos_hi, tanh(y + y_max2) / beta2), cos_hi);
    return {cos_lo, cos_hi};
}

// Kernels

// Squeezes the cos(theta) random number of a two_body_decay_com into the range
// com_cos_theta_window leaves open. The Jacobian is the fraction of the
// cos(theta) range kept; it is zero when the cuts leave nothing.
template <typename T>
KERNELSPEC void kernel_com_cos_theta_window(
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> pt_min,
    FSIn<T, 0> y_max1,
    FSIn<T, 0> y_max2,
    FOut<T, 0> r_out,
    FOut<T, 0> det
) {
    auto window = com_cos_theta_window<T>(m0, m1, m2, x1, x2, pt_min, y_max1, y_max2);
    auto width = max(window.second - window.first, 0.);
    r_out = 0.5 * (window.first + 1.) + 0.5 * width * r_cos_theta;
    det = 0.5 * width;
}

template <typename T>
KERNELSPEC void kernel_com_cos_theta_window_inverse(
    FIn<T, 0> r_out,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> pt_min,
    FSIn<T, 0> y_max1,
    FSIn<T, 0> y_max2,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> det
) {
    auto window = com_cos_theta_window<T>(m0, m1, m2, x1, x2, pt_min, y_max1, y_max2);
    auto width = max(window.second - window.first, EPS);
    r_cos_theta = (r_out - 0.5 * (window.first + 1.)) / (0.5 * width);
    det = 2. / width;
}

template <typename T>
KERNELSPEC void kernel_two_body_decay_com(
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    auto decay_out = two_body_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, p1_tmp);
    det = det_tmp;
    auto e2 = m0 - p1_tmp[0];
    p2[0] = max(e2, 0.);
    p2[1] = -p1_tmp[1];
    p2[2] = -p1_tmp[2];
    p2[3] = -p1_tmp[3];
}

template <typename T>
KERNELSPEC void kernel_two_body_decay_com_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> det
) {
    FourMom<T> p0_vec;
    for (int i = 0; i < 4; ++i) {
        p0_vec[i] = p1[i] + p2[i];
    }
    auto decay_out =
        two_body_decay_inverse<T>(p0_vec, load_mom<T>(p1), load_mom<T>(p2));
    r_phi = decay_out.first;
    r_cos_theta = decay_out.second;
    m0 = decay_out.third;
    m1 = decay_out.fourth;
    m2 = decay_out.fifth;
    det = decay_out.sixth;
}

template <typename T>
KERNELSPEC void kernel_two_body_decay(
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 1> p0,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    auto decay_out = two_body_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, boost<T>(p1_tmp, load_mom<T>(p0), 1.));
    det = det_tmp;
    auto e2 = p0[0] - p1[0];
    p2[0] = max(e2, 0.);
    p2[1] = p0[1] - p1[1];
    p2[2] = p0[2] - p1[2];
    p2[3] = p0[3] - p1[3];
}

template <typename T>
KERNELSPEC void kernel_two_body_decay_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 1> p0,
    FOut<T, 0> det
) {
    FourMom<T> p0_vec;
    for (int i = 0; i < 4; ++i) {
        p0_vec[i] = p1[i] + p2[i];
        p0[i] = p0_vec[i];
    }
    auto p1_com = boost<T>(load_mom<T>(p1), p0_vec, -1.);
    auto decay_out = two_body_decay_inverse<T>(p0_vec, p1_com, load_mom<T>(p2));
    r_phi = decay_out.first;
    r_cos_theta = decay_out.second;
    m0 = decay_out.third;
    m1 = decay_out.fourth;
    m2 = decay_out.fifth;
    det = decay_out.sixth;
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering_com(
    FIn<T, 0> r_phi,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto phi = PI * (2. * r_phi - 1.);
    auto scatter_out =
        p1com_from_tabs_phi<T>(load_mom<T>(pa), s_tot, phi, t_abs, m1, m2, ma_2, mb_2);
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    auto p1_rot = rotate<T>(p1_com, load_mom<T>(pa));
    store_mom<T>(p1, p1_rot);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_tot[i] - p1_rot[i];
    }
    det = det_tmp;
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering_com_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FOut<T, 0> r_phi,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto p1_rot = rotate_inverse<T>(load_mom<T>(p1), load_mom<T>(pa));
    auto scatter_out =
        phi_m1_m2_from_p1com<T>(p1_rot, load_mom<T>(p2), s_tot, ma_2, mb_2);
    auto phi = scatter_out.first;
    m1 = scatter_out.second;
    m2 = scatter_out.third;
    det = scatter_out.fourth;
    r_phi = phi / PI / 2. + 0.5;
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering(
    FIn<T, 0> r_phi,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto phi = PI * (2. * r_phi - 1.);
    auto scatter_out =
        p1com_from_tabs_phi<T>(pa_com, s_tot, phi, t_abs, m1, m2, ma_2, mb_2);
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    auto p1_rot = rotate<T>(p1_com, pa_com);
    // p1 and p2 back to back in the rest frame, the softer one boosted on its
    // mass shell and the other p_tot minus it; see boost_two_body.
    auto m_tot = sqrt(max(s_tot, EPS2));
    auto e2_com = 0.5 * (m_tot - (m1 - m2) * (m1 + m2) / m_tot);
    FourMom<T> p2_rot{max(e2_com, 0.), -p1_rot[1], -p1_rot[2], -p1_rot[3]};
    auto p_out = boost_two_body<T>(p1_rot, m1 * m1, p2_rot, m2 * m2, p_tot, s_tot);
    store_mom<T>(p1, p_out.first);
    store_mom<T>(p2, p_out.second);
    det = det_tmp;
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FOut<T, 0> r_phi,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto p1_com = boost<T>(load_mom<T>(p1), p_tot, -1.);
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    auto p1_rot = rotate_inverse<T>(p1_com, pa_com);
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto scatter_out =
        phi_m1_m2_from_p1com<T>(p1_rot, load_mom<T>(p2), s_tot, ma_2, mb_2);
    auto phi = scatter_out.first;
    m1 = scatter_out.second;
    m2 = scatter_out.third;
    det = scatter_out.fourth;
    r_phi = phi / PI / 2. + 0.5;
}

template <typename T>
KERNELSPEC void kernel_double_t_scattering(
    FIn<T, 0> r_phi,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t1_abs,
    FIn<T, 0> t2_abs,
    FIn<T, 0> m1,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    // pa + pb -> p1 + p2 with |t1| = -(pa-p1)^2, |t2| = -(pb-p1)^2, phi.
    // Assumes pa, pb massless. m1 fixed; m2 derived.
    //
    // Canonical frame: (pa+pb) rest frame, with pa along +z. The formulas:
    //   E1   = (t1_abs + t2_abs + 2 m1^2) / (2 sqrt(s))
    //   pz1  = (t2_abs - t1_abs)          / (2 sqrt(s))
    //   pt^2 = (t1_abs t2_abs + (t1_abs + t2_abs) m1^2 + m1^4 - s m1^2) / s
    // Then rotate so canonical-z aligns with pa_com, boost back to lab.
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    auto s = lsquare<T>(p_tot);
    auto sqrts = sqrt(max(s, EPS2));
    auto m1_2 = m1 * m1;

    auto e1 = (t1_abs + t2_abs + 2. * m1_2) / (2. * sqrts);
    auto pz1 = (t2_abs - t1_abs) / (2. * sqrts);
    auto pt2_raw =
        (t1_abs * t2_abs + (t1_abs + t2_abs) * m1_2 + m1_2 * m1_2 - s * m1_2) / s;
    auto pt2 = max(pt2_raw, 0.);
    auto pt = sqrt(pt2);
    auto phi = PI * (2. * r_phi - 1.);

    FourMom<T> p1_canon{e1, pt * cos(phi), pt * sin(phi), pz1};

    auto p1_rot = rotate<T>(p1_canon, pa_com);
    auto p1_lab = boost<T>(p1_rot, p_tot, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_tot[i] - p1_lab[i];
    }

    det = PI / (2. * max(s, EPS));
}

template <typename T>
KERNELSPEC void kernel_double_t_scattering_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FOut<T, 0> r_phi,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    auto p1_com = boost<T>(load_mom<T>(p1), p_tot, -1.);
    auto p1_canon = rotate_inverse<T>(p1_com, pa_com);
    auto phi = atan2(p1_canon[2], p1_canon[1]);
    r_phi = phi / PI / 2. + 0.5;
    auto s = lsquare(p_tot);
    det = 2. * max(s, EPS) / PI;
}

} // namespace kernels
} // namespace madspace
