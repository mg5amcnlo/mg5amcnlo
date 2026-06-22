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
    auto pp = 0.5 * where(m1 * m2 == 0., m0 - fabs(ed), sqrt(max(pp2, EPS)));
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    auto e1 = 0.5 * (m0 + ed);
    FourMom<T> p1{
        max(e1, 0.),
        pp * sin_theta * cos(phi),
        pp * sin_theta * sin(phi),
        pp * cos_theta
    };

    auto det = PI * pp / m0_clip;
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

// Kernels

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
    auto p1_lab = boost<T>(p1_rot, p_tot, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_tot[i] - p1_lab[i];
    }
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
