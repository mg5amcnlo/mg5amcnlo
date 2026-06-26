#include "madspace/phasespace/three_particle.hpp"

using namespace madspace;

ThreeBodyDecay::ThreeBodyDecay(bool com) :
    Mapping(
        "ThreeBodyDecay",
        [&] {
            NamedVector<Type> input_types{
                {"random_energy1", batch_float},
                {"random_energy2", batch_float},
                {"random_phi", batch_float},
                {"random_cos_theta", batch_float},
                {"random_beta", batch_float},
                {"mass0", batch_float},
                {"mass1", batch_float},
                {"mass2", batch_float},
                {"mass3", batch_float},
            };
            if (!com) {
                input_types.push_back("com_momentum", batch_four_vec);
            }
            return input_types;
        }(),
        {{"momentum1", batch_four_vec},
         {"momentum2", batch_four_vec},
         {"momentum3", batch_four_vec}},
        {}
    ),
    _com(com) {}

Mapping::Result ThreeBodyDecay::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto [p1, p2, p3, det] = _com
        ? fb.three_body_decay_com(
              inputs["random_energy1"],
              inputs["random_energy2"],
              inputs["random_phi"],
              inputs["random_cos_theta"],
              inputs["random_beta"],
              inputs["mass0"],
              inputs["mass1"],
              inputs["mass2"],
              inputs["mass3"]
          )
        : fb.three_body_decay(
              inputs["random_energy1"],
              inputs["random_energy2"],
              inputs["random_phi"],
              inputs["random_cos_theta"],
              inputs["random_beta"],
              inputs["mass0"],
              inputs["mass1"],
              inputs["mass2"],
              inputs["mass3"],
              inputs["com_momentum"]
          );
    return {{{"momentum1", p1}, {"momentum2", p2}, {"momentum3", p3}}, det};
}

Mapping::Result ThreeBodyDecay::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    if (_com) {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, det] =
            fb.three_body_decay_com_inverse(
                inputs["momentum1"], inputs["momentum2"], inputs["momentum3"]
            );
        return {
            {input_types().keys(),
             {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3}},
            det
        };
    } else {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0, det] =
            fb.three_body_decay_inverse(
                inputs["momentum1"], inputs["momentum2"], inputs["momentum3"]
            );
        return {
            {input_types().keys(),
             {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0}},
            det
        };
    }
}

TwoToThreeParticleScattering::TwoToThreeParticleScattering(
    double t_invariant_power,
    double t_mass,
    double t_width,
    double s_invariant_power,
    double s_mass,
    double s_width
) :
    Mapping(
        "TwoToThreeParticleScattering",
        {{"random_choice", batch_float},
         {"random_s23", batch_float},
         {"random_t1", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        {{"momentum_in1", batch_four_vec},
         {"momentum_in2", batch_four_vec},
         {"momentum3", batch_four_vec}}
    ),
    _t_invariant(t_invariant_power, t_mass, t_width),
    _s_invariant(s_invariant_power, s_mass, s_width) {}

Mapping::Result TwoToThreeParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_choice = inputs.at(0), r_s23 = inputs.at(1), r_t1 = inputs.at(2),
         m1 = inputs.at(3), m2 = inputs.at(4);
    auto p_a = conditions.at(0), p_b = conditions.at(1), p_3 = conditions.at(2);
    auto [t1_min, t1_max] = fb.t_inv_min_max(p_a, fb.sub(p_b, p_3), m1, m2);
    auto t_inv_result = _t_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});
    auto [s23_min, s23_max] =
        fb.s23_min_max(p_a, p_b, p_3, t_inv_result["invariant"], m1, m2);
    auto s23_inv_result = _s_invariant.build_forward(fb, {r_s23}, {s23_min, s23_max});
    auto det_inv = fb.mul(t_inv_result["det"], s23_inv_result["det"]);
    auto [index_choice, index_det] = fb.sample_discrete(r_choice, 2);
    auto [p1, p2, det_scatter] = fb.two_to_three_particle_scattering(
        index_choice,
        p_a,
        p_b,
        p_3,
        s23_inv_result["invariant"],
        t_inv_result["invariant"],
        m1,
        m2
    );
    auto det_scatter_23 = fb.mul(index_det, det_scatter);
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_inv, det_scatter_23)};
}

Mapping::Result TwoToThreeParticleScattering::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_a = conditions.at(0), p_b = conditions.at(1), p_3 = conditions.at(2);
    auto [t1_abs, t1_min, t1_max] =
        fb.t_inv_value_and_min_max(p_a, fb.sub(p_b, p_3), p1, p2);
    auto t_inv_result = _t_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});
    auto [s23, s23_min, s23_max] =
        fb.s23_value_and_min_max(p_a, p_b, p_3, t1_abs, p1, p2);
    auto s23_inv_result = _s_invariant.build_inverse(fb, {s23}, {s23_min, s23_max});
    auto det_inv = fb.mul(t_inv_result["det"], s23_inv_result["det"]);
    auto [m1, m2, index_choice, det_scatter] =
        fb.two_to_three_particle_scattering_inverse(p1, p2, p_3, p_a, p_b, t1_abs, s23);
    auto [r_choice, index_det] = fb.sample_discrete_inverse(index_choice, 2);
    auto det_scatter_23 = fb.mul(index_det, det_scatter);
    return {
        {{"random_choice", r_choice},
         {"random_s23", s23_inv_result["random"]},
         {"random_t1", t_inv_result["random"]},
         {"mass1", m1},
         {"mass2", m2}},
        fb.mul(det_inv, det_scatter_23)
    };
}
