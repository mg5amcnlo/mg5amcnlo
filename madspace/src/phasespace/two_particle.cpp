#include "madspace/phasespace/two_particle.hpp"

using namespace madspace;

TwoBodyDecay::TwoBodyDecay(bool com) :
    Mapping(
        "TwoBodyDecay",
        [&] {
            NamedVector<Type> input_types{
                {"random_phi", batch_float},
                {"random_cos_theta", batch_float},
                {"mass0", batch_float},
                {"mass1", batch_float},
                {"mass2", batch_float},
            };
            if (!com) {
                input_types.push_back("com_momentum", batch_four_vec);
            }
            return input_types;
        }(),
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        {}
    ),
    _com(com) {}

Mapping::Result TwoBodyDecay::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_cos_theta = inputs.at(1);
    auto m0 = inputs.at(2), m1 = inputs.at(3), m2 = inputs.at(4);
    auto [p1, p2, det] = _com
        ? fb.two_body_decay_com(r_phi, r_cos_theta, m0, m1, m2)
        : fb.two_body_decay(r_phi, r_cos_theta, m0, m1, m2, inputs.at(5));
    return {{{"momentum1", p1}, {"momentum2", p2}}, det};
}

Mapping::Result TwoBodyDecay::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    if (_com) {
        auto [r_phi, r_cos_theta, m0, m1, m2, det] =
            fb.two_body_decay_com_inverse(p1, p2);
        return {
            {{"random_phi", r_phi},
             {"random_cos_theta", r_cos_theta},
             {"mass0", m0},
             {"mass1", m1},
             {"mass2", m2}},
            det
        };
    } else {
        auto [r_phi, r_cos_theta, m0, m1, m2, p0, det] =
            fb.two_body_decay_inverse(p1, p2);
        return {
            {{"random_phi", r_phi},
             {"random_cos_theta", r_cos_theta},
             {"mass0", m0},
             {"mass1", m1},
             {"mass2", m2},
             {"com_momentum", p0}},
            det
        };
    }
}

TwoToTwoParticleScattering::TwoToTwoParticleScattering(
    bool com, double invariant_power, double mass, double width
) :
    Mapping(
        "TwoToTwoParticleScattering",
        {{"random_phi", batch_float},
         {"random_inv", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        {{"momentum_in1", batch_four_vec}, {"momentum_in2", batch_four_vec}}
    ),
    _com(com),
    _invariant(invariant_power, mass, width) {}

Mapping::Result TwoToTwoParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2),
         m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_min, t_max] = fb.t_inv_min_max(p_in1, p_in2, m1, m2);
    auto t_result = _invariant.build_forward(fb, {r_inv}, {t_min, t_max});
    auto [p1, p2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com(
              r_phi, p_in1, p_in2, t_result["invariant"], m1, m2
          )
        : fb.two_to_two_particle_scattering(
              r_phi, p_in1, p_in2, t_result["invariant"], m1, m2
          );
    return {
        {{"momentum1", p1}, {"momentum2", p2}}, fb.mul(t_result["det"], det_scatter)
    };
}

Mapping::Result TwoToTwoParticleScattering::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_abs, t_min, t_max] = fb.t_inv_value_and_min_max(p_in1, p_in2, p1, p2);
    auto t_result = _invariant.build_inverse(fb, {t_abs}, {t_min, t_max});
    auto [r_phi, m1, m2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com_inverse(p1, p2, p_in1, p_in2)
        : fb.two_to_two_particle_scattering_inverse(p1, p2, p_in1, p_in2);
    return {
        {{"random_phi", r_phi},
         {"random_inv", t_result["random"]},
         {"mass1", m1},
         {"mass2", m2}},
        fb.mul(t_result["det"], det_scatter)
    };
}

DoubleT::DoubleT(
    double t1_invariant_power,
    double t1_mass,
    double t1_width,
    double t2_invariant_power,
    double t2_mass,
    double t2_width
) :
    Mapping(
        "DoubleT",
        {{"random_phi", batch_float},
         {"random_t1", batch_float},
         {"random_t2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        {{"momentum_in1", batch_four_vec},
         {"momentum_in2", batch_four_vec},
         {"mass1", batch_float},
         {"mass_rest_min", batch_float}}
    ),
    _t1_invariant(t1_invariant_power, t1_mass, t1_width),
    _t2_invariant(t2_invariant_power, t2_mass, t2_width) {}

Mapping::Result DoubleT::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_t1 = inputs.at(1), r_t2 = inputs.at(2);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto m1 = conditions.at(2);
    auto mir_min = conditions.at(3);

    auto [t1_min, t1_max] = fb.t1_inv_min_max_doublet(p_in1, p_in2, m1, mir_min);
    auto t1_result = _t1_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});

    auto [t2_min, t2_max] =
        fb.t2_inv_min_max_doublet(p_in1, p_in2, m1, mir_min, t1_result["invariant"]);
    auto t2_result = _t2_invariant.build_forward(fb, {r_t2}, {t2_min, t2_max});

    auto [p1, p2, det_scatter] = fb.double_t_scattering(
        r_phi, p_in1, p_in2, t1_result["invariant"], t2_result["invariant"], m1
    );

    auto det_inv = fb.mul(t1_result["det"], t2_result["det"]);
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result DoubleT::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto m1 = conditions.at(2);
    auto mir_min = conditions.at(3);

    auto [r_phi, det_scatter] = fb.double_t_scattering_inverse(p1, p2, p_in1, p_in2);

    auto [t1_abs, t1_min, t1_max] =
        fb.t1_inv_value_and_min_max_doublet(p_in1, p_in2, p1, m1, mir_min);
    auto t1_result = _t1_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});

    auto [t2_abs, t2_min, t2_max] =
        fb.t2_inv_value_and_min_max_doublet(p_in1, p_in2, p1, m1, mir_min, t1_abs);
    auto t2_result = _t2_invariant.build_inverse(fb, {t2_abs}, {t2_min, t2_max});

    auto det_inv = fb.mul(t1_result["det"], t2_result["det"]);
    return {
        {{"random_phi", r_phi},
         {"random_t1", t1_result["random"]},
         {"random_t2", t2_result["random"]}},
        fb.mul(det_inv, det_scatter)
    };
}
