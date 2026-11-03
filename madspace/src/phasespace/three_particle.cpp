#include "madspace/phasespace/three_particle.hpp"

#include <stdexcept>
#include <utility>

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
    double s_width,
    bool has_cut,
    bool arcsine_s23,
    bool p12_condition,
    bool incoming_recoil,
    bool cut_aware_branch
) :
    Mapping(
        "TwoToThreeParticleScattering",
        {{"discrete_choice", batch_int},
         {"random_s23", batch_float},
         {"random_t1", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec},
                {p12_condition ? "momentum12" : "momentum_in2", batch_four_vec},
                {"momentum3", batch_four_vec}
            };
            if (has_cut) {
                cond.push_back("etmin_1", batch_float);
                cond.push_back("etmin_2", batch_float);
                cond.push_back("drcut", batch_float);
                cond.push_back("s23_min_cut", batch_float);
            }
            return cond;
        }()
    ),
    _t_invariant(t_invariant_power, t_mass, t_width),
    _s_invariant(s_invariant_power, s_mass, s_width),
    _s_power(s_invariant_power),
    _s_mass(s_mass),
    _has_cut(has_cut),
    _arcsine_s23(arcsine_s23),
    _p12_condition(p12_condition),
    _incoming_recoil(incoming_recoil),
    _cut_aware_branch(cut_aware_branch && has_cut) {
    // The block peels a particle off a t-channel chain, where no s-channel
    // resonance is expected; the arcsine sampling covers power laws only.
    if (arcsine_s23 && s_width != 0.) {
        throw std::invalid_argument(
            "TwoToThreeParticleScattering: arcsine_s23 does not support a "
            "Breit-Wigner s23 sampling (s_width != 0)"
        );
    }
    if (incoming_recoil && !p12_condition) {
        throw std::invalid_argument(
            "TwoToThreeParticleScattering: incoming_recoil requires p12_condition"
        );
    }
}

namespace {

Value negated(FunctionBuilder& fb, Value x) { return fb.sub(fb.sub(x, x), x); }

} // namespace

std::array<Value, 3> TwoToThreeParticleScattering::split_conditions(
    FunctionBuilder& fb, const NamedVector<Value>& conditions
) const {
    // The kernels take the outgoing system p_12 = p1 + p2 itself. By default it
    // is formed from the incoming momenta as pa + pb - p3; a caller that already
    // holds p_12 more precisely (ColorOrderedMapping, where it can be a soft
    // system next to the beams) passes it as the second condition instead.
    auto p_a = conditions.at(0), p_3 = conditions.at(2);
    if (_p12_condition) {
        auto p_12 = conditions.at(1);
        return {p_a, p_12, fb.sub(p_12, p_a)};
    }
    auto p_b = conditions.at(1);
    return {p_a, fb.sub(fb.add(p_a, p_b), p_3), fb.sub(p_b, p_3)};
}

std::array<Value, 4> TwoToThreeParticleScattering::sampling_ranges(
    FunctionBuilder& fb,
    Value s23_min,
    Value s23_max,
    Value m2_squared,
    const NamedVector<Value>& conditions
) const {
    // returns the sampled and the kinematic range of the sampled variable
    if (!_incoming_recoil) {
        return {s23_min, s23_max, s23_min, s23_max};
    }
    // v = -(p2 - k)^2 >= 0 runs over [-s23_max, -s23_min]
    auto v_phys_min = negated(fb, s23_max), v_phys_max = negated(fb, s23_min);
    auto v_min = v_phys_min, v_max = v_phys_max;
    if (_has_cut) {
        // |(p2 - k)^2| >= m_T(p2)^2 - m2^2 = p_T(p2)^2 for a massless k along
        // the beam axis: the ETmin floor of a massive particle bounds m_T
        v_min = fb.max(
            v_min, fb.max(fb.sub(fb.square(conditions.at(4)), m2_squared), Value(0.))
        );
        // an empty range keeps a small width; the weight is zero there, as
        // the point then lies outside the kinematic range
        auto width = fb.max(fb.mul(v_min, Value(1e-10)), Value(1e-300));
        v_max = fb.max(v_max, fb.add(v_min, width));
    }
    return {v_min, v_max, v_phys_min, v_phys_max};
}

Mapping::Result TwoToThreeParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto index_choice = inputs.at(0), r_s23 = inputs.at(1), r_t1 = inputs.at(2),
         m1 = inputs.at(3), m2 = inputs.at(4);
    auto p_3 = _incoming_recoil ? negated(fb, conditions.at(2)) : conditions.at(2);
    auto [p_a, p_12, p_c] = split_conditions(fb, conditions);
    auto [t1_min, t1_max] = _has_cut
        ? fb.t_inv_min_max_cut(p_a, p_c, m1, m2, conditions.at(3), conditions.at(4))
        : fb.t_inv_min_max(p_a, p_c, m1, m2);
    auto t_inv_result = _t_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});
    auto t1 = t_inv_result["invariant"];
    // The scattering kernel takes the position of s23 in its kinematic range,
    // u and 1 - u. Without cuts the sampled and the kinematic range coincide.
    std::array<Value, 4> ranges;
    if (_incoming_recoil) {
        auto [lo, hi] = fb.s23_min_max(p_a, p_12, p_3, t1, m1, m2);
        ranges = sampling_ranges(fb, lo, hi, fb.square(m2), conditions);
    } else if (_has_cut) {
        auto [s_min, s_max] = fb.s23_min_max_cut(
            p_a,
            p_12,
            p_3,
            t1,
            m1,
            m2,
            conditions.at(3),
            conditions.at(4),
            conditions.at(5),
            conditions.at(6)
        );
        auto [lo, hi] = fb.s23_min_max(p_a, p_12, p_3, t1, m1, m2);
        ranges = {s_min, s_max, lo, hi};
    } else {
        auto [lo, hi] = fb.s23_min_max(p_a, p_12, p_3, t1, m1, m2);
        ranges = {lo, hi, lo, hi};
    }
    auto [v_min, v_max, v_phys_min, v_phys_max] = ranges;
    Value u, u_c, det_s23;
    if (_arcsine_s23) {
        // arcsine map and importance sampling in one step, carrying u and
        // 1 - u at full relative precision up to the edges
        auto [u_out, u_c_out, det_out] = fb.s23_arcsine_sample(
            r_s23, v_min, v_max, v_phys_min, v_phys_max, Value(_s_power), Value(_s_mass)
        );
        u = u_out;
        u_c = u_c_out;
        det_s23 = det_out;
    } else {
        auto s23_inv_result = _s_invariant.build_forward(fb, {r_s23}, {v_min, v_max});
        det_s23 = s23_inv_result["det"];
        auto [u_out, u_c_out] =
            fb.s23_position(s23_inv_result["invariant"], v_phys_min, v_phys_max);
        u = u_out;
        u_c = u_c_out;
    }
    if (_incoming_recoil) {
        // v = -s23 runs the other way through the range of s23
        std::swap(u, u_c);
    }
    auto det_inv = fb.mul(t_inv_result["det"], det_s23);
    if (_cut_aware_branch) {
        auto [index_pass, factor, inv_factor] = fb.two_to_three_branch_choice(
            index_choice,
            p_a,
            p_12,
            p_3,
            u,
            u_c,
            t1,
            m1,
            m2,
            conditions.at(3),
            conditions.at(4)
        );
        index_choice = index_pass;
        det_inv = fb.mul(det_inv, factor);
    }
    auto [p1, p2, det_scatter] = fb.two_to_three_particle_scattering(
        index_choice, p_a, p_12, p_3, u, u_c, t1, m1, m2
    );
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result TwoToThreeParticleScattering::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_3 = _incoming_recoil ? negated(fb, conditions.at(2)) : conditions.at(2);
    auto [p_a, p_12, p_c] = split_conditions(fb, conditions);
    auto [t1_abs, t1_min, t1_max] = _has_cut
        ? fb.t_inv_value_and_min_max_cut(
              p_a, p_c, p1, p2, conditions.at(3), conditions.at(4)
          )
        : fb.t_inv_value_and_min_max(p_a, p_c, p1, p2);
    auto t_inv_result = _t_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});
    Value v;
    std::array<Value, 4> ranges;
    if (_incoming_recoil) {
        auto [s23, lo, hi] = fb.s23_value_and_min_max(p_a, p_3, t1_abs, p1, p2);
        v = negated(fb, s23);
        nested_vector2<double> factors_p2{{1.0}};
        auto m2_squared =
            fb.unstack(fb.invariants_from_momenta(fb.stack({p2}), factors_p2)).at(0);
        ranges = sampling_ranges(fb, lo, hi, m2_squared, conditions);
    } else if (_has_cut) {
        auto [s23, s_min, s_max] = fb.s23_value_and_min_max_cut(
            p_a,
            p_3,
            t1_abs,
            p1,
            p2,
            conditions.at(3),
            conditions.at(4),
            conditions.at(5),
            conditions.at(6)
        );
        v = s23;
        ranges = {s_min, s_max, s_min, s_max};
        if (_arcsine_s23) {
            auto [s23_phys, lo, hi] =
                fb.s23_value_and_min_max(p_a, p_3, t1_abs, p1, p2);
            ranges = {s_min, s_max, lo, hi};
        }
    } else {
        auto [s23, lo, hi] = fb.s23_value_and_min_max(p_a, p_3, t1_abs, p1, p2);
        v = s23;
        ranges = {lo, hi, lo, hi};
    }
    auto [v_min, v_max, v_phys_min, v_phys_max] = ranges;
    auto [m1, m2, index_choice, u, u_c, det_scatter] =
        fb.two_to_three_particle_scattering_inverse(p1, p2, p_3, p_a, p_12, t1_abs);
    Value det_branch = Value(1.);
    if (_cut_aware_branch) {
        auto [index_pass, factor, inv_factor] = fb.two_to_three_branch_choice(
            index_choice,
            p_a,
            p_12,
            p_3,
            u,
            u_c,
            t1_abs,
            m1,
            m2,
            conditions.at(3),
            conditions.at(4)
        );
        det_branch = inv_factor;
    }
    // position of the sampled variable in its kinematic range
    Value u_v = _incoming_recoil ? u_c : u, u_c_v = _incoming_recoil ? u : u_c;
    Value r_s23, det_s23;
    if (_arcsine_s23) {
        // u and 1 - u read off the azimuth, at full relative precision
        auto [r, det_r] = fb.s23_arcsine_sample_inverse(
            u_v,
            u_c_v,
            v_min,
            v_max,
            v_phys_min,
            v_phys_max,
            Value(_s_power),
            Value(_s_mass)
        );
        r_s23 = r;
        det_s23 = det_r;
    } else {
        auto s23_inv_result = _s_invariant.build_inverse(fb, {v}, {v_min, v_max});
        r_s23 = s23_inv_result["random"];
        det_s23 = s23_inv_result["det"];
    }
    auto det_inv = fb.mul(t_inv_result["det"], det_s23);
    if (_cut_aware_branch) {
        det_inv = fb.mul(det_inv, det_branch);
    }
    return {
        {{"discrete_choice", index_choice},
         {"random_s23", r_s23},
         {"random_t1", t_inv_result["random"]},
         {"mass1", m1},
         {"mass2", m2}},
        fb.mul(det_inv, det_scatter)
    };
}
