#include "madspace/phasespace/two_particle.hpp"

using namespace madspace;

namespace {
bool has_cos_theta_window(bool com, double pt_min, double y_max1, double y_max2) {
    return com && (pt_min > 0. || y_max1 >= 0. || y_max2 >= 0.);
}
} // namespace

TwoBodyDecay::TwoBodyDecay(bool com, double pt_min, double y_max1, double y_max2) :
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
        has_cos_theta_window(com, pt_min, y_max1, y_max2)
            ? NamedVector<Type>{{"x1", batch_float}, {"x2", batch_float}}
            : NamedVector<Type>{}
    ),
    _com(com),
    _window(has_cos_theta_window(com, pt_min, y_max1, y_max2)),
    _pt_min(pt_min),
    _y_max1(y_max1),
    _y_max2(y_max2) {}

Mapping::Result TwoBodyDecay::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_cos_theta = inputs.at(1);
    auto m0 = inputs.at(2), m1 = inputs.at(3), m2 = inputs.at(4);
    Value det_window = 1.;
    if (_window) {
        auto [r_window, det] = fb.com_cos_theta_window(
            r_cos_theta,
            m0,
            m1,
            m2,
            conditions.at(0),
            conditions.at(1),
            _pt_min,
            _y_max1,
            _y_max2
        );
        r_cos_theta = r_window;
        det_window = det;
    }
    auto [p1, p2, det] = _com
        ? fb.two_body_decay_com(r_phi, r_cos_theta, m0, m1, m2)
        : fb.two_body_decay(r_phi, r_cos_theta, m0, m1, m2, inputs.at(5));
    return {
        {{"momentum1", p1}, {"momentum2", p2}}, _window ? fb.mul(det, det_window) : det
    };
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
        if (_window) {
            auto [r_free, det_window] = fb.com_cos_theta_window_inverse(
                r_cos_theta,
                m0,
                m1,
                m2,
                conditions.at(0),
                conditions.at(1),
                _pt_min,
                _y_max1,
                _y_max2
            );
            r_cos_theta = r_free;
            det = fb.mul(det, det_window);
        }
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
    bool com,
    double invariant_power,
    double mass,
    double width,
    bool has_cut,
    double y_max1,
    double y_max2,
    double beam_sign
) :
    Mapping(
        "TwoToTwoParticleScattering",
        {{"random_phi", batch_float},
         {"random_inv", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec}, {"momentum_in2", batch_four_vec}
            };
            if (has_cut) {
                cond.push_back("etmin_1", batch_float);
                cond.push_back("etmin_2", batch_float);
            }
            if (y_max1 >= 0. || y_max2 >= 0.) {
                cond.push_back("x1", batch_float);
                cond.push_back("x2", batch_float);
            }
            return cond;
        }()
    ),
    _com(com),
    _invariant(invariant_power, mass, width),
    _has_cut(has_cut),
    _rapidity_window(y_max1 >= 0. || y_max2 >= 0.),
    _y_max1(y_max1),
    _y_max2(y_max2),
    _beam_sign(beam_sign) {}

std::array<Value, 2> TwoToTwoParticleScattering::rapidity_clamp(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t_min,
    Value t_max,
    Value m1,
    Value m2
) const {
    if (!_rapidity_window) {
        return {t_min, t_max};
    }
    std::size_t x_index = _has_cut ? 4 : 2;
    auto [lo, hi] = fb.t_rapidity_clamp(
        t_min,
        t_max,
        conditions.at(0),
        conditions.at(1),
        m1,
        m2,
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

Mapping::Result TwoToTwoParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2),
         m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_min_kin, t_max_kin] = _has_cut
        ? fb.t_inv_min_max_cut(p_in1, p_in2, m1, m2, conditions.at(2), conditions.at(3))
        : fb.t_inv_min_max(p_in1, p_in2, m1, m2);
    auto [t_min, t_max] = rapidity_clamp(fb, conditions, t_min_kin, t_max_kin, m1, m2);
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
    auto [t_abs, t_min_kin, t_max_kin] = _has_cut
        ? fb.t_inv_value_and_min_max_cut(
              p_in1, p_in2, p1, p2, conditions.at(2), conditions.at(3)
          )
        : fb.t_inv_value_and_min_max(p_in1, p_in2, p1, p2);
    auto [t_min, t_max] = _rapidity_window
        ? rapidity_clamp(
              fb, conditions, t_min_kin, t_max_kin, fb.obs_mass(p1), fb.obs_mass(p2)
          )
        : std::array<Value, 2>{t_min_kin, t_max_kin};
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
    double t2_width,
    bool has_cut,
    double y_max1,
    double y_max2,
    double beam_sign
) :
    Mapping(
        "DoubleT",
        {{"random_phi", batch_float},
         {"random_t1", batch_float},
         {"random_t2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec},
                {"momentum_in2", batch_four_vec},
                {"mass1", batch_float},
                {"mass_rest_min", batch_float}
            };
            if (has_cut) {
                cond.push_back("etmin_i", batch_float);
                cond.push_back("etmin_ir", batch_float);
            }
            if (y_max1 >= 0. || y_max2 >= 0.) {
                cond.push_back("x1", batch_float);
                cond.push_back("x2", batch_float);
            }
            return cond;
        }()
    ),
    _t1_invariant(t1_invariant_power, t1_mass, t1_width),
    _t2_invariant(t2_invariant_power, t2_mass, t2_width),
    _has_cut(has_cut),
    _rapidity_window(y_max1 >= 0. || y_max2 >= 0.),
    _y_max1(y_max1),
    _y_max2(y_max2),
    _beam_sign(beam_sign) {}

std::array<Value, 2> DoubleT::rapidity_clamp_t1(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t1_min,
    Value t1_max
) const {
    if (!_rapidity_window) {
        return {t1_min, t1_max};
    }
    std::size_t x_index = _has_cut ? 6 : 4;
    auto [lo, hi] = fb.t1_rapidity_clamp_doublet(
        t1_min,
        t1_max,
        conditions.at(0),
        conditions.at(1),
        conditions.at(2),
        conditions.at(3),
        _has_cut ? conditions.at(4) : Value(0.),
        _has_cut ? conditions.at(5) : Value(0.),
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

std::array<Value, 2> DoubleT::rapidity_clamp(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t2_min,
    Value t2_max,
    Value t1_abs
) const {
    if (!_rapidity_window) {
        return {t2_min, t2_max};
    }
    std::size_t x_index = _has_cut ? 6 : 4;
    auto [lo, hi] = fb.t2_rapidity_clamp_doublet(
        t2_min,
        t2_max,
        conditions.at(0),
        conditions.at(1),
        conditions.at(2),
        t1_abs,
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

Mapping::Result DoubleT::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_t1 = inputs.at(1), r_t2 = inputs.at(2);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto m1 = conditions.at(2);
    auto mir_min = conditions.at(3);
    auto etmin_i = _has_cut ? conditions.at(4) : Value(0.);
    auto etmin_ir = _has_cut ? conditions.at(5) : Value(0.);

    auto [t1_min_kin, t1_max_kin] =
        fb.t1_inv_min_max_doublet(p_in1, p_in2, m1, mir_min, etmin_i, etmin_ir);
    auto [t1_min, t1_max] = rapidity_clamp_t1(fb, conditions, t1_min_kin, t1_max_kin);
    auto t1_result = _t1_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});

    auto [t2_min_kin, t2_max_kin] = fb.t2_inv_min_max_doublet(
        p_in1, p_in2, m1, mir_min, t1_result["invariant"], etmin_i, etmin_ir
    );
    auto [t2_min, t2_max] =
        rapidity_clamp(fb, conditions, t2_min_kin, t2_max_kin, t1_result["invariant"]);
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
    auto etmin_i = _has_cut ? conditions.at(4) : Value(0.);
    auto etmin_ir = _has_cut ? conditions.at(5) : Value(0.);

    auto [r_phi, det_scatter] = fb.double_t_scattering_inverse(p1, p2, p_in1, p_in2);

    auto [t1_abs, t1_min_kin, t1_max_kin] = fb.t1_inv_value_and_min_max_doublet(
        p_in1, p_in2, p1, m1, mir_min, etmin_i, etmin_ir
    );
    auto [t1_min, t1_max] = rapidity_clamp_t1(fb, conditions, t1_min_kin, t1_max_kin);
    auto t1_result = _t1_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});

    auto [t2_abs, t2_min_kin, t2_max_kin] = fb.t2_inv_value_and_min_max_doublet(
        p_in1, p_in2, p1, m1, mir_min, t1_abs, etmin_i, etmin_ir
    );
    auto [t2_min, t2_max] =
        rapidity_clamp(fb, conditions, t2_min_kin, t2_max_kin, t1_abs);
    auto t2_result = _t2_invariant.build_inverse(fb, {t2_abs}, {t2_min, t2_max});

    auto det_inv = fb.mul(t1_result["det"], t2_result["det"]);
    return {
        {{"random_phi", r_phi},
         {"random_t1", t1_result["random"]},
         {"random_t2", t2_result["random"]}},
        fb.mul(det_inv, det_scatter)
    };
}
