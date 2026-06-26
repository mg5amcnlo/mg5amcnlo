#include "madspace/phasespace/luminosity.hpp"

using namespace madspace;

Luminosity::Luminosity(
    double s_lab,
    double s_hat_min,
    double s_hat_max,
    double invariant_power,
    double mass,
    double width
) :
    Mapping(
        "Luminosity",
        {{"r_s", batch_float}, {"r_x", batch_float}},
        {{"x1", batch_float}, {"x2", batch_float}, {"s_hat", batch_float}},
        {}
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _s_hat_max(s_hat_max == 0 ? s_lab : s_hat_max),
    _invariant(invariant_power, mass, width) {}

Mapping::Result Luminosity::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_s = inputs[0], r_x = inputs[1];
    auto result = _invariant.build_forward(fb, {r_s}, {_s_hat_min, _s_hat_max});
    auto s_hat = result["s"];
    auto [x1, x2, det_x] = fb.r_to_x1x2(r_x, s_hat, _s_lab);
    auto det = fb.mul(result["det"], det_x);
    return {{{"x1", x1}, {"x2", x2}, {"s_hat", s_hat}}, det};
}

Mapping::Result Luminosity::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto x1 = inputs[0], x2 = inputs[1], s_hat = inputs[2];
    auto result = _invariant.build_inverse(fb, {s_hat}, {_s_hat_min, _s_hat_max});
    auto [r_x, det_x] = fb.x1x2_to_r(x1, x2, _s_lab);
    auto det = fb.mul(result["det"], det_x);
    return {{{"r_s", result["r"]}, {"r_x", r_x}}, det};
}
