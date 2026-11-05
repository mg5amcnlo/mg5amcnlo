#include "madspace/phasespace/scale.hpp"

using namespace madspace;

EnergyScale::EnergyScale(
    std::size_t particle_count,
    DynamicalScaleType dynamical_scale_type,
    bool ren_scale_fixed,
    bool fact_scale_fixed,
    double ren_scale,
    double fact_scale1,
    double fact_scale2,
    double scale_factor,
    const std::optional<CustomFunction>& custom_scale,
    double min_scale,
    double max_scale
) :
    FunctionGenerator(
        "EnergyScale",
        {{"momenta", batch_four_vec_array(particle_count)}},
        {{"ren_scale", batch_float},
         {"fact_scale1", batch_float},
         {"fact_scale2", batch_float}}
    ),
    _dynamical_scale_type(dynamical_scale_type),
    _ren_scale_fixed(ren_scale_fixed),
    _fact_scale_fixed(fact_scale_fixed),
    _ren_scale(ren_scale),
    _fact_scale1(fact_scale1),
    _fact_scale2(fact_scale2),
    _scale_factor(scale_factor),
    _custom_scale(custom_scale),
    _min_scale(min_scale),
    _max_scale(max_scale) {
    if (custom_scale &&
        (custom_scale->type() != CustomFunction::scale ||
         custom_scale->dynamic_pdg_ids())) {
        throw std::invalid_argument(
            "custom scale must be of type scale, with a single PDG-id option"
        );
    }
}

EnergyScale::EnergyScale(
    const MLMClustering& clustering, double min_scale, double max_scale
) :
    FunctionGenerator(
        "EnergyScale",
        clustering.arg_types(),
        (min_scale > 0. || max_scale > 0.)
            ? [&] {
                  auto types = clustering.return_types();
                  types.push_back("scale_weight", batch_float);
                  return types;
              }()
            : clustering.return_types()
    ),
    _min_scale(min_scale),
    _max_scale(max_scale),
    _clustering(clustering) {}

// Clamp the scales into [min_scale, max_scale], the band the PDF grid covers.
// Outside it the density can be NaN, which a later veto weight cannot remove
// (NaN * 0 = NaN). An MLM clustering also vetoes events outside the range
// through scale_weight, as madevent's setclscales does for mu_F < 2 GeV.
NamedVector<Value>
EnergyScale::apply_scale_range(FunctionBuilder& fb, NamedVector<Value> scales) const {
    if (_min_scale <= 0. && _max_scale <= 0.) {
        return scales;
    }
    double low = _min_scale > 0. ? _min_scale : 0.;
    double high = _max_scale > 0. ? _max_scale : 1e30;
    Value weight;
    std::vector<const char*> names{"fact_scale1", "fact_scale2", "ren_scale"};
    if (_clustering && _clustering->pdf_reweighting()) {
        // Under pdf reweighting the density is evaluated at pdf_scale, which can
        // lie below the factorisation scale, so it is clamped (and vetoed) too.
        names.push_back("pdf_scale1");
        names.push_back("pdf_scale2");
    }
    for (auto name : names) {
        auto& scale = scales.at(name);
        if (_clustering) {
            auto pass = fb.cut_one(scale, low, high);
            weight = weight ? fb.mul(weight, pass) : pass;
        }
        auto batch_size = fb.batch_size({scale});
        scale = fb.max(scale, fb.full({low, batch_size}));
        scale = fb.min(scale, fb.full({high, batch_size}));
    }
    if (weight) {
        scales.push_back("scale_weight", weight);
    }
    return scales;
}

NamedVector<Value> EnergyScale::build_mlm_from_start_state(
    FunctionBuilder& fb,
    Value momenta,
    Value start_state,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    return apply_scale_range(
        fb,
        _clustering.value().build_from_start_state(
            fb, momenta, start_state, flavor_index, leg_flavors
        )
    );
}

NamedVector<Value> EnergyScale::build_mlm_with_flavors(
    FunctionBuilder& fb,
    Value momenta,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    return apply_scale_range(
        fb,
        _clustering.value().build_with_flavors(fb, momenta, flavor_index, leg_flavors)
    );
}

NamedVector<Value> EnergyScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto momenta = args.at(0);
    if (_clustering) {
        return apply_scale_range(fb, _clustering.value().build_function(fb, args));
    }
    if (_ren_scale_fixed && _fact_scale_fixed) {
        auto batch_size = fb.batch_size({momenta});
        return apply_scale_range(
            fb,
            {
                {"ren_scale", fb.full({_ren_scale, batch_size})},
                {"fact_scale1", fb.full({_fact_scale1, batch_size})},
                {"fact_scale2", fb.full({_fact_scale2, batch_size})},
            }
        );
    }
    auto batch_size = fb.batch_size({momenta});
    Value ren_scale, fact_scale1, fact_scale2;
    if (_custom_scale) {
        auto scales = _custom_scale->build_function(fb, {momenta});
        ren_scale = scales.at(0);
        fact_scale1 = scales.at(1);
        fact_scale2 = scales.at(2);
    } else {
        Value scale;
        switch (_dynamical_scale_type) {
        case transverse_energy:
            scale = fb.scale_transverse_energy(momenta);
            break;
        case transverse_mass:
            scale = fb.scale_transverse_mass(momenta);
            break;
        case half_transverse_mass:
            scale = fb.scale_half_transverse_mass(momenta);
            break;
        case partonic_energy:
            scale = fb.scale_partonic_energy(momenta);
            break;
        default:
            throw std::runtime_error("invalid dynamical scale type");
        }
        // dynamical scale only; a fixed scale is an absolute value
        if (_scale_factor != 1.) {
            scale = fb.mul(scale, _scale_factor);
        }
        ren_scale = fact_scale1 = fact_scale2 = scale;
    }
    if (_custom_scale && _scale_factor != 1.) {
        ren_scale = fb.mul(ren_scale, _scale_factor);
        fact_scale1 = fb.mul(fact_scale1, _scale_factor);
        fact_scale2 = fb.mul(fact_scale2, _scale_factor);
    }
    return apply_scale_range(
        fb,
        {{"ren_scale",
          _ren_scale_fixed ? fb.full({_ren_scale, batch_size}) : ren_scale},
         {"fact_scale1",
          _fact_scale_fixed ? fb.full({_fact_scale1, batch_size}) : fact_scale1},
         {"fact_scale2",
          _fact_scale_fixed ? fb.full({_fact_scale2, batch_size}) : fact_scale2}}
    );
}
