#include "madspace/phasespace/scale.hpp"

using namespace madspace;

EnergyScale::EnergyScale(
    std::size_t particle_count,
    DynamicalScaleType dynamical_scale_type,
    bool ren_scale_fixed,
    bool fact_scale_fixed,
    double ren_scale,
    double fact_scale1,
    double fact_scale2
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
    _fact_scale2(fact_scale2) {}

NamedVector<Value> EnergyScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto momenta = args.at(0);
    if (_ren_scale_fixed && _fact_scale_fixed) {
        auto batch_size = fb.batch_size({momenta});
        return {
            {"ren_scale", fb.full({_ren_scale, batch_size})},
            {"fact_scale1", fb.full({_fact_scale1, batch_size})},
            {"fact_scale2", fb.full({_fact_scale2, batch_size})},
        };
    }
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
    auto batch_size = fb.batch_size({momenta});
    return {
        {"ren_scale", _ren_scale_fixed ? fb.full({_ren_scale, batch_size}) : scale},
        {"fact_scale1",
         _fact_scale_fixed ? fb.full({_fact_scale1, batch_size}) : scale},
        {"fact_scale2", _fact_scale_fixed ? fb.full({_fact_scale2, batch_size}) : scale}
    };
}
