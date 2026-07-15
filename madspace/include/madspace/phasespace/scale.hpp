#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

class EnergyScale : public FunctionGenerator {
public:
    enum DynamicalScaleType {
        transverse_energy,
        transverse_mass,
        half_transverse_mass,
        partonic_energy
    };

    EnergyScale(std::size_t particle_count) :
        EnergyScale(particle_count, half_transverse_mass, false, false, 0., 0., 0.) {}
    EnergyScale(std::size_t particle_count, DynamicalScaleType type) :
        EnergyScale(particle_count, type, false, false, 0., 0., 0.) {}
    EnergyScale(std::size_t particle_count, double fixed_scale) :
        EnergyScale(
            particle_count,
            half_transverse_mass,
            true,
            true,
            fixed_scale,
            fixed_scale,
            fixed_scale
        ) {}
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType dynamical_scale_type,
        bool ren_scale_fixed,
        bool fact_scale_fixed,
        double ren_scale,
        double fact_scale1,
        double fact_scale2
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    DynamicalScaleType _dynamical_scale_type;
    bool _ren_scale_fixed;
    bool _fact_scale_fixed;
    double _ren_scale;
    double _fact_scale1;
    double _fact_scale2;
};

} // namespace madspace
