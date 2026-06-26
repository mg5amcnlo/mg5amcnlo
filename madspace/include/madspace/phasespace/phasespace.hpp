#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/chili.hpp"
#include "madspace/phasespace/cuts.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/luminosity.hpp"
#include "madspace/phasespace/rambo.hpp"
#include "madspace/phasespace/t_propagator_mapping.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

class PhaseSpaceMapping : public Mapping {
public:
    enum TChannelMode { propagator, rambo, chili };

    PhaseSpaceMapping(
        const Topology& topology,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode t_channel_mode = propagator,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::vector<std::vector<std::size_t>>& permutations = {}
    );

    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt
    );

    std::size_t random_dim() const {
        return 3 * _topology.outgoing_masses().size() - (_leptonic ? 4 : 2);
    }
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + 2;
    }
    std::size_t channel_count() const { return _permutations.size(); }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    Topology _topology;
    Cuts _cuts;
    double _pi_factors;
    double _sqrt_s_lab;
    bool _leptonic;
    bool _map_luminosity;
    std::vector<Invariant> _s_invariants;
    std::variant<TPropagatorMapping, FastRamboMapping, ChiliMapping, std::monostate>
        _t_mapping;
    std::vector<std::variant<TwoBodyDecay, ThreeBodyDecay, FastRamboMapping>> _s_decays;
    nested_vector2<me_int_t> _permutations;
};

} // namespace madspace
