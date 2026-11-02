#include "madspace/phasespace/channel_weights.hpp"

#include "madspace/util.hpp"

#include <cmath>
#include <utility>

using namespace madspace;

namespace {

std::vector<me_int_t> first_indices(std::size_t count) {
    std::vector<me_int_t> indices(count);
    for (std::size_t i = 0; i < count; ++i) {
        indices[i] = static_cast<me_int_t>(i);
    }
    return indices;
}

} // namespace

PropagatorChannelWeights::PropagatorChannelWeights(
    const std::vector<Topology>& topologies,
    const nested_vector3<std::size_t>& permutations,
    const nested_vector2<std::size_t>& channel_indices
) :
    FunctionGenerator(
        "PropagatorChannelWeights",
        {{"momenta",
          batch_four_vec_array(topologies.at(0).outgoing_masses().size() + 2)}},
        {{"channel_weights", batch_float_array([&]() {
              std::size_t channel_count = 0;
              for (auto& perm : permutations) {
                  channel_count += perm.size();
              }
              return channel_count;
          }())}}
    ) {
    std::size_t channel_count = return_types().at(0).shape.at(0);
    _invariant_indices.resize(channel_count);
    _masses.resize(channel_count);
    _widths.resize(channel_count);

    std::map<std::vector<me_int_t>, std::size_t> found_factors;
    std::size_t max_propagator_count = 0;
    for (auto [topology, chan_perms, indices] :
         zip(topologies, permutations, channel_indices)) {
        auto mom_terms = topology.propagator_momentum_terms();
        if (mom_terms.size() > max_propagator_count) {
            max_propagator_count = mom_terms.size();
        }
        for (auto [perm, index] : zip(chan_perms, indices)) {
            auto& chan_invariants = _invariant_indices.at(index);
            auto& chan_masses = _masses.at(index);
            auto& chan_widths = _widths.at(index);
            for (auto [factors, mass, width] : mom_terms) {
                std::vector<me_int_t> permuted_factors;
                for (std::size_t i : perm) {
                    permuted_factors.push_back(factors.at(i));
                }
                auto found = found_factors.find(permuted_factors);
                std::size_t inv_index;
                if (found == found_factors.end()) {
                    inv_index = _momentum_factors.size();
                    _momentum_factors.emplace_back(
                        permuted_factors.begin(), permuted_factors.end()
                    );
                    found_factors[permuted_factors] = inv_index;
                } else {
                    inv_index = found->second;
                }
                chan_masses.push_back(mass);
                chan_widths.push_back(width);
                chan_invariants.push_back(inv_index);
            }
        }
    }
    for (auto& masses : _masses) {
        masses.resize(max_propagator_count);
    }
    for (auto& widths : _widths) {
        widths.resize(max_propagator_count);
    }
    for (auto& invars : _invariant_indices) {
        invars.resize(max_propagator_count, -1);
    }
}

NamedVector<Value> PropagatorChannelWeights::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto invariants = fb.invariants_from_momenta(args["momenta"], _momentum_factors);
    auto channel_weights =
        fb.sde2_channel_weights(invariants, _masses, _widths, _invariant_indices);
    return {{"channel_weights", channel_weights}};
}

SubchannelWeights::SubchannelWeights(
    const nested_vector2<Topology>& topologies,
    const nested_vector3<std::size_t>& permutations,
    const nested_vector2<std::size_t>& channel_indices
) :
    FunctionGenerator(
        "SubchannelWeights",
        {{"momenta",
          batch_four_vec_array(topologies.at(0).at(0).outgoing_masses().size() + 2)},
         {"channel_weights_in", batch_float_array([&]() {
              std::size_t channel_count = 0;
              for (auto& perm : permutations) {
                  channel_count += perm.size();
              }
              return channel_count;
          }())}},
        {{"channel_weights_out", batch_float_array([&]() {
              std::size_t channel_count = 0;
              for (auto [perm, topos] : zip(permutations, topologies)) {
                  channel_count += perm.size() * topos.size();
              }
              return channel_count;
          }())}}
    ) {
    std::map<std::vector<me_int_t>, std::size_t> found_factors;
    std::size_t max_propagator_count = 0;

    for (auto [topos, perms, indices] :
         zip(topologies, permutations, channel_indices)) {
        if (topos.size() <= 1) {
            _channel_indices.insert(
                _channel_indices.end(), indices.begin(), indices.end()
            );
            _subchannel_indices.insert(_subchannel_indices.end(), indices.size(), -1);
            continue;
        }

        auto mom_terms = topos.at(0).propagator_momentum_terms(true);
        std::vector<std::size_t> on_shell_counts(mom_terms.size());
        nested_vector2<bool> on_shell_configs;
        for (auto& topo : topos) {
            std::size_t decay_index = 0;
            auto& configs = on_shell_configs.emplace_back();
            for (auto& decay : std::views::reverse(topo.decays())) {
                if (decay.index == 0) {
                    if (topo.t_integration_order().size() != 0) {
                        continue;
                    }
                } else if (decay.child_indices.size() == 0) {
                    continue;
                }
                on_shell_counts.at(decay_index) += decay.on_shell;
                configs.push_back(decay.on_shell);
                ++decay_index;
            }
        }

        for (auto [perm, index] : zip(perms, indices)) {
            std::vector<me_int_t> chan_invariants;
            std::vector<double> chan_masses, chan_widths;

            for (auto [mom_term, on_shell_count] : zip(mom_terms, on_shell_counts)) {
                if (on_shell_count == 0 || on_shell_count == topos.size()) {
                    continue;
                }
                auto& [factors, mass, width] = mom_term;
                std::vector<me_int_t> permuted_factors;
                for (std::size_t i : perm) {
                    permuted_factors.push_back(factors.at(i));
                }
                auto found = found_factors.find(permuted_factors);
                std::size_t inv_index;
                if (found == found_factors.end()) {
                    inv_index = _momentum_factors.size();
                    _momentum_factors.emplace_back(
                        permuted_factors.begin(), permuted_factors.end()
                    );
                    found_factors[permuted_factors] = inv_index;
                } else {
                    inv_index = found->second;
                }
                chan_masses.push_back(mass);
                chan_widths.push_back(width);
                chan_invariants.push_back(inv_index);
            }
            if (chan_masses.size() > max_propagator_count) {
                max_propagator_count = chan_masses.size();
            }
            _group_sizes.push_back(topos.size());
            for (auto [topo, configs] : zip(topos, on_shell_configs)) {
                auto& chan_on_shell = _on_shell.emplace_back();
                for (auto [on_shell, count] : zip(configs, on_shell_counts)) {
                    if (count != 0 && count != topos.size()) {
                        chan_on_shell.push_back(on_shell);
                    }
                }
                _channel_indices.push_back(index);
                _subchannel_indices.push_back(_masses.size());
                _masses.push_back(chan_masses);
                _widths.push_back(chan_widths);
                _invariant_indices.push_back(chan_invariants);
            }
        }
    }
    for (auto& masses : _masses) {
        masses.resize(max_propagator_count);
    }
    for (auto& widths : _widths) {
        widths.resize(max_propagator_count);
    }
    for (auto& invars : _invariant_indices) {
        invars.resize(max_propagator_count, -1);
    }
    for (auto& on_shell : _on_shell) {
        on_shell.resize(max_propagator_count);
    }
}

NamedVector<Value> SubchannelWeights::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto channel_weights_in = args.at(1);
    auto invariants = fb.invariants_from_momenta(args["momenta"], _momentum_factors);
    auto subchan_weights = fb.subchannel_weights(
        invariants, _masses, _widths, _invariant_indices, _on_shell, _group_sizes
    );
    auto channel_weights_out = fb.apply_subchannel_weights(
        channel_weights_in, subchan_weights, _channel_indices, _subchannel_indices
    );
    return {{"channel_weights_out", channel_weights_out}};
}

MappingChannelWeights::MappingChannelWeights(
    const std::vector<PhaseSpaceMapping>& mappings,
    const nested_vector2<double>& flavor_masks,
    const std::vector<me_int_t>& diagram_channels,
    const nested_vector2<double>& flavor_diagram_masks,
    const std::vector<PhaseSpaceMapping>& partner_mappings
) :
    FunctionGenerator(
        "MappingChannelWeights",
        [&] {
            if (mappings.empty()) {
                throw std::invalid_argument("MappingChannelWeights needs a channel");
            }
            NamedVector<Type> args{
                {"momenta", batch_four_vec_array(mappings.at(0).particle_count())},
                {"x1", batch_float},
                {"x2", batch_float},
                {"flavor", batch_int},
            };
            if (mappings.at(0).mirror_beams()) {
                args.push_back("mirror_index", batch_int);
            }
            return args;
        }(),
        [&] {
            NamedVector<Type> ret{
                {"channel_weights", batch_float_array(mappings.size())}
            };
            if (!diagram_channels.empty()) {
                ret.push_back("diagram_index", batch_int);
            }
            return ret;
        }()
    ),
    _mappings(mappings),
    _channel_count(mappings.size()),
    _flavor_masks(flavor_masks),
    _diagram_channels(diagram_channels),
    _flavor_diagram_masks(flavor_diagram_masks),
    _mirror_beams(mappings.at(0).mirror_beams()) {
    if (!partner_mappings.empty() && mappings.size() != 1) {
        // a learned weight of channels that also partner elsewhere is not a
        // partition of the contribution any more
        throw std::invalid_argument(
            "MappingChannelWeights: partner mappings need a single channel"
        );
    }
    _mappings.insert(_mappings.end(), partner_mappings.begin(), partner_mappings.end());
    for (auto& mapping : _mappings) {
        if (mapping.channel_count() > 1) {
            throw std::invalid_argument(
                "MappingChannelWeights: the mappings cannot have permutations"
            );
        }
        if (mapping.particle_count() != mappings.at(0).particle_count() ||
            mapping.mirror_beams() != _mirror_beams) {
            throw std::invalid_argument(
                "MappingChannelWeights: the mappings must take the same momenta"
            );
        }
    }
    for (auto& mask : flavor_masks) {
        if (mask.size() != _mappings.size()) {
            throw std::invalid_argument(
                "MappingChannelWeights: one flavor mask entry per channel and "
                "partner expected"
            );
        }
    }
    if (flavor_masks.empty()) {
        throw std::invalid_argument("MappingChannelWeights needs a flavor");
    }
    for (auto channel : diagram_channels) {
        if (channel < 0 || std::cmp_greater_equal(channel, _mappings.size())) {
            throw std::invalid_argument(
                "MappingChannelWeights: diagram channel out of range"
            );
        }
    }
    if (!diagram_channels.empty()) {
        if (flavor_diagram_masks.size() != flavor_masks.size()) {
            throw std::invalid_argument(
                "MappingChannelWeights: one diagram mask per flavor expected"
            );
        }
        for (auto& mask : flavor_diagram_masks) {
            if (mask.size() != diagram_channels.size()) {
                throw std::invalid_argument(
                    "MappingChannelWeights: one diagram mask entry per diagram "
                    "expected"
                );
            }
        }
    }
}

NamedVector<Value> MappingChannelWeights::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec conditions;
    if (_mirror_beams) {
        conditions.push_back(args.at("mirror_index"));
    }
    ValueVec densities;
    for (auto& mapping : _mappings) {
        auto inverse = mapping.build_inverse(
            fb, {args.at("momenta"), args.at("x1"), args.at("x2")}, conditions
        );
        // the discrete choices are sampled uniformly, each with probability 1/2
        Value density = fb.mul(
            inverse.at("det"),
            std::pow(0.5, static_cast<double>(mapping.discrete_dim()))
        );
        // the event is out of reach of the mapping if a random number is outside
        // the unit hypercube
        if (mapping.random_dim() > 0) {
            density = fb.mul(density, fb.cut_all(inverse.at("random"), 0., 1.));
        }
        densities.push_back(density);
    }
    Value flavor = args.at("flavor");
    Value channel_weights = fb.density_channel_weights(
        fb.stack(densities), fb.gather_vector(flavor, _flavor_masks)
    );
    NamedVector<Value> ret{
        {"channel_weights",
         has_partners() ? fb.select(channel_weights, first_indices(_channel_count))
                        : channel_weights}
    };
    if (!_diagram_channels.empty()) {
        Value probs = fb.mul(
            fb.select(channel_weights, _diagram_channels),
            fb.gather_vector(flavor, _flavor_diagram_masks)
        );
        Value random =
            fb
                .unstack(fb.random(
                    fb.batch_size({args.at("momenta")}), static_cast<me_int_t>(1)
                ))
                .at(0);
        auto [diagram, det] = fb.sample_discrete_probs(random, probs);
        ret.push_back("diagram_index", diagram);
    }
    return ret;
}
