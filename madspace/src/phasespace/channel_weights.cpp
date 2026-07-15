#include "madspace/phasespace/channel_weights.hpp"

#include "madspace/util.hpp"

using namespace madspace;

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
