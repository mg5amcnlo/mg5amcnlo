#include "madspace/phasespace/phasespace.hpp"
#include "madspace/constants.hpp"
#include "madspace/util.hpp"

using namespace madspace;

namespace {

struct DecayData {
    const Topology::Decay& decay;
    std::optional<Value> mass;
    std::optional<Value> mass2;
    std::vector<Value> min_masses;
    std::optional<Value> max_mass;
    std::vector<Value> max_mass_subtract;
    std::optional<Value> momentum;
    std::optional<Value> computed_mass;

    DecayData(const Topology::Decay& decay) : decay(decay) {}
};

void update_mass_min_max(
    FunctionBuilder& fb, std::vector<DecayData>& decay_data, std::size_t decay_index
) {
    // Update the minimum mass for the entire decay tree, based on the external
    // masses and already sampled masses.
    for (auto& data : std::views::reverse(decay_data)) {
        data.min_masses.clear();
        if (data.mass) {
            data.min_masses.push_back(data.mass.value());
        } else {
            for (std::size_t child_index : data.decay.child_indices) {
                auto& child_min_masses = decay_data.at(child_index).min_masses;
                data.min_masses.insert(
                    data.min_masses.end(),
                    child_min_masses.begin(),
                    child_min_masses.end()
                );
            }
            if (data.decay.e_min > 0.) {
                data.min_masses = {fb.max(data.decay.e_min, fb.sum(data.min_masses))};
            }
        }
    }

    // Go up the decay tree until propagator with known mass m_i is found. Keep track of
    // all the other nodes branching off. The maximum mass is given by m_max = M_i -
    // sum_{other nodes j} m_{min,j}
    auto& start_decay = decay_data.at(decay_index);
    auto current_decay = &start_decay;
    while (!current_decay->mass && current_decay->decay.index != 0) {
        std::size_t prev_index = current_decay->decay.index;
        current_decay = &decay_data.at(current_decay->decay.parent_index);
        for (std::size_t child_index : current_decay->decay.child_indices) {
            if (child_index == prev_index) {
                continue;
            }
            auto& child_min_masses = decay_data.at(child_index).min_masses;
            start_decay.max_mass_subtract.insert(
                start_decay.max_mass_subtract.end(),
                child_min_masses.begin(),
                child_min_masses.end()
            );
        }
    }
    start_decay.max_mass =
        current_decay->mass ? current_decay->mass : current_decay->max_mass;
}

nested_vector2<me_int_t> invert_permutations(nested_vector2<me_int_t> perms_in) {
    nested_vector2<me_int_t> perms_out(perms_in.size());
    for (auto [perm_in, perm_out] : zip(perms_in, perms_out)) {
        perm_out.resize(perm_in.size());
        std::iota(perm_out.begin(), perm_out.end(), 0);
        std::sort(perm_out.begin(), perm_out.end(), [&](me_int_t i, me_int_t j) {
            return perm_in.at(i) < perm_in.at(j);
        });
    }
    return perms_out;
}

} // namespace

PhaseSpaceMapping::PhaseSpaceMapping(
    const Topology& topology,
    double cm_energy,
    bool leptonic,
    double invariant_power,
    TChannelMode t_channel_mode,
    const std::optional<Cuts>& cuts,
    const std::vector<std::vector<std::size_t>>& permutations
) :
    Mapping(
        "PhaseSpaceMapping",
        {{"random",
          batch_float_array(
              3 * topology.outgoing_masses().size() - (leptonic ? 4 : 2)
          )}},
        {{"momenta", batch_four_vec_array(topology.outgoing_masses().size() + 2)},
         {"x1", batch_float},
         {"x2", batch_float}},
        permutations.size() > 1
            ? NamedVector<Type>{{"permutation_index", batch_int}}
            : NamedVector<Type>{}
    ),
    _topology(topology),
    _cuts(cuts.value_or(Cuts(topology.outgoing_masses().size() + 2))),
    _pi_factors(
        std::pow(2 * PI, 4 - 3 * static_cast<int>(topology.outgoing_masses().size()))
    ),
    _sqrt_s_lab(cm_energy),
    _leptonic(leptonic),
    _map_luminosity(
        !leptonic &&
        (_topology.t_propagator_count() == 0 ||
         t_channel_mode != PhaseSpaceMapping::chili)
    ),
    _t_mapping(std::monostate{}) {
    bool has_t_channel = _topology.t_propagator_count() > 0;
    struct DecayInfo {
        double m_min, pt_min, eta_max;
        std::optional<Invariant> invariant;
    };
    std::vector<DecayInfo> decay_info(_topology.decays().size());
    for (auto [index, m_min, pt_min, eta_max] :
         zip(_topology.outgoing_indices(),
             _topology.outgoing_masses(),
             _cuts.pt_min(),
             _cuts.eta_max())) {
        decay_info.at(index) = {m_min, pt_min, eta_max, std::nullopt};
    }
    for (auto [decay, info] :
         zip(std::views::reverse(_topology.decays()),
             std::views::reverse(decay_info))) {
        if (decay.child_indices.size() == 0) {
            continue;
        }

        bool is_com_decay = decay.index == 0;
        if (decay.index != 0 || !has_t_channel) {
            if (decay.child_indices.size() == 2) {
                _s_decays.push_back(TwoBodyDecay(is_com_decay));
            } else if (decay.child_indices.size() == 3) {
                _s_decays.push_back(ThreeBodyDecay(is_com_decay));
            } else {
                _s_decays.push_back(
                    FastRamboMapping(decay.child_indices.size(), false, is_com_decay)
                );
            }
        }

        double m_min = 0.;
        for (std::size_t child_index : decay.child_indices) {
            m_min += decay_info.at(child_index).m_min;
        }
        info.m_min = std::max(m_min, decay.e_min);
        info.pt_min = 0.;
        info.eta_max = std::numeric_limits<double>::infinity();

        if (!is_com_decay || _map_luminosity) {
            double mass = decay.width == 0. ? 0. : decay.mass;
            double width = decay.width;
            info.invariant = Invariant(invariant_power, mass, width);
        }
    }
    for (std::size_t index : _topology.decay_integration_order()) {
        auto& invariant = decay_info.at(index).invariant;
        if (invariant) {
            _s_invariants.push_back(invariant.value());
        }
    }

    double total_mass = 0.;
    for (std::size_t index : topology.decays().at(0).child_indices) {
        total_mass += decay_info.at(index).m_min;
    }
    double sqrt_s_hat_min = _cuts.sqrt_s_min();
    double s_hat_min =
        std::max(total_mass * total_mass, sqrt_s_hat_min * sqrt_s_hat_min);
    if (has_t_channel) {
        if (t_channel_mode == PhaseSpaceMapping::chili) {
            // |y| <= |eta|, so we can pass y_max = eta_max
            std::vector<double> eta_max, pt_min;
            for (std::size_t index : topology.decays().at(0).child_indices) {
                auto& info = decay_info.at(index);
                eta_max.push_back(info.eta_max);
                pt_min.push_back(info.pt_min);
            }
            _t_mapping =
                ChiliMapping(_topology.t_propagator_count() + 1, eta_max, pt_min);
        } else if (t_channel_mode == PhaseSpaceMapping::propagator ||
                   topology.t_propagator_count() < 2) {
            _t_mapping =
                TPropagatorMapping(_topology.t_integration_order(), invariant_power);
        } else if (t_channel_mode == PhaseSpaceMapping::rambo) {
            // TODO: add massless special case
            _t_mapping = FastRamboMapping(_topology.t_propagator_count() + 1, false);
        }
    }

    for (auto& perm : permutations) {
        _permutations.emplace_back(perm.begin(), perm.end());
    }
}

PhaseSpaceMapping::PhaseSpaceMapping(
    const std::vector<double>& external_masses,
    double cm_energy,
    bool leptonic,
    double invariant_power,
    TChannelMode mode,
    const std::optional<Cuts>& cuts
) :
    PhaseSpaceMapping(
        Topology([&] {
            if (external_masses.size() < 4) {
                throw std::invalid_argument("The number of masses must be at least 4");
            }
            std::vector<Diagram::Vertex> vertices;
            auto n_out = external_masses.size() - 2;
            vertices.push_back({
                {Diagram::incoming, 0},
                {Diagram::propagator, 0},
                {Diagram::outgoing, 0},
            });
            for (std::size_t i = 1; i < n_out - 1; ++i) {
                vertices.push_back({
                    {Diagram::propagator, i - 1},
                    {Diagram::propagator, i},
                    {Diagram::outgoing, i},
                });
            }
            vertices.push_back({
                {Diagram::incoming, 1},
                {Diagram::propagator, n_out - 2},
                {Diagram::outgoing, n_out - 1},
            });
            return Diagram(
                {external_masses.at(0), external_masses.at(1)},
                {external_masses.begin() + 2, external_masses.end()},
                std::vector<Propagator>(n_out - 1),
                vertices
            );
        }()),
        cm_energy,
        leptonic,
        invariant_power,
        mode,
        cuts
    ) {}

Mapping::Result PhaseSpaceMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto random_numbers = fb.unstack(inputs.at(0));
    auto r = random_numbers.begin();
    auto next_random = [&]() { return *(r++); };

    ValueVec dets{_pi_factors};
    Value x1 = 1.0, x2 = 1.0;

    // initialize masses and square masses
    std::vector<DecayData> decay_data(
        _topology.decays().begin(), _topology.decays().end()
    );
    for (auto [decay_index, mass] :
         zip(_topology.outgoing_indices(), _topology.outgoing_masses())) {
        auto& data = decay_data.at(decay_index);
        data.mass = mass;
        data.mass2 = mass * mass;
    }
    auto& root_data = decay_data.at(0);
    root_data.max_mass = _sqrt_s_lab;

    // sample decay s-invariants, following the integration order
    std::size_t invariant_index = 0;
    for (std::size_t decay_index : _topology.decay_integration_order()) {
        auto& decay = _topology.decays().at(decay_index);
        auto& data = decay_data.at(decay_index);
        update_mass_min_max(fb, decay_data, decay_index);
        auto s_min = fb.square(fb.sum(data.min_masses));
        auto sqrt_s_max = fb.sub(data.max_mass.value(), fb.sum(data.max_mass_subtract));
        if (data.decay.e_max > 0.) {
            sqrt_s_max = fb.min(sqrt_s_max, data.decay.e_max);
        }
        if (decay_index != 0 || _map_luminosity) {
            auto s_max = fb.square(sqrt_s_max);
            auto invariant =
                _s_invariants.at(invariant_index++)
                    .build_forward(fb, {next_random()}, {s_min, s_max});
            data.mass2 = invariant["invariant"];
            data.mass = fb.sqrt(data.mass2.value());
            dets.push_back(invariant["det"]);
        } else if (decay_index == 0) {
            data.mass2 = _sqrt_s_lab * _sqrt_s_lab;
            data.mass = _sqrt_s_lab;
        }
    }

    // sample momentum fractions
    auto sqrt_s_hat = root_data.mass.value();
    auto s_hat = root_data.mass2.value();
    if (_map_luminosity) {
        auto [x1_new, x2_new, det_x] =
            fb.r_to_x1x2(next_random(), s_hat, _sqrt_s_lab * _sqrt_s_lab);
        x1 = x1_new;
        x2 = x2_new;
        dets.push_back(det_x);
    }

    // if required, build t-channel part of phase space mapping
    ValueVec p_ext;
    std::visit(
        Overloaded{
            [&](auto& t_mapping) {
                ValueVec args, conds;
                for (std::size_t i = 0; i < t_mapping.random_dim(); ++i) {
                    args.push_back(next_random());
                }
                conds.push_back(sqrt_s_hat);
                for (std::size_t index : decay_data.at(0).decay.child_indices) {
                    conds.push_back(decay_data.at(index).mass.value());
                }
                auto t_result = t_mapping.build_forward(fb, args, conds);
                std::size_t result_index;
                using TMapping = std::decay_t<decltype(t_mapping)>;
                if constexpr (std::is_same_v<TMapping, FastRamboMapping>) {
                    auto [p1, p2] = fb.com_p_in(sqrt_s_hat);
                    p_ext = {p1, p2};
                    result_index = 0;
                } else {
                    p_ext = {t_result.at(0), t_result.at(1)};
                    result_index = 2;
                }
                for (std::size_t index : decay_data.at(0).decay.child_indices) {
                    decay_data.at(index).momentum = t_result.at(result_index);
                    ++result_index;
                }
                dets.push_back(t_result["det"]);

                if constexpr (std::is_same_v<TMapping, ChiliMapping>) {
                    auto [x1_new, x2_new] = fb.momenta_to_x1x2(
                        fb.stack({t_result.at(0), t_result.at(1)}), _sqrt_s_lab
                    );
                    x1 = x1_new;
                    x2 = x2_new;
                }
            },
            [&](std::monostate) {
                auto [p1, p2] = fb.com_p_in(sqrt_s_hat);
                p_ext = {p1, p2};
            }
        },
        _t_mapping
    );

    // go through decays and generate momenta
    std::size_t decay_map_index = _s_decays.size();
    for (auto& data : decay_data) {
        if (data.decay.child_indices.size() == 0) {
            continue;
        }
        if (data.decay.index == 0 &&
            !std::holds_alternative<std::monostate>(_t_mapping)) {
            continue;
        }
        std::visit(
            [&](auto& decay_map) {
                ValueVec decay_args{r, r += decay_map.random_dim()};
                decay_args.push_back(data.mass.value());
                for (std::size_t child_index : data.decay.child_indices) {
                    decay_args.push_back(decay_data.at(child_index).mass.value());
                }
                if (data.decay.index != 0) {
                    decay_args.push_back(data.momentum.value());
                }
                auto k_out = decay_map.build_forward(fb, decay_args, {});
                for (auto [child_index, k] : zip(data.decay.child_indices, k_out)) {
                    decay_data.at(child_index).momentum = k;
                }
                dets.push_back(k_out["det"]);
            },
            _s_decays.at(--decay_map_index)
        );
    }

    // collect outgoing momenta
    for (std::size_t decay_index : _topology.outgoing_indices()) {
        p_ext.push_back(decay_data.at(decay_index).momentum.value());
    }
    auto p_ext_stack = fb.stack(p_ext);

    // permute momenta if permutations are given
    if (_permutations.size() > 1) {
        p_ext_stack = fb.permute_momenta(p_ext_stack, _permutations, conditions.at(0));
    } else if (_permutations.size() == 1 &&
               !std::is_sorted(
                   _permutations.at(0).begin(), _permutations.at(0).end()
               )) {
        p_ext_stack =
            fb.permute_momenta(p_ext_stack, _permutations, static_cast<me_int_t>(0));
    }

    // boost into correct frame and apply cuts
    auto p_ext_lab = _map_luminosity ? fb.boost_beam(p_ext_stack, x1, x2) : p_ext_stack;
    dets.push_back(_cuts.build_function(fb, {p_ext_lab}).at(0));
    auto ps_weight = fb.cut_unphysical(fb.product(dets), p_ext_lab, x1, x2);
    return {{{"momenta", p_ext_lab}, {"x1", x1}, {"x2", x2}}, ps_weight};
}

Mapping::Result PhaseSpaceMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value p_ext_lab = inputs.at(0), x1 = inputs.at(1), x2 = inputs.at(2);
    Value p_ext_stack =
        _map_luminosity ? fb.boost_beam_inverse(p_ext_lab, x1, x2) : p_ext_lab;

    // permute momenta if permutations are given
    if (_permutations.size() > 1) {
        p_ext_stack = fb.permute_momenta(
            p_ext_stack, invert_permutations(_permutations), conditions.at(0)
        );
    } else if (_permutations.size() == 1 &&
               !std::is_sorted(
                   _permutations.at(0).begin(), _permutations.at(0).end()
               )) {
        p_ext_stack = fb.permute_momenta(
            p_ext_stack, invert_permutations(_permutations), static_cast<me_int_t>(0)
        );
    }

    // initialize momenta, masses and square masses
    ValueVec p_ext = fb.unstack(p_ext_stack);
    std::vector<DecayData> decay_data(
        _topology.decays().begin(), _topology.decays().end()
    );
    for (auto [decay_index, mass, momentum] :
         zip(_topology.outgoing_indices(),
             _topology.outgoing_masses(),
             std::span(p_ext.begin() + 2, p_ext.end()))) {
        auto& data = decay_data.at(decay_index);
        data.mass = mass;
        data.mass2 = mass * mass;
        data.computed_mass = mass;
        data.momentum = momentum;
    }
    auto& root_data = decay_data.at(0);
    root_data.max_mass = _sqrt_s_lab;

    // go through decays and recover random numbers from momenta
    ValueVec random_out_reversed;
    ValueVec dets{1. / _pi_factors};
    for (std::size_t decay_map_index = 0;
         auto& data : std::views::reverse(decay_data)) {
        if (data.decay.child_indices.size() == 0) {
            continue;
        }
        if (data.decay.index == 0 &&
            !std::holds_alternative<std::monostate>(_t_mapping)) {
            continue;
        }
        std::visit(
            [&](auto& decay_map) {
                ValueVec decay_args;
                for (auto child_index : data.decay.child_indices) {
                    decay_args.push_back(decay_data.at(child_index).momentum.value());
                }
                auto decay_out = decay_map.build_inverse(fb, decay_args, {});
                data.computed_mass = decay_out.at(decay_map.random_dim());
                random_out_reversed.insert(
                    random_out_reversed.end(),
                    decay_out.rend() - decay_map.random_dim(),
                    decay_out.rend()
                );
                if (data.decay.index != 0) {
                    data.momentum = decay_out.at(decay_out.size() - 2);
                }
                dets.push_back(decay_out["det"]);
            },
            _s_decays.at(decay_map_index++)
        );
    }

    // if required, build inverse t-channel part of phase space mapping
    std::visit(
        Overloaded{
            [&](auto& t_mapping) {
                ValueVec args, conds;
                using TMapping = std::decay_t<decltype(t_mapping)>;
                if constexpr (!std::is_same_v<TMapping, FastRamboMapping>) {
                    args.push_back(p_ext.at(0));
                    args.push_back(p_ext.at(1));
                }
                Value e_cm = std::is_same_v<TMapping, ChiliMapping>
                    ? Value(_sqrt_s_lab)
                    : fb.obs_mass(fb.add(p_ext.at(0), p_ext.at(1)));
                conds.push_back(e_cm);
                decay_data.at(0).computed_mass = e_cm;
                for (std::size_t index : decay_data.at(0).decay.child_indices) {
                    args.push_back(decay_data.at(index).momentum.value());
                    conds.push_back(decay_data.at(index).computed_mass.value());
                }
                auto t_result = t_mapping.build_inverse(fb, args, conds);
                random_out_reversed.insert(
                    random_out_reversed.end(),
                    t_result.rend() - t_mapping.random_dim(),
                    t_result.rend()
                );
                dets.push_back(t_result["det"]);
            },
            [&](std::monostate) {}
        },
        _t_mapping
    );

    if (_map_luminosity) {
        auto [r, det_x] = fb.x1x2_to_r(x1, x2, _sqrt_s_lab * _sqrt_s_lab);
        random_out_reversed.push_back(r);
        dets.push_back(det_x);
    }

    // recover random numbers for s-invariants, following the integration order
    ValueVec random_out;
    std::size_t invariant_index = 0;
    for (std::size_t decay_index : _topology.decay_integration_order()) {
        auto& decay = _topology.decays().at(decay_index);
        auto& data = decay_data.at(decay_index);
        update_mass_min_max(fb, decay_data, decay_index);
        auto s_min = fb.square(fb.sum(data.min_masses));
        auto sqrt_s_max = fb.sub(data.max_mass.value(), fb.sum(data.max_mass_subtract));
        if (data.decay.e_max > 0.) {
            sqrt_s_max = fb.min(sqrt_s_max, data.decay.e_max);
        }
        if (decay_index != 0 || _map_luminosity) {
            auto s_max = fb.square(sqrt_s_max);
            data.mass = data.computed_mass.value();
            data.mass2 = fb.square(data.mass.value());
            auto invariant =
                _s_invariants.at(invariant_index++)
                    .build_inverse(fb, {data.mass2.value()}, {s_min, s_max});
            random_out.push_back(invariant["random"]);
            dets.push_back(invariant["det"]);
        } else if (decay_index == 0) {
            data.mass2 = _sqrt_s_lab * _sqrt_s_lab;
            data.mass = _sqrt_s_lab;
        }
    }

    random_out.insert(
        random_out.end(), random_out_reversed.rbegin(), random_out_reversed.rend()
    );
    return {{{"random", fb.stack(random_out)}}, fb.product(dets)};
}
