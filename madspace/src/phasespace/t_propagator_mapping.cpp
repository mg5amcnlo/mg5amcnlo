#include "madspace/phasespace/t_propagator_mapping.hpp"

#include "madspace/util.hpp"

using namespace madspace;

TPropagatorMapping::TPropagatorMapping(
    const std::vector<std::size_t>& integration_order, double invariant_power
) :
    Mapping(
        "TPropagatorMapping",
        [&] {
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < 3 * integration_order.size() - 1; ++i) {
                input_types.push_back(std::format("random{}", i), batch_float);
            }
            return input_types;
        }(),
        [&] {
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < integration_order.size() + 3; ++i) {
                output_types.push_back(std::format("momentum{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            for (std::size_t i = 0; i < integration_order.size() + 1; ++i) {
                cond_types.push_back(std::format("mass{}", i), batch_float);
            }
            return cond_types;
        }()
    ),
    _integration_order(integration_order),
    _com_scattering(true, invariant_power),
    _lab_scattering(false, invariant_power) {
    std::size_t next_index_low = 0;
    std::size_t next_index_high = integration_order.size() - 1;
    for (std::size_t index : integration_order) {
        if (index == next_index_high) {
            _sample_sides.push_back(true);
            --next_index_high;
        } else if (index == next_index_low) {
            _sample_sides.push_back(false);
            ++next_index_low;
        } else {
            throw std::invalid_argument("Invalid integration order");
        }
    }
}

Mapping::Result TPropagatorMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto r = inputs.begin();
    auto next_random = [&]() { return *(r++); };
    ValueVec dets;

    ValueVec mass_sum_invariants;
    if (_integration_order.size() > 1) {
        // compute sums of outgoing masses, starting from those sampled last
        std::size_t last_index = _integration_order.back();
        ValueVec min_masses{fb.add(m_out.at(last_index), m_out.at(last_index + 1))};
        ValueVec max_masses_subtract;
        for (std::size_t i = _integration_order.size() - 2; i > 0; --i) {
            Value next_mass = m_out.at(_integration_order.at(i) + _sample_sides.at(i));
            min_masses.push_back(fb.add(min_masses.back(), next_mass));
            max_masses_subtract.push_back(next_mass);
        }
        max_masses_subtract.push_back(
            m_out.at(_integration_order.at(0) + _sample_sides.at(0))
        );

        // sample intermediate invariant masses
        auto max_mass = e_cm;
        for (auto [min_mass, max_mass_subtract] :
             zip(std::views::reverse(min_masses),
                 std::views::reverse(max_masses_subtract))) {
            auto s_min = fb.square(min_mass);
            auto s_max = fb.square(fb.sub(max_mass, max_mass_subtract));
            auto s_result =
                _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            auto mass = fb.sqrt(s_result["invariant"]);
            mass_sum_invariants.push_back(mass);
            dets.push_back(s_result["det"]);
            max_mass = mass;
        }
    }
    mass_sum_invariants.push_back(m_out.at(_integration_order.back()));

    // construct initial state momenta
    auto [p1, p2] = fb.com_p_in(e_cm);
    ValueVec p_ext(_integration_order.size() + 3);
    p_ext.at(0) = p1;
    p_ext.at(1) = p2;
    auto p1_rest = p1, p2_rest = p2;

    // sample t-invariants and build momenta of t-channel part of the diagram
    Value k_rest;
    bool first = true;
    for (auto [index, side, mass_sum] :
         zip(_integration_order, _sample_sides, mass_sum_invariants)) {
        auto& scattering = first ? _com_scattering : _lab_scattering;
        first = false;
        std::size_t sampled_index = index + side;
        auto mass = m_out.at(sampled_index);
        auto ks = scattering.build_forward(
            fb,
            {next_random(), next_random(), mass_sum, mass},
            {side ? p1_rest : p2_rest, side ? p2_rest : p1_rest}
        );
        k_rest = ks.at(0);
        auto k = ks.at(1);
        p_ext.at(sampled_index + 2) = k;
        dets.push_back(ks["det"]);
        if (side) {
            p2_rest = fb.sub(p2_rest, k);
        } else {
            p1_rest = fb.sub(p1_rest, k);
        }
    }
    p_ext.at(_integration_order.back() + 2) = k_rest;
    return {{output_types().keys(), p_ext}, fb.product(dets)};
}

Mapping::Result TPropagatorMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    std::size_t n_out = _integration_order.size() + 1;
    ValueVec random_out;
    ValueVec dets;

    // compute s-invariants from momenta
    if (_integration_order.size() > 1) {
        nested_vector2<double> invariant_factors;
        std::size_t last_index = _integration_order.back();
        std::vector<double> marked_indices(n_out + 2);
        marked_indices.at(last_index + 2) = 1;
        marked_indices.at(last_index + 3) = 1;
        invariant_factors.push_back(marked_indices);
        for (std::size_t i = _integration_order.size() - 2; i > 0; --i) {
            marked_indices.at(_integration_order.at(i) + _sample_sides.at(i) + 2) = 1;
            invariant_factors.push_back(marked_indices);
        }
        ValueVec mass_sum_invariants = fb.unstack(
            fb.invariants_from_momenta(fb.stack(inputs.values()), invariant_factors)
        );

        // compute sums of outgoing masses, starting from those sampled last
        ValueVec min_masses{fb.add(m_out.at(last_index), m_out.at(last_index + 1))};
        ValueVec max_masses_subtract;
        for (std::size_t i = _integration_order.size() - 2; i > 0; --i) {
            Value next_mass = m_out.at(_integration_order.at(i) + _sample_sides.at(i));
            min_masses.push_back(fb.add(min_masses.back(), next_mass));
            max_masses_subtract.push_back(next_mass);
        }
        max_masses_subtract.push_back(
            m_out.at(_integration_order.at(0) + _sample_sides.at(0))
        );

        // recover random numbers from intermediate invariant masses
        auto max_mass = e_cm;
        for (auto [mass2, min_mass, max_mass_subtract] :
             zip(std::views::reverse(mass_sum_invariants),
                 std::views::reverse(min_masses),
                 std::views::reverse(max_masses_subtract))) {
            auto s_min = fb.square(min_mass);
            auto s_max = fb.square(fb.sub(max_mass, max_mass_subtract));
            auto result = _uniform_invariant.build_inverse(fb, {mass2}, {s_min, s_max});
            random_out.push_back(result["random"]);
            dets.push_back(result["det"]);
            max_mass = fb.sqrt(mass2);
        }
    }

    // sample t-invariants and build momenta of t-channel part of the diagram
    Value k_rest = fb.add(inputs.at(0), inputs.at(1));
    Value p1_rest = inputs.at(0), p2_rest = inputs.at(1);
    bool first = true;
    for (auto [index, side] : zip(_integration_order, _sample_sides)) {
        auto& scattering = first ? _com_scattering : _lab_scattering;
        first = false;
        std::size_t sampled_index = index + side;
        auto mass = m_out.at(sampled_index);
        auto k = inputs.at(sampled_index + 2);
        k_rest = fb.sub(k_rest, k);
        auto rs = scattering.build_inverse(
            fb, {k_rest, k}, {side ? p1_rest : p2_rest, side ? p2_rest : p1_rest}
        );
        random_out.push_back(rs.at(0));
        random_out.push_back(rs.at(1));
        dets.push_back(rs["det"]);
        if (side) {
            p2_rest = fb.sub(p2_rest, k);
        } else {
            p1_rest = fb.sub(p1_rest, k);
        }
    }

    return {{input_types().keys(), random_out}, fb.product(dets)};
}
