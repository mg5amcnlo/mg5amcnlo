#include "madspace/phasespace/color_ordered_mapping.hpp"

#include <algorithm>
#include <stdexcept>

#include "madspace/util.hpp"

using namespace madspace;

namespace {

// Cyclically rotate the color order so that particle 0 is first, then
// split the rest into two sets: particles strictly between 0 and 1 in the
// cyclic order go into set1, particles after 1 (wrapping around) go into
// set2.
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
split_sets_from_color_order(const std::vector<std::size_t>& color_order) {
    std::size_t n = color_order.size();
    if (n < 4) {
        throw std::invalid_argument(
            "ColorOrderedMapping requires at least 4 particles (2 beams + 2 outgoing)"
        );
    }
    auto it = std::find(color_order.begin(), color_order.end(), 0u);
    if (it == color_order.end()) {
        throw std::invalid_argument("color_order must contain particle 0");
    }
    std::size_t i0 = std::distance(color_order.begin(), it);
    std::vector<std::size_t> rotated;
    rotated.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
        rotated.push_back(color_order[(i0 + k) % n]);
    }
    auto it1 = std::find(rotated.begin(), rotated.end(), 1u);
    if (it1 == rotated.end()) {
        throw std::invalid_argument("color_order must contain particle 1");
    }
    std::size_t i1 = std::distance(rotated.begin(), it1);
    std::vector<std::size_t> set1, set2;
    for (std::size_t k = 1; k < i1; ++k) {
        std::size_t p = rotated[k];
        if (p <= 1) {
            throw std::invalid_argument("invalid color_order");
        }
        set1.push_back(p - 2);
    }
    for (std::size_t k = i1 + 1; k < n; ++k) {
        std::size_t p = rotated[k];
        if (p <= 1) {
            throw std::invalid_argument("invalid color_order");
        }
        set2.push_back(p - 2);
    }
    // An empty set is allowed: it means particles 0 and 1 are adjacent in the
    // color order, so every outgoing particle sits on a single side. With n >= 4
    // (>= 2 outgoing) at most one of the two sets can be empty.
    if (set1.empty() && set2.empty()) {
        throw std::invalid_argument(
            "ColorOrderedMapping: at least one set must be non-empty"
        );
    }
    return {set1, set2};
}

std::size_t n_intermediate_masses_for_set_size(std::size_t k) {
    return (k >= 2) ? (k - 2) : 0;
}

std::size_t n_block_randoms_for_set_size(std::size_t k) {
    if (k <= 1) {
        return 0;
    }
    // First peel: 2->2 LAB (2 randoms); each subsequent peel: 2->3 (3 randoms).
    return 2 + 3 * (k - 2);
}

} // namespace

ColorOrderedMapping::ColorOrderedMapping(
    const std::vector<std::size_t>& color_order,
    double t_invariant_power,
    double s_invariant_power
) :
    Mapping(
        "ColorOrderedMapping",
        [&] {
            auto [s1, s2] = split_sets_from_color_order(color_order);
            bool use_single_chain = s1.empty() || s2.empty();
            bool use_double_t =
                !use_single_chain && ((s1.size() == 1) != (s2.size() == 1));
            // single chain: 0 set-masses + 0 central randoms (mass is e_cm, no central
            // block). DoubleT branch: 0 set-masses + 3 central randoms. Standard
            // branch: 1 set-mass per multi-particle side + 2 central randoms.
            std::size_t n_set_masses = (use_single_chain || use_double_t)
                ? 0u
                : (s1.size() >= 2 ? 1u : 0u) + (s2.size() >= 2 ? 1u : 0u);
            std::size_t n_central = use_single_chain ? 0u : (use_double_t ? 3u : 2u);
            std::size_t n_intermediate_masses =
                n_intermediate_masses_for_set_size(s1.size()) +
                n_intermediate_masses_for_set_size(s2.size());
            std::size_t n_walk = n_block_randoms_for_set_size(s1.size()) +
                n_block_randoms_for_set_size(s2.size());
            std::size_t total =
                n_set_masses + n_intermediate_masses + n_central + n_walk;
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < total; ++i) {
                input_types.push_back(std::format("random{}", i), batch_float);
            }
            return input_types;
        }(),
        [&] {
            std::size_t n_out = color_order.size() - 2;
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < n_out + 2; ++i) {
                output_types.push_back(std::format("momentum{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            std::size_t n_out = color_order.size() - 2;
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            for (std::size_t i = 0; i < n_out; ++i) {
                cond_types.push_back(std::format("mass{}", i), batch_float);
            }
            return cond_types;
        }()
    ),
    _n_out(color_order.size() - 2),
    _com_scattering(true, t_invariant_power),
    _lab_scattering(false, t_invariant_power),
    _two_to_three(t_invariant_power, 0., 0., s_invariant_power, 0., 0.),
    _double_t(t_invariant_power, 0., 0., t_invariant_power, 0., 0.) {
    auto [s1, s2] = split_sets_from_color_order(color_order);
    _set1 = s1;
    _set2 = s2;
    _use_single_chain = s1.empty() || s2.empty();
    _use_double_t = !_use_single_chain && ((s1.size() == 1) != (s2.size() == 1));
    std::size_t n_set_masses = (_use_single_chain || _use_double_t)
        ? 0u
        : (s1.size() >= 2 ? 1u : 0u) + (s2.size() >= 2 ? 1u : 0u);
    std::size_t n_central = _use_single_chain ? 0u : (_use_double_t ? 3u : 2u);
    std::size_t n_intermediate_masses = n_intermediate_masses_for_set_size(s1.size()) +
        n_intermediate_masses_for_set_size(s2.size());
    std::size_t n_walk = n_block_randoms_for_set_size(s1.size()) +
        n_block_randoms_for_set_size(s2.size());
    _random_dim = n_set_masses + n_intermediate_masses + n_central + n_walk;
}

Mapping::Result ColorOrderedMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto r = inputs.begin();
    auto next_random = [&]() { return *(r++); };
    ValueVec dets;

    // Phase 1a + Phase 2: set composite masses and the central block.
    // Three topologies:
    //   * single chain: one set empty -> no central block; the full final
    //     state (mass e_cm) is a single t-channel chain seeded off the beams.
    //   * DoubleT: exactly one set has size 1; defers the multi-particle
    //     side's mass until after the central block.
    //   * standard: both sets non-empty -> central 2->2 scattering.
    auto [pa, pb] = fb.com_p_in(e_cm);
    Value m_set1, m_set2;
    Value P_set1, P_set2;
    auto masses_of = [&](const std::vector<std::size_t>& idxs) {
        ValueVec v;
        v.reserve(idxs.size());
        for (auto i : idxs) {
            v.push_back(m_out.at(i));
        }
        return fb.sum(v);
    };
    if (_use_single_chain) {
        // No mass is sampled (the full system mass is fixed at e_cm) and no
        // central block runs. The empty set's composite momentum is the zero
        // four-vector, so the usual R_b = beam - P_other below reduces to the
        // bare opposite beam, exactly seeding a single t-channel chain.
        Value zero4 = fb.sub(pa, pa);
        Value P_full = fb.add(pa, pb);
        if (_set2.empty()) {
            m_set1 = e_cm;
            P_set1 = P_full;
            P_set2 = zero4;
        } else {
            m_set2 = e_cm;
            P_set2 = P_full;
            P_set1 = zero4;
        }
    } else if (_use_double_t) {
        // DoubleT: pa, pb -> (single, recoil). The single particle's mass
        // is m_single (fixed); the recoil mass must be >= mir_min (sum of
        // recoil-side outgoing masses).
        Value mass_sum_set1 = masses_of(_set1);
        Value mass_sum_set2 = masses_of(_set2);
        if (_set1.size() == 1) {
            m_set1 = m_out.at(_set1[0]);
        }
        if (_set2.size() == 1) {
            m_set2 = m_out.at(_set2[0]);
        }
        bool single_is_set1 = (_set1.size() == 1);
        Value m_single = single_is_set1 ? m_set1 : m_set2;
        Value mir_min = single_is_set1 ? mass_sum_set2 : mass_sum_set1;
        auto central = _double_t.build_forward(
            fb,
            {next_random(), next_random(), next_random()},
            {pa, pb, m_single, mir_min}
        );
        Value p_single = central.at(0);
        Value p_recoil = central.at(1);
        dets.push_back(central["det"]);
        if (single_is_set1) {
            P_set1 = p_single;
            P_set2 = p_recoil;
            // Derive multi-side m_set from the recoil momentum for Phase 1b.
            nested_vector2<double> factors_recoil{{1.0}};
            auto m2 =
                fb.unstack(
                      fb.invariants_from_momenta(fb.stack({p_recoil}), factors_recoil)
                )
                    .at(0);
            m_set2 = fb.sqrt(m2);
        } else {
            P_set2 = p_single;
            P_set1 = p_recoil;
            nested_vector2<double> factors_recoil{{1.0}};
            auto m2 =
                fb.unstack(
                      fb.invariants_from_momenta(fb.stack({p_recoil}), factors_recoil)
                )
                    .at(0);
            m_set1 = fb.sqrt(m2);
        }
    } else {
        Value mass_sum_set1 = masses_of(_set1);
        Value mass_sum_set2 = masses_of(_set2);
        if (_set1.size() >= 2) {
            auto s_min = fb.square(mass_sum_set1);
            auto s_max = fb.square(fb.sub(e_cm, mass_sum_set2));
            auto res =
                _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            m_set1 = fb.sqrt(res["invariant"]);
            dets.push_back(res["det"]);
        } else {
            m_set1 = m_out.at(_set1[0]);
        }
        if (_set2.size() >= 2) {
            auto s_min = fb.square(mass_sum_set2);
            auto s_max = fb.square(fb.sub(e_cm, m_set1));
            auto res =
                _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            m_set2 = fb.sqrt(res["invariant"]);
            dets.push_back(res["det"]);
        } else {
            m_set2 = m_out.at(_set2[0]);
        }
        auto central = _com_scattering.build_forward(
            fb, {next_random(), next_random(), m_set1, m_set2}, {pa, pb}
        );
        P_set1 = central.at(0);
        P_set2 = central.at(1);
        dets.push_back(central["det"]);
    }

    // R_b for each walk: the beam minus what the central block emitted on the other
    // side.
    Value R_b_for_set1 = fb.sub(pb, P_set2);
    Value R_b_for_set2 = fb.sub(pa, P_set1);

    // Phase 1b: pre-sample intermediate rest masses for each walk.
    // For each step j in 0..k-3 (last is the residual mass(s[k-1])):
    //   m_min = sum masses [set[j+1]..set[k-1]]
    //   m_max = m_rest_j - mass[set[j]],  m_rest_0 = m_set
    // Sampling from [m_min, m_set - sum_peeled] independently would
    // over-count by (k-2)! by ignoring the monotonic ordering.
    // Sampled after the central block: DoubleT only fixes m_set then.
    auto sample_intermediate_masses =
        [&](const std::vector<std::size_t>& s, Value m_set) -> ValueVec {
        std::size_t k = s.size();
        ValueVec res;
        if (k <= 2) {
            return res;
        }
        Value prev_mass = m_set;
        for (std::size_t j = 0; j < k - 2; ++j) {
            Value m_min = m_out.at(s[j + 1]);
            for (std::size_t i = j + 2; i < k; ++i) {
                m_min = fb.add(m_min, m_out.at(s[i]));
            }
            auto s_min = fb.square(m_min);
            auto s_max = fb.square(fb.sub(prev_mass, m_out.at(s[j])));
            auto r =
                _uniform_invariant.build_forward(fb, {next_random()}, {s_min, s_max});
            Value m_rest = fb.sqrt(r["invariant"]);
            res.push_back(m_rest);
            dets.push_back(r["det"]);
            prev_mass = m_rest;
        }
        return res;
    };

    // Empty set (single-chain case) contributes no intermediate masses; its
    // m_set is never assigned, so skip the call rather than pass it through.
    ValueVec rest_masses_set1, rest_masses_set2;
    if (!_set1.empty()) {
        rest_masses_set1 = sample_intermediate_masses(_set1, m_set1);
    }
    if (!_set2.empty()) {
        rest_masses_set2 = sample_intermediate_masses(_set2, m_set2);
    }

    // Phase 3: peel-off walks
    ValueVec p_out(_n_out);

    auto walk =
        [&](const std::vector<std::size_t>& s,
            Value P_set,
            Value R_b,
            const ValueVec& rest_masses) {
            std::size_t k = s.size();
            if (k == 1) {
                p_out[s[0]] = P_set;
                return;
            }
            Value R_a = fb.sub(P_set, R_b);
            Value im1;
            bool first = true;
            for (std::size_t j = 0; j < k - 1; ++j) {
                // New-rest mass: intermediate (rest_masses[j]) or final residual (j ==
                // k-2).
                Value m_rest = (j < k - 2) ? rest_masses[j] : m_out.at(s[k - 1]);
                Value m_peel = m_out.at(s[j]);
                // Block convention: pa-side carries the chain (mass m_rest), pb-side
                // carries the peeled particle (mass m_peel). R_a is the active leg
                // and gets decremented by peeled; R_b is constant.
                //
                // 2->3 kernel internally subtracts p_3 = im1 from pa+pb. R_a already
                // has im1 subtracted from our previous step, so we pass pb = R_a + im1
                // to recover p_12 = R_b + R_a inside the kernel.
                if (first) {
                    auto ks = _lab_scattering.build_forward(
                        fb, {next_random(), next_random(), m_rest, m_peel}, {R_b, R_a}
                    );
                    Value peeled = ks.at(1);
                    p_out[s[j]] = peeled;
                    R_a = fb.sub(R_a, peeled);
                    im1 = peeled;
                    dets.push_back(ks["det"]);
                    first = false;
                } else {
                    Value pb_for_block = fb.add(R_a, im1);
                    auto ks = _two_to_three.build_forward(
                        fb,
                        {next_random(), next_random(), next_random(), m_rest, m_peel},
                        {R_b, pb_for_block, im1}
                    );
                    Value peeled = ks.at(1);
                    p_out[s[j]] = peeled;
                    R_a = fb.sub(R_a, peeled);
                    im1 = peeled;
                    dets.push_back(ks["det"]);
                }
            }
            // Last particle = R_a + R_b (= what remains after all explicit peels).
            p_out[s[k - 1]] = fb.add(R_a, R_b);
        };

    if (!_set1.empty()) {
        walk(_set1, P_set1, R_b_for_set1, rest_masses_set1);
    }
    if (!_set2.empty()) {
        walk(_set2, P_set2, R_b_for_set2, rest_masses_set2);
    }

    // Assemble outputs: momentum0, momentum1 = beams; momentum_{2+i} = outgoing i.
    ValueVec p_ext;
    p_ext.reserve(_n_out + 2);
    p_ext.push_back(pa);
    p_ext.push_back(pb);
    for (std::size_t i = 0; i < _n_out; ++i) {
        p_ext.push_back(p_out[i]);
    }

    return {{output_types().keys(), p_ext}, fb.product(dets)};
}

Mapping::Result ColorOrderedMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    ValueVec random_out;
    ValueVec dets;

    Value pa = inputs.at(0);
    Value pb = inputs.at(1);
    auto p_outgoing = [&](std::size_t i) { return inputs.at(2 + i); };

    // Factor matrix: extract all needed invariant masses in one fb call.
    // Layout: m^2(P_set1), m^2(P_set2), then per-set rest-system masses.
    // idx_* below record where each band starts.
    nested_vector2<double> invariant_factors;
    auto n_inputs = _n_out + 2;
    auto factor_set = [&](const std::vector<std::size_t>& idxs) {
        std::vector<double> row(n_inputs, 0.0);
        for (auto idx : idxs) {
            row.at(idx + 2) = 1.0;
        }
        return row;
    };
    int idx_m2_set1 = -1, idx_m2_set2 = -1;
    int idx_rest_set1_start = -1, idx_rest_set2_start = -1;
    if (_set1.size() >= 2) {
        idx_m2_set1 = invariant_factors.size();
        invariant_factors.push_back(factor_set(_set1));
    }
    if (_set2.size() >= 2) {
        idx_m2_set2 = invariant_factors.size();
        invariant_factors.push_back(factor_set(_set2));
    }
    if (_set1.size() >= 3) {
        idx_rest_set1_start = invariant_factors.size();
        for (std::size_t j = 0; j < _set1.size() - 2; ++j) {
            std::vector<std::size_t> sub(_set1.begin() + j + 1, _set1.end());
            invariant_factors.push_back(factor_set(sub));
        }
    }
    if (_set2.size() >= 3) {
        idx_rest_set2_start = invariant_factors.size();
        for (std::size_t j = 0; j < _set2.size() - 2; ++j) {
            std::vector<std::size_t> sub(_set2.begin() + j + 1, _set2.end());
            invariant_factors.push_back(factor_set(sub));
        }
    }
    ValueVec invariants;
    if (!invariant_factors.empty()) {
        invariants = fb.unstack(
            fb.invariants_from_momenta(fb.stack(inputs.values()), invariant_factors)
        );
    }

    auto sum_momenta = [&](const std::vector<std::size_t>& s) -> Value {
        Value p = p_outgoing(s[0]);
        for (std::size_t k = 1; k < s.size(); ++k) {
            p = fb.add(p, p_outgoing(s[k]));
        }
        return p;
    };
    // An empty set (single-chain case) has zero composite momentum, so that
    // R_b = beam - P_other below reduces to the bare opposite beam.
    Value P_set1 = _set1.empty() ? fb.sub(pa, pa) : sum_momenta(_set1);
    Value P_set2 = _set2.empty() ? fb.sub(pa, pa) : sum_momenta(_set2);

    // Phase 1a inverse: recover set-mass randoms (none for DoubleT or single
    // chain). m_set values come from invariants_from_momenta for multi-particle
    // sides.
    Value m_set1, m_set2;
    auto masses_of = [&](const std::vector<std::size_t>& idxs) {
        ValueVec v;
        v.reserve(idxs.size());
        for (auto i : idxs) {
            v.push_back(m_out.at(i));
        }
        return fb.sum(v);
    };
    if (_use_single_chain) {
        // Full-system mass is fixed at e_cm; no set-mass random to recover.
        if (_set2.empty()) {
            m_set1 = e_cm;
        } else {
            m_set2 = e_cm;
        }
    } else if (_use_double_t) {
        if (_set1.size() == 1) {
            m_set1 = m_out.at(_set1[0]);
            m_set2 = fb.sqrt(invariants.at(idx_m2_set2));
        } else {
            m_set2 = m_out.at(_set2[0]);
            m_set1 = fb.sqrt(invariants.at(idx_m2_set1));
        }
    } else {
        Value mass_sum_set1 = masses_of(_set1);
        Value mass_sum_set2 = masses_of(_set2);
        if (_set1.size() >= 2) {
            auto s_min = fb.square(mass_sum_set1);
            auto s_max = fb.square(fb.sub(e_cm, mass_sum_set2));
            auto m2_set1 = invariants.at(idx_m2_set1);
            auto res = _uniform_invariant.build_inverse(fb, {m2_set1}, {s_min, s_max});
            random_out.push_back(res["random"]);
            dets.push_back(res["det"]);
            m_set1 = fb.sqrt(m2_set1);
        } else {
            m_set1 = m_out.at(_set1[0]);
        }
        if (_set2.size() >= 2) {
            auto s_min = fb.square(mass_sum_set2);
            auto s_max = fb.square(fb.sub(e_cm, m_set1));
            auto m2_set2 = invariants.at(idx_m2_set2);
            auto res = _uniform_invariant.build_inverse(fb, {m2_set2}, {s_min, s_max});
            random_out.push_back(res["random"]);
            dets.push_back(res["det"]);
            m_set2 = fb.sqrt(m2_set2);
        } else {
            m_set2 = m_out.at(_set2[0]);
        }
    }

    // Phase 2 inverse: central block. Comes before Phase 1b because the
    // forward emits central randoms first (so multi-side m_set is known by 1b).
    // Single chain has no central block and emits no central randoms.
    if (_use_single_chain) {
        // nothing to recover
    } else if (_use_double_t) {
        bool single_is_set1 = (_set1.size() == 1);
        Value p_single = single_is_set1 ? P_set1 : P_set2;
        Value p_recoil = single_is_set1 ? P_set2 : P_set1;
        Value m_single = single_is_set1 ? m_set1 : m_set2;
        Value mir_min = single_is_set1 ? masses_of(_set2) : masses_of(_set1);
        auto central = _double_t.build_inverse(
            fb, {p_single, p_recoil}, {pa, pb, m_single, mir_min}
        );
        random_out.push_back(central.at(0));
        random_out.push_back(central.at(1));
        random_out.push_back(central.at(2));
        dets.push_back(central["det"]);
    } else {
        auto central = _com_scattering.build_inverse(fb, {P_set1, P_set2}, {pa, pb});
        random_out.push_back(central.at(0));
        random_out.push_back(central.at(1));
        dets.push_back(central["det"]);
    }

    // Phase 1b inverse: recover intermediate rest-mass randoms
    auto recover_intermediate_masses =
        [&](const std::vector<std::size_t>& s, Value m_set, int idx_start) {
            std::size_t k = s.size();
            if (k <= 2) {
                return;
            }
            // Mirror the forward chaining: prev_mass starts at m_set, then becomes
            // the (sqrt of the) actual recovered m2 at each step.
            Value prev_mass = m_set;
            for (std::size_t j = 0; j < k - 2; ++j) {
                Value m_min = m_out.at(s[j + 1]);
                for (std::size_t i = j + 2; i < k; ++i) {
                    m_min = fb.add(m_min, m_out.at(s[i]));
                }
                auto s_min = fb.square(m_min);
                auto s_max = fb.square(fb.sub(prev_mass, m_out.at(s[j])));
                auto m2 = invariants.at(idx_start + j);
                auto res = _uniform_invariant.build_inverse(fb, {m2}, {s_min, s_max});
                random_out.push_back(res["random"]);
                dets.push_back(res["det"]);
                prev_mass = fb.sqrt(m2);
            }
        };

    if (!_set1.empty()) {
        recover_intermediate_masses(_set1, m_set1, idx_rest_set1_start);
    }
    if (!_set2.empty()) {
        recover_intermediate_masses(_set2, m_set2, idx_rest_set2_start);
    }

    Value R_b_for_set1 = fb.sub(pb, P_set2);
    Value R_b_for_set2 = fb.sub(pa, P_set1);

    // Phase 3 inverse: peel-off walks
    auto walk_inverse = [&](const std::vector<std::size_t>& s, Value P_set, Value R_b) {
        std::size_t k = s.size();
        if (k == 1) {
            return;
        }
        Value R_a = fb.sub(P_set, R_b);
        Value im1;
        bool first = true;
        for (std::size_t j = 0; j < k - 1; ++j) {
            Value peeled = p_outgoing(s[j]);
            // p1_out is the chain carrier (mass m_rest); by conservation
            // p1_out = R_a + R_b - peeled.
            Value p1_out = fb.sub(fb.add(R_a, R_b), peeled);
            if (first) {
                auto rs =
                    _lab_scattering.build_inverse(fb, {p1_out, peeled}, {R_b, R_a});
                random_out.push_back(rs.at(0));
                random_out.push_back(rs.at(1));
                dets.push_back(rs["det"]);
                first = false;
            } else {
                // Mirror the forward's pb restoration: the 2->3 kernel
                // internally subtracts p_3 = im1, so we pass pb = R_a + im1
                // to get p_12 = R_b + R_a (the remaining-to-produce system).
                Value pb_for_block = fb.add(R_a, im1);
                auto rs = _two_to_three.build_inverse(
                    fb, {p1_out, peeled}, {R_b, pb_for_block, im1}
                );
                random_out.push_back(rs.at(0));
                random_out.push_back(rs.at(1));
                random_out.push_back(rs.at(2));
                dets.push_back(rs["det"]);
            }
            R_a = fb.sub(R_a, peeled);
            im1 = peeled;
        }
    };

    if (!_set1.empty()) {
        walk_inverse(_set1, P_set1, R_b_for_set1);
    }
    if (!_set2.empty()) {
        walk_inverse(_set2, P_set2, R_b_for_set2);
    }

    return {{input_types().keys(), random_out}, fb.product(dets)};
}
