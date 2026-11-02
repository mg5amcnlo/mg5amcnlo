#include "madspace/phasespace/phasespace.hpp"
#include "madspace/constants.hpp"
#include "madspace/util.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

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

// Masses of the external momenta in the order they are handed to boost_beam,
// i.e. after the channel permutation. A position whose mass differs between
// permutations gets -1, and boost_beam reads that mass off the momentum.
std::vector<double>
lab_masses(const Topology& topology, const nested_vector2<me_int_t>& permutations) {
    std::vector<double> masses = topology.incoming_masses();
    const auto& out = topology.outgoing_masses();
    masses.insert(masses.end(), out.begin(), out.end());
    if (permutations.empty()) {
        return masses;
    }
    std::vector<double> result(masses.size());
    for (std::size_t i = 0; i < masses.size(); ++i) {
        result.at(i) = masses.at(permutations.at(0).at(i));
        for (const auto& perm : permutations) {
            if (masses.at(perm.at(i)) != result.at(i)) {
                result.at(i) = -1.;
            }
        }
    }
    return result;
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

namespace {
// Number of discrete two-solution choices for the t-channel (opt-in r_disc):
// non-zero only for color_ordered with at least one 2->3 step.
std::size_t ps_discrete_dim(
    const Topology& topology,
    PhaseSpaceMapping::TChannelMode mode,
    const std::optional<ColorOrderedMapping::Options>& color_ordered
) {
    if (mode == PhaseSpaceMapping::color_ordered && color_ordered &&
        topology.t_propagator_count() >= 1) {
        return ColorOrderedMapping(*color_ordered).discrete_dim();
    }
    return 0;
}

// The floor every pair of final-state particles puts on its invariant mass:
// the pair mass cuts m_inv_min, raised wherever the pt and delta R cuts imply
// more. All tables are indexed by outgoing position. For two massless
// particles
//     m^2 = 2 pt_i pt_j (cosh(d_eta) - cos(d_phi)),
// and on d_eta^2 + d_phi^2 >= R^2 the bracket is smallest at d_eta = 0,
// d_phi = R (moving along the circle towards d_phi = 0 costs more in cosh than
// it gains in cos, since sin b <= b), so
//     m >= sqrt(2 pt_i,min pt_j,min (1 - cos R)).
// That is a consequence of the cuts, not an approximation to them, so it can
// bound the sampling without changing the integral. It needs d_phi = R to be
// reachable (R <= pi), and it needs massless legs: with a mass, delta R is
// measured in pseudorapidity while the invariant depends on the rapidity, and
// the bound no longer follows.
std::vector<std::vector<double>> pair_mass_floors(
    std::vector<std::vector<double>> floors,
    const std::vector<double>& pt_min,
    const std::vector<std::vector<double>>& dr_min,
    const std::vector<double>& masses
) {
    for (std::size_t i = 0; i < floors.size(); ++i) {
        if (masses.at(i) != 0.) {
            continue;
        }
        for (std::size_t j = i + 1; j < floors.size(); ++j) {
            double r = dr_min.at(i).at(j);
            if (masses.at(j) != 0. || r <= 0. || r > PI) {
                continue;
            }
            double floor =
                std::sqrt(2. * pt_min.at(i) * pt_min.at(j) * (1. - std::cos(r)));
            if (floor > floors.at(i).at(j)) {
                floors.at(i).at(j) = floor;
                floors.at(j).at(i) = floor;
            }
        }
    }
    return floors;
}

// The smallest sum_i sqrt(m_i^2 + pt_i^2) the final state can have when every
// pt_i >= pt_min_i. The beams carry no transverse momentum, so the outgoing
// transverse momenta add up to zero, and that needs the largest to be no more
// than the sum of the others. When the cuts alone leave that unsatisfied (one
// hard cut, little else cut), the others have to make up the deficit, and the
// cheapest way is to spread it over the massive ones, at equal pt_i / m_i,
// whose transverse mass grows more slowly than pt; with no massive particle
// to take it, it costs its full size.
double min_transverse_mass_sum(
    const std::vector<double>& masses, const std::vector<double>& pt_min
) {
    double sum = 0.;
    auto transverse_mass = [](double mass, double pt) {
        return std::sqrt(mass * mass + pt * pt);
    };
    for (auto [mass, pt] : zip(masses, pt_min)) {
        sum += transverse_mass(mass, pt);
    }
    if (pt_min.empty()) {
        return sum;
    }
    std::size_t hardest =
        std::distance(pt_min.begin(), std::max_element(pt_min.begin(), pt_min.end()));
    double deficit =
        2. * pt_min.at(hardest) - std::accumulate(pt_min.begin(), pt_min.end(), 0.);
    if (deficit <= 0.) {
        return sum;
    }
    // raise every massive particle other than the hardest to pt = t m (where
    // that is more than its cut), with t such that the raises add up to the
    // deficit
    std::vector<std::size_t> massive;
    double mass_sum = 0., massive_pt_sum = 0., t_high = 0.;
    for (std::size_t i = 0; i < masses.size(); ++i) {
        if (i != hardest && masses.at(i) > 0.) {
            massive.push_back(i);
            mass_sum += masses.at(i);
            massive_pt_sum += pt_min.at(i);
            t_high = std::max(t_high, pt_min.at(i) / masses.at(i));
        }
    }
    if (massive.empty()) {
        return sum + deficit;
    }
    auto raised = [&](double t) {
        double total = 0.;
        for (std::size_t i : massive) {
            total += std::max(0., t * masses.at(i) - pt_min.at(i));
        }
        return total;
    };
    double t_low = 0.;
    t_high = std::max(t_high, (deficit + massive_pt_sum) / mass_sum);
    for (int iteration = 0; iteration < 200; ++iteration) {
        double t = 0.5 * (t_low + t_high);
        (raised(t) < deficit ? t_low : t_high) = t;
    }
    for (std::size_t i : massive) {
        double pt = std::max(pt_min.at(i), t_low * masses.at(i));
        sum += transverse_mass(masses.at(i), pt) -
            transverse_mass(masses.at(i), pt_min.at(i));
    }
    return sum;
}
} // namespace

PhaseSpaceMapping::PhaseSpaceMapping(
    const Topology& topology,
    double cm_energy,
    bool leptonic,
    double invariant_power,
    TChannelMode t_channel_mode,
    const std::optional<Cuts>& cuts,
    const std::vector<std::vector<std::size_t>>& permutations,
    const std::optional<ColorOrderedMapping::Options>& color_ordered,
    double beam_rapidity,
    bool mirror_beams,
    std::optional<double> s_hat_power
) :
    Mapping(
        "PhaseSpaceMapping",
        [&] {
            NamedVector<Type> in{
                {"random",
                 batch_float_array(
                     PhaseSpaceMapping::random_dim_for(topology, leptonic)
                 )}
            };
            // Opt-in discrete channel: only declared when the t-channel strategy
            // actually has discrete two-solution choices (color_ordered).
            std::size_t nd = ps_discrete_dim(topology, t_channel_mode, color_ordered);
            if (nd > 0) {
                in.push_back(
                    "discrete",
                    Type{DataType::dt_int32, batch_size, {static_cast<int>(nd)}}
                );
            }
            return in;
        }(),
        {{"momenta",
          batch_four_vec_array(
              topology.outgoing_masses().size() + topology.incoming_masses().size()
          )},
         {"x1", batch_float},
         {"x2", batch_float}},
        [&] {
            NamedVector<Type> conditions;
            if (permutations.size() > 1) {
                conditions.push_back("permutation_index", batch_int);
            }
            if (mirror_beams) {
                conditions.push_back("mirror_index", batch_int);
            }
            return conditions;
        }()
    ),
    _topology(topology),
    _cuts(cuts.value_or(
        Cuts(topology.outgoing_masses().size() + topology.incoming_masses().size())
    )),
    _pi_factors(
        std::pow(2 * PI, 4 - 3 * static_cast<int>(topology.outgoing_masses().size()))
    ),
    _sqrt_s_lab(cm_energy),
    _beam_rapidity(beam_rapidity),
    _mirror_beams(mirror_beams),
    _leptonic(leptonic),
    // A decay has no beams to sample momentum fractions for: the root
    // virtuality is fixed at the decaying particle's mass (passed as
    // cm_energy) and there is no boost into a lab frame.
    _map_luminosity(
        !leptonic && !topology.is_decay() &&
        (_topology.t_propagator_count() == 0 ||
         t_channel_mode != PhaseSpaceMapping::chili)
    ),
    _two_to_one(topology.outgoing_masses().size() == 1),
    _t_mapping(std::monostate{}) {
    // the chain order and settings belong to the color-ordered mode only
    if ((t_channel_mode == PhaseSpaceMapping::color_ordered) !=
        color_ordered.has_value()) {
        throw std::invalid_argument(
            t_channel_mode == PhaseSpaceMapping::color_ordered
                ? "PhaseSpaceMapping: the color_ordered mode needs color_ordered "
                  "options (the chain order)"
                : "PhaseSpaceMapping: color_ordered options are only used by the "
                  "color_ordered mode"
        );
    }
    if ((_beam_rapidity != 0. || _mirror_beams) && _topology.is_decay()) {
        throw std::invalid_argument("a decay has no beams to give a rapidity");
    }
    if (_two_to_one && _topology.outgoing_masses().at(0) <= 0.) {
        // s_hat = m^2 = 0 is outside the luminosity (tau = 0)
        throw std::invalid_argument(
            "a 2 -> 1 process needs a massive final-state particle"
        );
    }
    bool has_t_channel = _topology.t_propagator_count() > 0;
    struct DecayInfo {
        double m_min, pt_min, eta_max;
        std::optional<Invariant> invariant;
    };
    std::vector<DecayInfo> decay_info(_topology.decays().size());

    // The cuts are indexed by the position of a particle in the event the
    // mapping returns, and that event is the topology's own momenta reordered
    // by the channel permutation (output[i] = topology[perm[i]], see
    // permute_momenta). The same mapping serves every permutation, so a bound
    // is handed to the sampler only in a form that holds for all of them:
    // the weakest of the per-permutation bounds. Without permutations the
    // topology order is the event order.
    //
    // Cuts indexes its per-particle tables by outgoing position, counting two
    // incoming particles. A decay topology has one, so the tables cannot be
    // read against it at all - and a decay has no cuts to apply anyway.
    constexpr std::size_t no_index = static_cast<std::size_t>(-1);
    const std::size_t n_in = _topology.incoming_masses().size();
    const std::size_t n_out = _topology.outgoing_masses().size();
    const bool read_cuts = n_in == 2;
    // event_position[p][t]: where permutation p puts the topology's outgoing
    // particle t, as an outgoing position of the event (no_index if p sends
    // it to an incoming slot)
    std::vector<std::vector<std::size_t>> event_position;
    if (permutations.empty()) {
        event_position.emplace_back(n_out);
        std::iota(event_position.back().begin(), event_position.back().end(), 0);
    }
    for (const auto& perm : permutations) {
        auto& position = event_position.emplace_back(n_out, no_index);
        for (std::size_t i = n_in; i < perm.size(); ++i) {
            if (perm.at(i) >= n_in && perm.at(i) - n_in < n_out) {
                position.at(perm.at(i) - n_in) = i - n_in;
            }
        }
    }
    std::vector<double> cut_pt_min, cut_eta_max;
    std::vector<std::vector<double>> cut_m_inv_min, cut_dr_min;
    std::vector<Cuts::PairMassAny> cut_pair_mass_any;
    if (read_cuts) {
        cut_pt_min = _cuts.pt_min();
        cut_eta_max = _cuts.eta_max();
        cut_m_inv_min = _cuts.m_inv_min();
        cut_dr_min = _cuts.dr_min();
        cut_pair_mass_any = _cuts.pair_mass_any_min();
    }
    // per-particle and pairwise tables in topology order, valid for every
    // permutation (an index the cut tables do not cover reads as no cut)
    auto at_or = [](const std::vector<double>& v, std::size_t i, double none) {
        return i < v.size() ? v.at(i) : none;
    };
    auto pair_at =
        [](const std::vector<std::vector<double>>& m, std::size_t i, std::size_t j) {
            return i < m.size() && j < m.at(i).size() ? m.at(i).at(j) : 0.;
        };
    constexpr double inf = std::numeric_limits<double>::infinity();
    std::vector<double> topo_pt_min(n_out, inf), topo_eta_max(n_out, 0.);
    std::vector<std::vector<double>> topo_m_inv_min(
        n_out, std::vector<double>(n_out, inf)
    );
    std::vector<std::vector<double>> topo_dr_min(
        n_out, std::vector<double>(n_out, inf)
    );
    for (const auto& position : event_position) {
        for (std::size_t t = 0; t < n_out; ++t) {
            std::size_t i = position.at(t);
            topo_pt_min.at(t) = std::min(topo_pt_min.at(t), at_or(cut_pt_min, i, 0.));
            topo_eta_max.at(t) =
                std::max(topo_eta_max.at(t), at_or(cut_eta_max, i, inf));
            for (std::size_t u = 0; u < n_out; ++u) {
                std::size_t j = position.at(u);
                topo_m_inv_min.at(t).at(u) =
                    std::min(topo_m_inv_min.at(t).at(u), pair_at(cut_m_inv_min, i, j));
                topo_dr_min.at(t).at(u) =
                    std::min(topo_dr_min.at(t).at(u), pair_at(cut_dr_min, i, j));
            }
        }
    }
    // The pt and delta R cuts of two massless particles bound their invariant
    // mass too (pair_mass_floors); from here on that floor counts as a cut.
    topo_m_inv_min = pair_mass_floors(
        topo_m_inv_min, topo_pt_min, topo_dr_min, _topology.outgoing_masses()
    );
    for (auto [index, m_min, pt_min, eta_max] :
         zip(_topology.outgoing_indices(),
             _topology.outgoing_masses(),
             topo_pt_min,
             topo_eta_max)) {
        decay_info.at(index) = {m_min, pt_min, eta_max, std::nullopt};
    }

    // A cut on the invariant mass of a pair is also a statement about the
    // phase space, not only about which events to keep afterwards. Wherever
    // both members of the pair come out of the same propagator, the cut is a
    // floor on that propagator's invariant and can be handed straight to the
    // sampler, which is the difference between generating the region the cut
    // allows and throwing away nearly everything generated.
    constexpr std::size_t no_leaf = static_cast<std::size_t>(-1);
    std::vector<std::vector<std::size_t>> node_leaves(_topology.decays().size());
    {
        std::vector<std::size_t> decay_to_outgoing(_topology.decays().size(), no_leaf);
        for (std::size_t out_pos = 0; out_pos < _topology.outgoing_indices().size();
             ++out_pos) {
            decay_to_outgoing.at(_topology.outgoing_indices().at(out_pos)) = out_pos;
        }
        // children always sit at a higher index than their parent, so one
        // backwards pass builds every leaf set
        for (std::size_t d = _topology.decays().size(); d-- > 0;) {
            const auto& decay = _topology.decays().at(d);
            auto& leaves = node_leaves.at(d);
            if (decay.child_indices.empty()) {
                if (decay_to_outgoing.at(d) != no_leaf) {
                    leaves.push_back(decay_to_outgoing.at(d));
                }
                continue;
            }
            for (std::size_t child : decay.child_indices) {
                const auto& child_leaves = node_leaves.at(child);
                leaves.insert(leaves.end(), child_leaves.begin(), child_leaves.end());
            }
            std::sort(leaves.begin(), leaves.end());
        }
    }
    // A bound on the absolute rapidity of every node whose leaves all have a
    // pseudorapidity cut, and negative otherwise. A leaf has |y| <= |eta|, and
    // the rapidity of a sum of momenta, tanh(y) = sum_i m_T,i sinh(y_i) /
    // sum_i m_T,i cosh(y_i), is a weighted mean of the tanh(y_i), so it never
    // exceeds the largest |y_i|. This holds for massive particles too.
    std::vector<double> node_y_max(node_leaves.size(), -1.);
    for (std::size_t d = 0; d < node_leaves.size(); ++d) {
        if (node_leaves.at(d).empty()) {
            continue;
        }
        double y_max = 0.;
        for (std::size_t leaf : node_leaves.at(d)) {
            double eta_max = topo_eta_max.at(leaf);
            if (!std::isfinite(eta_max)) {
                y_max = -1.;
                break;
            }
            y_max = std::max(y_max, eta_max);
        }
        node_y_max.at(d) = y_max;
    }
    // The rapidity of the partonic system, log(x1 / x2) / 2, is the root's. The
    // cuts hold in the lab frame and x1, x2 live in the beams' centre-of-mass
    // frame, which is boosted by beam_rapidity: |y_cm| <= |y_lab| + |y0|. (The
    // bounds on the children, in the partonic frame, are unaffected.)
    if (_map_luminosity && read_cuts) {
        double y_root = node_y_max.at(0);
        _y_max_lab = y_root >= 0. ? y_root + std::abs(beam_rapidity) : y_root;
    }

    // The floor a set of final-state particles (topology outgoing positions)
    // inherits from the pair mass cuts. A pair contributes at least its cut and
    // every other particle at least its mass, and for future-pointing momenta
    // the invariant mass of a sum is at least the sum of the invariant masses:
    //   * a cut every pair must pass (topo_m_inv_min) gives
    //         m(leaves) >= cut(i, j) + sum_{k != i, j} m_k
    //     for each cut pair (i, j) among the leaves;
    //   * a CutMode::any cut, which only one of its pairs has to pass, gives
    //     the smallest of those bounds over its pairs - provided every one of
    //     them lies among the leaves, or the passing pair may lie elsewhere.
    //   * every pair together: expanding (sum p_k)^2 with
    //     2 p_i.p_j = m_ij^2 - m_i^2 - m_j^2 gives the exact identity
    //         m(leaves)^2 = sum_{i<j} m_ij^2 - (n - 2) sum_k m_k^2,
    //     and m_ij is at least its cut and at least m_i + m_j. With several
    //     cut pairs among the leaves this beats any single pair: four leptons
    //     with every pair above 50 GeV are at least sqrt(6) * 50 GeV heavy.
    // Each holds, so the largest is the floor. The any-mode cuts are read per
    // permutation and the weakest result is kept, like the tables above.
    const auto& out_masses = _topology.outgoing_masses();
    auto pair_floor = [&](const std::vector<std::size_t>& leaves) {
        double leaf_mass_sum = 0.;
        for (std::size_t leaf : leaves) {
            leaf_mass_sum += out_masses.at(leaf);
        }
        auto bound = [&](std::size_t t, std::size_t u, double cut) {
            return cut + leaf_mass_sum - out_masses.at(t) - out_masses.at(u);
        };
        double floor = 0.;
        double pair_mass2_sum = 0., leaf_mass2_sum = 0.;
        bool any_cut = false;
        for (std::size_t leaf : leaves) {
            leaf_mass2_sum += out_masses.at(leaf) * out_masses.at(leaf);
        }
        for (std::size_t a = 0; a < leaves.size(); ++a) {
            for (std::size_t b = a + 1; b < leaves.size(); ++b) {
                std::size_t t = leaves.at(a), u = leaves.at(b);
                double cut = topo_m_inv_min.at(t).at(u);
                double pair_min = out_masses.at(t) + out_masses.at(u);
                if (cut > 0.) {
                    any_cut = true;
                    floor = std::max(floor, bound(t, u, cut));
                    pair_min = std::max(pair_min, cut);
                }
                pair_mass2_sum += pair_min * pair_min;
            }
        }
        if (any_cut && leaves.size() > 2) {
            double n_other = static_cast<double>(leaves.size()) - 2.;
            double mass2 = pair_mass2_sum - n_other * leaf_mass2_sum;
            if (mass2 > 0.) {
                floor = std::max(floor, std::sqrt(mass2));
            }
        }
        if (cut_pair_mass_any.empty()) {
            return floor;
        }
        double any_floor = inf;
        for (const auto& position : event_position) {
            // topology position of every event outgoing position
            std::vector<std::size_t> topo_of(n_out, no_index);
            for (std::size_t t = 0; t < n_out; ++t) {
                if (position.at(t) < n_out) {
                    topo_of.at(position.at(t)) = t;
                }
            }
            double perm_floor = 0.;
            for (const auto& item : cut_pair_mass_any) {
                double item_floor = inf;
                for (auto [i, j] : item.pairs) {
                    std::size_t t = i < n_out ? topo_of.at(i) : no_index;
                    std::size_t u = j < n_out ? topo_of.at(j) : no_index;
                    if (t == no_index || u == no_index ||
                        !std::binary_search(leaves.begin(), leaves.end(), t) ||
                        !std::binary_search(leaves.begin(), leaves.end(), u)) {
                        item_floor = 0.;
                        break;
                    }
                    item_floor = std::min(item_floor, bound(t, u, item.min));
                }
                perm_floor = std::max(perm_floor, item_floor);
            }
            any_floor = std::min(any_floor, perm_floor);
        }
        return std::max(floor, any_floor);
    };
    // e_min is the propagator's own floor on its invariant mass, and
    // update_mass_min_max already carries it into every s_min the sampler uses
    // and into what the parents subtract, so raising it here is all that is
    // needed for the cut to shape the integration. A floor at or above the top
    // of an on-shell window (e_max) leaves the propagator nothing to sample:
    // the cuts reject the whole channel. It is then not handed on as an
    // inverted range but reported through empty(), so the channel can be
    // dropped instead of failing to find a single passing point.
    for (std::size_t d = 1; d < node_leaves.size(); ++d) {
        const auto& decay = _topology.decays().at(d);
        if (decay.child_indices.empty()) {
            continue;
        }
        double floor = pair_floor(node_leaves.at(d));
        if (floor <= 0.) {
            continue;
        }
        if (decay.e_max > 0. && floor >= decay.e_max) {
            _empty = true;
            continue;
        }
        _topology.raise_decay_e_min(d, floor);
    }

    // The same thing one level up: a floor on the total invariant mass of the
    // final state is a floor on the root propagator, which is the s-hat the
    // luminosity mapping samples. Handing it over is the difference between
    // sampling the region the cut allows and sampling everything and throwing
    // nearly all of it away; the Invariant's Jacobian follows the range it is
    // given, so the integral is unchanged and only the efficiency moves.
    //
    // Two sources of such a floor: a cut on sqrt(s_hat) itself, and the pair
    // cuts, every one of which bounds the total exactly as it bounds a
    // propagator above.
    double sqrt_s_hat_min =
        std::max(read_cuts ? _cuts.sqrt_s_min() : 0., pair_floor(node_leaves.at(0)));
    // Only the luminosity mapping samples the root virtuality, and the root's
    // e_min is read nowhere else: build_forward_impl gives the root its s_min
    // only when it samples it, and the other propagators bound themselves by
    // the root's mass, never by its floor. A leptonic collision has s_hat fixed
    // at s_lab and chili reconstructs it from the momenta it has already
    // generated, so in neither case is there a range to narrow -- the cut
    // stays a filter there. A floor at or above the beam energy leaves nothing
    // to sample at all, and is left to the filter too rather than handed on as
    // an empty range.
    if (_map_luminosity && sqrt_s_hat_min > 0. && sqrt_s_hat_min < _sqrt_s_lab) {
        _topology.raise_decay_e_min(0, sqrt_s_hat_min);
    }
    // One more floor on s_hat, from the pt cuts: in the partonic
    // centre-of-mass frame sqrt(s_hat) is the sum of the outgoing energies,
    // each at least the transverse mass, and the transverse momenta are the
    // same there as in the lab, so
    //     sqrt(s_hat) >= sum_i sqrt(m_i^2 + pt_i^2),
    // minimised over pt_i >= pt_i,min with the pt_i adding up to zero
    // (min_transverse_mass_sum). Without a pt cut this is just the sum of the
    // masses, which the sampler already respects.
    if (_map_luminosity && read_cuts) {
        bool has_pt_cut = std::any_of(
            topo_pt_min.begin(), topo_pt_min.end(), [](double pt) { return pt > 0.; }
        );
        double transverse_mass_sum =
            min_transverse_mass_sum(_topology.outgoing_masses(), topo_pt_min);
        if (has_pt_cut && transverse_mass_sum < _sqrt_s_lab) {
            _topology.raise_decay_e_min(0, transverse_mass_sum);
        }
    }
    for (auto [decay, info] :
         zip(std::views::reverse(_topology.decays()),
             std::views::reverse(decay_info))) {
        if (decay.child_indices.size() == 0) {
            continue;
        }

        bool is_com_decay = decay.index == 0;
        if (decay.index != 0 || !has_t_channel) {
            if (decay.child_indices.size() == 2 && is_com_decay && read_cuts) {
                // The root decays in the partonic centre-of-mass frame, which
                // is the lab up to a boost along the beam, so the pt cut of a
                // final-state child and the rapidity bound of any child
                // restrict its polar angle (TwoBodyDecay). A composite child
                // has no pt bound of its own.
                double pt_min = 0.;
                for (std::size_t child_index : decay.child_indices) {
                    pt_min = std::max(pt_min, decay_info.at(child_index).pt_min);
                }
                _s_decays.push_back(TwoBodyDecay(
                    true,
                    pt_min,
                    node_y_max.at(decay.child_indices.at(0)),
                    node_y_max.at(decay.child_indices.at(1))
                ));
            } else if (decay.child_indices.size() == 2) {
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
            // The root is s_hat, which the PDF convolution samples. A
            // color-ordered chain samples it as AmpliCol does, with
            // 1/s_hat^1.2, when masses or cuts bound it from below; without a
            // bound that density piles the points up at the regulator of
            // Invariant (and AmpliCol falls back to a flat one).
            double s_hat_default =
                t_channel_mode == PhaseSpaceMapping::color_ordered && info.m_min > 0.
                ? 1.2
                : invariant_power;
            double power =
                is_com_decay ? s_hat_power.value_or(s_hat_default) : invariant_power;
            info.invariant = Invariant(power, mass, width, decay.flat_window);
        }
    }
    for (std::size_t index : _topology.decay_integration_order()) {
        auto& invariant = decay_info.at(index).invariant;
        if (invariant) {
            _s_invariants.push_back(invariant.value());
        }
    }

    if (has_t_channel) {
        // Per-child pt_min (and eta_max), ordered to match the mass conditions
        // handed to the t-channel mapping (leaf children carry their pt cut;
        // composite children were reset to 0 above).
        std::vector<double> eta_max, pt_min, y_max;
        for (std::size_t index : topology.decays().at(0).child_indices) {
            auto& info = decay_info.at(index);
            // eta cuts hold in the lab frame, the t-channel is generated in the
            // beams' centre-of-mass frame: |y_cm| <= |eta_lab| + |beam_rapidity|
            eta_max.push_back(info.eta_max + std::abs(beam_rapidity));
            pt_min.push_back(info.pt_min);
            // rapidity bound of the child, composites included (node_y_max)
            y_max.push_back(node_y_max.at(index));
        }
        if (t_channel_mode == PhaseSpaceMapping::chili) {
            // |y| <= |eta|, so we can pass y_max = eta_max
            _t_mapping =
                ChiliMapping(_topology.t_propagator_count() + 1, eta_max, pt_min);
        } else if (t_channel_mode == PhaseSpaceMapping::color_ordered) {
            // Reorder the per-pair cut matrices (indexed by topology outgoing
            // position) into the child order in which masses/pt are handed to the
            // chain, mirroring the pt_min reordering above. Composite (non-leaf)
            // children carry no pairwise cut.
            const auto& out_idx = topology.outgoing_indices();
            const auto& child_indices = topology.decays().at(0).child_indices;
            std::vector<std::size_t> child_to_out(
                child_indices.size(), std::numeric_limits<std::size_t>::max()
            );
            for (std::size_t a = 0; a < child_indices.size(); ++a) {
                auto it =
                    std::find(out_idx.begin(), out_idx.end(), child_indices.at(a));
                if (it != out_idx.end()) {
                    child_to_out.at(a) = std::distance(out_idx.begin(), it);
                }
            }
            // already in topology order and valid for every permutation
            const auto& m_inv_full = topo_m_inv_min;
            const auto& dr_full = topo_dr_min;
            std::size_t nc = child_to_out.size();
            std::vector<std::vector<double>> m_inv_co(nc, std::vector<double>(nc, 0.));
            std::vector<std::vector<double>> dr_co(nc, std::vector<double>(nc, 0.));
            for (std::size_t a = 0; a < nc; ++a) {
                if (child_to_out.at(a) >= m_inv_full.size()) {
                    continue;
                }
                for (std::size_t b = 0; b < nc; ++b) {
                    if (child_to_out.at(b) >= m_inv_full.size()) {
                        continue;
                    }
                    m_inv_co.at(a).at(b) =
                        m_inv_full.at(child_to_out.at(a)).at(child_to_out.at(b));
                    // Delta R uses pseudorapidities, which a longitudinal boost
                    // does not shift uniformly for massive particles: a lab
                    // cut is no bound in the beams' frame. The cut itself is
                    // still applied to the lab momenta.
                    if (beam_rapidity == 0.) {
                        dr_co.at(a).at(b) =
                            dr_full.at(child_to_out.at(a)).at(child_to_out.at(b));
                    }
                }
            }
            // The blocks of the chain that scatter the two beams (central 2->2,
            // double-t, first peel of a single chain) take the rapidity bounds.
            // An unset chain power is 1 with cuts, as in AmpliCol, and
            // invariant_power without: then the ranges of the momentum transfers
            // and s23 invariants reach zero, where 1/|x| piles the points up at
            // the regulator of Invariant (AmpliCol falls back to a flat density).
            auto options = *color_ordered;
            auto positive = [](double x) { return x > 0.; };
            bool chain_cuts = std::any_of(pt_min.begin(), pt_min.end(), positive);
            for (std::size_t a = 0; a < nc; ++a) {
                chain_cuts |=
                    std::any_of(
                        m_inv_co.at(a).begin(), m_inv_co.at(a).end(), positive
                    ) ||
                    std::any_of(dr_co.at(a).begin(), dr_co.at(a).end(), positive);
            }
            if (!options.chain_invariant_power) {
                options.chain_invariant_power = chain_cuts ? 1. : invariant_power;
            }
            _t_mapping = ColorOrderedMapping(options, pt_min, m_inv_co, dr_co, y_max);
        } else if (t_channel_mode == PhaseSpaceMapping::propagator ||
                   topology.t_propagator_count() < 2) {
            // The first scattering of the chain is between the two beams in
            // the partonic centre-of-mass frame, so the rapidity bounds of the
            // particle it peels and of the recoil narrow its |t| range.
            _t_mapping = TPropagatorMapping(
                _topology.t_integration_order(), invariant_power, pt_min, y_max
            );
        } else if (t_channel_mode == PhaseSpaceMapping::rambo) {
            // TODO: add massless special case
            _t_mapping = FastRamboMapping(_topology.t_propagator_count() + 1, false);
        }
    }

    // Random-number budget: identical computation to the declared input shape.
    _n_discrete = ps_discrete_dim(_topology, t_channel_mode, color_ordered);

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
    const std::optional<Cuts>& cuts,
    const std::optional<ColorOrderedMapping::Options>& color_ordered,
    double beam_rapidity,
    bool mirror_beams,
    std::optional<double> s_hat_power
) :
    PhaseSpaceMapping(
        Topology([&] {
            if (external_masses.size() < 3) {
                throw std::invalid_argument("The number of masses must be at least 3");
            }
            if (external_masses.size() == 3) {
                return Diagram(
                    {external_masses.at(0), external_masses.at(1)},
                    {external_masses.at(2)},
                    {},
                    {{
                        {Diagram::incoming, 0},
                        {Diagram::incoming, 1},
                        {Diagram::outgoing, 0},
                    }}
                );
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
        cuts,
        {},
        color_ordered,
        beam_rapidity,
        mirror_beams,
        s_hat_power
    ) {}

Mapping::Result PhaseSpaceMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto random_numbers = fb.unstack(inputs.at(0));
    auto r = random_numbers.begin();
    auto next_random = [&]() { return *(r++); };
    // Opt-in discrete channel: present as inputs.at(1) only when the t-channel
    // strategy declared discrete choices (color_ordered). These are passed
    // through to the t-channel mapping after its continuous randoms.
    ValueVec discrete_numbers;
    if (inputs.size() > 1) {
        discrete_numbers = fb.unstack(inputs.at(1));
    }
    auto d = discrete_numbers.begin();
    auto next_discrete = [&]() { return *(d++); };

    ValueVec dets{_pi_factors};
    Value x1 = 1.0, x2 = 1.0;

    // permute, boost into the lab frame and apply the cuts
    auto finish = [&](Value p_ext_stack) -> Result {
        if (_permutations.size() > 1) {
            p_ext_stack = fb.permute_momenta(
                p_ext_stack, _permutations, conditions.at("permutation_index")
            );
        } else if (_permutations.size() == 1 &&
                   !std::is_sorted(
                       _permutations.at(0).begin(), _permutations.at(0).end()
                   )) {
            p_ext_stack = fb.permute_momenta(
                p_ext_stack, _permutations, static_cast<me_int_t>(0)
            );
        }
        auto p_ext_lab = to_lab(fb, p_ext_stack, x1, x2, conditions);
        dets.push_back(_cuts.build_function(fb, {p_ext_lab}).at(0));
        auto ps_weight = fb.cut_unphysical(fb.product(dets), p_ext_lab, x1, x2);
        // Without sampled momentum fractions (leptonic, a decay) x1 = x2 = 1
        // stay constants above. They are returned with the batch dimension of
        // their type like every output, so that the outputs of channels can
        // be concatenated (MultiChannelMapping).
        auto as_batch = [&](Value x) {
            return std::holds_alternative<std::monostate>(x.literal_value)
                ? x
                : fb.full({x, fb.batch_size({p_ext_lab})});
        };
        return {
            {{"momenta", p_ext_lab}, {"x1", as_batch(x1)}, {"x2", as_batch(x2)}},
            ps_weight
        };
    };

    if (_two_to_one) {
        // dPhi_1 = 2 pi delta(s_hat - m^2) (the 2 pi is _pi_factors): s_hat is
        // fixed and the one random number left is the rapidity, through the
        // split of tau = m^2 / s between x1 and x2.
        double mass = _topology.outgoing_masses().at(0);
        auto [x1_new, x2_new, det_x] =
            fb.r_to_x1x2(next_random(), mass * mass, _sqrt_s_lab * _sqrt_s_lab);
        x1 = x1_new;
        x2 = x2_new;
        dets.push_back(det_x);
        auto [p1, p2] = fb.com_p_in(mass);
        return finish(fb.stack({p1, p2, fb.add(p1, p2)}));
    }

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
    if (_map_luminosity && _y_max_lab >= 0.) {
        auto [x1_new, x2_new, det_x] = fb.r_to_x1x2_window(
            next_random(), s_hat, _sqrt_s_lab * _sqrt_s_lab, _y_max_lab
        );
        x1 = x1_new;
        x2 = x2_new;
        dets.push_back(det_x);
    } else if (_map_luminosity) {
        auto [x1_new, x2_new, det_x] =
            fb.r_to_x1x2(next_random(), s_hat, _sqrt_s_lab * _sqrt_s_lab);
        x1 = x1_new;
        x2 = x2_new;
        dets.push_back(det_x);
    }

    auto lab_x = lab_fractions(fb, x1, x2, conditions);
    Value x1_lab = lab_x.first, x2_lab = lab_x.second;

    // if required, build t-channel part of phase space mapping
    ValueVec p_ext;
    std::visit(
        Overloaded{
            [&](auto& t_mapping) {
                ValueVec args, conds;
                for (std::size_t i = 0; i < t_mapping.random_dim(); ++i) {
                    args.push_back(next_random());
                }
                // Discrete choices follow the continuous randoms, matching the
                // t-channel mapping's input_types order [random..., discrete...].
                for (std::size_t j = 0; j < t_mapping.discrete_dim(); ++j) {
                    args.push_back(next_discrete());
                }
                conds.push_back(sqrt_s_hat);
                for (std::size_t index : decay_data.at(0).decay.child_indices) {
                    conds.push_back(decay_data.at(index).mass.value());
                }
                using TMapping = std::decay_t<decltype(t_mapping)>;
                if constexpr (std::is_same_v<TMapping, TPropagatorMapping> ||
                              std::is_same_v<TMapping, ColorOrderedMapping>) {
                    if (t_mapping.has_rapidity_window()) {
                        conds.push_back(x1_lab);
                        conds.push_back(x2_lab);
                    }
                }
                auto t_result = t_mapping.build_forward(fb, args, conds);
                std::size_t result_index;
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
                if (_topology.is_decay()) {
                    // Single incoming particle, at rest in the frame the decay
                    // products are generated in: p_in = (M, 0, 0, 0), which is
                    // exactly the sum of the two back-to-back beam momenta
                    // com_p_in builds for sqrt(s_hat) = M.
                    p_ext = {fb.add(p1, p2)};
                } else {
                    p_ext = {p1, p2};
                }
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
                // a root decay restricted by the cuts reads the boost to the lab
                ValueVec decay_conds;
                if (decay_map.condition_types().size() == 2) {
                    decay_conds = {x1_lab, x2_lab};
                }
                auto k_out = decay_map.build_forward(fb, decay_args, decay_conds);
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
    return finish(fb.stack(p_ext));
}

std::pair<Value, Value> PhaseSpaceMapping::lab_fractions(
    FunctionBuilder& fb, Value x1, Value x2, const NamedVector<Value>& conditions
) const {
    if (_beam_rapidity == 0.) {
        return {x1, x2};
    }
    double exp_plus = std::exp(_beam_rapidity), exp_minus = std::exp(-_beam_rapidity);
    if (!_mirror_beams) {
        return {fb.mul(x1, exp_plus), fb.mul(x2, exp_minus)};
    }
    auto mirror_index = conditions.at("mirror_index");
    return {
        fb.mul(x1, fb.gather(mirror_index, std::vector<double>{exp_plus, exp_minus})),
        fb.mul(x2, fb.gather(mirror_index, std::vector<double>{exp_minus, exp_plus}))
    };
}

Value PhaseSpaceMapping::to_lab(
    FunctionBuilder& fb,
    Value momenta,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
) const {
    // boost_beam boosts by the rapidity 0.5 ln(x1 / x2). Without a mirror the
    // partonic and beam boosts are one: scaling x1 by exp(y) and x2 by
    // exp(-y) adds the beam rapidity y.
    Value masses(lab_masses(_topology, _permutations));
    double exp_plus = std::exp(_beam_rapidity), exp_minus = std::exp(-_beam_rapidity);
    if (_map_luminosity && !_mirror_beams && _beam_rapidity != 0.) {
        return fb.boost_beam(
            momenta, masses, fb.mul(x1, exp_plus), fb.mul(x2, exp_minus)
        );
    }
    if (_map_luminosity) {
        momenta = fb.boost_beam(momenta, masses, x1, x2);
    }
    if (_mirror_beams) {
        // a rotation: the masses of the legs are unchanged by it
        momenta = fb.mirror_momenta(momenta, conditions.at("mirror_index"));
    }
    if (_beam_rapidity != 0.) {
        momenta = fb.boost_beam(momenta, masses, exp_plus, exp_minus);
    }
    return momenta;
}

Value PhaseSpaceMapping::from_lab(
    FunctionBuilder& fb,
    Value momenta,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
) const {
    Value masses(lab_masses(_topology, _permutations));
    double exp_plus = std::exp(_beam_rapidity), exp_minus = std::exp(-_beam_rapidity);
    if (_map_luminosity && !_mirror_beams && _beam_rapidity != 0.) {
        return fb.boost_beam_inverse(
            momenta, masses, fb.mul(x1, exp_plus), fb.mul(x2, exp_minus)
        );
    }
    if (_beam_rapidity != 0.) {
        momenta = fb.boost_beam_inverse(momenta, masses, exp_plus, exp_minus);
    }
    if (_mirror_beams) {
        // a rotation: the masses of the legs are unchanged by it
        momenta = fb.mirror_momenta(momenta, conditions.at("mirror_index"));
    }
    if (_map_luminosity) {
        momenta = fb.boost_beam_inverse(momenta, masses, x1, x2);
    }
    return momenta;
}

Mapping::Result PhaseSpaceMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value p_ext_lab = inputs.at(0), x1 = inputs.at(1), x2 = inputs.at(2);
    if (_two_to_one) {
        auto [r, det_x] = fb.x1x2_to_r(x1, x2, _sqrt_s_lab * _sqrt_s_lab);
        return {{{"random", fb.stack({r})}}, fb.mul(det_x, 1. / _pi_factors)};
    }
    Value p_ext_stack = from_lab(fb, p_ext_lab, x1, x2, conditions);
    auto lab_x = lab_fractions(fb, x1, x2, conditions);
    Value x1_lab = lab_x.first, x2_lab = lab_x.second;

    // permute momenta if permutations are given
    if (_permutations.size() > 1) {
        p_ext_stack = fb.permute_momenta(
            p_ext_stack,
            invert_permutations(_permutations),
            conditions.at("permutation_index")
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
    for (auto [decay_index, mass, momentum] : zip(
             _topology.outgoing_indices(),
             _topology.outgoing_masses(),
             std::span(p_ext.begin() + _topology.incoming_masses().size(), p_ext.end())
         )) {
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
    ValueVec discrete_out;
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
                ValueVec decay_conds;
                if (decay_map.condition_types().size() == 2) {
                    decay_conds = {x1_lab, x2_lab};
                }
                auto decay_out = decay_map.build_inverse(fb, decay_args, decay_conds);
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
                if constexpr (std::is_same_v<TMapping, TPropagatorMapping> ||
                              std::is_same_v<TMapping, ColorOrderedMapping>) {
                    if (t_mapping.has_rapidity_window()) {
                        conds.push_back(x1_lab);
                        conds.push_back(x2_lab);
                    }
                }
                auto t_result = t_mapping.build_inverse(fb, args, conds);
                random_out_reversed.insert(
                    random_out_reversed.end(),
                    t_result.rend() - t_mapping.random_dim(),
                    t_result.rend()
                );
                // Discrete choices sit at forward positions [nc, nc+nd) in
                // t_result (after the continuous randoms, before "det").
                for (std::size_t j = 0; j < t_mapping.discrete_dim(); ++j) {
                    discrete_out.push_back(t_result.at(t_mapping.random_dim() + j));
                }
                dets.push_back(t_result["det"]);
            },
            [&](std::monostate) {}
        },
        _t_mapping
    );

    if (_map_luminosity && _y_max_lab >= 0.) {
        auto [r, det_x] =
            fb.x1x2_to_r_window(x1, x2, _sqrt_s_lab * _sqrt_s_lab, _y_max_lab);
        random_out_reversed.push_back(r);
        dets.push_back(det_x);
    } else if (_map_luminosity) {
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
    NamedVector<Value> result{{"random", fb.stack(random_out)}};
    if (!discrete_out.empty()) {
        result.push_back("discrete", fb.stack(discrete_out));
    }
    return {result, fb.product(dets)};
}
