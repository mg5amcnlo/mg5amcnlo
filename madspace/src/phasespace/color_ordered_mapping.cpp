#include "madspace/phasespace/color_ordered_mapping.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
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
        rotated.push_back(color_order.at((i0 + k) % n));
    }
    auto it1 = std::find(rotated.begin(), rotated.end(), 1u);
    if (it1 == rotated.end()) {
        throw std::invalid_argument("color_order must contain particle 1");
    }
    std::size_t i1 = std::distance(rotated.begin(), it1);
    std::vector<std::size_t> set1, set2;
    for (std::size_t k = 1; k < i1; ++k) {
        std::size_t p = rotated.at(k);
        if (p <= 1) {
            throw std::invalid_argument("invalid color_order");
        }
        set1.push_back(p - 2);
    }
    for (std::size_t k = i1 + 1; k < n; ++k) {
        std::size_t p = rotated.at(k);
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
    // Continuous randoms only. First peel: 2->2 LAB (2 randoms). Each subsequent
    // peel: 2->3 (2 randoms).
    return 2 + 2 * (k - 2);
}

// A side of exactly two particles is one 2->3 step off the opposite beam when
// the other side is not empty (AmpliCol), else a 2->2 peel.
bool two_particle_step(std::size_t k, std::size_t k_other) {
    return k == 2 && k_other > 0;
}

std::size_t n_discrete_for_set_size(std::size_t k, std::size_t k_other) {
    // One discrete two-solution choice per 2->3 step: every peel after the first,
    // and the single step of a side of two particles.
    if (two_particle_step(k, k_other)) {
        return 1;
    }
    return (k >= 2) ? (k - 2) : 0;
}

// s in [s_min, s_max], sampled with the given Invariant, and its weight. The
// cut-aware bounds of a mass can leave an empty range, which gets zero weight
// (and a small width, so that the sampling stays finite).
std::array<Value, 2> sample_mass(
    FunctionBuilder& fb, const Invariant& invariant, Value r, Value s_min, Value s_max
) {
    auto ok = fb.cut_one(fb.sub(s_max, s_min), Value(0.), Value(1e300));
    auto width = fb.max(fb.mul(s_min, Value(1e-10)), Value(1e-300));
    auto res =
        invariant.build_forward(fb, {r}, {s_min, fb.max(s_max, fb.add(s_min, width))});
    return {res["invariant"], fb.mul(res["det"], ok)};
}

std::array<Value, 2> recover_mass(
    FunctionBuilder& fb, const Invariant& invariant, Value s, Value s_min, Value s_max
) {
    auto ok = fb.cut_one(fb.sub(s_max, s_min), Value(0.), Value(1e300));
    auto width = fb.max(fb.mul(s_min, Value(1e-10)), Value(1e-300));
    auto res =
        invariant.build_inverse(fb, {s}, {s_min, fb.max(s_max, fb.add(s_min, width))});
    return {res["random"], fb.mul(res["det"], ok)};
}

// Range of the squared mass s_a of a system a that makes up the full final
// state together with a system b, from the energies in the partonic frame,
// E_a = (s_hat + s_a - s_b) / (2 sqrt(s_hat)) >= e_min_a and the same for b
// (AmpliCol's gent_one_step). e_min is a floor on the transverse mass: the sum
// of those of the particles, and the square root of the mass floor. With s_b
// still to be sampled (m_b empty) its floor and e_min_b bound it.
std::array<Value, 2> full_system_mass_range(
    FunctionBuilder& fb,
    Value e_cm,
    Value floor_a,
    Value e_min_a,
    Value floor_b,
    Value e_min_b,
    std::optional<Value> m_b
) {
    auto lower = fb.add(
        m_b ? fb.square(*m_b) : floor_b,
        fb.mul(e_cm, fb.sub(fb.add(e_min_a, e_min_a), e_cm))
    );
    auto s_min = fb.max(floor_a, lower);
    Value s_max;
    if (m_b) {
        // s_b + s_hat - 2 sqrt(s_hat) e_b, as (sqrt(s_hat) - e_b)^2 + s_b - e_b^2:
        // without a cut e_b = m_b, and the range of a heavy b keeps its
        // relative precision
        auto e_b = fb.max(*m_b, e_min_b);
        s_max = fb.add(
            fb.square(fb.sub(e_cm, e_b)), fb.mul(fb.sub(*m_b, e_b), fb.add(*m_b, e_b))
        );
    } else {
        s_max = fb.square(fb.sub(e_cm, e_min_b));
    }
    return {s_min, s_max};
}

double mat_at(const std::vector<std::vector<double>>& m, std::size_t i, std::size_t j) {
    if (i < m.size() && j < m.at(i).size()) {
        return m.at(i).at(j);
    }
    return 0.0;
}

// x1, x2 for a block that declared them (two more conditions than given)
void append_beam_fractions(
    ValueVec& cond, const Mapping& block, const Value& x1, const Value& x2
) {
    if (block.condition_types().size() == cond.size() + 2) {
        cond.push_back(x1);
        cond.push_back(x2);
    }
}

} // namespace

// Whether any cut (pt, invariant-mass, or dR) is active. With no cut the cut
// kernels reduce to the plain ones, so we skip them to keep the map exactly
// invertible and avoid boosted-frame round-trip noise.
static bool has_any_cut(
    const std::vector<double>& pt_min,
    const std::vector<std::vector<double>>& m_inv_min,
    const std::vector<std::vector<double>>& dr_min
) {
    for (double p : pt_min) {
        if (p > 0.0) {
            return true;
        }
    }
    for (const auto& row : m_inv_min) {
        for (double v : row) {
            if (v > 0.0) {
                return true;
            }
        }
    }
    for (const auto& row : dr_min) {
        for (double v : row) {
            if (v > 0.0) {
                return true;
            }
        }
    }
    return false;
}

// Power of the momentum transfers and s23 invariants of the chain: as given,
// else 1 with cuts (AmpliCol) and 0.8 without, where their ranges reach zero
// and 1/|x| would pile the points up at the regulator of Invariant.
static double chain_power(
    const ColorOrderedMapping::Options& options,
    const std::vector<double>& pt_min,
    const std::vector<std::vector<double>>& m_inv_min,
    const std::vector<std::vector<double>>& dr_min
) {
    return options.chain_invariant_power.value_or(
        has_any_cut(pt_min, m_inv_min, dr_min) ? 1. : 0.8
    );
}

double ColorOrderedMapping::pt2(std::size_t i) const {
    double p = (i < _pt_min.size()) ? _pt_min.at(i) : 0.0;
    return p * p;
}

double ColorOrderedMapping::cut_floor(const std::vector<std::size_t>& subset) const {
    if (subset.size() < 2) {
        return 0.0;
    }
    // Sum over distinct pairs of the per-pair invariant-mass floor implied by
    // m_inv_min and the pt/dR cut, exactly as gen23's setup_PS_cuts.
    double cut = 0.0;
    for (std::size_t a = 0; a + 1 < subset.size(); ++a) {
        for (std::size_t b = a + 1; b < subset.size(); ++b) {
            std::size_t i = subset.at(a), j = subset.at(b);
            double ss = mat_at(_m_inv_min, i, j);
            double pti = (i < _pt_min.size()) ? _pt_min.at(i) : 0.0;
            double ptj = (j < _pt_min.size()) ? _pt_min.at(j) : 0.0;
            double dr = mat_at(_dr_min, i, j);
            cut += std::max(ss * ss, 2.0 * pti * ptj * (1.0 - std::cos(dr)));
        }
    }
    double npart = static_cast<double>(subset.size());
    // gen23 uses npart/(npart-1) for the full final state and 1/2 for proper
    // subsets.
    double scaling = (subset.size() == _n_out) ? npart / (npart - 1.0) : 0.5;
    return cut * scaling;
}

std::array<Value, 2> ColorOrderedMapping::set_floors(
    FunctionBuilder& fb, const std::vector<std::size_t>& set, const ValueVec& m_out
) const {
    Value mass_sum = m_out.at(set.at(0));
    for (std::size_t i = 1; i < set.size(); ++i) {
        mass_sum = fb.add(mass_sum, m_out.at(set.at(i)));
    }
    auto floor = fb.square(mass_sum);
    double fl = cut_floor(set);
    if (fl > 0.0) {
        floor = fb.max(floor, Value(fl));
    }
    if (!_has_cut) {
        return {floor, fb.sqrt(floor)};
    }
    auto e_min = fb.sqrt(fb.add(fb.square(m_out.at(set.at(0))), Value(pt2(set.at(0)))));
    for (std::size_t i = 1; i < set.size(); ++i) {
        e_min = fb.add(
            e_min,
            fb.sqrt(fb.add(fb.square(m_out.at(set.at(i))), Value(pt2(set.at(i)))))
        );
    }
    return {floor, fb.max(e_min, fb.sqrt(floor))};
}

std::array<Value, 2> ColorOrderedMapping::rest_mass_range(
    FunctionBuilder& fb,
    const std::vector<std::size_t>& s,
    std::size_t j,
    Value prev_mass,
    Value e_cm,
    const ValueVec& m_out
) const {
    // rest system {s[j+1], ..., s[k-1]} after peeling s[j] off a system of
    // mass prev_mass
    std::vector<std::size_t> rest(s.begin() + j + 1, s.end());
    auto [floor, e_min] = set_floors(fb, rest, m_out);
    if (_use_single_chain && j == 0) {
        // the rest and the first particle make up the full final state
        auto [floor_p, e_min_p] = set_floors(fb, {s.at(0)}, m_out);
        return full_system_mass_range(
            fb, e_cm, floor, e_min, floor_p, e_min_p, m_out.at(s.at(0))
        );
    }
    return {floor, fb.square(fb.sub(prev_mass, m_out.at(s.at(j))))};
}

// A set of particles is bounded by the largest of their bounds when each has
// one: the rapidity of a sum of momenta is a weighted mean of theirs.
ColorOrderedMapping::RapidityBounds ColorOrderedMapping::rapidity_bounds(
    const std::vector<std::size_t>& color_order, const std::vector<double>& y_max
) {
    RapidityBounds bounds;
    auto [s1, s2] = split_sets_from_color_order(color_order);
    auto bound_of = [&](auto begin, auto end) {
        if (begin == end) {
            return -1.;
        }
        double y = 0.;
        for (auto it = begin; it != end; ++it) {
            if (*it >= y_max.size() || y_max.at(*it) < 0.) {
                return -1.;
            }
            y = std::max(y, y_max.at(*it));
        }
        return y;
    };
    if (s1.empty() || s2.empty()) {
        // the walk of the non-empty set starts from its own beam (beam 1 for
        // set1, beam 2 for set2) against the other one
        const auto& s = s1.empty() ? s2 : s1;
        bounds.chain_peeled = bound_of(s.begin(), s.begin() + 1);
        bounds.chain_rest = bound_of(s.begin() + 1, s.end());
        bounds.chain_beam_sign = s1.empty() ? -1. : 1.;
    } else if ((s1.size() == 1) != (s2.size() == 1)) {
        const auto& single = s1.size() == 1 ? s1 : s2;
        const auto& recoil = s1.size() == 1 ? s2 : s1;
        bounds.double_t = true;
        bounds.central1 = bound_of(single.begin(), single.end());
        bounds.central2 = bound_of(recoil.begin(), recoil.end());
    } else {
        bounds.central1 = bound_of(s1.begin(), s1.end());
        bounds.central2 = bound_of(s2.begin(), s2.end());
    }
    return bounds;
}

ColorOrderedMapping::ColorOrderedMapping(
    const Options& options,
    const std::vector<double>& pt_min,
    const std::vector<std::vector<double>>& m_inv_min,
    const std::vector<std::vector<double>>& dr_min,
    const std::vector<double>& y_max
) :
    ColorOrderedMapping(
        rapidity_bounds(options.color_order, y_max), options, pt_min, m_inv_min, dr_min
    ) {}

ColorOrderedMapping::ColorOrderedMapping(
    const RapidityBounds& bounds,
    const Options& options,
    const std::vector<double>& pt_min,
    const std::vector<std::vector<double>>& m_inv_min,
    const std::vector<std::vector<double>>& dr_min
) :
    Mapping(
        "ColorOrderedMapping",
        [&] {
            auto [s1, s2] = split_sets_from_color_order(options.color_order);
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
            std::size_t n_discrete = n_discrete_for_set_size(s1.size(), s2.size()) +
                n_discrete_for_set_size(s2.size(), s1.size());
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < total; ++i) {
                input_types.push_back(std::format("random{}", i), batch_float);
            }
            // One int per 2->3 step (the two-solution choice), after the randoms.
            for (std::size_t j = 0; j < n_discrete; ++j) {
                input_types.push_back(std::format("discrete{}", j), batch_int);
            }
            return input_types;
        }(),
        [&] {
            std::size_t n_out = options.color_order.size() - 2;
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < n_out + 2; ++i) {
                output_types.push_back(std::format("momentum{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            std::size_t n_out = options.color_order.size() - 2;
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            for (std::size_t i = 0; i < n_out; ++i) {
                cond_types.push_back(std::format("mass{}", i), batch_float);
            }
            if (bounds.active()) {
                cond_types.push_back("x1", batch_float);
                cond_types.push_back("x2", batch_float);
            }
            return cond_types;
        }()
    ),
    _n_out(options.color_order.size() - 2),
    _pt_min(pt_min),
    _m_inv_min(m_inv_min),
    _dr_min(dr_min),
    _has_cut(has_any_cut(pt_min, m_inv_min, dr_min)),
    _rapidity_window(bounds.active()),
    _mass_invariant(0.5),
    _com_scattering(
        true,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min),
        bounds.double_t ? -1. : bounds.central1,
        bounds.double_t ? -1. : bounds.central2,
        1.
    ),
    _lab_scattering(
        false,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min)
    ),
    _chain_scattering(
        false,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min),
        bounds.chain_rest,
        bounds.chain_peeled,
        bounds.chain_beam_sign
    ),
    _two_to_three(
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min),
        true,
        true,
        false,
        options.cut_aware_branch
    ),
    _two_to_three_beam(
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min),
        true,
        true,
        true,
        options.cut_aware_branch
    ),
    _double_t(
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        chain_power(options, pt_min, m_inv_min, dr_min),
        0.,
        0.,
        has_any_cut(pt_min, m_inv_min, dr_min),
        bounds.double_t ? bounds.central1 : -1.,
        bounds.double_t ? bounds.central2 : -1.,
        1.
    ) {
    auto [s1, s2] = split_sets_from_color_order(options.color_order);
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
    _discrete_dim = n_discrete_for_set_size(s1.size(), s2.size()) +
        n_discrete_for_set_size(s2.size(), s1.size());
}

Mapping::Result ColorOrderedMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.begin() + 1 + _n_out);
    Value x1 = _rapidity_window ? conditions.at(1 + _n_out) : Value(1.);
    Value x2 = _rapidity_window ? conditions.at(2 + _n_out) : Value(1.);
    auto r = inputs.begin();
    auto next_random = [&]() { return *(r++); };
    auto r_disc = inputs.begin() + _random_dim;
    auto next_discrete = [&]() { return *(r_disc++); };
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
            m_set1 = m_out.at(_set1.at(0));
        }
        if (_set2.size() == 1) {
            m_set2 = m_out.at(_set2.at(0));
        }
        bool single_is_set1 = (_set1.size() == 1);
        Value m_single = single_is_set1 ? m_set1 : m_set2;
        Value mir_min = single_is_set1 ? mass_sum_set2 : mass_sum_set1;
        // ETmin floors (pt-cut tightening of the central t1/t2 bounds): the lone
        // particle's ET and the summed ET of the recoil set. Zero (inert) with no cut.
        Value etmin_i = Value(0.);
        Value etmin_ir = Value(0.);
        if (_has_cut) {
            auto etmin_p = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            std::size_t single_idx = single_is_set1 ? _set1.at(0) : _set2.at(0);
            const auto& recoil = single_is_set1 ? _set2 : _set1;
            etmin_i = etmin_p(single_idx);
            etmin_ir = etmin_p(recoil.at(0));
            for (std::size_t j = 1; j < recoil.size(); ++j) {
                etmin_ir = fb.add(etmin_ir, etmin_p(recoil.at(j)));
            }
        }
        ValueVec dt_cond{pa, pb, m_single, mir_min};
        if (_has_cut) {
            dt_cond.push_back(etmin_i);
            dt_cond.push_back(etmin_ir);
        }
        append_beam_fractions(dt_cond, _double_t, x1, x2);
        auto central = _double_t.build_forward(
            fb, {next_random(), next_random(), next_random()}, dt_cond
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
        // Composite masses: set1 first, then set2 in what is left. Both make up
        // the full final state, so the energies of the two sides bound them
        // (AmpliCol's shatminmax and gent_one_step), with the mass and
        // transverse-mass floors of the cuts.
        auto [floor1, e_min1] = set_floors(fb, _set1, m_out);
        auto [floor2, e_min2] = set_floors(fb, _set2, m_out);
        if (_set1.size() >= 2) {
            auto [s_min, s_max] = full_system_mass_range(
                fb, e_cm, floor1, e_min1, floor2, e_min2, std::nullopt
            );
            auto [s, det] =
                sample_mass(fb, _mass_invariant, next_random(), s_min, s_max);
            m_set1 = fb.sqrt(s);
            dets.push_back(det);
        } else {
            m_set1 = m_out.at(_set1.at(0));
        }
        if (_set2.size() >= 2) {
            auto [s_min, s_max] = full_system_mass_range(
                fb, e_cm, floor2, e_min2, floor1, e_min1, m_set1
            );
            auto [s, det] =
                sample_mass(fb, _mass_invariant, next_random(), s_min, s_max);
            m_set2 = fb.sqrt(s);
            dets.push_back(det);
        } else {
            m_set2 = m_out.at(_set2.at(0));
        }
        // ETmin floors for the central 2->2: each output system must carry enough
        // transverse energy for its jets, so the floor is the summed ET of the set
        // (a valid outer bound since ET_sys >= sum ET_constituents). Zero with no cut.
        Value etmin_set1 = Value(0.);
        Value etmin_set2 = Value(0.);
        if (_has_cut) {
            auto etmin_p = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            etmin_set1 = etmin_p(_set1.at(0));
            for (std::size_t j = 1; j < _set1.size(); ++j) {
                etmin_set1 = fb.add(etmin_set1, etmin_p(_set1.at(j)));
            }
            etmin_set2 = etmin_p(_set2.at(0));
            for (std::size_t j = 1; j < _set2.size(); ++j) {
                etmin_set2 = fb.add(etmin_set2, etmin_p(_set2.at(j)));
            }
        }
        ValueVec com_cond{pa, pb};
        if (_has_cut) {
            com_cond.push_back(etmin_set1);
            com_cond.push_back(etmin_set2);
        }
        append_beam_fractions(com_cond, _com_scattering, x1, x2);
        auto central = _com_scattering.build_forward(
            fb, {next_random(), next_random(), m_set1, m_set2}, com_cond
        );
        P_set1 = central.at(0);
        P_set2 = central.at(1);
        dets.push_back(central["det"]);
    }

    // R_b for each walk: the beam minus what the central block emitted on the other
    // side.
    Value R_b_for_set1 = fb.sub(pb, P_set2);
    Value R_b_for_set2 = fb.sub(pa, P_set1);

    // Phase 1b: intermediate rest masses of each walk, for steps j in 0..k-3.
    // Each is bounded by the previous one (rest_mass_range), which avoids
    // (k-2)! over-counting. Sampled after the central block, since DoubleT
    // only fixes m_set there.
    auto sample_intermediate_masses =
        [&](const std::vector<std::size_t>& s, Value m_set) -> ValueVec {
        std::size_t k = s.size();
        ValueVec res;
        if (k <= 2) {
            return res;
        }
        Value prev_mass = m_set;
        for (std::size_t j = 0; j < k - 2; ++j) {
            auto [s_min, s_max] = rest_mass_range(fb, s, j, prev_mass, e_cm, m_out);
            auto [s_rest, det] =
                sample_mass(fb, _mass_invariant, next_random(), s_min, s_max);
            Value m_rest = fb.sqrt(s_rest);
            res.push_back(m_rest);
            dets.push_back(det);
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
            std::size_t k_other,
            Value P_set,
            Value R_b,
            Value beam_own,
            Value beam_other,
            const ValueVec& rest_masses) {
            std::size_t k = s.size();
            if (k == 1) {
                p_out.at(s.at(0)) = P_set;
                return;
            }
            auto etmin_particle = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            if (two_particle_step(k, k_other)) {
                // AmpliCol's gen23_one_step(set(i), inext, i, 3 - i): p1 = s[0]
                // with t1 = (beam_own - p1)^2, p2 = s[1] with the momentum
                // transfer (p2 - beam_other)^2, the two invariants next to the
                // beams in the color order.
                ValueVec cond{beam_own, P_set, beam_other};
                if (_has_cut) {
                    cond.push_back(etmin_particle(s.at(0)));
                    cond.push_back(etmin_particle(s.at(1)));
                    cond.push_back(Value(0.));
                    cond.push_back(Value(0.));
                }
                auto ks = _two_to_three_beam.build_forward(
                    fb,
                    {next_discrete(),
                     next_random(),
                     next_random(),
                     m_out.at(s.at(0)),
                     m_out.at(s.at(1))},
                    cond
                );
                p_out.at(s.at(0)) = ks.at(0);
                p_out.at(s.at(1)) = ks.at(1);
                dets.push_back(ks["det"]);
                return;
            }
            Value R_a = fb.sub(P_set, R_b);
            ValueVec etmin_suffix;
            if (_has_cut) {
                etmin_suffix.resize(k);
                etmin_suffix.at(k - 1) = etmin_particle(s.at(k - 1));
                for (std::size_t j = k - 1; j-- > 0;) {
                    etmin_suffix.at(j) =
                        fb.add(etmin_particle(s.at(j)), etmin_suffix.at(j + 1));
                }
            }
            Value im1, chain;
            bool first = true;
            for (std::size_t j = 0; j < k - 1; ++j) {
                // Rest mass after this peel: rest_masses[j], or the last particle's
                // mass for j == k-2.
                Value m_rest = (j < k - 2) ? rest_masses[j] : m_out.at(s.at(k - 1));
                Value m_peel = m_out.at(s.at(j));
                // Block convention: the first output is the chain (mass m_rest),
                // the second the peeled particle (mass m_peel). Each block returns
                // chain + peeled = its incoming system by construction, so the
                // outgoing momenta add up to P_set, and the chain is handed on as
                // the system of the next block.
                if (first) {
                    // First peel is a 2->2 LAB block, projecting the own-side on-z
                    // beam R_a (AmpliCol: gent_one_step beam i). Cut: ETmin of the
                    // recoil chain (etmin_1) then the peeled particle (etmin_2).
                    ValueVec cond{R_a, R_b};
                    if (_has_cut) {
                        cond.push_back(etmin_suffix.at(j + 1));
                        cond.push_back(etmin_particle(s.at(j)));
                    }
                    // in a single chain R_a, R_b are the beams themselves
                    const auto& first_peel =
                        _use_single_chain ? _chain_scattering : _lab_scattering;
                    append_beam_fractions(cond, first_peel, x1, x2);
                    auto ks = first_peel.build_forward(
                        fb, {next_random(), next_random(), m_rest, m_peel}, cond
                    );
                    Value peeled = ks.at(1);
                    p_out.at(s.at(j)) = peeled;
                    im1 = peeled;
                    chain = ks.at(0);
                    dets.push_back(ks["det"]);
                    first = false;
                } else {
                    // The system still to be resolved, p_12 of this block, is the
                    // chain side of the previous block. R_a + R_b is the same
                    // momentum, but formed from beam-sized ones; for a soft system
                    // its absolute error is of order the beam energy.
                    ValueVec cond{beam_other, chain, im1};
                    if (_has_cut) {
                        cond.push_back(etmin_suffix.at(j + 1));
                        cond.push_back(etmin_particle(s.at(j)));
                        cond.push_back(Value(mat_at(_dr_min, s.at(j - 1), s.at(j))));
                        cond.push_back(Value(cut_floor({s.at(j - 1), s.at(j)})));
                    }
                    auto ks = _two_to_three.build_forward(
                        fb,
                        {next_discrete(), next_random(), next_random(), m_rest, m_peel},
                        cond
                    );
                    Value peeled = ks.at(1);
                    p_out.at(s.at(j)) = peeled;
                    im1 = peeled;
                    chain = ks.at(0);
                    dets.push_back(ks["det"]);
                }
            }
            // Last particle: the chain side of the last block.
            p_out.at(s.at(k - 1)) = chain;
        };

    if (!_set1.empty()) {
        walk(_set1, _set2.size(), P_set1, R_b_for_set1, pa, pb, rest_masses_set1);
    }
    if (!_set2.empty()) {
        walk(_set2, _set1.size(), P_set2, R_b_for_set2, pb, pa, rest_masses_set2);
    }

    // Assemble outputs: momentum0, momentum1 = beams; momentum_{2+i} = outgoing i.
    ValueVec p_ext;
    p_ext.reserve(_n_out + 2);
    p_ext.push_back(pa);
    p_ext.push_back(pb);
    for (std::size_t i = 0; i < _n_out; ++i) {
        p_ext.push_back(p_out.at(i));
    }

    return {{output_types().keys(), p_ext}, fb.product(dets)};
}

Mapping::Result ColorOrderedMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.begin() + 1 + _n_out);
    Value x1 = _rapidity_window ? conditions.at(1 + _n_out) : Value(1.);
    Value x2 = _rapidity_window ? conditions.at(2 + _n_out) : Value(1.);
    ValueVec random_out;
    ValueVec discrete_out;
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
        Value p = p_outgoing(s.at(0));
        for (std::size_t k = 1; k < s.size(); ++k) {
            p = fb.add(p, p_outgoing(s.at(k)));
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
            m_set1 = m_out.at(_set1.at(0));
            m_set2 = fb.sqrt(invariants.at(idx_m2_set2));
        } else {
            m_set2 = m_out.at(_set2.at(0));
            m_set1 = fb.sqrt(invariants.at(idx_m2_set1));
        }
    } else {
        auto [floor1, e_min1] = set_floors(fb, _set1, m_out);
        auto [floor2, e_min2] = set_floors(fb, _set2, m_out);
        if (_set1.size() >= 2) {
            auto [s_min, s_max] = full_system_mass_range(
                fb, e_cm, floor1, e_min1, floor2, e_min2, std::nullopt
            );
            auto m2_set1 = invariants.at(idx_m2_set1);
            auto [r, det] = recover_mass(fb, _mass_invariant, m2_set1, s_min, s_max);
            random_out.push_back(r);
            dets.push_back(det);
            m_set1 = fb.sqrt(m2_set1);
        } else {
            m_set1 = m_out.at(_set1.at(0));
        }
        if (_set2.size() >= 2) {
            auto [s_min, s_max] = full_system_mass_range(
                fb, e_cm, floor2, e_min2, floor1, e_min1, m_set1
            );
            auto m2_set2 = invariants.at(idx_m2_set2);
            auto [r, det] = recover_mass(fb, _mass_invariant, m2_set2, s_min, s_max);
            random_out.push_back(r);
            dets.push_back(det);
            m_set2 = fb.sqrt(m2_set2);
        } else {
            m_set2 = m_out.at(_set2.at(0));
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
        Value etmin_i = Value(0.);
        Value etmin_ir = Value(0.);
        if (_has_cut) {
            auto etmin_p = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            std::size_t single_idx = single_is_set1 ? _set1.at(0) : _set2.at(0);
            const auto& recoil = single_is_set1 ? _set2 : _set1;
            etmin_i = etmin_p(single_idx);
            etmin_ir = etmin_p(recoil.at(0));
            for (std::size_t j = 1; j < recoil.size(); ++j) {
                etmin_ir = fb.add(etmin_ir, etmin_p(recoil.at(j)));
            }
        }
        ValueVec dt_cond{pa, pb, m_single, mir_min};
        if (_has_cut) {
            dt_cond.push_back(etmin_i);
            dt_cond.push_back(etmin_ir);
        }
        append_beam_fractions(dt_cond, _double_t, x1, x2);
        auto central = _double_t.build_inverse(fb, {p_single, p_recoil}, dt_cond);
        random_out.push_back(central.at(0));
        random_out.push_back(central.at(1));
        random_out.push_back(central.at(2));
        dets.push_back(central["det"]);
    } else {
        Value etmin_set1 = Value(0.);
        Value etmin_set2 = Value(0.);
        if (_has_cut) {
            auto etmin_p = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            etmin_set1 = etmin_p(_set1.at(0));
            for (std::size_t j = 1; j < _set1.size(); ++j) {
                etmin_set1 = fb.add(etmin_set1, etmin_p(_set1.at(j)));
            }
            etmin_set2 = etmin_p(_set2.at(0));
            for (std::size_t j = 1; j < _set2.size(); ++j) {
                etmin_set2 = fb.add(etmin_set2, etmin_p(_set2.at(j)));
            }
        }
        ValueVec com_cond{pa, pb};
        if (_has_cut) {
            com_cond.push_back(etmin_set1);
            com_cond.push_back(etmin_set2);
        }
        append_beam_fractions(com_cond, _com_scattering, x1, x2);
        auto central = _com_scattering.build_inverse(fb, {P_set1, P_set2}, com_cond);
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
                auto [s_min, s_max] = rest_mass_range(fb, s, j, prev_mass, e_cm, m_out);
                auto m2 = invariants.at(idx_start + j);
                auto [r, det] = recover_mass(fb, _mass_invariant, m2, s_min, s_max);
                random_out.push_back(r);
                dets.push_back(det);
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

    // Phase 3 inverse: peel-off walks, mirroring the forward (first peel projects
    // the own-side beam R_a, 2->3 peels project beam_other).
    auto walk_inverse =
        [&](const std::vector<std::size_t>& s,
            std::size_t k_other,
            Value P_set,
            Value R_b,
            Value beam_own,
            Value beam_other) {
            std::size_t k = s.size();
            if (k == 1) {
                return;
            }
            auto etmin_particle = [&](std::size_t idx) {
                return fb.sqrt(fb.add(fb.square(m_out.at(idx)), Value(pt2(idx))));
            };
            if (two_particle_step(k, k_other)) {
                ValueVec cond{beam_own, P_set, beam_other};
                if (_has_cut) {
                    cond.push_back(etmin_particle(s.at(0)));
                    cond.push_back(etmin_particle(s.at(1)));
                    cond.push_back(Value(0.));
                    cond.push_back(Value(0.));
                }
                auto rs = _two_to_three_beam.build_inverse(
                    fb, {p_outgoing(s.at(0)), p_outgoing(s.at(1))}, cond
                );
                discrete_out.push_back(rs.at(0));
                random_out.push_back(rs.at(1));
                random_out.push_back(rs.at(2));
                dets.push_back(rs["det"]);
                return;
            }
            Value R_a = fb.sub(P_set, R_b);
            // ETmin suffix sums for the recoil chain (see forward walk); only built
            // when cuts are active.
            ValueVec etmin_suffix;
            if (_has_cut) {
                etmin_suffix.resize(k);
                etmin_suffix.at(k - 1) = etmin_particle(s.at(k - 1));
                for (std::size_t j = k - 1; j-- > 0;) {
                    etmin_suffix.at(j) =
                        fb.add(etmin_particle(s.at(j)), etmin_suffix.at(j + 1));
                }
            }
            // Chain systems, mirroring the forward: chain_sums[j] is the sum of
            // the outgoing momenta s[j..k-1], summed from the soft end so that
            // it keeps the relative precision of the momenta it is made of.
            ValueVec chain_sums(k);
            chain_sums.at(k - 1) = p_outgoing(s.at(k - 1));
            for (std::size_t j = k - 1; j-- > 0;) {
                chain_sums.at(j) = fb.add(p_outgoing(s.at(j)), chain_sums.at(j + 1));
            }
            Value im1;
            bool first = true;
            for (std::size_t j = 0; j < k - 1; ++j) {
                Value peeled = p_outgoing(s.at(j));
                // p1_out is the chain carrier (mass m_rest)
                Value p1_out = chain_sums.at(j + 1);
                if (first) {
                    // Same projection as the forward 2->2 block: own-side beam R_a.
                    ValueVec cond{R_a, R_b};
                    if (_has_cut) {
                        cond.push_back(etmin_suffix.at(j + 1));
                        cond.push_back(etmin_particle(s.at(j)));
                    }
                    const auto& first_peel =
                        _use_single_chain ? _chain_scattering : _lab_scattering;
                    append_beam_fractions(cond, first_peel, x1, x2);
                    auto rs = first_peel.build_inverse(fb, {p1_out, peeled}, cond);
                    random_out.push_back(rs.at(0));
                    random_out.push_back(rs.at(1));
                    dets.push_back(rs["det"]);
                    first = false;
                } else {
                    // Mirror the forward: project beam_other, with the chain system
                    // s[j..k-1] as p_12 of the block.
                    ValueVec cond{beam_other, chain_sums.at(j), im1};
                    if (_has_cut) {
                        cond.push_back(etmin_suffix.at(j + 1));
                        cond.push_back(etmin_particle(s.at(j)));
                        cond.push_back(Value(mat_at(_dr_min, s.at(j - 1), s.at(j))));
                        cond.push_back(Value(cut_floor({s.at(j - 1), s.at(j)})));
                    }
                    auto rs = _two_to_three.build_inverse(fb, {p1_out, peeled}, cond);
                    // rs.at(0) is the discrete choice index (int) emitted by the
                    // 2->3 inverse; rs.at(1), rs.at(2) are the continuous r_s23, r_t1.
                    discrete_out.push_back(rs.at(0));
                    random_out.push_back(rs.at(1));
                    random_out.push_back(rs.at(2));
                    dets.push_back(rs["det"]);
                }
                im1 = peeled;
            }
        };

    if (!_set1.empty()) {
        walk_inverse(_set1, _set2.size(), P_set1, R_b_for_set1, pa, pb);
    }
    if (!_set2.empty()) {
        walk_inverse(_set2, _set1.size(), P_set2, R_b_for_set2, pb, pa);
    }

    // input_types is [random0..random_{N-1}, discrete0..discrete_{M-1}], so the
    // returned values must be the continuous randoms followed by the discrete ints.
    ValueVec all_out = random_out;
    all_out.insert(all_out.end(), discrete_out.begin(), discrete_out.end());
    return {{input_types().keys(), all_out}, fb.product(dets)};
}
