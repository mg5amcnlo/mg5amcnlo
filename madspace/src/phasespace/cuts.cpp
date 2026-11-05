#include "madspace/phasespace/cuts.hpp"

#include "madspace/compgraphs/type.hpp"
#include "madspace/util.hpp"

#include <algorithm>

using namespace madspace;

namespace {

void check_custom_cut(const std::optional<CustomFunction>& custom_cut) {
    if (custom_cut &&
        (custom_cut->type() != CustomFunction::cut || custom_cut->dynamic_pdg_ids())) {
        throw std::invalid_argument(
            "custom cut must be of type cut, with a single PDG-id option"
        );
    }
}

} // namespace

Cuts::Cuts(
    const std::vector<CutItem>& cut_data,
    const std::optional<CustomFunction>& custom_cut
) :
    FunctionGenerator(
        "Cuts", cut_data.at(0).observable.arg_types(), {{"mask", batch_float}}
    ),
    _cut_data(cut_data),
    _custom_cut(custom_cut) {
    check_custom_cut(custom_cut);
}

Cuts::Cuts(
    std::size_t particle_count, const std::optional<CustomFunction>& custom_cut
) :
    FunctionGenerator(
        "Cuts",
        {{"momenta", batch_four_vec_array(particle_count)}},
        {{"mask", batch_float}}
    ),
    _custom_cut(custom_cut) {
    check_custom_cut(custom_cut);
}

NamedVector<Value>
Cuts::build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const {
    ValueVec weights;
    for (auto& item : _cut_data) {
        if (item.observable.not_found()) {
            continue;
        }
        Value obs = item.observable.build_function(fb, args).at(0);
        if (obs.type.shape.size() == 0) {
            weights.push_back(fb.cut_one(obs, item.min, item.max));
        } else if (item.mode == CutMode::all) {
            weights.push_back(fb.cut_all(obs, item.min, item.max));
        } else {
            weights.push_back(fb.cut_any(obs, item.min, item.max));
        }
    }
    if (_custom_cut) {
        weights.push_back(_custom_cut->build_function(fb, {args.at(0)}).at(0));
    }
    return {{"mask", fb.product(weights)}};
}

namespace {

// The accessors hand bounds to the mappings as hard floors on a fixed particle
// or pair, which is only valid if every selected object must satisfy the bound.
// Ordered selections (a rank, not a particle) and CutMode::any cuts over
// several objects stay filters.
bool binds_each_object(const Cuts::CutItem& item, std::size_t object_count) {
    return !item.observable.ordered() &&
        (item.mode == Cuts::CutMode::all || object_count == 1);
}

} // namespace

double Cuts::sqrt_s_min() const {
    double sqrt_s_min = 0.;
    for (auto& item : _cut_data) {
        if (item.observable.observable() == Observable::obs_sqrt_s &&
            sqrt_s_min < item.min) {
            sqrt_s_min = item.min;
        }
    }
    return sqrt_s_min;
}

std::vector<double> Cuts::eta_max() const {
    std::vector<double> eta_max(
        arg_types().at(0).shape.at(0) - 2, std::numeric_limits<double>::infinity()
    );
    for (auto& item : _cut_data) {
        double item_max = std::numeric_limits<double>::infinity();
        if (item.observable.observable() == Observable::obs_eta_abs) {
            item_max = item.max;
        } else if (item.observable.observable() == Observable::obs_eta) {
            item_max = std::max(-item.min, item.max);
        } else {
            continue;
        }
        auto indices = item.observable.simple_observable_indices();
        if (!binds_each_object(item, indices.size())) {
            continue;
        }
        for (std::size_t index : indices) {
            if (index < 2) {
                continue;
            }
            double& limit = eta_max.at(index - 2);
            if (limit > item_max) {
                limit = item_max;
            }
        }
    }
    return eta_max;
}

std::vector<double> Cuts::pt_min() const {
    std::vector<double> pt_min(arg_types().at(0).shape.at(0) - 2, 0.);
    for (auto& item : _cut_data) {
        if (item.observable.observable() != Observable::obs_pt) {
            continue;
        }
        auto indices = item.observable.simple_observable_indices();
        if (!binds_each_object(item, indices.size())) {
            continue;
        }
        for (std::size_t index : indices) {
            if (index < 2) {
                continue;
            }
            double& limit = pt_min.at(index - 2);
            if (limit < item.min) {
                limit = item.min;
            }
        }
    }
    return pt_min;
}

namespace {

using PairList = std::vector<std::pair<std::size_t, std::size_t>>;

// The pairs (indices including the beams) whose invariant mass an observable
// measures: the pair observables, and "mass" of summed momenta when the sum is
// a pair (one group of exactly two particles, or one pair per element of two
// groups). Empty for anything else.
PairList mass_pairs(const Observable& o) {
    PairList pairs;
    const auto& idx = o.indices();
    switch (o.observable()) {
    case Observable::obs_mass:
        if (!o.sum_momenta()) {
            break;
        }
        if (idx.size() == 1 && idx.at(0).size() == 2) {
            pairs.emplace_back(idx.at(0).at(0), idx.at(0).at(1));
            break;
        }
        [[fallthrough]];
    case Observable::obs_pair_mass:
    case Observable::obs_sfos_pair_mass:
        if (idx.size() == 2) {
            for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
                pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
            }
        }
        break;
    default:
        break;
    }
    return pairs;
}

PairList delta_r_pairs(const Observable& o) {
    PairList pairs;
    const auto& idx = o.indices();
    if (o.observable() == Observable::obs_delta_r && idx.size() == 2) {
        for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
            pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
        }
    }
    return pairs;
}

} // namespace

std::vector<std::vector<double>>
Cuts::pairwise_min(const std::function<PairList(const Observable&)>& pairs) const {
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.));
    for (auto& item : _cut_data) {
        if (item.observable.not_found()) {
            continue;
        }
        auto item_pairs = pairs(item.observable);
        if (item_pairs.empty() || !binds_each_object(item, item_pairs.size())) {
            continue;
        }
        for (auto [i, j] : item_pairs) {
            if (i < 2 || j < 2) {
                continue;
            }
            i -= 2;
            j -= 2;
            if (i < n && j < n && item.min > out.at(i).at(j)) {
                out.at(i).at(j) = item.min;
                out.at(j).at(i) = item.min;
            }
        }
    }
    return out;
}

std::vector<std::vector<double>> Cuts::m_inv_min() const {
    return pairwise_min(mass_pairs);
}

std::vector<Cuts::PairMassAny> Cuts::pair_mass_any_min() const {
    // The CutMode::any pair mass cuts that m_inv_min leaves out: they bound no
    // pair in particular, but any set of particles holding all of their pairs.
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<PairMassAny> out;
    for (auto& item : _cut_data) {
        if (item.observable.not_found() || item.mode != CutMode::any ||
            item.observable.ordered() || !(item.min > 0.)) {
            continue;
        }
        auto item_pairs = mass_pairs(item.observable);
        if (item_pairs.size() < 2) {
            // nothing, or one pair, which m_inv_min already reports
            continue;
        }
        PairMassAny bound{{}, item.min};
        for (auto [i, j] : item_pairs) {
            if (i < 2 || j < 2 || i - 2 >= n || j - 2 >= n) {
                // a pair the outgoing particles cannot account for could be
                // the one that passes, so the cut bounds nothing
                bound.pairs.clear();
                break;
            }
            bound.pairs.emplace_back(i - 2, j - 2);
        }
        if (!bound.pairs.empty()) {
            out.push_back(std::move(bound));
        }
    }
    return out;
}

std::vector<std::vector<double>> Cuts::dr_min() const {
    return pairwise_min(delta_r_pairs);
}
