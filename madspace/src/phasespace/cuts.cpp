#include "madspace/phasespace/cuts.hpp"

#include "madspace/compgraphs/type.hpp"

using namespace madspace;

Cuts::Cuts(const std::vector<CutItem>& cut_data) :
    FunctionGenerator(
        "Cuts", cut_data.at(0).observable.arg_types(), {{"mask", batch_float}}
    ),
    _cut_data(cut_data) {}

Cuts::Cuts(std::size_t particle_count) :
    FunctionGenerator(
        "Cuts",
        {{"momenta", batch_four_vec_array(particle_count)}},
        {{"mask", batch_float}}
    ) {}

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
    return {{"mask", fb.product(weights)}};
}

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
        for (std::size_t index : item.observable.simple_observable_indices()) {
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
        for (std::size_t index : item.observable.simple_observable_indices()) {
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

std::vector<std::vector<double>> Cuts::m_inv_min() const {
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.));
    for (auto& item : _cut_data) {
        if (item.observable.observable() != Observable::obs_mass) {
            continue;
        }
        const auto& idx = item.observable.indices();
        if (!item.observable.sum_momenta() || idx.size() != 1 ||
            idx.at(0).size() != 2) {
            continue;
        }
        std::size_t i = idx.at(0).at(0), j = idx.at(0).at(1);
        if (i < 2 || j < 2) {
            continue;
        }
        i -= 2;
        j -= 2;
        if (i < n && j < n && item.min > m.at(i).at(j)) {
            m.at(i).at(j) = item.min;
            m.at(j).at(i) = item.min;
        }
    }
    return m;
}

std::vector<std::vector<double>> Cuts::dr_min() const {
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<std::vector<double>> dr(n, std::vector<double>(n, 0.));
    for (auto& item : _cut_data) {
        if (item.observable.observable() != Observable::obs_delta_r) {
            continue;
        }
        const auto& idx = item.observable.indices();
        if (idx.size() != 2) {
            continue;
        }
        for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
            std::size_t i = idx.at(0).at(k), j = idx.at(1).at(k);
            if (i < 2 || j < 2) {
                continue;
            }
            i -= 2;
            j -= 2;
            if (i < n && j < n && item.min > dr.at(i).at(j)) {
                dr.at(i).at(j) = item.min;
                dr.at(j).at(i) = item.min;
            }
        }
    }
    return dr;
}
