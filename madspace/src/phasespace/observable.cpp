#include "madspace/phasespace/observable.hpp"

#include <ranges>
#include <set>

using namespace madspace;

namespace {

int observable_type(Observable::ObservableOption observable) {
    switch (observable) {
    case Observable::obs_e:
    case Observable::obs_px:
    case Observable::obs_py:
    case Observable::obs_pz:
    case Observable::obs_mass:
    case Observable::obs_pt:
    case Observable::obs_p_mag:
    case Observable::obs_phi:
    case Observable::obs_theta:
    case Observable::obs_y:
    case Observable::obs_y_abs:
    case Observable::obs_eta:
    case Observable::obs_eta_abs:
        return 1;
    case Observable::obs_delta_eta:
    case Observable::obs_delta_phi:
    case Observable::obs_delta_r:
        return 2;
    case Observable::obs_sqrt_s:
        return 0;
    }
}

Value build_observable(
    FunctionBuilder& fb,
    Observable::ObservableOption observable,
    std::vector<Value> momenta
) {
    switch (observable) {
    case Observable::obs_e:
        return fb.obs_e(momenta.at(0));
    case Observable::obs_px:
        return fb.obs_px(momenta.at(0));
    case Observable::obs_py:
        return fb.obs_py(momenta.at(0));
    case Observable::obs_pz:
        return fb.obs_pz(momenta.at(0));
    case Observable::obs_mass:
        return fb.obs_mass(momenta.at(0));
    case Observable::obs_pt:
        return fb.obs_pt(momenta.at(0));
    case Observable::obs_p_mag:
        return fb.obs_p_mag(momenta.at(0));
    case Observable::obs_phi:
        return fb.obs_phi(momenta.at(0));
    case Observable::obs_theta:
        return fb.obs_theta(momenta.at(0));
    case Observable::obs_y:
        return fb.obs_y(momenta.at(0));
    case Observable::obs_y_abs:
        return fb.obs_y_abs(momenta.at(0));
    case Observable::obs_eta:
        return fb.obs_eta(momenta.at(0));
    case Observable::obs_eta_abs:
        return fb.obs_eta_abs(momenta.at(0));
    case Observable::obs_delta_eta:
        return fb.obs_delta_eta(momenta.at(0), momenta.at(1));
    case Observable::obs_delta_phi:
        return fb.obs_delta_phi(momenta.at(0), momenta.at(1));
    case Observable::obs_delta_r:
        return fb.obs_delta_r(momenta.at(0), momenta.at(1));
    case Observable::obs_sqrt_s:
        return fb.obs_sqrt_s(momenta.at(0));
    }
}

std::tuple<nested_vector2<me_int_t>, nested_vector2<me_int_t>, Type> build_indices(
    const std::vector<int>& pids,
    Observable::ObservableOption observable,
    const nested_vector2<int>& select_pids,
    bool sum_momenta,
    bool sum_observable,
    const std::optional<Observable::ObservableOption>& order_observable,
    const std::vector<int>& order_indices,
    bool ignore_incoming
) {
    nested_vector2<int> selected_indices;
    for (auto& selection : select_pids) {
        auto& indices = selected_indices.emplace_back();
        std::size_t i = ignore_incoming ? 2 : 0;
        for (int pid : pids | std::views::drop(i)) {
            if (std::find(selection.begin(), selection.end(), pid) != selection.end()) {
                indices.push_back(i);
            }
            ++i;
        }
    }

    int obs_type = observable_type(observable);
    switch (obs_type) {
    case 0:
        if (selected_indices.size() != 0) {
            throw std::invalid_argument("observable requires no selection of PIDs");
        }
        break;
    case 1:
        if (selected_indices.size() == 0) {
            throw std::invalid_argument(
                "observable requires at least one selection of pids"
            );
        } else if (selected_indices.size() > 1 && !sum_momenta && !sum_observable) {
            throw std::invalid_argument(
                "multiple selected pids only allowed when summing over momenta or "
                "observables"
            );
        }
        break;
    case 2:
        if (selected_indices.size() != 1 && selected_indices.size() != 2) {
            throw std::invalid_argument(
                "observable requires one or two selections of pids"
            );
        }
        if (selected_indices.size() == 1) {
            selected_indices.push_back(selected_indices.at(0));
        }
        break;
    }

    bool empty = false;
    for (auto& indices : selected_indices) {
        if (indices.size() == 0) {
            empty = true;
            break;
        }
    }
    if (empty && obs_type != 0) {
        return {{}, {}, batch_float};
    }

    nested_vector2<me_int_t> ret_order_indices(selected_indices.size());
    if (order_observable) {
        if (observable_type(order_observable.value()) != 1) {
            throw std::invalid_argument(
                "pair-wise or event-level observable cannot be used for sorting"
            );
        }
        if (order_indices.size() != selected_indices.size()) {
            throw std::invalid_argument(
                "an order index must be provided for every item in select_pids"
            );
        }
        for (auto [order_index_in, order_indices_out, indices] :
             zip(order_indices, ret_order_indices, selected_indices)) {
            if (order_index_in == 0) {
                continue;
            }
            order_indices_out = indices;
            if (std::abs(order_index_in) > indices.size()) {
                throw std::invalid_argument(
                    "absolute value of order index must be smaller or equal to number "
                    "of selected PIDs"
                );
            }
            indices = {static_cast<me_int_t>(
                order_index_in < 0
                    ? -order_index_in - 1
                    : indices.size() - order_index_in
            )};
        }
    }

    nested_vector2<me_int_t> ret_indices(selected_indices.size());
    if (selected_indices.size() > 1) {
        // from the cartesian product of the n sets of indices I, J, K, ...
        // find all unique sets {i, j, k, ...} of indices with n elements
        std::vector<std::size_t> product_indices(selected_indices.size());
        std::set<std::set<std::size_t>> found_indices;
        std::set<std::size_t> index_set;
        bool done = false;
        while (!done) {
            index_set.clear();
            std::size_t max_set_size = 0;
            for (auto [prod_index, indices, ord_indices] :
                 zip(product_indices, selected_indices, ret_order_indices)) {
                if (ord_indices.size() > 0) {
                    continue;
                }
                index_set.insert(indices.at(prod_index));
                ++max_set_size;
            }
            bool keep_indices =
                index_set.size() == max_set_size && !found_indices.contains(index_set);
            if (keep_indices) {
                found_indices.insert(index_set);
            }
            bool carry = true;
            for (auto [prod_index, indices, out_indices] :
                 zip(product_indices, selected_indices, ret_indices)) {
                if (keep_indices) {
                    out_indices.push_back(indices.at(prod_index));
                }
                if (carry) {
                    ++prod_index;
                    carry = false;
                }
                if (prod_index == indices.size()) {
                    prod_index = 0;
                    carry = true;
                }
            }
            done = carry;
        }
    } else {
        ret_indices = selected_indices;
    }
    Type ret_type =
        (obs_type == 1 &&
         (ret_indices.size() > 1 || !(sum_momenta || sum_observable))) ||
            obs_type == 2
        ? batch_float_array(ret_indices.at(0).size())
        : batch_float;
    return {ret_indices, ret_order_indices, ret_type};
}

} // namespace

const std::vector<int> Observable::jet_pids{1, 2, 3, 4, -1, -2, -3, -4, 21};
const std::vector<int> Observable::bottom_pids{-5, 5};
const std::vector<int> Observable::lepton_pids{11, 13, 15, -11, -13, -15};
const std::vector<int> Observable::missing_pids{12, 14, 16, -12, -14, -16};
const std::vector<int> Observable::photon_pids{22};

Observable::Observable(
    const std::vector<int>& pids,
    ObservableOption observable,
    const nested_vector2<int>& select_pids,
    bool sum_momenta,
    bool sum_observable,
    const std::optional<ObservableOption>& order_observable,
    const std::vector<int>& order_indices,
    bool ignore_incoming,
    const std::string& name
) :
    Observable(
        build_indices(
            pids,
            observable,
            select_pids,
            sum_momenta,
            sum_observable,
            order_observable,
            order_indices,
            ignore_incoming
        ),
        pids,
        observable,
        sum_momenta,
        sum_observable,
        order_observable,
        ignore_incoming,
        name
    ) {}

Observable::Observable(
    std::tuple<nested_vector2<me_int_t>, nested_vector2<me_int_t>, Type>
        indices_and_type,
    const std::vector<int>& pids,
    ObservableOption observable,
    bool sum_momenta,
    bool sum_observable,
    const std::optional<ObservableOption>& order_observable,
    bool ignore_incoming,
    const std::string& name
) :
    FunctionGenerator(
        "Observable",
        {{"momenta", batch_four_vec_array(pids.size())}},
        {{"observable", std::get<2>(indices_and_type)}}
    ),
    _observable(observable),
    _indices(std::get<0>(indices_and_type)),
    _order_observable(order_observable),
    _order_indices(std::get<1>(indices_and_type)),
    _sum_momenta(sum_momenta),
    _sum_observable(sum_observable),
    _name(name) {}

NamedVector<Value> Observable::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return {
        {"observable", {[&]() -> Value {
             if (not_found()) {
                 return 0.;
             }
             Value momenta = args.at(0);
             int obs_type = observable_type(_observable);
             if (obs_type == 0) {
                 return build_observable(fb, _observable, {momenta});
             }

             ValueVec selected_momenta;
             for (auto [indices, order_indices] : zip(_indices, _order_indices)) {
                 Value sel_indices;
                 if (order_indices.size() > 0) {
                     Value order = fb.argsort(build_observable(
                         fb,
                         _order_observable.value(),
                         {fb.select_vector(momenta, order_indices)}
                     ));
                     sel_indices =
                         fb.select_int(order_indices, fb.select_int(order, indices));
                 } else {
                     sel_indices = indices;
                 }
                 selected_momenta.push_back(fb.select_vector(momenta, sel_indices));
             }

             if (obs_type == 2) {
                 return build_observable(fb, _observable, selected_momenta);
             }

             if (selected_momenta.size() == 1) {
                 if (_sum_momenta) {
                     selected_momenta.at(0) =
                         fb.reduce_sum_vector(selected_momenta.at(0));
                 }
                 Value obs = build_observable(fb, _observable, selected_momenta);
                 if (_sum_observable && !_sum_momenta) {
                     obs = fb.reduce_sum(obs);
                 }
                 return obs;
             }

             if (_sum_momenta) {
                 return build_observable(fb, _observable, {fb.sum(selected_momenta)});
             }

             ValueVec observables;
             for (auto& momentum : selected_momenta) {
                 observables.push_back(build_observable(fb, _observable, {momentum}));
             }
             return fb.sum(observables);
         }()}}
    };
}

bool Observable::not_found() const {
    return observable_type(_observable) != 0 && _indices.size() == 0;
}
