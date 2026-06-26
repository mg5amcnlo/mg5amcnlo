#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/util.hpp"

#include <vector>

namespace madspace {

class Observable : public FunctionGenerator {
public:
    static const std::vector<int> jet_pids;
    static const std::vector<int> bottom_pids;
    static const std::vector<int> lepton_pids;
    static const std::vector<int> missing_pids;
    static const std::vector<int> photon_pids;

    enum ObservableOption {
        obs_e,
        obs_px,
        obs_py,
        obs_pz,
        obs_mass,
        obs_pt,
        obs_p_mag,
        obs_phi,
        obs_theta,
        obs_y,
        obs_y_abs,
        obs_eta,
        obs_eta_abs,
        obs_delta_eta,
        obs_delta_phi,
        obs_delta_r,
        obs_sqrt_s
    };

    Observable(
        const std::vector<int>& pids,
        ObservableOption observable,
        const nested_vector2<int>& select_pids,
        bool sum_momenta = false,
        bool sum_observable = false,
        const std::optional<ObservableOption>& order_observable = std::nullopt,
        const std::vector<int>& order_indices = {},
        bool ignore_incoming = true,
        const std::string& name = ""
    );
    ObservableOption observable() const { return _observable; }
    std::vector<std::size_t> simple_observable_indices() const {
        if (_sum_momenta || _sum_observable || _indices.size() != 1) {
            return {};
        } else {
            return {_indices.at(0).begin(), _indices.at(0).end()};
        }
    }
    std::string name() const { return _name; }
    bool not_found() const;

private:
    Observable(
        std::tuple<nested_vector2<me_int_t>, nested_vector2<me_int_t>, Type>
            indices_and_type,
        const std::vector<int>& pids,
        ObservableOption observable,
        bool sum_momenta,
        bool sum_observable,
        const std::optional<ObservableOption>& order_observable,
        bool ignore_incoming,
        const std::string& name
    );
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    ObservableOption _observable;
    nested_vector2<me_int_t> _indices;
    std::optional<ObservableOption> _order_observable;
    nested_vector2<me_int_t> _order_indices;
    bool _sum_momenta;
    bool _sum_observable;
    std::string _name;
};

} // namespace madspace
