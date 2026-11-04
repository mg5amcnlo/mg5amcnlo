#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/util.hpp"

#include <vector>

namespace madspace {

/**
 * Kinematic observable evaluated for a selection of particles.
 *
 * @p pids lists the PDG ids of all particles of the process, in the order of
 * the momenta. Each entry of @p select_pids is a set of PDG ids that selects
 * matching particles; selections are combined into tuples whose momenta can be
 * summed, and one @ref ObservableOption is evaluated per tuple [1]. Several PDG
 * types may be combined, for example all jet flavours. The available
 * observables are single-momentum functions (`e`, `px`, `py`, `pz`, `mass`, `pt`,
 * `p_mag`, `phi`, `theta`, `y`, `y_abs`, `eta`, `eta_abs`), pair functions
 * (`delta_eta`, `delta_phi`, `delta_r`, `pair_mass`, `sfos_pair_mass`), and
 * the event-level `sqrt_s`. `sfos_pair_mass` is `pair_mass` restricted to
 * same-flavour opposite-sign pairs, i.e. pairs whose PDG ids are `id` and
 * `-id` (the l+ l- pairs of MadEvent's `mmll`); it needs the signed PDG ids
 * of the actual flavours in @p pids and cannot be combined with ordering. The
 * same mechanism defines the observables used by @ref Cuts and
 * @ref ObservableHistograms.
 *
 * `batch` is the leading batch dimension. `n` is `len(pids)` and `k` the
 * number of tuples.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the momenta of all particles.
 *
 * **Returns**
 * - `observable` – `float`, shape `(batch,)` or `(batch, k)` – the observable
 *   value, one per tuple unless @p sum_momenta or @p sum_observable collapses
 *   them. It is 0 if the selection matches no particle.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.4)
 */
class Observable : public FunctionGenerator {
public:
    /// PDG ids of the light quarks and the gluon, grouped as a jet.
    static const std::vector<int> jet_pids;
    /// PDG ids of the b quark.
    static const std::vector<int> bottom_pids;
    /// PDG ids of the charged leptons.
    static const std::vector<int> lepton_pids;
    /// PDG ids treated as missing energy (neutrinos).
    static const std::vector<int> missing_pids;
    /// PDG id of the photon.
    static const std::vector<int> photon_pids;

    /// Kinematic quantity computed by an @ref Observable.
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
        obs_pair_mass,
        obs_sqrt_s,
        obs_sfos_pair_mass
    };

    /**
     * @param pids            PDG ids of all particles of the process, in the
     *                         order of the momenta.
     * @param observable       Which @ref ObservableOption to compute.
     * @param select_pids      One set of PDG ids per selection. Single-momentum
     *                         observables need at least one selection, pair
     *                         observables one or two (one is paired with
     *                         itself) and `sqrt_s` none.
     * @param sum_momenta      Sum the four-momenta within each tuple before
     *                         evaluating the observable.
     * @param sum_observable   Sum the computed observable over all selections.
     * @param order_observable Optional observable used to sort the particles
     *                         of each selection before @p order_indices is
     *                         applied.
     * @param order_indices    Per selection, the position to keep after
     *                         sorting by @p order_observable: positive values
     *                         count from the end of the sorted list (1 is the
     *                         last), negative ones from the start (-1 is the
     *                         first), 0 applies no ordering.
     * @param ignore_incoming  Exclude the two incoming partons from the
     *                         selection.
     * @param name             Label for the returned value.
     */
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
    /// The kinematic quantity this observable computes.
    ObservableOption observable() const { return _observable; }
    /// Per-selection lists of particle indices the observable is evaluated on.
    const nested_vector2<me_int_t>& indices() const { return _indices; }
    /// Whether the tuple momenta are summed before evaluation.
    bool sum_momenta() const { return _sum_momenta; }
    /// Whether the selection is sorted first (e.g. the leading jet). The
    /// entries of indices() are then positions in the sorted list, not
    /// particle indices.
    bool ordered() const { return _order_observable.has_value(); }
    /// Flat particle-index list for a single ungrouped, unsummed selection;
    /// empty otherwise.
    std::vector<std::size_t> simple_observable_indices() const {
        if (_sum_momenta || _sum_observable || _indices.size() != 1) {
            return {};
        } else {
            return {_indices.at(0).begin(), _indices.at(0).end()};
        }
    }
    /// The label passed to the constructor.
    std::string name() const { return _name; }
    /// True when a selection matches no particle, so the observable is 0.
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
