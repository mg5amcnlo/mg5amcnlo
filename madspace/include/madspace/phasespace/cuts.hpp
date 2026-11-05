#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/observable.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace madspace {

/**
 * Fiducial phase-space cuts, as a compute-graph acceptance mask.
 *
 * Builds a per-event `mask` (1 for events passing every cut, 0 otherwise) from
 * a list of @ref Cuts::CutItem "cut items" [1]. Each item evaluates one
 * @ref Observable for its particle selection and requires the result to lie in
 * `[min, max]`. When the selection yields several values, @ref CutMode decides
 * whether all or any of them must pass. See Sec. 3.2.4 of [1]. An optional
 * @ref CustomFunction of type CustomFunction::cut multiplies its value into the
 * mask.
 *
 * The accessors re-expose the active lower bounds in the layout the
 * phase-space mappings consume. `pt_min()` and `eta_max()` are indexed by
 * outgoing particle, with index 0 the first outgoing particle and the two
 * beams excluded. `m_inv_min()` and `dr_min()` are symmetric `n_out * n_out`
 * matrices indexed by outgoing-particle pair. A bound of `0`, or infinity for
 * `eta_max()`, means the cut is inactive. Only bounds that every selected
 * object must satisfy are reported: an ordered selection (a rank, not a
 * particle) or a CutMode::any cut over several objects stays a filter.
 *
 * `batch` is the leading batch dimension. `n_particles` counts the incoming and
 * outgoing particles.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n_particles, 4)` – the event momenta.
 *
 * **Returns**
 * - `mask` – `float`, shape `(batch,)` – 1 if every cut is satisfied, else 0.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.4)
 */
class Cuts : public FunctionGenerator {
public:
    /// Whether all or at least one of a multi-object selection must pass.
    enum CutMode { any, all };
    /// One entry in a @ref Cuts list: an @ref Observable and its allowed range.
    struct CutItem {
        /// The observable to test.
        Observable observable;
        /// Lower bound (default: none).
        double min = -std::numeric_limits<double>::infinity();
        /// Upper bound (default: none).
        double max = std::numeric_limits<double>::infinity();
        /// Whether all or any selected objects must satisfy the bound.
        CutMode mode = CutMode::all;
    };

    /// @param cut_data    The cut items to apply.
    /// @param custom_cut  Optional user-supplied cut function.
    Cuts(
        const std::vector<CutItem>& cut_data,
        const std::optional<CustomFunction>& custom_cut = std::nullopt
    );
    /// Build a mask with no cut items.
    /// @param particle_count  Number of external particles.
    /// @param custom_cut      Optional user-supplied cut function.
    Cuts(
        std::size_t particle_count,
        const std::optional<CustomFunction>& custom_cut = std::nullopt
    );
    /// Largest required partonic center-of-mass energy, or 0 if unconstrained.
    double sqrt_s_min() const;
    /// Per-outgoing-particle maximum pseudorapidity (infinity where inactive).
    std::vector<double> eta_max() const;
    /// Per-outgoing-particle minimum transverse momentum (0 where inactive).
    std::vector<double> pt_min() const;
    /// Symmetric matrix of minimum pair invariant masses (0 where inactive).
    std::vector<std::vector<double>> m_inv_min() const;
    /// Symmetric matrix of minimum pair `delta_r` separations (0 where inactive).
    std::vector<std::vector<double>> dr_min() const;

    /// A pair mass cut with CutMode::any over several pairs: at least one of
    /// `pairs` (outgoing indices, beams excluded) has invariant mass >= `min`.
    struct PairMassAny {
        /// The pairs, as outgoing indices; one of them has to pass.
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        /// Lower bound on the invariant mass of the passing pair.
        double min;
    };
    /// The CutMode::any pair mass cuts over several pairs, which bound no pair
    /// in particular and are therefore absent from m_inv_min().
    std::vector<PairMassAny> pair_mass_any_min() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::vector<double>> pairwise_min(
        const std::function<
            std::vector<std::pair<std::size_t, std::size_t>>(const Observable&)>& pairs
    ) const;

    std::vector<CutItem> _cut_data;
    std::optional<CustomFunction> _custom_cut;
};

} // namespace madspace
