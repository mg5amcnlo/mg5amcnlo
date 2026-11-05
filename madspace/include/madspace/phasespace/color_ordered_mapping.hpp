#pragma once

#include <array>
#include <optional>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

/**
 * Leading-color phase-space mapping following a color-ordered particle chain.
 *
 * Generates the full final state as a chain of @f$(\tilde s, t)@f$ two-particle
 * blocks (@ref TwoToThreeParticleScattering) laid out along a color-ordered
 * permutation of the particles, peeling one on-shell momentum off at a time
 * from each beam, with a @ref TwoToTwoParticleScattering or @ref DoubleT
 * central block joining the two sides [1] (no central block if all outgoing
 * particles sit on one side). It encodes no specific diagram
 * topology. The `discrete_i` inputs select, per @f$2 \to 3@f$ peel, which of
 * the two branches to take. See Sec. 2.2.5 of [1], following [2].
 *
 * `batch` is the leading batch dimension. `n_out` is the number of outgoing
 * particles and `n = n_out + 2` the total.
 *
 * **Inputs**
 * - `random_i` - `float`, shape `(batch,)` – the continuous random numbers
 *   (`random_dim()` of them).
 * - `discrete_i` - `int`, shape `(batch,)` – the two-solution choices
 *   (`discrete_dim()` of them).
 *
 * **Conditions**
 * - `com_energy` - `float`, shape `(batch,)` – total collision energy.
 * - `mass_i` - `float`, shape `(batch,)` – the `n_out` outgoing masses.
 * - `x1`, `x2` - `float`, shape `(batch,)` – beam momentum fractions. Present
 *   only when a rapidity bound is used (see the constructor).
 *
 * **Outputs**
 * - `momentum_i` - `float`, shape `(batch, 4)` – the `n` momenta (incoming
 *   beams first, then outgoing).
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.5)
 * - [2] R. Frederix, T. Vitos, "Leading-colour-based unweighted event
 *   generation for multi-parton tree-level processes",
 *   https://arxiv.org/abs/2409.12128
 */
class ColorOrderedMapping : public Mapping {
public:
    /**
     * Settings of the chain. Besides these, the chain follows AmpliCol's
     * gen23 [2]: a side of exactly two particles (with the other side not
     * empty) is one 2->3 step with the opposite beam as the previous emission,
     * the composite and rest masses are sampled as `1/sqrt(s)` (flat in the
     * mass), and the s23 random numbers of the 2->3 steps are remapped with the
     * arcsine map of @ref TwoToThreeParticleScattering.
     */
    struct Options {
        /// 0-based permutation of `{0, ..., n-1}` (`n = n_out + 2`) giving the
        /// chain order; particles 0 and 1 are the incoming beams. See
        /// @ref Topology.
        std::vector<std::size_t> color_order;
        /// Exponent of the `1/|x|^p` sampling of the momentum transfers and
        /// the s23 invariants of the chain; see @ref Invariant. Unset: 1 with
        /// cuts (AmpliCol) and 0.8 without, where these ranges reach zero and
        /// `1/|x|` would pile the points up at the regulator of
        /// @ref Invariant.
        std::optional<double> chain_invariant_power = std::nullopt;
        /// In every 2->3 step, keep the azimuthal solution that passes the pt
        /// cuts, and pick with the discrete choice only when both pass, as
        /// AmpliCol does. The weight assumes uniformly sampled discrete
        /// choices: turn it off when they are sampled with a learned
        /// distribution.
        bool cut_aware_branch = true;
    };

    /**
     * @param options           Chain order and settings; see @ref Options.
     * @param pt_min            Per-outgoing-particle minimum transverse
     *                          momentum; empty disables the cut. See @ref Cuts.
     * @param m_inv_min         Symmetric matrix of minimum pair invariant
     *                          masses; empty disables the cut. See @ref Cuts.
     * @param dr_min            Symmetric matrix of minimum pair `delta_r`
     *                          separations; empty disables the cut. See
     *                          @ref Cuts.
     * @param y_max             Per-outgoing-particle bound on the absolute lab
     *                          rapidity, negative (or empty) for none. Used
     *                          where a block scatters the two beams in the
     *                          partonic centre-of-mass frame, so that its
     *                          momentum transfers fix the light-cone
     *                          components of what it emits: the central 2->2
     *                          (bounds of the two sides), the double-t central
     *                          block (|t2| at fixed |t1|) and the first peel of
     *                          a single chain. A side or rest system is bounded
     *                          when each of its particles is. The mapping then
     *                          takes the beam momentum fractions `x1`, `x2`
     *                          (`float`, shape `(batch,)`) as two further
     *                          conditions.
     */
    ColorOrderedMapping(
        const Options& options,
        const std::vector<double>& pt_min = {},
        const std::vector<std::vector<double>>& m_inv_min = {},
        const std::vector<std::vector<double>>& dr_min = {},
        const std::vector<double>& y_max = {}
    );

    /// Number of continuous unit-hypercube inputs consumed by the forward map.
    std::size_t random_dim() const { return _random_dim; }
    /// Number of discrete two-solution choices (one per 2->3 peel).
    std::size_t discrete_dim() const override { return _discrete_dim; }
    /// Whether rapidity bounds narrow a block (and the mapping therefore takes
    /// `x1`, `x2` as conditions).
    bool has_rapidity_window() const { return _rapidity_window; }

private:
    // Rapidity bounds of the blocks that scatter the two beams (negative:
    // none). central1/central2 are the two sides of the central 2->2, or the
    // single particle and the recoil of the double-t block; chain_* belong to
    // the first peel of a single chain.
    struct RapidityBounds {
        double central1 = -1., central2 = -1.;
        double chain_rest = -1., chain_peeled = -1., chain_beam_sign = 1.;
        bool double_t = false;
        bool active() const {
            return central1 >= 0. || central2 >= 0. || chain_rest >= 0. ||
                chain_peeled >= 0.;
        }
    };
    static RapidityBounds rapidity_bounds(
        const std::vector<std::size_t>& color_order, const std::vector<double>& y_max
    );
    // The public constructor with the rapidity bounds worked out once.
    ColorOrderedMapping(
        const RapidityBounds& bounds,
        const Options& options,
        const std::vector<double>& pt_min,
        const std::vector<std::vector<double>>& m_inv_min,
        const std::vector<std::vector<double>>& dr_min
    );

    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    // pt^2 of outgoing particle i (0 if no pt cut on it).
    double pt2(std::size_t i) const;
    // Cut-derived invariant-mass^2 floor (gen23 invm_min, without the mass^2
    // term, which is applied separately) for a subset of outgoing particles.
    // Returns 0 when cuts are disabled or the subset has fewer than 2 members.
    double cut_floor(const std::vector<std::size_t>& subset) const;
    // Floor on the squared mass of a set of outgoing particles (masses and
    // cuts), and floor on its transverse mass: the sum of those of the
    // particles with their pt cuts, at least the square root of the former.
    std::array<Value, 2> set_floors(
        FunctionBuilder& fb, const std::vector<std::size_t>& set, const ValueVec& m_out
    ) const;
    // Range of the squared mass of the rest system of a walk over the set s
    // after its peel j, from a system of mass prev_mass.
    std::array<Value, 2> rest_mass_range(
        FunctionBuilder& fb,
        const std::vector<std::size_t>& s,
        std::size_t j,
        Value prev_mass,
        Value e_cm,
        const ValueVec& m_out
    ) const;

    // 0-indexed outgoing-particle indices (values in {0,...,n_out-1}).
    // _set1 contains the outgoing particles attached to beam 0's side,
    // _set2 those attached to beam 1's side, in peel order.
    std::vector<std::size_t> _set1;
    std::vector<std::size_t> _set2;
    std::size_t _n_out;
    std::size_t _random_dim;
    // Number of discrete two-solution choices (one per 2->3 peel). They are
    // separate batch_int inputs after the continuous randoms.
    std::size_t _discrete_dim;
    // True iff exactly one of (set1, set2) has size 1 (and the other >= 2).
    // In that case the central block is DoubleT instead of 2->2.
    bool _use_double_t;
    // True iff one of (set1, set2) is empty, i.e. particles 0 and 1 are
    // adjacent in the color order and all outgoing particles sit on one side.
    // In that case there is no central block at all: the full final state is
    // produced as a single t-channel chain seeded directly off the beams.
    bool _use_single_chain;

    // Cut configuration (empty => all bounds resolve to 0 = no cut).
    std::vector<double> _pt_min;
    std::vector<std::vector<double>> _m_inv_min;
    std::vector<std::vector<double>> _dr_min;
    bool _has_cut;
    bool _rapidity_window;

    Invariant _mass_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
    // first peel of a single chain, whose incoming momenta are the beams
    TwoToTwoParticleScattering _chain_scattering;
    TwoToThreeParticleScattering _two_to_three;
    // the 2->3 step of a side of two particles, with an incoming recoil
    TwoToThreeParticleScattering _two_to_three_beam;
    DoubleT _double_t;
};

} // namespace madspace
