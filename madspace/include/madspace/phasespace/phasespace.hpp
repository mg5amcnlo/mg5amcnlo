#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/chili.hpp"
#include "madspace/phasespace/color_ordered_mapping.hpp"
#include "madspace/phasespace/cuts.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/luminosity.hpp"
#include "madspace/phasespace/rambo.hpp"
#include "madspace/phasespace/t_propagator_mapping.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

/**
 * Full @f$2 \to n@f$ phase-space mapping for one diagram topology.
 *
 * Assembles a complete mapping from the elementary building blocks along the
 * recursive decomposition of the tree-level phase space (Sec. 2.2 of [1]),
 *
 * @f[
 *   \int \mathrm{d}\Phi_n =
 *   \Big[\textstyle\prod_i \int \mathrm{d}s_i\Big]
 *   \Big[\textstyle\prod_j \int \mathrm{d}\Phi^{(\phi,t)}_{2,j}\Big]
 *   \Big[\textstyle\prod_k \int \mathrm{d}\Phi^{(\tilde s,t)}_{2,k}\Big]
 *   \Big[\textstyle\prod_l \int \mathrm{d}\Phi_{3,l}\Big]
 *   \Big[\textstyle\prod_m \int \mathrm{d}\Phi^{(\phi,\theta)}_{2,m}\Big].
 * @f]
 *
 * The time-like invariants are sampled with @ref Invariant, the s-channel
 * decays with @ref TwoBodyDecay / @ref ThreeBodyDecay, and the PDF convolution
 * with @ref Luminosity. @ref TChannelMode selects the t-channel strategy. The
 * second constructor takes a flat list of external masses instead of a
 * @ref Topology and is used for the topology-free @ref TChannelMode::rambo and
 * @ref TChannelMode::chili modes.
 *
 * `batch` is the leading batch dimension. `n_out` is the number of outgoing
 * particles.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the unit-hypercube
 *   coordinates.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices. Present only when `discrete_dim()` is nonzero.
 *
 * **Conditions**
 * - `permutation_index` – `int`, shape `(batch,)` – which permutation channel
 *   to use. Present only when more than one permutation is given.
 * - `mirror_index` – `int`, shape `(batch,)` – 1 to take leg 1 from beam 2 and
 *   leg 2 from beam 1. Present only when `mirror_beams` is set; it follows
 *   `permutation_index`.
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n_out + 2, 4)` – all momenta (incoming
 *   beams first, then outgoing).
 * - `x1` – `float`, shape `(batch,)` – first parton momentum fraction (1 for
 *   every event if `leptonic` and for a decay).
 * - `x2` – `float`, shape `(batch,)` – second parton momentum fraction (1 for
 *   every event if `leptonic` and for a decay).
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2)
 */
class PhaseSpaceMapping : public Mapping {
public:
    /// How the space-like (t-channel) part of the phase space is generated.
    enum TChannelMode {
        propagator,   ///< A chain of 2->2 blocks (@ref TPropagatorMapping).
        rambo,        ///< @ref FastRamboMapping on the t-channel legs.
        chili,        ///< Collider coordinates (@ref ChiliMapping).
        color_ordered ///< A color-ordered chain (@ref ColorOrderedMapping).
    };

    /**
     * @param topology        The diagram topology to follow. See @ref Topology.
     * @param cm_energy        Total collision energy.
     * @param leptonic         If true, skip the PDF convolution (no `x1`/`x2`
     *                         sampling).
     * @param invariant_power  Exponent of the `1/s^p` sampling of every
     *                         time-like invariant. See @ref Invariant. The
     *                         chain of @ref TChannelMode::color_ordered takes
     *                         its powers from @p color_ordered.
     * @param t_channel_mode   The @ref TChannelMode strategy.
     * @param cuts             Fiducial cuts to fold into the sampling. See
     *                         @ref Cuts.
     * @param permutations     One outgoing-particle permutation per channel.
     *                         More than one enables the `permutation_index`
     *                         condition.
     * @param color_ordered    Chain order and settings of
     *                         @ref TChannelMode::color_ordered, required there
     *                         and not allowed for the other modes. An unset
     *                         `chain_invariant_power` is 1 with cuts and
     *                         @p invariant_power without. See
     *                         @ref ColorOrderedMapping::Options.
     * @param beam_rapidity    Rapidity of the beams' centre-of-mass frame in the
     *                         lab frame the momenta are returned in:
     *                         0.5 ln(E1 / E2) for beam energies E1 (along +z)
     *                         and E2, with @p cm_energy = 2 sqrt(E1 E2). The
     *                         phase space is generated in the beams' frame and
     *                         boosted into the lab before the cuts, so rapidity
     *                         cuts act in the lab.
     * @param mirror_beams     Add the `mirror_index` condition: for index 1 the
     *                         event is rotated by pi about the x axis in the
     *                         beams' frame (leg 1 from beam 2, leg 2 from beam 1)
     *                         before the lab boost and the cuts. With identical
     *                         beams that is a rotation of the lab event too.
     * @param s_hat_power      Exponent of the `1/s_hat^p` sampling of the
     *                         partonic invariant s_hat in the PDF convolution
     *                         (see @ref Luminosity), in every mode. Unset, it is
     *                         1.2 for @ref TChannelMode::color_ordered when
     *                         masses or cuts bound s_hat from below (as
     *                         AmpliCol), and @p invariant_power otherwise. A
     *                         resonant s_hat is sampled with its Breit-Wigner
     *                         either way.
     */
    PhaseSpaceMapping(
        const Topology& topology,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode t_channel_mode = propagator,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::vector<std::vector<std::size_t>>& permutations = {},
        const std::optional<ColorOrderedMapping::Options>& color_ordered = std::nullopt,
        double beam_rapidity = 0.,
        bool mirror_beams = false,
        std::optional<double> s_hat_power = std::nullopt
    );

    /**
     * @param external_masses  Masses of all external particles, the two
     *                         beams first. A trivial topology is built from
     *                         them.
     * @param cm_energy        Total collision energy.
     * @param leptonic         If true, skip the PDF convolution.
     * @param invariant_power  Exponent of the `1/s^p` invariant sampling. See
     *                         @ref Invariant. The chain of
     *                         @ref TChannelMode::color_ordered takes its
     *                         powers from @p color_ordered.
     * @param mode             The @ref TChannelMode strategy (default
     *                         @ref TChannelMode::rambo).
     * @param cuts             Fiducial cuts. See @ref Cuts.
     * @param color_ordered    Chain order and settings of
     *                         @ref TChannelMode::color_ordered, required there
     *                         and not allowed for the other modes. An unset
     *                         `chain_invariant_power` is 1 with cuts and
     *                         @p invariant_power without. See
     *                         @ref ColorOrderedMapping::Options.
     * @param beam_rapidity    Rapidity of the beams' centre-of-mass frame in the
     *                         lab frame the momenta are returned in:
     *                         0.5 ln(E1 / E2) for beam energies E1 (along +z)
     *                         and E2, with @p cm_energy = 2 sqrt(E1 E2). The
     *                         phase space is generated in the beams' frame and
     *                         boosted into the lab before the cuts, so rapidity
     *                         cuts act in the lab.
     * @param mirror_beams     Add the `mirror_index` condition: for index 1 the
     *                         event is rotated by pi about the x axis in the
     *                         beams' frame (leg 1 from beam 2, leg 2 from beam 1)
     *                         before the lab boost and the cuts. With identical
     *                         beams that is a rotation of the lab event too.
     * @param s_hat_power      Exponent of the `1/s_hat^p` sampling of the
     *                         partonic invariant s_hat in the PDF convolution
     *                         (see @ref Luminosity), in every mode. Unset, it is
     *                         1.2 for @ref TChannelMode::color_ordered when
     *                         masses or cuts bound s_hat from below (as
     *                         AmpliCol), and @p invariant_power otherwise. A
     *                         resonant s_hat is sampled with its Breit-Wigner
     *                         either way.
     */
    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::optional<ColorOrderedMapping::Options>& color_ordered = std::nullopt,
        double beam_rapidity = 0.,
        bool mirror_beams = false,
        std::optional<double> s_hat_power = std::nullopt
    );

    /**
     * Number of continuous unit-hypercube inputs a forward map over @p topology
     * consumes.
     *
     * A 1 -> n decay and a leptonic (fixed-s) 2 -> n collision both have 3n-4
     * degrees of freedom. A hadronic 2 -> n adds the beam momentum fractions,
     * i.e. one further sampled invariant on top of s_hat, giving 3n-2. A
     * hadronic 2 -> 1 collision keeps a single random number, the rapidity of
     * the produced particle, since s_hat is fixed at its mass.
     *
     * @param topology  The decay topology.
     * @param leptonic  If true, skip the PDF convolution.
     * @throws std::invalid_argument for a 2 -> 1 topology without PDFs.
     */
    static std::size_t random_dim_for(const Topology& topology, bool leptonic) {
        if (topology.outgoing_masses().size() == 1 &&
            (leptonic || topology.is_decay())) {
            throw std::invalid_argument(
                "a 2 -> 1 process needs beams with a PDF: with fixed beam "
                "energies s_hat is fixed and there is nothing to integrate"
            );
        }
        return 3 * topology.outgoing_masses().size() -
            ((leptonic || topology.is_decay()) ? 4 : 2);
    }
    /// Number of continuous unit-hypercube inputs consumed by the forward map.
    std::size_t random_dim() const { return random_dim_for(_topology, _leptonic); }
    /// Number of discrete channel choices.
    std::size_t discrete_dim() const override { return _n_discrete; }
    /// Total number of particles, incoming and outgoing.
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + _topology.incoming_masses().size();
    }
    /// Number of permutation channels.
    std::size_t channel_count() const { return _permutations.size(); }
    /// Rapidity of the beams' centre-of-mass frame in the lab frame.
    double beam_rapidity() const { return _beam_rapidity; }
    /// Whether the mapping takes the `mirror_index` condition.
    bool mirror_beams() const { return _mirror_beams; }
    /// The cuts, applied to the lab-frame momenta.
    const Cuts& cuts() const { return _cuts; }
    /// True when the cuts exclude the whole region this mapping samples: a
    /// pair mass cut puts a propagator's floor at or above the upper end of
    /// its on-shell window. Every point then fails the cuts, so the channel
    /// contributes nothing and should be dropped rather than sampled.
    bool empty() const { return _empty; }

private:
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
    // partonic centre-of-mass frame <-> lab frame (partonic boost, mirror,
    // beam_rapidity boost)
    // x1, x2 scaled so that log(x1 / x2) / 2 is the rapidity of the partonic
    // system in the lab frame, which is what the rapidity windows of the
    // decays and the t-channel need (a mirrored event has the opposite sign of
    // the beam rapidity relative to the frame the mapping works in)
    std::pair<Value, Value> lab_fractions(
        FunctionBuilder& fb, Value x1, Value x2, const NamedVector<Value>& conditions
    ) const;
    Value to_lab(
        FunctionBuilder& fb,
        Value momenta,
        Value x1,
        Value x2,
        const NamedVector<Value>& conditions
    ) const;
    Value from_lab(
        FunctionBuilder& fb,
        Value momenta,
        Value x1,
        Value x2,
        const NamedVector<Value>& conditions
    ) const;

    Topology _topology;
    Cuts _cuts;
    double _pi_factors;
    double _sqrt_s_lab;
    double _beam_rapidity;
    bool _mirror_beams;
    bool _leptonic;
    bool _map_luminosity;
    bool _two_to_one;
    // bound on the absolute rapidity of the partonic system, log(x1 / x2) / 2,
    // implied by the cuts; negative when they imply none
    double _y_max_lab = -1.;
    bool _empty = false;
    std::size_t _n_discrete;
    std::vector<Invariant> _s_invariants;
    std::variant<
        TPropagatorMapping,
        FastRamboMapping,
        ChiliMapping,
        ColorOrderedMapping,
        std::monostate>
        _t_mapping;
    std::vector<std::variant<TwoBodyDecay, ThreeBodyDecay, FastRamboMapping>> _s_decays;
    nested_vector2<me_int_t> _permutations;
};

} // namespace madspace
