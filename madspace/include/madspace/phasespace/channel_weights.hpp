#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * Local multi-channel weights from propagator denominators.
 *
 * Implements the single-diagram-enhanced (SDE) multi-channel weights
 * @f$\alpha_i(x)@f$ of Eq. (2.7) of [1]: each channel weight is proportional to
 * the product of the Breit-Wigner propagator denominators of its topology,
 * evaluated at the current phase-space point [2, 3]. Only the forward mappings
 * are needed, not their inverses (Sec. 2.1.1 of [1]).
 *
 * `batch` is the leading batch dimension. `c` indexes the channels.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – all momenta of the event.
 *
 * **Returns**
 * - `channel_weights` – `float`, shape `(batch, c)` – the normalized
 *   per-channel weights @f$\alpha_i(x)@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 * - [2] F. Maltoni, T. Stelzer, "MadEvent: automatic event generation with
 *   MadGraph", https://arxiv.org/abs/hep-ph/0208156
 * - [3] O. Mattelaer, K. Ostrolenk, "Speeding up MadGraph5_aMC@NLO",
 *   https://arxiv.org/abs/2102.00773
 */
class PropagatorChannelWeights : public FunctionGenerator {
public:
    /**
     * @param topologies      One @ref Topology per channel.
     * @param permutations    Per topology, the outgoing-particle permutations;
     *                        each one is a separate output channel.
     * @param channel_indices Per topology, the output channel index of each of
     *                        its permutations.
     */
    PropagatorChannelWeights(
        const std::vector<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
};

/**
 * Splits each channel weight among its on-shell sub-channels.
 *
 * A single decay topology admits several sampling orders, one per set of
 * propagators that can be simultaneously on shell (Sec. 2.2.7 of [1]). This
 * distributes an incoming per-channel weight over those sub-channels in
 * proportion to their on-shell propagator denominators.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – all momenta of the event.
 * - `channel_weights_in` – `float`, shape `(batch, c_in)` – the per-channel
 *   weights to split.
 *
 * **Returns**
 * - `channel_weights_out` – `float`, shape `(batch, c_out)` – the per-sub-channel
 *   weights.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.7)
 */
class SubchannelWeights : public FunctionGenerator {
public:
    /**
     * @param topologies      Per channel, its list of on-shell sub-channel
     *                        @ref Topology "topologies".
     * @param permutations    Per channel, the outgoing-particle permutations.
     * @param channel_indices Per channel, the input channel index of each
     *                        permutation.
     */
    SubchannelWeights(
        const nested_vector2<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

    /// Number of output channels (the sub-channels).
    std::size_t channel_count() const { return _channel_indices.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<me_int_t> _on_shell;
    std::vector<me_int_t> _group_sizes;
    std::vector<me_int_t> _channel_indices;
    std::vector<me_int_t> _subchannel_indices;
};

/**
 * Local multi-channel weights from the sampling densities of the channel
 * mappings.
 *
 * The weight of channel @f$c@f$ for an event of flavour @f$f@f$ is
 * @f$\alpha_c(x) = m_{fc} g_c(x) / \sum_k m_{fk} g_k(x)@f$ [1, 2], where
 * @f$g_k@f$ is the density with which the @ref PhaseSpaceMapping of channel
 * @f$k@f$ samples the event, read off its inverse with uniformly sampled
 * discrete choices, and @f$m_{fk}@f$ is 1 if channel @f$k@f$ integrates
 * flavour @f$f@f$, else 0. The adaptive samplers do not enter, so the weights
 * of one channel do not depend on the state of the others. A channel whose
 * mapping cannot produce the event gets the weight 0. If no channel of the
 * flavour has a valid density, the weights are uniform over these channels.
 *
 * Partner mappings enter the sums over @f$k@f$ like channels but are not
 * channels of this integrand: they are sampled elsewhere, by integrands that
 * weight the same contribution with their own density in the numerator. The
 * weights of the channels then add up to less than one, and the integrand
 * applies them as a factor even with a single channel.
 *
 * Optionally, it also samples a diagram for the event record: diagram
 * @f$d@f$ belongs to channel or partner @f$c_d@f$ and is drawn with a
 * probability proportional to @f$\alpha_{c_d}@f$ among the diagrams of the
 * flavour.
 *
 * `batch` is the leading batch dimension. `c` indexes the channels.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the lab-frame momenta.
 * - `x1`, `x2` – `float`, shape `(batch,)` – the momentum fractions returned
 *   by the mapping that generated the event.
 * - `flavor` – `int`, shape `(batch,)` – the flavour index.
 * - `mirror_index` – `int`, shape `(batch,)` – the orientation of the beams.
 *   Present only if the mappings are built with `mirror_beams`.
 *
 * **Returns**
 * - `channel_weights` – `float`, shape `(batch, c)` – the normalized
 *   per-channel weights, without those of the partners.
 * - `diagram_index` – `int`, shape `(batch,)` – the sampled diagram. Present
 *   only if diagrams are given.
 *
 * **References**
 * - [1] R. Kleiss, R. Pittau, "Weight optimization in multichannel Monte
 *   Carlo", https://arxiv.org/abs/hep-ph/9405257
 * - [2] R. Frederix, T. Vitos, "Leading-colour-based unweighted event
 *   generation for multi-parton tree-level processes",
 *   https://arxiv.org/abs/2409.12128
 */
class MappingChannelWeights : public FunctionGenerator {
public:
    /**
     * @param mappings              The mapping of each channel, without
     *                              permutations. They take the same momenta.
     * @param flavor_masks          Per flavour, 1 for each channel, then
     *                              each partner, that integrates it, else 0.
     * @param diagram_channels      The channel or partner of each diagram,
     *                              the partners numbered after the channels.
     * @param flavor_diagram_masks  Per flavour, 1 for each diagram it can be
     *                              written with, else 0.
     * @param partner_mappings      Mappings that only enter the weights,
     *                              without permutations; they take the same
     *                              momenta.
     */
    MappingChannelWeights(
        const std::vector<PhaseSpaceMapping>& mappings,
        const nested_vector2<double>& flavor_masks,
        const std::vector<me_int_t>& diagram_channels = {},
        const nested_vector2<double>& flavor_diagram_masks = {},
        const std::vector<PhaseSpaceMapping>& partner_mappings = {}
    );

    /// Number of channels.
    std::size_t channel_count() const { return _channel_count; }
    /// Whether partner mappings take part of the weight.
    bool has_partners() const { return _mappings.size() > _channel_count; }
    /// Whether the beams' orientation is an argument.
    bool mirror_beams() const { return _mirror_beams; }
    /// Whether a diagram is sampled.
    bool has_diagrams() const { return !_diagram_channels.empty(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    // the channels, then the partners
    std::vector<PhaseSpaceMapping> _mappings;
    std::size_t _channel_count;
    nested_vector2<double> _flavor_masks;
    std::vector<me_int_t> _diagram_channels;
    nested_vector2<double> _flavor_diagram_masks;
    bool _mirror_beams;
};

} // namespace madspace
