#pragma once

#include "madspace/phasespace/channel_weight_network.hpp"
#include "madspace/phasespace/channel_weights.hpp"
#include "madspace/phasespace/cross_section.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/discrete_flow.hpp"
#include "madspace/phasespace/discrete_sampler.hpp"
#include "madspace/phasespace/flow.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/phasespace.hpp"
#include "madspace/phasespace/unweighter.hpp"
#include "madspace/phasespace/vegas.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * The full single-channel integrand, from random numbers to a weighted event.
 *
 * Chains all the pieces of a leading-order calculation into one
 * @ref FunctionGenerator. It draws the unit-hypercube numbers, optionally
 * reshapes them with an adaptive @ref VegasMapping or @ref Flow, runs the
 * @ref PhaseSpaceMapping, evaluates the @ref DifferentialCrossSection (matrix
 * element, PDFs, scales), applies the @ref Cuts and the multi-channel weights,
 * and returns the event weight with the momenta and the sampled discrete
 * quantities [1]. This is the object that is integrated and unweighted.
 *
 * @warning For internal use within MadGraph. Its interface (in particular the
 *          long constructor and the argument / return layout) is not part of
 *          the stable public API and may change without notice.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `batch_size` – the number of events to generate.
 *
 * **Returns** (the common set; @p madnis_training selects a different set for
 * the training loss)
 * - `weight` – `float`, shape `(batch,)` – the event weight.
 * - `momenta` – `float`, shape `(batch, particle_count, 4)` – the momenta.
 * - `color_index`, `helicity_index`, `diagram_index`, `flavor_index` – `int`,
 *   shape `(batch,)` – the sampled discrete quantities. Indices the matrix
 *   element does not return are -1. `subprocess_index` (`int`) follows when
 *   @p flavor_diff_xs_indices is given.
 * - `ren_scale`, `alpha_qcd`, `x1`, `x2`, `fact_scale1`, `fact_scale2` –
 *   `float`, shape `(batch,)` – scales and momentum fractions (`x1`, `x2` and
 *   the factorisation scales only for beams with a PDF). Depending on the
 *   energy scale and @p partial_weights, `partial_weight_product`,
 *   `cluster_scales`, `alphas_scales` and the `ladder_*` arrays follow.
 * - `random` – `float`, shape `(batch, mapping.random_dim())` – the underlying
 *   random numbers.
 * - `matrix_element` – `float`, shape `(batch,)` – the squared matrix element,
 *   present only if the differential cross sections return it, followed by
 *   `matrix_element_flavor` – `int` – the flavor index passed to it, and with
 *   several differential cross sections `matrix_element_index` – `int` – the
 *   index of the one evaluated.
 * - `bias` – `float`, shape `(batch,)` – the event bias, present only with a
 *   custom bias. `weight` then includes the bias factor.
 *
 * Only `matrix_element_out` is required from the matrix element.
 * `diagram_amp2_out` is needed for diagram-based channel weights, i.e. with
 * several channels and neither @p prop_chan_weights nor
 * @p mapping_chan_weights.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class Integrand : public FunctionGenerator {
public:
    /// The optional adaptive reparametrization of the continuous latent space.
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping, Flow>;
    /// The optional adaptive sampler for a set of discrete choices.
    using AdaptiveDiscrete =
        std::variant<std::monostate, DiscreteSampler, DiscreteFlow>;
    /// The optional parton density: an LHAPDF grid or a user-supplied library.
    using PdfSource = std::variant<std::monostate, PdfGrid, CustomPdf>;

    /// The @ref MatrixElement inputs an `Integrand` supplies.
    inline static const std::vector<MatrixElement::MatrixElementInput>
        matrix_element_inputs = {
            MatrixElement::momenta_in,
            MatrixElement::alpha_s_in,
            MatrixElement::flavor_in,
            MatrixElement::random_color_in,
            MatrixElement::random_helicity_in,
            MatrixElement::random_diagram_in,
        };
    /// The @ref MatrixElement outputs an `Integrand` can use.
    inline static const std::vector<MatrixElement::MatrixElementOutput>
        matrix_element_outputs = {
            MatrixElement::matrix_element_out,
            MatrixElement::diagram_amp2_out,
            MatrixElement::color_index_out,
            MatrixElement::helicity_index_out,
            MatrixElement::diagram_index_out,
        };

    /**
     * @param mapping                          The phase-space mapping. Its
     *                                          discrete choices, if any, are
     *                                          sampled uniformly.
     * @param diff_xs                           The differential cross sections;
     *                                          @p flavor_diff_xs_indices picks
     *                                          one per flavour when several.
     * @param adaptive_map                      Optional continuous adaptive
     *                                          sampler (@ref VegasMapping or
     *                                          @ref Flow).
     * @param discrete_sym                      Optional adaptive sampler for
     *                                          the permutation channel.
     * @param discrete_flavor                   Optional adaptive sampler for
     *                                          the flavour choice.
     * @param pid_options                       Allowed initial-parton flavour
     *                                          combinations.
     * @param pdf_grid                          Optional shared @ref PdfGrid or
     *                                          @ref CustomPdf.
     * @param running_coupling                  Shared @ref RunningCoupling
     *                                          (dereferenced unconditionally).
     * @param energy_scale                      Shared @ref EnergyScale
     *                                          (dereferenced unconditionally).
     * @param prop_chan_weights                 Optional
     *                                          @ref PropagatorChannelWeights.
     * @param subchan_weights                   Optional @ref SubchannelWeights.
     * @param chan_weight_net                   Optional
     *                                          @ref ChannelWeightNetwork.
     * @param first_chan_weight_remap           Remap applied to the channel
     *                                          weights before the network.
     * @param first_remapped_chan_count         Channel count after that remap.
     * @param second_chan_weight_remap          Remap applied before the network,
     *                                          after the subchannel weights.
     * @param second_remapped_chan_count        Channel count after that remap.
     * @param madnis_training                   If true, return the quantities
     *                                          needed for the @ref MadnisLoss
     *                                          instead of a finished event.
     * @param drop_cuts_and_rescale             If true, return only the events
     *                                          that pass the cuts, and divide
     *                                          adaptive_prob by the acceptance
     *                                          fraction.
     * @param partial_weights                   If true, emit the product of the
     *                                          beam densities (partial_weight_product)
     *                                          for the systematics.
     *                                          Ignored under MLM pdf
     *                                          reweighting, whose densities are
     *                                          not at the factorisation scale.
     * @param channel_indices                   Global indices of this
     *                                          integrand's channels.
     * @param active_flavors                    For each permutation channel,
     *                                          the flavour indices it covers.
     * @param flavor_remap                      Per flavour index, the flavour
     *                                          index passed to @p diff_xs.
     * @param flavor_factors                    Per-flavour multiplicative
     *                                          weight factors.
     * @param flavor_mirror                     Per-flavour beam-swap flags.
     * @param flavor_diff_xs_indices            Per flavour, the index into
     *                                          @p diff_xs.
     * @param flavor_subproc_indices            Per-flavour sub-process indices
     *                                          (output subprocess_index).
     * @param flavor_per_subproc_remap          Per-flavour index written as
     *                                          flavor_index in the output.
     * @param compressed_channel_weight_count   Channel-weight entries kept per
     *                                          event.
     * @param custom_bias                       Optional user-supplied event
     *                                          bias of type CustomFunction::bias.
     *                                          Its PDG-id options must match
     *                                          @p pid_options.
     * @param pdf_grid2                         PDF of the second beam when it
     *                                          differs from @p pdf_grid (which
     *                                          is then the first beam's). Its
     *                                          globals must be registered with
     *                                          the prefix @ref pdf2_prefix.
     *                                          Not allowed with a @ref CustomPdf,
     *                                          which receives the beam index.
     * @param mapping_chan_weights              Optional
     *                                          @ref MappingChannelWeights, in
     *                                          place of @p prop_chan_weights.
     *                                          Its sampled diagram is the
     *                                          diagram index of the event when
     *                                          the matrix element returns none.
     * @param mlm_history_qcd_power             Power of alpha_s in |M|^2 when it
     *                                          is the same for every diagram,
     *                                          -1 otherwise. With an MLM history
     *                                          picked per diagram, the matrix
     *                                          element at the scales of that
     *                                          history is then the first
     *                                          evaluation rescaled by the alpha_s
     *                                          and PDF ratios instead of a second
     *                                          evaluation.
     */
    Integrand(
        const PhaseSpaceMapping& mapping,
        const std::vector<DifferentialCrossSection>& diff_xs,
        const AdaptiveMapping& adaptive_map = std::monostate{},
        const AdaptiveDiscrete& discrete_sym = std::monostate{},
        const AdaptiveDiscrete& discrete_flavor = std::monostate{},
        const nested_vector2<me_int_t>& pid_options = {},
        const PdfSource& pdf_grid = std::monostate{},
        const std::optional<RunningCoupling>& running_coupling = std::nullopt,
        const std::optional<EnergyScale>& energy_scale = std::nullopt,
        const std::optional<PropagatorChannelWeights>& prop_chan_weights = std::nullopt,
        const std::optional<SubchannelWeights>& subchan_weights = std::nullopt,
        const std::optional<ChannelWeightNetwork>& chan_weight_net = std::nullopt,
        const nested_vector2<me_int_t>& first_chan_weight_remap = {},
        std::size_t first_remapped_chan_count = 0,
        const std::vector<me_int_t>& second_chan_weight_remap = {},
        std::size_t second_remapped_chan_count = 0,
        bool madnis_training = false,
        bool drop_cuts_and_rescale = false,
        bool partial_weights = false,
        const std::vector<std::size_t>& channel_indices = {},
        const nested_vector2<std::size_t>& active_flavors = {},
        const std::vector<std::size_t>& flavor_remap = {},
        const std::vector<double>& flavor_factors = {},
        const std::vector<bool>& flavor_mirror = {},
        const std::vector<std::size_t>& flavor_diff_xs_indices = {},
        const std::vector<std::size_t>& flavor_subproc_indices = {},
        const std::vector<std::size_t>& flavor_per_subproc_remap = {},
        std::size_t compressed_channel_weight_count = 50,
        const std::optional<CustomFunction>& custom_bias = std::nullopt,
        const std::optional<PdfGrid>& pdf_grid2 = std::nullopt,
        const std::optional<MappingChannelWeights>& mapping_chan_weights = std::nullopt,
        int mlm_history_qcd_power = -1
    );
    /// Globals prefix under which the second beam's PDF grid is registered
    /// (PdfGrid::initialize_globals) when `pdf_grid2` is given.
    inline static const std::string pdf2_prefix = "beam2";
    /// Total number of particles, incoming and outgoing.
    std::size_t particle_count() const { return _mapping.particle_count(); }
    /// Whether the integrand emits the @ref MadnisLoss training quantities.
    bool madnis_training() const { return _madnis_training; }
    /// Global name of the VEGAS grid, if an adaptive @ref VegasMapping is used.
    std::optional<std::string> vegas_grid_name() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->grid_name();
        } else {
            return std::nullopt;
        }
    }
    /// Dimension of the adaptive @ref VegasMapping, or 0 if none.
    std::size_t vegas_dimension() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->dimension();
        } else {
            return 0;
        }
    }
    /// Bin count of the adaptive @ref VegasMapping, or 0 if none.
    std::size_t vegas_bin_count() const {
        if (auto vegas = std::get_if<VegasMapping>(&_adaptive_map)) {
            return vegas->bin_count();
        } else {
            return 0;
        }
    }
    /// The phase-space mapping.
    const PhaseSpaceMapping& mapping() const { return _mapping; }
    /// The differential cross sections.
    const std::vector<DifferentialCrossSection>& diff_xs() const { return _diff_xs; }
    /// The continuous adaptive sampler.
    const AdaptiveMapping& adaptive_map() const { return _adaptive_map; }
    /// The permutation-channel adaptive sampler.
    const AdaptiveDiscrete& discrete_sym() const { return _discrete_sym; }
    /// The flavour-choice adaptive sampler.
    const AdaptiveDiscrete& discrete_flavor() const { return _discrete_flavor; }
    /// The shared energy scale, if any.
    const std::optional<EnergyScale>& energy_scale() const { return _energy_scale; }
    /// The propagator-based channel weights, if any.
    const std::optional<PropagatorChannelWeights>& prop_chan_weights() const {
        return _prop_chan_weights;
    }
    /// The mapping-density channel weights, if any.
    const std::optional<MappingChannelWeights>& mapping_chan_weights() const {
        return _mapping_chan_weights;
    }
    /// The channel-weight network, if any.
    const std::optional<ChannelWeightNetwork>& chan_weight_net() const {
        return _chan_weight_net;
    }
    /// Number of random numbers per event, including those for the permutation,
    /// flavour and beam-orientation choices (not the mapping's own discrete ones).
    const std::size_t random_dim() const { return _random_dim; }
    /// The dimensions of the latent inputs and, per input, whether it is a
    /// float (true) or an integer index (false).
    std::tuple<std::vector<std::size_t>, std::vector<bool>> latent_dims() const;
    /// Global indices of this integrand's channels.
    const std::vector<me_int_t>& channel_indices() const { return _channel_indices; }
    /// Flavour indices this integrand covers.
    const std::vector<std::size_t>& active_flavors() const { return _active_flavors; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
    NamedVector<Type> compute_channel_part_ret_types() const;
    // Whether build_channel_part multiplies anything into weight_after_cuts
    // (flavor or mirror sampling, merging weights, scale veto). Otherwise it
    // is one and the common part leaves it out.
    bool has_channel_weight_after_cuts() const;
    NamedVector<Value>
    build_channel_part(FunctionBuilder& fb, const NamedVector<Value>& args) const;
    NamedVector<Value>
    build_common_part(FunctionBuilder& fb, const NamedVector<Value>& channel_out) const;
    // The beam densities at the given scales, for the sampled flavour.
    std::array<Value, 2> evaluate_pdfs(
        FunctionBuilder& fb,
        const NamedVector<Value>& scales,
        const std::array<Value, 2>& x,
        Value flavor_id
    ) const;
    // The factors an MLM clustering puts on the event weight: merging cut,
    // alpha_s and pdf reweighting, each with its veto. With `ladder`, also the
    // pdf reweighting ladder (see emits_pdf_ladder in integrand.cpp).
    ValueVec mlm_weights(
        FunctionBuilder& fb,
        const NamedVector<Value>& scales,
        const std::array<Value, 2>& x,
        Value flavor_id,
        NamedVector<Value>* ladder = nullptr
    ) const;

    PhaseSpaceMapping _mapping;
    std::vector<DifferentialCrossSection> _diff_xs;
    AdaptiveMapping _adaptive_map;
    AdaptiveDiscrete _discrete_sym;
    AdaptiveDiscrete _discrete_flavor;
    nested_vector2<me_int_t> _pid_options;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    // with different PDFs per beam: each leg's density from the other beam,
    // for the events whose orientation is mirrored
    std::array<std::optional<PartonDensity>, 2> _pdfs_swapped;
    std::array<std::vector<me_int_t>, 2> _pdf_indices;
    // Density over every flavour a beam line can have on the MLM clustering
    // ladder, picked per event by index.
    std::optional<PartonDensity> _pdf_rw;
    // Index into _pdf_rw for (flavour class from the mlm kernel, flavour option),
    // flattened as class * option_count + option; offsets hold class * option_count.
    std::vector<me_int_t> _pdf_rw_table;
    std::vector<me_int_t> _pdf_rw_class_offsets;
    // pdg id of each parton of _pdf_rw, as a float for gather
    std::vector<double> _pdf_rw_pdgs;
    std::optional<RunningCoupling> _running_coupling;
    std::optional<EnergyScale> _energy_scale;
    // One row with a 1 for every diagram that has its own MLM clustering history.
    nested_vector2<double> _mlm_diagram_mask;
    // MLMClustering::diagram_start_states padded to the diagram count.
    std::vector<me_int_t> _mlm_start_states;
    // see mlm_history_qcd_power in the constructor; -1 re-evaluates
    int _mlm_history_qcd_power = -1;
    // abs pdg id of every external leg per flavour option, row-major. Empty when
    // the clustering does not depend on the flavours.
    std::vector<me_int_t> _mlm_leg_flavors;
    std::optional<PropagatorChannelWeights> _prop_chan_weights;
    std::optional<MappingChannelWeights> _mapping_chan_weights;
    std::optional<SubchannelWeights> _subchan_weights;
    std::optional<ChannelWeightNetwork> _chan_weight_net;
    nested_vector2<me_int_t> _first_chan_weight_remap;
    me_int_t _first_remapped_chan_count;
    std::vector<me_int_t> _second_chan_weight_remap;
    me_int_t _second_remapped_chan_count;
    std::size_t _compressed_channel_weight_count;
    bool _madnis_training;
    bool _drop_cuts_and_rescale;
    bool _partial_weights;
    std::vector<me_int_t> _channel_indices;
    me_int_t _random_dim;
    std::size_t _latent_dim;
    std::vector<std::size_t> _active_flavors;
    nested_vector2<double> _active_flavors_mask;
    std::vector<me_int_t> _flavor_remap;
    std::vector<double> _flavor_factors;
    std::vector<double> _flavor_mirror_factors;
    bool _has_mirror;
    NamedVector<Type> _channel_part_ret_types;
    std::vector<me_int_t> _flavor_diff_xs_indices;
    std::vector<me_int_t> _flavor_subproc_indices;
    std::vector<me_int_t> _flavor_per_subproc_remap;
    std::optional<CustomFunction> _custom_bias;

    friend class IntegrandProbability;
    friend class IntegrandChannelPart;
    friend class IntegrandCommonPart;
    friend class IntegrandConcatenator;
    friend class MultiChannelIntegrand;
};

/**
 * The channel-dependent first half of an @ref Integrand.
 *
 * `madspace` splits the integrand so the parts that differ between channels
 * (the phase-space mapping and adaptive samplers) run per channel, while the
 * shared part (the matrix element) runs once on the merged batch. This is the
 * per-channel part; @ref IntegrandCommonPart is the shared one and
 * @ref IntegrandConcatenator stitches the results back together.
 *
 * **Arguments**
 * - the inputs of @p integrand for one channel; see @ref Integrand.
 *
 * **Returns**
 * - the intermediate quantities @ref IntegrandCommonPart consumes.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class IntegrandChannelPart : public FunctionGenerator {
public:
    /// @param integrand  The integrand to take the channel part of.
    IntegrandChannelPart(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

/**
 * The channel-independent second half of an @ref Integrand.
 *
 * Runs the matrix element and the weight assembly on the accepted events of
 * the batch merged across channels (see @ref IntegrandConcatenator).
 *
 * **Arguments**
 * - the intermediate quantities produced by @ref IntegrandChannelPart.
 *
 * **Returns**
 * - the same quantities as @ref Integrand.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class IntegrandCommonPart : public FunctionGenerator {
public:
    /// @param integrand  The integrand to take the common part of.
    IntegrandCommonPart(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

/**
 * Concatenates two @ref IntegrandChannelPart outputs along the batch.
 *
 * Unless `drop_cuts_and_rescale` is set, the `indices_acc` of the second one
 * are offset by the batch size of the first, so the result can be fed to
 * @ref IntegrandCommonPart.
 *
 * **Arguments**
 * - two sets of channel-part outputs, with the keys prefixed `arg1_` and
 *   `arg2_`.
 *
 * **Returns**
 * - one set of channel-part outputs.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class IntegrandConcatenator : public FunctionGenerator {
public:
    /// @param integrand  The integrand whose common part is being re-split.
    IntegrandConcatenator(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    const Integrand& _integrand;
};

/**
 * Evaluates several @ref Integrand channels on one merged batch.
 *
 * Runs the channel part of each @ref Integrand on its slice of the batch,
 * concatenates the results and runs the common part of the first integrand once
 * on the merged batch.
 *
 * **Arguments**
 * - `batch_sizes` – the number of events per channel.
 *
 * **Returns**
 * - the per-event quantities of @ref Integrand, plus `return_batch_sizes` when
 *   @p return_sizes is set.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 */
class MultiChannelIntegrand : public FunctionGenerator {
public:
    /// @param integrands    The per-channel integrands.
    /// @param return_sizes  If true, also return the per-channel batch sizes.
    MultiChannelIntegrand(
        const std::vector<std::shared_ptr<Integrand>>& integrands,
        bool return_sizes = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::shared_ptr<Integrand>> _integrands;
    bool _return_sizes;
};

/**
 * The sampling density of an @ref Integrand at a given phase-space point.
 *
 * Evaluates the probability with which the integrand's adaptive samplers and
 * permutation / flavour selection would have produced a given point. Used for
 * the density of stored points in MadNIS training.
 *
 * **Arguments**
 * - `latent`, `channel_index_in_group`, and with an adaptive flavour sampler
 *   `discrete_flavor_index` (plus `pdf_prior` if the integrand has one).
 *
 * **Returns**
 * - `prob` – `float`, shape `(batch,)` – the sampling density.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class IntegrandProbability : public FunctionGenerator {
public:
    /// @param integrand  The integrand to take the sampling density of.
    IntegrandProbability(const Integrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    Integrand::AdaptiveMapping _adaptive_map;
    Integrand::AdaptiveDiscrete _discrete_sym;
    Integrand::AdaptiveDiscrete _discrete_flavor;
    std::size_t _permutation_count;
    std::size_t _flavor_count;
    bool _has_pdf_prior;
};

} // namespace madspace
