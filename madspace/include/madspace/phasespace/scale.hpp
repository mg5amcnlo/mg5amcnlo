#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/mlm_clustering.hpp"

namespace madspace {

/**
 * Renormalization and factorization scales for an event.
 *
 * Computes the renormalization scale and the two factorization scales from the
 * event momenta. Each can be fixed to a constant or set to a dynamical choice
 * (Sec. 3.2.6 of [1], matching the choices in [2]):
 *
 * @f[
 *   \mu = E_\mathrm{T} = \sum_i \frac{E_i\,p_{\mathrm{T},i}}{|\vec p_i|},
 *   \qquad
 *   \mu = H_\mathrm{T} = \sum_i \sqrt{m_i^2 + p_{\mathrm{T},i}^2},
 * @f]
 *
 * with sums over the outgoing particles, and @ref DynamicalScaleType selecting
 * @f$E_\mathrm{T}@f$ (`transverse_energy`), @f$H_\mathrm{T}@f$ (`transverse_mass`),
 * @f$H_\mathrm{T}/2@f$ (`half_transverse_mass`) or @f$\sqrt{\hat s}@f$
 * (`partonic_energy`). The dynamical scale is multiplied by `scale_factor`,
 * e.g. @f$H_\mathrm{T}/4@f$ is `half_transverse_mass` with `scale_factor = 0.5`.
 * A @ref CustomFunction of type CustomFunction::scale replaces the dynamical
 * choice (also multiplied by `scale_factor`), like `user_dynamical_scale` in
 * MadGraph5.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, particle_count, 4)` – the event momenta.
 *
 * **Returns**
 * - `ren_scale` – `float`, shape `(batch,)` – renormalization scale.
 * - `fact_scale1` – `float`, shape `(batch,)` – factorization scale for the
 *   first beam.
 * - `fact_scale2` – `float`, shape `(batch,)` – factorization scale for the
 *   second beam.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.6)
 * - [2] V. Hirschi, O. Mattelaer, "Automated event generation for loop-induced
 *   processes", https://arxiv.org/abs/1507.00020
 */
class EnergyScale : public FunctionGenerator {
public:
    /// Dynamical scale choice; see the class description for the formulas.
    enum DynamicalScaleType {
        transverse_energy,    ///< Total transverse energy @f$E_\mathrm{T}@f$.
        transverse_mass,      ///< Sum of transverse masses @f$H_\mathrm{T}@f$.
        half_transverse_mass, ///< @f$H_\mathrm{T}/2@f$.
        partonic_energy       ///< Partonic energy @f$\sqrt{\hat s}@f$.
    };

    /// Dynamical @ref half_transverse_mass for all three scales.
    /// @param particle_count  Number of external particles.
    EnergyScale(std::size_t particle_count) :
        EnergyScale(particle_count, half_transverse_mass, false, false, 0., 0., 0.) {}
    /// Dynamical scale of type @p type for all three scales.
    /// @param particle_count  Number of external particles.
    /// @param type            The dynamical scale choice.
    /// @param min_scale       Floor on the scales, see the full constructor.
    /// @param max_scale       Ceiling on the scales, see the full constructor.
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType type,
        double min_scale = 0.,
        double max_scale = 0.
    ) :
        EnergyScale(
            particle_count,
            type,
            false,
            false,
            0.,
            0.,
            0.,
            1.,
            std::nullopt,
            min_scale,
            max_scale
        ) {}
    /// A single fixed value for all three scales.
    /// @param particle_count  Number of external particles.
    /// @param fixed_scale     The constant scale value.
    /// @param min_scale       Floor on the scales, see the full constructor.
    /// @param max_scale       Ceiling on the scales, see the full constructor.
    EnergyScale(
        std::size_t particle_count,
        double fixed_scale,
        double min_scale = 0.,
        double max_scale = 0.
    ) :
        EnergyScale(
            particle_count,
            half_transverse_mass,
            true,
            true,
            fixed_scale,
            fixed_scale,
            fixed_scale,
            1.,
            std::nullopt,
            min_scale,
            max_scale
        ) {}
    /**
     * @param particle_count       Number of external particles.
     * @param dynamical_scale_type Scale choice used where a scale is not fixed.
     * @param ren_scale_fixed      Whether the renormalization scale is fixed.
     * @param fact_scale_fixed     Whether the factorization scales are fixed.
     * @param ren_scale            Fixed renormalization-scale value.
     * @param fact_scale1          Fixed factorization scale for the first beam.
     * @param fact_scale2          Fixed factorization scale for the second beam.
     * @param scale_factor         Factor on the dynamical or custom scale; fixed
     *                             scales are left alone.
     * @param custom_scale         Optional user-supplied function that replaces
     *                             the dynamical scale choice.
     * @param min_scale            Floor on mu_R and mu_F. The scales are clamped
     *                             to it, so that a pdf is never asked for a
     *                             density below the bottom of its grid. Zero
     *                             disables it.
     * @param max_scale            Upper end of the same range, normally the top
     *                             of the PDF grid. Zero disables it.
     */
    EnergyScale(
        std::size_t particle_count,
        DynamicalScaleType dynamical_scale_type,
        bool ren_scale_fixed,
        bool fact_scale_fixed,
        double ren_scale,
        double fact_scale1,
        double fact_scale2,
        double scale_factor = 1.,
        const std::optional<CustomFunction>& custom_scale = std::nullopt,
        double min_scale = 0.,
        double max_scale = 0.
    );
    /**
     * Scales from an MLM clustering of the event (dynamical_scale_choice mlm).
     *
     * @param clustering  The compiled clustering histories of the subprocess.
     * @param min_scale   As in the full constructor; here an event whose scales
     *                    fall outside the range is also vetoed, through an
     *                    extra `scale_weight` output, as setclscales does.
     * @param max_scale   As in the full constructor.
     */
    EnergyScale(
        const MLMClustering& clustering, double min_scale = 0., double max_scale = 0.
    );

    /// Whether the scales come from an MLM clustering of the event.
    bool is_mlm() const { return _clustering.has_value(); }
    bool mlm_alphas_reweighting() const {
        return _clustering && _clustering->alphas_scheme() != AlphasScheme::none;
    }
    // Whether the consumer has pdf ratios to apply. The kernel walks the
    // clustering ladder under ScaleScheme::madevent, and under clustering_mean
    // with clustering_mean_pdf_ladder; otherwise it leaves every reweighting
    // slot inert and pdf_scale equal to fact_scale, so the ratios would all
    // come out as exactly one.
    bool mlm_pdf_reweighting() const {
        return _clustering && _clustering->pdf_reweighting() &&
            (_clustering->scale_scheme() == ScaleScheme::madevent ||
             _clustering->clustering_mean_pdf_ladder());
    }
    const std::vector<int>& mlm_pdf_absolute_pdgs() const {
        return _clustering.value().pdf_absolute_pdgs();
    }
    // Whether the MLM history is picked per event from the diagram weights,
    // which moves the clustering after a first matrix element evaluation.
    bool mlm_history_per_diagram() const {
        return _clustering &&
            _clustering->clustering_history() != ClusteringHistory::all_diagrams;
    }
    const std::vector<me_int_t>& mlm_diagram_start_states() const {
        return _clustering.value().diagram_start_states();
    }
    // The scales of the MLM clustering walked from one diagram's start state
    // (see MLMClustering::diagram_start_states), with the same scale range
    // applied as the function itself applies.
    // flavor_index and leg_flavors as in MLMClustering::build_from_start_state.
    NamedVector<Value> build_mlm_from_start_state(
        FunctionBuilder& fb,
        Value momenta,
        Value start_state,
        Value flavor_index = Value(),
        const std::vector<me_int_t>& leg_flavors = {}
    ) const;
    // The MLM history over every diagram, restricted to the flavours of each
    // event (MLMClustering::build_with_flavors), with the scale range applied.
    NamedVector<Value> build_mlm_with_flavors(
        FunctionBuilder& fb,
        Value momenta,
        Value flavor_index,
        const std::vector<me_int_t>& leg_flavors
    ) const;
    bool mlm_flavor_dependent() const {
        return _clustering && _clustering->has_flavor_dependent_clusterings();
    }
    // The band the pdf grid covers, for whoever has to keep a scale inside it.
    double min_scale() const { return _min_scale; }
    double max_scale() const { return _max_scale; }

    bool has_scale_range() const { return _min_scale > 0. || _max_scale > 0.; }
    // Whether the scales come with a scale_weight output vetoing an event
    // outside the range. Only the MLM clustering has one: madevent drops a
    // merged event whose factorisation scale falls under 2 GeV, but for any
    // other scale choice it lets LHAPDF freeze the density instead, which is
    // what clamping the scale does here.
    bool has_scale_veto() const { return is_mlm() && has_scale_range(); }

private:
    NamedVector<Value>
    apply_scale_range(FunctionBuilder& fb, NamedVector<Value> scales) const;

    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    // The MLM constructor leaves these at their defaults: the clustering
    // decides the scales there.
    DynamicalScaleType _dynamical_scale_type = half_transverse_mass;
    bool _ren_scale_fixed = false;
    bool _fact_scale_fixed = false;
    double _ren_scale = 0.;
    double _fact_scale1 = 0.;
    double _fact_scale2 = 0.;
    double _scale_factor = 1.;
    std::optional<CustomFunction> _custom_scale;
    double _min_scale = 0.;
    double _max_scale = 0.;
    std::optional<MLMClustering> _clustering;
};

} // namespace madspace
