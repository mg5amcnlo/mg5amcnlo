#pragma once

#include <array>
#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/matrix_element.hpp"

namespace madspace {

/**
 * One re- or unweighting step after the first unweighting of an
 * @ref EventGenerator.
 *
 * The step evaluates its own matrix element on the events that survived the
 * previous steps and multiplies their weights by the ratio
 * @f$r = |M_\text{step}|^2 / |M_\text{previous}|^2@f$. All other factors of the
 * weight (PDFs, flux, Jacobian, channel weights, bias) are those of the first
 * step and cancel in the ratio. An unweighting step then unweights the events
 * with respect to the product of the ratios since the last unweighting step,
 * times the overweight factor carried from it, with its own maximum weight. A
 * reweighting step only multiplies the weights.
 *
 * Instead of the matrix element, the matrix elements of a step can return the
 * ratio directly, with the output `matrix_element_ratio_out`. Its matrix element
 * is then the ratio times the one of the previous step, which is the reference
 * for the ratio of the next step.
 *
 * Typical uses are a fast surrogate of the matrix element in the integrand,
 * followed by an unweighting step with the full matrix element, or a
 * leading-color integrand reweighted to full color.
 */
struct WeightingStep {
    /// What the step does with the ratio of the matrix elements.
    enum Mode {
        unweight, ///< Unweight with respect to the ratio, with its own maximum.
        reweight  ///< Multiply the event weight by the ratio.
    };

    /// One matrix element per matrix element of the integrand, called for the
    /// events with that `matrix_element_index`. The random inputs are passed
    /// in, so they must not be sampled internally.
    std::vector<MatrixElement> matrix_elements;
    /// Unweighting or reweighting.
    Mode mode = unweight;
    /// Like @ref GeneratorConfig::max_overweight_truncation, for this step.
    double max_overweight_truncation = 0.01;
    /// Like @ref GeneratorConfig::freeze_max_weight_after, for this step.
    std::size_t freeze_max_weight_after = 10000;
    /// Color, helicity or diagram index outputs that replace the values of the
    /// earlier steps. Without, an index is taken from the first step whose
    /// matrix element returns it.
    std::vector<MatrixElement::MatrixElementOutput> override_outputs;

    /// Whether the matrix elements return the ratio to the previous step
    /// instead of the matrix element.
    bool returns_ratio() const;
};

/**
 * Evaluates the matrix elements of a @ref WeightingStep.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, particle_count, 4)`.
 * - `alpha_s` – `float`, shape `(batch,)`.
 * - `flavor` – `int`, shape `(batch,)` – flavor index of the matrix element.
 * - `matrix_element_index` – `int`, shape `(batch,)` – which matrix element to
 *   call, only with several matrix elements.
 * - `color_index`, `helicity_index`, `diagram_index` – `int`, shape `(batch,)`
 *   – the indices of the earlier steps.
 * - `random` – `float`, shape `(batch, 4 step_count)` – four random numbers
 *   per step: for the color, helicity and diagram sampling of the matrix
 *   element and for the unweighting.
 *
 * **Returns**
 * - `matrix_element` – `float`, shape `(batch,)` – the matrix element, or the
 *   ratio to the previous step if the step returns ratios.
 * - `color_index`, `helicity_index`, `diagram_index` – `int`, shape `(batch,)`
 *   – the indices the step writes, otherwise the input values.
 * - `unweight_random` – `float`, shape `(batch,)` – the random number of this
 *   step for the unweighting.
 */
class WeightingStepFunction : public FunctionGenerator {
public:
    /// Random numbers per event and step: color, helicity, diagram, unweighting.
    static constexpr std::size_t random_count = 4;

    /**
     * @param step            The step.
     * @param particle_count  Number of particles of the events.
     * @param write_indices   Whether the color, helicity and diagram indices
     *                        returned by the matrix elements are written.
     * @param step_index      Position of the step, from 0.
     * @param step_count      Number of steps, which sets the shape of `random`.
     */
    WeightingStepFunction(
        const WeightingStep& step,
        std::size_t particle_count,
        const std::array<bool, 3>& write_indices,
        std::size_t step_index = 0,
        std::size_t step_count = 1
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<MatrixElement> _matrix_elements;
    std::array<bool, 3> _write_indices;
    std::size_t _step_index;
};

} // namespace madspace
