#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * A user-supplied function of the event momenta, loaded as a UMAMI library.
 *
 * Custom cuts, scales and biases use the UMAMI interface as a generic batched
 * function. The library is loaded like a matrix element, with
 * Context::load_matrix_element. It receives the momenta
 * (`UMAMI_IN_MOMENTA`), the particle count (`UMAMI_IN_PARTICLE_COUNT`) and the
 * PDG ids of the external particles (`UMAMI_IN_PDG_IDS`). It returns one value
 * per event under `UMAMI_OUT_CUT` or `UMAMI_OUT_BIAS`, or a scale returns the
 * three values `UMAMI_OUT_REN_SCALE`, `UMAMI_OUT_FACT_SCALE_1` and
 * `UMAMI_OUT_FACT_SCALE_2`.
 *
 * With a single PDG-id option the ids are constant. With several options the
 * ids are selected per event by the `flavor_index` argument.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, particle_count, 4)` – the event momenta.
 * - `flavor_index` – `int`, shape `(batch,)` – index into the PDG-id options.
 *   Present only with more than one option.
 *
 * **Returns**
 * - `cut` or `bias` – `float`, shape `(batch,)` – the function value.
 * - `ren_scale`, `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – the
 *   scales, for a function of type `scale`.
 */
class CustomFunction : public FunctionGenerator {
public:
    /// What the custom function computes.
    enum CustomFunctionType {
        cut,   ///< Acceptance factor, 0 rejects the event.
        scale, ///< Renormalization and factorization scales.
        bias   ///< Event bias.
    };

    /**
     * @param function_index   Index of the loaded library in the context.
     * @param type             What the function computes.
     * @param particle_count   Number of external particles.
     * @param pdg_id_options   PDG ids of the external particles, one list per
     *                         flavor option. Empty passes zeros.
     */
    CustomFunction(
        std::size_t function_index,
        CustomFunctionType type,
        std::size_t particle_count,
        const nested_vector2<me_int_t>& pdg_id_options = {}
    );
    /**
     * Take the index from an already-loaded library.
     * @param function_api     The loaded library.
     * @param type             What the function computes.
     * @param particle_count   Number of external particles.
     * @param pdg_id_options   See the other constructor.
     */
    CustomFunction(
        const MatrixElementApi& function_api,
        CustomFunctionType type,
        std::size_t particle_count,
        const nested_vector2<me_int_t>& pdg_id_options = {}
    ) :
        CustomFunction(function_api.index(), type, particle_count, pdg_id_options) {}

    /// Index of the loaded library in the context.
    std::size_t function_index() const { return _function_index; }
    /// What the function computes.
    CustomFunctionType type() const { return _type; }
    /// Whether the PDG ids are selected per event by a `flavor_index` argument.
    bool dynamic_pdg_ids() const { return _pdg_id_options.size() > 1; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _function_index;
    CustomFunctionType _type;
    std::size_t _particle_count;
    nested_vector2<me_int_t> _pdg_id_options;
};

/**
 * A user-supplied parton density, loaded as a UMAMI library.
 *
 * Passed in place of a @ref PdfGrid. The library receives the momentum
 * fraction, the factorization scale, the parton PDG id and the beam index. It
 * returns @f$x f(x, q)@f$ under `UMAMI_OUT_PDF`.
 */
struct CustomPdf {
    /// @param function_index  Index of the loaded library in the context.
    CustomPdf(std::size_t function_index) : function_index(function_index) {}
    /// @param function_api  The loaded library.
    CustomPdf(const MatrixElementApi& function_api) :
        function_index(function_api.index()) {}
    /// Index of the loaded library in the context.
    std::size_t function_index;
};

} // namespace madspace
