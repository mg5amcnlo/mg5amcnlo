#pragma once

#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/scale.hpp"

namespace madspace {

/**
 * The fully differential cross section @f$f(x)@f$.
 *
 * Assembles the integrand of the phase-space integral: a @ref MatrixElement
 * evaluation multiplied by the flux factor, the @ref PartonDensity values for
 * the two beams, and @f$\alpha_s@f$ from a @ref RunningCoupling, all at the
 * @ref EnergyScale of the event [1]. The `Cached*` variants of the PDF and
 * scale arguments reuse values already computed elsewhere in the graph.
 *
 * With @p decay the generator instead describes the decay of a single incoming
 * particle. The differential rate is then @f$|M|^2 / (2 M)@f$ with
 * @f$M@f$ = @p cm_energy, in GeV, rather than a hadronic cross section in pb.
 * There are no beams, hence no momentum fractions and no PDFs.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the arguments of @p matrix_element, plus
 * - `x1`, `x2` – `float`, shape `(batch,)` – parton momentum fractions (when
 *   @p input_momentum_fraction is true).
 * - `pdf_id` – `int`, shape `(batch,)` – selected flavour combination.
 * - `pdf1`, `pdf2` – `float`, shape `(batch,)` – cached PDF values (with a
 *   `CachedPdf` argument).
 * - `alpha_s` – `float`, shape `(batch,)` – cached coupling (with a
 *   `CachedScale` argument).
 *
 * **Returns**
 * - the return values of @p matrix_element, scaled to the differential cross
 *   section.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2)
 */
class DifferentialCrossSection : public FunctionGenerator {
public:
    /// Argument tag: take the PDF value from a value already in the graph.
    struct CachedPdf {};
    /// Argument tag: take the scale / coupling from a value already in the graph.
    struct CachedScale {};

    /**
     * @param matrix_element          The squared matrix element.
     * @param cm_energy                Total collision energy.
     * @param running_coupling         Optional @f$\alpha_s@f$ source.
     * @param energy_scale             An @ref EnergyScale, a `CachedScale` tag,
     *                                 or nothing.
     * @param pid_options              Allowed flavour combinations of the two
     *                                 initial partons.
     * @param pdf1                     A @ref PdfGrid, a `CachedPdf` tag, or
     *                                 nothing, for the first beam.
     * @param pdf2                     Likewise for the second beam.
     * @param input_momentum_fraction  If true, `x1` / `x2` are taken as inputs
     *                                 rather than reconstructed.
     * @param decay                    If true, describe the decay of a single
     *                                 incoming particle instead of a collision.
     * @param matrix_element_output    If true, also return the squared matrix
     *                                 element itself (before PDFs and flux) as
     *                                 a last output, `raw_matrix_element`.
     */
    DifferentialCrossSection(
        const MatrixElement& matrix_element,
        double cm_energy,
        const std::optional<RunningCoupling>& running_coupling,
        const std::variant<std::monostate, EnergyScale, CachedScale>& energy_scale =
            std::monostate{},
        const nested_vector2<me_int_t>& pid_options = {},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf1 = std::monostate{},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf2 = std::monostate{},
        bool input_momentum_fraction = true,
        bool decay = false,
        bool matrix_element_output = false
    );

    /// The allowed initial-parton flavour combinations.
    const nested_vector2<me_int_t>& pid_options() const { return _pid_options; }
    /// Whether beam @p pdf_index (0 or 1) has a PDF attached.
    bool has_pdf(std::size_t pdf_index) const { return _has_pdf.at(pdf_index); }
    /// The matrix element.
    const MatrixElement& matrix_element() const { return _matrix_element; }
    /// The strong-coupling source, if any.
    const std::optional<RunningCoupling>& running_coupling() const {
        return _running_coupling;
    }
    /// Whether the squared matrix element is returned as a last output.
    bool matrix_element_output() const { return _matrix_element_output; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<me_int_t> _pid_options;
    MatrixElement _matrix_element;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<bool, 2> _has_pdf;
    std::optional<RunningCoupling> _running_coupling;
    double _e_cm;
    std::variant<std::monostate, EnergyScale, CachedScale> _energy_scale;
    bool _input_momentum_fraction;
    bool _decay;
    bool _matrix_element_output;
};

} // namespace madspace
