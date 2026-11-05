#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"

namespace madspace {

/**
 * A parton-density grid in the LHAPDF format, loaded from a file.
 *
 * `madspace` carries a built-in PDF interpolator compatible with the LHAPDF
 * grid format so it needs no external PDF library [1, 2]. This struct holds the
 * loaded grid; @ref PartonDensity does the interpolation. `initialize_globals`
 * uploads the grid to a @ref Context.
 */
struct PdfGrid {
    /// Momentum-fraction grid knots.
    std::vector<double> x;
    /// `log(x)` grid knots.
    std::vector<double> logx;
    /// Scale grid knots.
    std::vector<double> q;
    /// `log(q^2)` grid knots.
    std::vector<double> logq2;
    /// PDG ids of the flavours in the grid.
    std::vector<int> pids;
    /// Grid values, indexed by `(q, x)` grid point (`q_index * x.size() + x_index`),
    /// then by flavour.
    std::vector<std::vector<double>> values;
    /// Number of `q` intervals in each sub-grid region.
    std::vector<std::size_t> region_sizes;

    /// Load an LHAPDF-format grid.
    /// @param file  Path to the grid file.
    PdfGrid(const std::string& file);
    /// Number of `(x, q)` grid points, including padding, as stored in the
    /// coefficient tensor.
    std::size_t grid_point_count() const;
    /// Number of scale knots, counting the knots shared by two regions once.
    std::size_t q_count() const;
    /// Fill @p tensor with the interpolation coefficients.
    void initialize_coefficients(Tensor tensor) const;
    /// Fill @p tensor with the `log(x)` knots.
    void initialize_logx(Tensor tensor) const;
    /// Fill @p tensor with the `log(q^2)` knots.
    void initialize_logq2(Tensor tensor) const;
    /// Shape of the coefficient tensor, optionally with a leading batch axis.
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    /// Shape of the `log(x)` tensor.
    std::vector<std::size_t> logx_shape(bool batch_dim = false) const;
    /// Shape of the `log(q^2)` tensor.
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    /// Upload the grid to @p context as compute-graph globals under @p prefix.
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

/**
 * Parton-density interpolation as a compute-graph function.
 *
 * Interpolates a @ref PdfGrid at the sampled @f$(x, q)@f$, returning
 * @f$x f(x, q)@f$ for the requested flavours; the numerical output agrees with
 * LHAPDF to double precision [1] (Sec. 3.2.3 of [2]). With a @ref CustomPdf,
 * the user-supplied library is called for each requested flavour instead.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `x` – `float`, shape `(batch,)` – momentum fraction.
 * - `q` – `float`, shape `(batch,)` – factorization scale.
 * - `flavor_index` – `int`, shape `(batch,)` – index into @p pids selecting
 *   the flavour to return. Present only when @p dynamic_pid is true.
 *
 * **Returns**
 * - `pdf` – `float`, shape `(batch,)` when @p dynamic_pid is true, otherwise
 *   `(batch, len(pids))` – the interpolated `x f(x, q)`.
 *
 * **References**
 * - [1] A. Buckley et al., "LHAPDF6: parton density access in the LHC
 *   precision era", https://arxiv.org/abs/1412.7420
 * - [2] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.3)
 */
class PartonDensity : public FunctionGenerator {
public:
    /**
     * @param grid         The parton-density grid to interpolate.
     * @param pids          PDG ids of the flavours to return.
     * @param dynamic_pid   If true, the single flavour of @p pids selected per
     *                      event by `flavor_index` is returned.
     * @param prefix        Prefix for the grid global names.
     */
    PartonDensity(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        bool dynamic_pid = false,
        const std::string& prefix = ""
    );
    /**
     * @param custom_pdf    The user-supplied parton density.
     * @param pids          PDG ids of the flavours to return.
     * @param beam_index    Index of the beam (0 or 1), passed to the library.
     * @param dynamic_pid   If true, the single flavour of @p pids selected per
     *                      event by `flavor_index` is returned.
     */
    PartonDensity(
        const CustomPdf& custom_pdf,
        const std::vector<int>& pids,
        int beam_index,
        bool dynamic_pid = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<me_int_t> _pid_indices;
    std::vector<me_int_t> _pids;
    std::optional<CustomPdf> _custom_pdf;
    me_int_t _beam_index = 0;
    bool _dynamic_pid;
    std::string _prefix;
    std::vector<std::size_t> _logx_shape;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

/**
 * A strong-coupling grid in the LHAPDF format, loaded from a file.
 *
 * Holds the @f$\alpha_s(q)@f$ knots of an LHAPDF `.info` set;
 * @ref RunningCoupling does the interpolation.
 */
struct AlphaSGrid {
    /// Scale grid knots.
    std::vector<double> q;
    /// `log(q^2)` grid knots.
    std::vector<double> logq2;
    /// `alpha_s` values at the knots.
    std::vector<double> values;
    /// Number of `q` intervals in each sub-grid region.
    std::vector<std::size_t> region_sizes;

    /// Load an LHAPDF-format `alpha_s` grid.
    /// @param file  Path to the grid file.
    AlphaSGrid(const std::string& file);
    /// Number of scale knots, counting the knots shared by two regions once.
    std::size_t q_count() const;
    /// Fill @p tensor with the interpolation coefficients.
    void initialize_coefficients(Tensor tensor) const;
    /// Fill @p tensor with the `log(q^2)` knots.
    void initialize_logq2(Tensor tensor) const;
    /// Shape of the coefficient tensor, optionally with a leading batch axis.
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    /// Shape of the `log(q^2)` tensor.
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    /// Upload the grid to @p context as compute-graph globals under @p prefix.
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

/**
 * Strong-coupling interpolation as a compute-graph function.
 *
 * Interpolates an @ref AlphaSGrid to return @f$\alpha_s(q)@f$ at the sampled
 * scale (Sec. 3.2.3 of [1]).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `q` – `float`, shape `(batch,)` – renormalization scale.
 *
 * **Returns**
 * - `alpha_s` – `float`, shape `(batch,)` – the interpolated @f$\alpha_s(q)@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.3)
 */
class RunningCoupling : public FunctionGenerator {
public:
    /// @param grid    The `alpha_s` grid to interpolate.
    /// @param prefix  Prefix for the grid global names.
    RunningCoupling(const AlphaSGrid& grid, const std::string& prefix = "");

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::string _prefix;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

} // namespace madspace
