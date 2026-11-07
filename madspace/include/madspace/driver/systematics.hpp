#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/util.hpp"

namespace madspace {

/// One PDF member entering the variations. The nominal member (the set and
/// member the events were generated with) is described by
/// `SystematicsConfig::nominal_*` and must not be listed here.
struct PdfMemberSpec {
    /// LHAPDF set name.
    std::string set_name;
    /// LHAPDF id of the set (`SetIndex` in the `.info` file).
    int set_lhaid = 0;
    /// Member number inside the set.
    int member = 0;
    /// Grid data file, `<set>_<mmmm>.dat`.
    std::string grid_file;
    /// Set info file, `<set>.info` (carries the set's `alpha_s`).
    std::string info_file;
    /// `ErrorType` of the set (`replicas`, `hessian`, ...).
    std::string error_type;
    /// Human-readable description of the member.
    std::string description;
};

/// Configuration of the scale and PDF variations computed by @ref
/// SystematicsCalculator.
struct SystematicsConfig {
    /// Renormalization scale factors; `1.0` is the nominal point.
    std::vector<double> mur{1.};
    /// Factorization scale factors; `1.0` is the nominal point.
    std::vector<double> muf{1.};
    /// Emission scale factors of an MLM merged sample, madevent's `alpsfact`:
    /// the alpha_s of every clustering vertex is evaluated at the factor times
    /// the vertex scale. `1.0` is the nominal point; the other values are
    /// written in an "Emission scale variation" group. Only for events whose
    /// record carries the vertex scales (ParticleRecord::f_alphas_scales).
    std::vector<double> alps{1.};
    /// If true, the events may carry an MLM pdf reweighting ladder
    /// (ParticleRecord::f_pdf_ladder): every parton of the PDF set is then
    /// made available, since a ladder step can be any of them.
    bool pdf_ladder = false;
    /// Range the varied scales of the pdf ladder are clamped to, normally
    /// the Q range of the PDF grid, where LHAPDF freezes the density for
    /// madevent; `0` disables a bound.
    double ladder_min_scale = 0.;
    /// See @ref ladder_min_scale.
    double ladder_max_scale = 0.;
    /// If true, vary every `(mur, muf)` combination; if false, vary one
    /// factor at a time.
    bool together = true;
    /// Alternative dynamical scale choices, combined with `mur`/`muf`; `1` =
    /// sum of transverse energies, `2` = sum of transverse masses, `3` = half
    /// of that, `4` = partonic center-of-mass energy.
    std::vector<int> dyn_scales;
    /// Factor already applied to the nominal dynamical scale; the alternative
    /// dynamical scales are multiplied by it too.
    double scale_factor = 1.;
    /// Varied PDF members, in output order.
    std::vector<PdfMemberSpec> pdf_members;
    /// LHAPDF set name the events were generated with.
    std::string nominal_set_name;
    /// LHAPDF id of the nominal set.
    int nominal_lhaid = 0;
    /// `ErrorType` of the nominal set.
    std::string nominal_error_type;
    /// Human-readable description of the nominal set.
    std::string nominal_description;
    /// If false, the beams are leptonic: no PDF at all, only `alpha_s`
    /// variations.
    bool has_pdf = true;
    /// If true, also write the per-event reweighting inputs (`x1`, `x2`,
    /// scales, PDG ids) to the output.
    bool write_inputs = false;
    /// Id of the first variation weight.
    int first_id = 1;
};

/// Per (unmerged) subprocess information needed to reweight its events.
struct SubprocessSystArgs {
    /// Power of `alpha_s` in `|M|^2`; `-1` if it differs between diagrams.
    int qcd_power = -1;
    /// `beam_pdgs[flavor_index]` is `{pdg of parton 1, pdg of parton 2}`.
    nested_vector2<int> beam_pdgs;
};

/// One weight variation computed by @ref SystematicsCalculator — a scale
/// choice, a PDF member, or the nominal point.
struct Variation {
    /// LHEF `<weight>` id.
    int id;
    /// Renormalization scale factor.
    double mur;
    /// Factorization scale factor.
    double muf;
    /// Index into @ref SystematicsCalculator::members; `-1` for the nominal
    /// PDF member.
    int pdf_index;
    /// Dynamical scale choice; `-1` for the scale the events were generated
    /// with.
    int dyn = -1;
    /// Emission scale factor of the MLM alpha_s vertices (madevent's
    /// `alpsfact`); `1` for the nominal vertex scales.
    double alps = 1.;
    /// Whether this is a scale variation (a non-nominal `mur`, `muf` or `dyn`
    /// at the nominal PDF member).
    bool is_scale() const {
        return pdf_index == -1 && (mur != 1. || muf != 1. || dyn != -1);
    }
    /// Whether this is an emission scale variation (a non-nominal `alps` at
    /// the nominal scales and PDF member).
    bool is_emission() const { return pdf_index == -1 && !is_scale() && alps != 1.; }
    /// Whether this is the nominal point (nominal scale and PDF member).
    bool is_nominal() const { return pdf_index == -1 && !is_scale() && !is_emission(); }
};

/// The variations belonging to one PDF set, for computing an uncertainty band.
struct PdfGroupInfo {
    /// LHAPDF set name.
    std::string set_name;
    /// LHAPDF id of the set.
    int set_lhaid;
    /// `ErrorType` of the set.
    std::string error_type;
    /// `(member number, index into SystematicsCalculator::variations)` pairs;
    /// member `0` is the central member.
    std::vector<std::pair<int, std::size_t>> members;
};

/**
 * Computes LHEF weight variations (scale and PDF systematics) of combined,
 * already-generated events.
 *
 * The PDFs and `alpha_s` are evaluated with the regular batched madspace
 * functions (@ref PartonDensity, @ref RunningCoupling) on `context` (a CPU
 * context; one is created when none is given): the grids of the nominal PDF
 * and of every varied member are registered as globals of that context under
 * a private prefix. Built once per run from a @ref SystematicsConfig, then
 * reused for every combined event via @ref compute.
 */
class SystematicsCalculator {
public:
    /**
     * @param config           The scale and PDF variations to compute.
     * @param subproc_args     Per-subprocess bookkeeping needed to reweight
     *                         its events; indexed by the events'
     *                         subprocess index.
     * @param nominal_pdf      The PDF the events were generated with;
     *                         `nullopt` for leptonic beams.
     * @param nominal_alpha_s  The `alpha_s` grid the events were generated
     *                         with.
     * @param context          Context the PDF and `alpha_s` runtimes are
     *                         built on; a private CPU context if `nullptr`.
     * @param matrix_elements  One matrix element per subprocess (may be
     *                         empty or `nullopt`), enabling the exact
     *                         renormalization-scale variation of
     *                         subprocesses whose `|M|^2` mixes several
     *                         powers of `alpha_s` (`qcd_power == -1`): it is
     *                         re-evaluated at the varied `alpha_s` on
     *                         `context`. Must take `(momenta, alpha_s,
     *                         flavor)` and return the matrix element only.
     * @param me_flavor_remap  `me_flavor_remap[subprocess][flavor_index]` is
     *                         the flavor passed to that subprocess's matrix
     *                         element.
     * @param nominal_pdf2     PDF of the second beam when its set differs from
     *                         @p nominal_pdf (which is then the first beam's).
     *                         PDF member variations are then dropped with a
     *                         warning; the scale variations use both sets.
     */
    SystematicsCalculator(
        const SystematicsConfig& config,
        const std::vector<SubprocessSystArgs>& subproc_args,
        const std::optional<PdfGrid>& nominal_pdf,
        const std::optional<AlphaSGrid>& nominal_alpha_s,
        ContextPtr context = nullptr,
        const std::vector<std::optional<MatrixElement>>& matrix_elements = {},
        const nested_vector2<me_int_t>& me_flavor_remap = {},
        const std::optional<PdfGrid>& nominal_pdf2 = std::nullopt
    );

    /// The configuration passed to the constructor.
    const SystematicsConfig& config() const { return _config; }
    /// Every variation to compute, in weight order.
    const std::vector<Variation>& variations() const { return _variations; }
    /// Number of variations, i.e. weights per event.
    std::size_t weight_count() const { return _variations.size(); }
    /// LHEF `<weight>` ids, in the order of @ref variations.
    std::vector<int> weight_ids() const;
    /// PDF members actually used: the configured ones minus those ignored
    /// with a warning. The nominal member is not listed.
    const std::vector<PdfMemberSpec>& members() const { return _members; }
    /// Warnings emitted while building the variation list.
    const std::vector<std::string>& warnings() const { return _warnings; }
    /// Indices into @ref variations of the scale variations (the envelope
    /// group).
    std::vector<std::size_t> scale_variation_indices() const;
    /// Indices into @ref variations of the MLM emission scale variations
    /// (their own envelope group).
    std::vector<std::size_t> emission_variation_indices() const;
    /// The PDF groups, one per set with members among @ref variations.
    std::vector<PdfGroupInfo> pdf_groups() const;

    /**
     * Compute the weights of every variation for the events of `buffer`.
     *
     * `buffer` must carry the combined-event layout (weight, subprocess
     * index, event data, momenta and the partial weight columns). `weights`
     * is resized to `event_count * weight_count()`, row-major per event, and
     * holds the varied event weights, not the ratios to the nominal weight.
     * Thread-safe.
     */
    void compute(EventBuffer& buffer, std::vector<double>& weights) const;
    /// The reweighting inputs of one event of `buffer`.
    LOReweightInfo reweight_info(EventBuffer& buffer, std::size_t event_index) const;
    /// Accumulate the cross sections of every variation over `buffer`'s
    /// events and their `weights`. Thread-safe.
    void accumulate(EventBuffer& buffer, const std::vector<double>& weights);

    /// `<initrwgt>` header content (without the enclosing tag), following the
    /// conventions of `systematics.py` (weight groups and attributes).
    std::string initrwgt() const;
    /// JSON description of the variations and, once events were accumulated,
    /// the per-variation cross sections and the scale/PDF uncertainties.
    nlohmann::json summary() const;

    /**
     * Combined PDF uncertainty for the given LHAPDF `error_type`.
     *
     * @param error_type     LHAPDF `ErrorType` (`replicas`, `hessian`, ...).
     * @param central        Central value; used when member `0` is absent
     *                       from `member_values` (not used for `replicas`).
     * @param member_values  `(member number, value)` pairs; member `0` is
     *                       the central member.
     * @return `(central, up, down)`; for `replicas` the first entry is the
     *         replica mean. `NaN` entries when they cannot be computed.
     */
    static std::tuple<double, double, double> pdf_uncertainty(
        const std::string& error_type,
        std::optional<double> central,
        const std::vector<std::pair<int, double>>& member_values
    );
    /// Dynamical scale `dyn` (`1`-`4`; see @ref SystematicsConfig::dyn_scales)
    /// of an event from its `momenta` (`momenta[particle][component]`, `E px
    /// py pz`, incoming particles first).
    static double
    dynamical_scale(int dyn, const std::vector<std::array<double, 4>>& momenta);
    /// Human-readable name of dynamical scale choice `dyn`.
    static std::string dyn_scale_name(int dyn);
    /// `value` formatted for LHEF text output.
    static std::string format_number(double value);

private:
    // batched PDF evaluation of one grid (nominal or member): x f(x, q) for the
    // PIDs used by the events, addressed by their slot in `pids`
    struct PdfEvaluator {
        RuntimePtr runtime;
        std::vector<int> pids;
        std::size_t alpha_s_index; // alpha_s grid of the set
    };
    struct MatrixElementData {
        RuntimePtr runtime;
        std::size_t particle_count;
        std::vector<me_int_t> flavor_remap;
    };

    SystematicsConfig _config;
    std::vector<SubprocessSystArgs> _subproc_args;
    std::vector<PdfMemberSpec> _members;
    std::vector<Variation> _variations;
    std::vector<std::string> _warnings;
    bool _mur_supported;

    ContextPtr _context;
    std::string _prefix;
    std::optional<PdfEvaluator> _nominal_pdf;  // has_pdf only
    std::optional<PdfEvaluator> _nominal_pdf2; // second beam, if its set differs
    std::vector<PdfEvaluator> _member_pdfs;    // one per member
    std::vector<RuntimePtr> _alpha_s_runtimes; // index 0: nominal set
    std::vector<std::optional<MatrixElementData>> _matrix_elements;
    mutable std::mutex _runtime_mutex;

    std::mutex _accumulate_mutex;
    double _nominal_sum = 0.;
    std::vector<double> _variation_sums;
    std::size_t _event_count = 0;

    void build_variations();
    PdfEvaluator make_pdf_evaluator(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        const std::string& name,
        std::size_t alpha_s_index
    );
    // batched x f(x, q) of `evaluator` for the given points; `slots` index
    // evaluator.pids
    std::vector<double> evaluate_pdf(
        const PdfEvaluator& evaluator,
        const std::vector<double>& x,
        const std::vector<double>& q,
        const std::vector<me_int_t>& slots
    ) const;
    std::vector<double>
    evaluate_alpha_s(std::size_t alpha_s_index, const std::vector<double>& q) const;
    // |M|^2 of the events `indices` of `buffer` (all of subprocess `subproc`) at
    // the given alpha_s values, via the matrix element runtime
    std::vector<double> matrix_elements(
        int subproc,
        EventBuffer& buffer,
        const std::vector<std::size_t>& indices,
        const std::vector<double>& alpha_s
    ) const;
    std::vector<std::array<double, 4>>
    event_momenta(EventBuffer& buffer, std::size_t event_index) const;
};

void to_json(nlohmann::json& j, const PdfMemberSpec& spec);
void from_json(const nlohmann::json& j, PdfMemberSpec& spec);
void to_json(nlohmann::json& j, const SystematicsConfig& config);
void from_json(const nlohmann::json& j, SystematicsConfig& config);
void to_json(nlohmann::json& j, const SubprocessSystArgs& args);
void from_json(const nlohmann::json& j, SubprocessSystArgs& args);
void to_json(nlohmann::json& j, const Variation& variation);

} // namespace madspace
