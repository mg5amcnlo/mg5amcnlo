/* ################################################### */
/* #                                                 # */
/* #     ___  ___          _ _  ______               # */
/* #     |  \/  |         | | | | ___ \              # */
/* #     | .  . | __ _  __| | |_| |_/ /_____  __     # */
/* #     | |\/| |/ _` |/ _` | __|    // _ \ \/ /     # */
/* #     | |  | | (_| | (_| | |_| |\ \  __/>  <      # */
/* #     \_|  |_/\__,_|\__,_|\__\_| \_\___/_/\_\     # */
/* #                                                 # */
/* #    ******************************************   # */
/* #    *    MadGraph7 internal PDF reweighting   *   # */
/* #    ******************************************   # */
/* ################################################### */
//
// pdfware wraps madspace's internal PDF interpolators (PdfGrid/PartonDensity
// and AlphaSGrid/RunningCoupling, madspace/phasespace/pdf.hpp) into REX::tea
// weightors, so that PDF-only reweighting can run through the same
// REX::tea::reweightor machinery madtrex::Driver/TupperWare already use
//

#pragma once

#include "madtrex.hpp"

#include "madspace/driver/backend.hpp"
#include "madspace/phasespace/pdf.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace madtrex {

// pids fixes, once and for all, which incoming-parton PDG codes this PdfSet
// can be evaluated for; evaluating at a PDG code outside that list throws. 
// default_pids() uses gluon + 5-flavor set unless caller overrides it.
class PdfSet {
public:
    PdfSet(
        madspace::ContextPtr context,
        const std::string& grid_file,
        std::vector<int> pids,
        const std::string& alpha_s_grid_file,
        const std::string& prefix
    );

    PdfSet(
        madspace::ContextPtr context,
        const madspace::PdfGrid& grid,
        std::vector<int> pids,
        const madspace::AlphaSGrid* alpha_s_grid,
        const std::string& existing_globals_prefix,
        const std::string& label
    );

    bool has_alpha_s() const { return static_cast<bool>(_alpha_s_runtime); }
    const std::vector<int>& pids() const { return _pids; }

    // Evaluates the parton density for `x.size()` independent (x, q, pdg)
    // triples in one batched call. x/q/pdgs must be the same size; every
    // entry of pdgs must be one of pids(), or this throws.
    std::vector<double> eval_pdf(
        const std::vector<double>& x,
        const std::vector<double>& q,
        const std::vector<long int>& pdgs
    ) const;

    // Evaluates alpha_s at each entry of q. Throws if has_alpha_s() is
    // false (this PdfSet was constructed without an alpha_s_grid_file).
    std::vector<double> eval_alpha_s(const std::vector<double>& q) const;

private:
    madspace::ContextPtr _context;
    std::string _label;
    std::vector<int> _pids;
    std::unordered_map<long int, madspace::me_int_t> _pid_index;

    std::optional<madspace::PartonDensity> _density;
    madspace::RuntimePtr _density_runtime;
    std::optional<madspace::RunningCoupling> _alpha_s;
    madspace::RuntimePtr _alpha_s_runtime;
};

// A scale (muF or muR) to evaluate a PdfSet/alpha_s at, either exactly (an
// "absolute" value in GeV, the same for every event regardless of what it
// actually generated at) or relative to each event's own recorded scale (a
// multiplicative "factor", eg 2.0 for twice the event's own muF/muR -- the
// pre-existing, and still default, behaviour).
//
// The single-argument constructor is intentionally not explicit: a bare
// double (eg passed to PdfDriver::set_baseline()/add_hypothesis(), which
// take ScaleChoice by value) is always taken as a *factor*, so every
// existing call site that used to pass a plain muF_scale_factor/
// muR_scale_factor double keeps compiling and behaving identically.
struct ScaleChoice {
    double value = 1.0;
    bool absolute = false;

    ScaleChoice() = default;
    ScaleChoice(double v) : value(v), absolute(false) {}
    ScaleChoice(double v, bool is_absolute) : value(v), absolute(is_absolute) {}

    static ScaleChoice factor(double v) { return ScaleChoice(v, false); }
    static ScaleChoice fixed(double v) { return ScaleChoice(v, true); }

    // Combines this scale with a further multiplicative factor, eg a seven-
    // point envelope's 0.5/1/2 (see PdfDriver::add_scale_variation_envelope()):
    // a factor scale's own value is simply multiplied through as usual; an
    // absolute scale's value is *also* just multiplied by extra_factor (ie
    // "vary around this fixed central value"), staying absolute.
    ScaleChoice scaled_by(double extra_factor) const {
        return ScaleChoice(value * extra_factor, absolute);
    }

    // The concrete scale to evaluate at for one event whose own recorded
    // muF_/muR_ is nominal: value itself if absolute, else nominal * value.
    double resolve(double nominal) const { return absolute ? value : value * nominal; }
};

struct PdfHypothesis {
    std::string launch_name;
    std::shared_ptr<PdfSet> beam1_pdf_set;
    std::shared_ptr<PdfSet> beam2_pdf_set;
    ScaleChoice muF_scale{1.0, false};
    ScaleChoice muR_scale{1.0, false};
};

// Which of a seven-point scale-variation envelope's weights (see
// PdfDriver::add_scale_variation_envelope()) actually get written to the
// output LHE by PdfDriver::write_weights(). Other hypotheses added via
// add_hypothesis() are unaffected regardless of this setting -- it only
// ever touches launch names registered through
// add_scale_variation_envelope().
enum class ScaleVariationOutput {
    none,     // none of the seven (default): still computed (needed to find
              // the envelope, and to feed rwgt_xSec/rwgt_xErr), just not written
    envelope, // only the two whose total cross section (reweightor::rwgt_xSec)
              // is smallest/largest among the seven
    all       // all seven
};

// Builds a REX::tea::weightor evaluating hyp's PDF-luminosity factor --
// pdf(x1, muF; pdg1) * pdf(x2, muF; pdg2), further multiplied by
// alpha_s(muR)^alpha_s_power if alpha_s_power != 0 -- for every event 
std::shared_ptr<REX::tea::weightor> make_pdf_weightor(
    const PdfHypothesis& hyp, double beam1_energy, double beam2_energy, int alpha_s_power
);

class PdfDriver {
public:
    PdfDriver();
    explicit PdfDriver(madspace::ContextPtr context);

    PdfDriver& load_pdf_set(
        const std::string& name,
        const std::string& grid_file,
        const std::string& alpha_s_grid_file = "",
        std::vector<int> pids = default_pids()
    );

    // As load_pdf_set(), but reuses an already-parsed grid
    PdfDriver& add_pdf_set(
        const std::string& name,
        const madspace::PdfGrid& grid,
        const madspace::AlphaSGrid* alpha_s_grid = nullptr,
        std::vector<int> pids = default_pids(),
        const std::string& existing_globals_prefix = ""
    );

    PdfDriver& set_alpha_s_power(int power);

    // Designates which (previously load_pdf_set()'d) PDF configuration and
    // scale choice the loaded events' own weight belongs to. muF_scale/
    // muR_scale each accept either a plain double (an implicit
    // ScaleChoice::factor(), ie a multiplier of the event's own recorded
    // scale -- unchanged pre-existing behaviour) or an explicit
    // ScaleChoice::fixed(value) for an absolute scale in GeV. Single-name
    // form uses the same PDF set for both beams (symmetric colliders).
    PdfDriver& set_baseline(
        const std::string& pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );
    // Two-name form uses independent PDF sets per beam (e.g. ep colliders, or
    // asymmetric proton-ion running where beam 2 uses a nuclear PDF set).
    PdfDriver& set_baseline(
        const std::string& beam1_pdf_set_name,
        const std::string& beam2_pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );

    // Single-name: same PDF set for both beams. See set_baseline() for
    // muF_scale/muR_scale's absolute-vs-factor ScaleChoice semantics.
    PdfDriver& add_hypothesis(
        const std::string& launch_name,
        const std::string& pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );
    // Two-name: independent PDF sets per beam.
    PdfDriver& add_hypothesis(
        const std::string& launch_name,
        const std::string& beam1_pdf_set_name,
        const std::string& beam2_pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );

    // Registers the standard seven-point scale-variation envelope against
    // pdf_set_name: add_hypothesis() calls for every (muR_factor,
    // muF_factor) pair in {0.5, 1, 2}^2 except the two "uncorrelated"
    // extremes (0.5, 2) and (2, 0.5), each combined onto the given baseline
    // muF_scale/muR_scale via ScaleChoice::scaled_by() -- ie the pair (1, 1)
    // reduces to exactly (muF_scale, muR_scale), and this works the same way
    // whether the baseline is a factor or an absolute value (see
    // ScaleChoice::scaled_by()'s own doc comment). Launch names are
    // group_name + "_muR<r>_muF<f>". Which of these seven actually get
    // written to the output LHE is controlled by
    // set_scale_variation_output(); all seven are always computed
    // regardless (needed to determine the envelope in the first place).
    // group_name must be unique among add_scale_variation_envelope() calls
    // on this PdfDriver.
    PdfDriver& add_scale_variation_envelope(
        const std::string& group_name,
        const std::string& pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );
    // Two-name: independent PDF sets per beam.
    PdfDriver& add_scale_variation_envelope(
        const std::string& group_name,
        const std::string& beam1_pdf_set_name,
        const std::string& beam2_pdf_set_name,
        ScaleChoice muF_scale = ScaleChoice::factor(1.0),
        ScaleChoice muR_scale = ScaleChoice::factor(1.0)
    );

    // Which of every add_scale_variation_envelope() group's seven weights
    // actually get written to the output LHE by write_weights(). Defaults
    // to ScaleVariationOutput::none. Applies uniformly to every registered
    // group; other hypotheses added via add_hypothesis() are unaffected
    // regardless.
    PdfDriver& set_scale_variation_output(ScaleVariationOutput mode);
    ScaleVariationOutput scale_variation_output() const;

    // Fixes the beam energies used to derive each event's x1/x2 (see
    // make_pdf_weightor), overriding whatever load_events() would otherwise
    // take from the loaded lhe's own init block. Needed whenever that block
    // doesn't carry real beam energies, eg MadGraph7's internal binary
    // EventFile format: pass real beam energies here, or via meta's
    // beam1_energy/beam2_energy at load_events() instead.
    PdfDriver& set_beam_energies(double beam1_energy, double beam2_energy);

    // Sniffs path for MadGraph7's internal binary EventFile format or LHEF
    // XML, then builds the owned REX::tea::reweightor against the result.
    // meta supplies init-block information (in particular beam energies)
    // for the binary path; ignored when path is XML.
    // Must be called after set_baseline() and every add_hypothesis() call it
    // should cover; requires beam energies via set_beam_energies(), meta, or the
    // loaded XML's own <init> block; otherwise throws. 
    // May be called again to rebuild against a different event source.
    PdfDriver& load_events(const std::string& path, const madspace::LHEMeta& meta = {});

    bool binary_output() const { return _binary_output; }
    PdfDriver& set_binary_output(bool binary);

    // Before actually writing anything, drops whichever of every
    // add_scale_variation_envelope() group's weight columns
    // set_scale_variation_output() says shouldn't be written (see its own
    // doc comment) from every event's wgts_ and from weight_ids -- so
    // those columns are entirely absent from the output, not just zeroed.
    // Idempotent: safe to call write_weights() more than once.
    PdfDriver& write_weights(const std::string& path);

    // Drops every currently-registered weight column whose id is *not* in
    // keep_names, from both weight_ids and every event's wgts_ -- a public,
    // name-driven generalization of write_weights()'s own
    // add_scale_variation_envelope()-keyed filtering, for callers that
    // build their own hypothesis set directly via add_hypothesis() (so have
    // no add_scale_variation_envelope() group for that mechanism to key
    // off) but still want only some of the resulting weights actually
    // written. A name not currently present is silently ignored. Must be
    // called after get().run() (so weight_ids/events are populated) and
    // before write_weights(); idempotent, like write_weights()'s own
    // filtering.
    //
    // Python callers in particular need this: the bound weight_ids
    // property replaces its whole shared_ptr on assignment, which would
    // desync it from each event's own (separately shared_ptr-aliased)
    // weight_ids -- this method instead erases in place, the same way
    // write_weights()'s internal filtering does, so the aliasing (and thus
    // every event's own weight_ids) stays correct.
    PdfDriver& keep_weights(const std::vector<std::string>& keep_names);

    // End-to-end convenience: load_events(events_path, meta), run the
    // reweighting (REX::tea::reweightor's full setup/run_all_iterations/
    // finalise_reweighting loop), then write_weights(output_path).
    PdfDriver& reweight(
        const std::string& events_path,
        const std::string& output_path,
        const madspace::LHEMeta& meta = {}
    );

    madspace::ContextPtr context() const { return _context; }
    // Access to the owned REX::tea::reweightor once load_events() has built
    // it. Throws otherwise.
    REX::tea::reweightor& get();
    const REX::tea::reweightor& get() const;
    bool built() const noexcept { return static_cast<bool>(_reweightor); }

    // The muted-by-default warning logger shared with the built reweightor
    // (and, transitively, its procReweightor -- see
    // REX::tea::reweightor::extract_sorter()). Muted (false) from
    // construction, independent of whether load_events() has been called
    // yet; call verbose().set_verbose(true) to unmute. Re-linked into the
    // reweightor's own verbose on every load_events()/rebuild.
    REX::verbosity& verbose() { return _verbose; }
    const REX::verbosity& verbose() const { return _verbose; }
    // Plain-bool convenience for verbose().set_verbose(v)/is_verbose().
    PdfDriver& set_verbose(bool v) {
        _verbose.set_verbose(v);
        return *this;
    }
    bool is_verbose() const { return _verbose.is_verbose(); }

    // The standard gluon + 5-quark-flavor (and antiquarks) QCD-collider PDG
    // set: {21, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5}.
    static std::vector<int> default_pids();

private:
    madspace::ContextPtr _context;
    std::unordered_map<std::string, std::shared_ptr<PdfSet>> _pdf_sets;
    int _alpha_s_power = 0;
    std::optional<PdfHypothesis> _baseline;
    std::vector<PdfHypothesis> _hypotheses;
    // One entry per add_scale_variation_envelope() call, each the seven
    // launch_names it registered into _hypotheses, in the same order.
    std::vector<std::vector<std::string>> _scale_variation_groups;
    ScaleVariationOutput _scale_variation_output = ScaleVariationOutput::none;
    double _beam1_energy = 0.0, _beam2_energy = 0.0;
    bool _beam_energies_set = false;
    bool _binary_output = false;
    REX::verbosity _verbose{false};
    std::unique_ptr<REX::tea::reweightor> _reweightor;

    const std::shared_ptr<PdfSet>& find_pdf_set(const std::string& name) const;
    void filter_scale_variation_weights();
};

} // namespace madtrex
