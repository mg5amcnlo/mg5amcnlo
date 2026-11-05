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
/* #    * MadGraph7 teaRex Reweighting executors *   # */
/* #    ******************************************   # */
/* ################################################### */
//
// MadtRex is the primary MadGraph7 driver for data-parallel
// event reweighting, using the Rex/teaRex libraries wrapper
// with the TupperWare and WareHouse classes to simplify the
// interface between MadGraph7 and Rex/teaRex.
// MadtRex provides immediate access to SLHA parameter rewegighting
// with UMAMI-compliant matrix element libraries through the 
// MadtRex driver type, which can load data directly from
// provided paths to relevant process information, i.e. LHE
// files, reweighting cards, SLHA parameter cards, and the
// utilised scattering amplitude libraries.
// The driver is minimal, only wrapping the REX::tea::reweightor
// and TupperWare types with necessary plumbing and bureaucracy
// to utilise scattering amplitudes safely.

#pragma once
#include "binary_io.hpp"
#include "param_handler.hpp"
#include "tupperware.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace madtrex {

using namespace madspace;

    // Which on-disk representation a set of reweighting events came from, or
    // should be written back out as: LHEF XML, or MadGraph7's default
    // internal binary EventFile format (io.hpp).
    enum class LheFormat { xml, binary };

    // Peeks at path's first bytes to tell a madspace binary EventFile (NumPy
    // magic "\x93NUMPY", see io.cpp's write_event_header) apart from LHEF
    // XML, without relying on the file extension. Throws if path cannot be
    // opened.
    LheFormat detect_lhe_format(const std::string& path);

    std::shared_ptr<REX::lhe> load_lhe_xml(const std::string& path);
    std::shared_ptr<REX::lhe> load_lhe_xml(std::istream& stream);
    // Dispatches to load_lhe_xml or madtrex::load_lhe_binary based on
    // detect_lhe_format(path).
    std::shared_ptr<REX::lhe> load_lhe(const std::string& path);

    // ---------------------------------------------------------------------
    // Splitting a job across independent shards, and recombining them
    // ---------------------------------------------------------------------
    //
    // Large reweighting jobs can be split into n_shards independent pieces
    // (eg to run one per cluster node) via split_events(), each piece
    // reweighted completely independently (via the ordinary Driver API,
    // pointed at that one shard -- no new API needed for that half), and
    // the n_shards results stitched back together via combine_shards(),
    // which is also where -- and the *only* place where -- the combined
    // sample's cross section and error actually get computed.

    // The result of split_events(): one binary EventFile path per shard
    // (empty if that shard ended up with zero events, eg n_shards exceeded
    // the source's event count -- no file is created for it), and the
    // event count actually written to each, both in shard order.
    struct SplitResult {
        std::vector<std::string> paths;
        std::vector<std::size_t> counts;
    };

    // Splits the event source at path (auto-detected XML or MadGraph7's
    // internal binary EventFile via detect_lhe_format()) into n_shards
    // separate binary EventFile shards on disk, named
    // "<output_prefix>_shard<i>.npy" for i in [0, n_shards) -- always
    // binary for the shards regardless of path's own format, via
    // REX::split_source() + open_lhe_sink_binary(), so each one is
    // independently a load_events()-ready input for its own, completely
    // separate Driver/job (eg one per cluster node) reweighting only that
    // shard. path is streamed directly either way (REX::open_lhe_source()
    // for XML, open_lhe_source_binary() for binary; meta is only consulted
    // for the latter, matching load_events()), so memory use stays
    // O(batch_size) regardless of path's total size. particle_count bounds
    // every shard's fixed per-event particle-slot count (see
    // open_lhe_sink_binary()); left at 0, each shard determines its own
    // from whichever event lands there first.
    //
    // This only ever splits *input* events, never weights: recombining
    // per-shard reweighting results (and computing the combined sample's
    // cross section/error exactly once, over the totality) is
    // combine_shards()'s job.
    SplitResult split_events(
        const std::string& path,
        const std::string& output_prefix,
        std::size_t n_shards,
        std::size_t particle_count = 0,
        std::size_t batch_size = 10000,
        const madspace::LHEMeta& meta = {}
    );

    // Recombines n_shards shards previously produced by split_events(),
    // each independently reweighted (eg via a Driver::reweight()/
    // write_weights() call per shard, with binary_output() left true --
    // the natural, and only currently supported, output for a per-shard
    // job in this workflow), back into a single result: for shard i, loads
    // its original events from shard_event_paths[i] (via
    // load_lhe_binary()) and its reweighted columns from
    // "<shard_weight_prefixes[i]>_<weight_id>.npy" per entry of
    // launch_names, in order (via load_weights_binary()), reattaches them
    // as that shard's events' wgts_, then appends all n_shards shards'
    // events, in order, into one combined REX::tea::reweightor.
    //
    // Cross sections and errors (reweightor::calc_xSecs()/calc_xErrs())
    // are computed exactly once here, over the full recombined sample --
    // this is the *only* place in the split/reweight/recombine workflow
    // that does so. A per-shard job must never compute (or, worse, write
    // out) its own cross-section/error estimate over just its own
    // fraction of the events: unlike a per-event weight, cross section
    // and (especially) its error do not additively decompose across an
    // arbitrary partition of the sample, so combining separately-computed
    // per-shard estimates afterward would silently be statistically
    // wrong. Using save_weights_binary() as every shard's own output
    // format (as this workflow requires) already sidesteps the temptation
    // structurally -- it never writes xSec_/xSecErr_/rwgt_xSec/rwgt_xErr
    // at all, only raw per-event weight columns.
    //
    // Writes the combined result to output_path as LHEF XML
    // (REX::write_lhef) unless binary_output is true, in which case
    // save_weights_binary() is used instead, the same convention
    // Driver::write_weights() follows. meta supplies init-block
    // information the same way it does for load_events()/load_lhe_binary()
    // (the binary format itself carries none); consulted once, for
    // shard_event_paths[0], not per shard. Returns the combined reweightor
    // (already finalised: calc_xSecs()/calc_xErrs() already run) for
    // callers that want to inspect rwgt_xSec/rwgt_xErr/norm_factor
    // themselves afterward.
    REX::tea::reweightor combine_shards(
        const std::vector<std::string>& shard_event_paths,
        const std::vector<std::string>& shard_weight_prefixes,
        const std::vector<std::string>& launch_names,
        const std::string& output_path,
        bool binary_output = false,
        const madspace::LHEMeta& meta = {}
    );

    // ---------------------------------------------------------------------
    // MadMatrix/UMAMI subprocess JSON loading
    // ---------------------------------------------------------------------
    //
    // MadMatrix compiles each physical subprocess of a MadGraph7 process (eg
    // the "gg > ttx" piece of "p p > 2j") into its own UMAMI-compliant matrix
    // element library, and describes the resulting set of subprocesses in a
    // PROCESS_DIRECTORY/SubProcesses/subprocesses.json file: a JSON array with
    // one entry per subprocess, giving its incoming/outgoing PDG codes, the
    // (device-templated) path to its compiled library, the flavor channels it
    // covers (eg one library often serves several external quark flavors, up
    // to the color/coupling factors UMAMI computes internally), and the full
    // list of helicity combinations it was generated for.
    //
    // The types and functions below parse that file and turn it into
    // REX::tea-ready building blocks: SubProcessSpec is the parsed form of one
    // entry, make_event_checker() builds the flavor/helicity-aware
    // REX::eventBelongs for it, resolve_me_path() finds the concrete library
    // file the device-templated me_path refers to, and MadtRex ties all of
    // this together with a madspace::Context and a madtrex::WareHouse into a
    // single driver for one process directory.

    // One entry of a subprocess's JSON "flavors" array, expanded so that
    // mirrored entries (incoming beams swapped, eg q qbar vs qbar q sharing one
    // library) become a second, explicit FlavorChannel instead of needing
    // special-casing at match time. `pdgs` is the exact, positional external-leg
    // PDG tuple expected of an event's own particle slot order (incoming legs,
    // in event order, followed by outgoing legs, in event order) for this
    // specific variant; `mirrored` records whether this variant is the
    // incoming-swapped copy of a JSON option, so per-event helicity lookups
    // know to undo the swap before consulting SubProcessSpec::helicities, which
    // is always recorded in the un-mirrored leg order.
    struct FlavorChannel {
        int index = -1;
        std::vector<long int> pdgs;
        bool mirrored = false;
    };

    // Parsed form of one entry of a subprocesses.json array: one physical
    // MadMatrix subprocess, carrying enough information to load its matrix
    // element library and build a flavor/helicity-aware event checker for it.
    // Phase-space-only JSON fields (channels, color_flows, pdg_color_types) are
    // intentionally not retained here: they describe multichannel phase-space
    // generation, not the reweighting-facing contract between an event and a
    // UMAMI matrix element call.
    struct SubProcessSpec {
        std::vector<long int> incoming;
        std::vector<long int> outgoing;
        std::string me_path_template; // eg "lib/libmadmatrix_P1_gg_ttx_{device}.so", relative to PROCESS_DIRECTORY
        std::string source_path;      // JSON "path" field; informational only, may be stale if the process directory was moved since generation
        std::size_t diagram_count = 0;
        std::vector<FlavorChannel> flavors;       // expanded, incl. mirror variants
        std::vector<std::vector<int>> helicities; // as given in JSON, un-mirrored leg order
    };

    // Parses a MadMatrix subprocesses.json file (eg
    // PROCESS_DIRECTORY/SubProcesses/subprocesses.json) into one
    // SubProcessSpec per top-level array entry.
    std::vector<SubProcessSpec> parse_subprocesses_json(const std::string& subprocesses_json_path);

    // Builds the flavor/helicity-aware REX::eventBelongs for a single
    // SubProcessSpec: one template event per distinct (pre-mirror-expansion)
    // flavor option, matched via the default external_legs_comparator (ie PDG
    // multiset per status, which mirroring doesn't change), with
    // flavor_index/helicity_index hash functions that positionally re-match
    // the specific (possibly mirrored) variant to report the UMAMI flavor
    // index and, if the event carries genuine per-particle helicities in its
    // spin_ fields (LHEF-style +-1; anything else is treated as "no fixed
    // helicity"), the position of that helicity combination in
    // SubProcessSpec::helicities. Events without fixed helicities are left
    // with helicity_ unset (REX::npos truncated into event::helicity_'s -1
    // sentinel), meaning the matrix element should sum over helicities.
    REX::eventBelongs make_event_checker(const SubProcessSpec& spec);

    // Given PROCESS_DIRECTORY and a me_path template containing literal
    // "{device}" and/or "{fptype}" placeholders (see me_lib_format in
    // MadMatrix's own madmatrix/output.py, and MADMATRIX_LIB in
    // madmatrix.mk), lists whichever concrete files actually exist on disk
    // (eg "libmadmatrix_sm_P1_gg_ttx_cppnone_m.so") and resolves which one
    // to use via device_priority/fptype_priority: device is resolved
    // first -- the first device_priority entry with a matching file wins;
    // if none match but exactly one device variant was found (for any
    // fptype), that one is used regardless -- then, among files matching
    // the chosen device, fptype is resolved the same way via
    // fptype_priority. A template with only one of the two placeholders
    // (or neither, eg an already-fully-resolved path) resolves that one
    // placeholder alone, or is returned as-is if it exists. Throws (no
    // match at any stage, or several found and no priority set) listing
    // what was found, so the caller can call
    // Driver::set_device_priority(...)/set_fptype_priority(...) and try
    // again.
    std::string resolve_me_path(
        const std::string& process_directory,
        const std::string& me_path_template,
        const std::vector<std::string>& device_priority,
        const std::vector<std::string>& fptype_priority = {}
    );

    // Driver is the top-level driver tying together JSON-described MadMatrix
    // subprocesses, the madspace::Context that loads and owns their UMAMI
    // matrix element libraries, and the madtrex::WareHouse of
    // REX::tea::procReweightors built from them. One Driver instance
    // corresponds to one MadGraph7 process directory (eg PROCMG7_sm_2):
    // load_process_directory()/load_process() may be called more than once to
    // register subprocesses from several subprocesses.json files into the same
    // WareHouse before calling load_events() (or warehouse().build(lhe)
    // directly, if the events are already in hand as a REX::lhe rather than
    // on disk).
    class Driver {
    public:
        Driver();
        explicit Driver(madspace::ContextPtr context);

        // Priority order used to disambiguate multiple {device}-suffixed
        // library variants found for the same subprocess (eg prefer
        // "cppavx2" over "cppnone" once vectorized kernels exist). Earlier
        // entries win. Empty by default, meaning: auto-use a lone match,
        // throw on ambiguity.
        Driver& set_device_priority(std::vector<std::string> priority);
        const std::vector<std::string>& device_priority() const;

        // As device_priority(), but disambiguates multiple {fptype}-suffixed
        // library variants found for the same subprocess (see MADMATRIX_LIB
        // in madmatrix.mk, SUPPORTED_FPTYPES therein for the full set of
        // legal values: "d", "f", "m"), among whichever ones exist for the
        // already-resolved device (see resolve_me_path()). Defaults to
        // {"m", "d", "f"}, matching madmatrix.mk's own FPTYPE=m default; an
        // explicitly empty priority means: auto-use a lone match, throw on
        // ambiguity.
        Driver& set_fptype_priority(std::vector<std::string> priority);
        const std::vector<std::string>& fptype_priority() const;

        // Priority order used to resolve which {fptype}-suffixed library
        // variant load_process()/load_process_directory() loads for each
        // subprocess's *normaliser* (REX::tea::procReweightor's own
        // normalisation weightor -- see TupperWare::set_normaliser()), kept
        // deliberately independent of fptype_priority() (used for the
        // reweight-*target* api instead). This is a fundamental assumption
        // of how reweighting works here: every launch must be normalised
        // against the sample's actual, original configuration -- never
        // against whatever it's being reweighted TO -- exactly as ordinary
        // physics-parameter reweighting already does (normaliser = the
        // as-loaded param_card, untouched by any launch's "set BLOCK ID
        // VALUE" override, which only ever gets applied to the reweight-
        // target api via ParamHandler). A change of *fptype* additionally
        // needs a differently-*compiled* library for that role, which is
        // exactly what this priority controls.
        //
        // Left unset (the default) means: reuse fptype_priority() for the
        // normaliser too, ie load_process() shares one api between both
        // roles -- correct and harmless when a card never changes fptype
        // (the sample's own fptype IS the target fptype in that case), and
        // preserves this class's pre-existing behavior for every caller
        // that never calls this setter. Call this explicitly with the
        // sample's *original* fptype (madtrex.py's run_parameter_
        // reweighting() reads this from the process directory's own
        // run_card.toml [run] fptype) whenever fptype_priority() targets a
        // *different* fptype than the sample was actually generated at --
        // load_process() then loads a second, independent api for the
        // normaliser role, deliberately never registered via add_api()
        // (see its own implementation comment), so ParamHandler's per-
        // launch parameter overrides can never touch it.
        Driver& set_normaliser_fptype_priority(std::vector<std::string> priority);
        const std::vector<std::string>& normaliser_fptype_priority() const;

        // Helicity reweighting (true, the default) or helicity summation
        // (false) for subprocesses loaded by load_process()/
        // load_process_directory() from here on (it does not retroactively
        // affect subprocesses already loaded). With helicity reweighting,
        // each event is reweighted at its own helicity: the matrix elements
        // are evaluated for the helicity combination of the event's LHE
        // SPINUP column (UMAMI_IN_HELICITY_INDEX) rather than summed over
        // helicities. A matrix element without per-event helicity support
        // falls back to helicity summation with a one-time warning on stderr;
        // a sample without helicity information (any particle with a SPINUP
        // other than -1, 0, +1, eg 9) is an error -- reweight it with
        // helicity_reweighting off. See default_input_keys() and
        // input_keys_for().
        Driver& set_helicity_reweighting(bool helicity_reweighting);
        bool helicity_reweighting() const;
        // The process directory (and param_card, default its Cards/
        // param_card.dat) whose matrix elements subprocesses loaded from
        // here on are normalised against, instead of their own: the
        // original process of a "change model" reweighting, whose events
        // were generated with it. The weights are then the new model's
        // matrix element over the original one, MadGraph's reweighting
        // convention. Every subprocess loaded must have an identical
        // counterpart there (same legs, flavor channels and helicity
        // table), or load_process() throws; its library is the one
        // normaliser_fptype_priority() resolves, and it is evaluated the
        // same way as the reweighted one (both at the event's helicity, or
        // both helicity-summed). Empty (the default): the loaded process
        // directory's own.
        Driver& set_normaliser_process_directory(
            const std::string& process_directory, const std::string& param_card = ""
        );
        const std::string& normaliser_process_directory() const;
        // The param_card the normaliser is loaded with: the one the events
        // were generated with, when the param_card the reweighted matrix
        // elements start from (load_process()'s) is another one. Empty (the
        // default): with a normaliser process directory, its Cards/
        // param_card.dat, else load_process()'s param_card. Also set by
        // set_normaliser_process_directory().
        Driver& set_normaliser_param_card(const std::string& param_card);
        const std::string& normaliser_param_card() const;

        // Whether subprocesses loaded from here on normalise against the
        // squared matrix elements the events were generated with, when the
        // event file stores them (the "matrix_element" column of a madspace
        // binary event file) and they are the quantity the normaliser
        // computes, instead of evaluating the normaliser again (see
        // TupperWare::set_stored_normalisation()). True by default; turn it
        // off to always evaluate the normaliser.
        Driver& set_use_stored_normalisation(bool use_stored_normalisation);
        bool use_stored_normalisation() const;
        // Former names of set_helicity_reweighting()/helicity_reweighting().
        Driver& set_use_helicity(bool use_helicity);
        bool use_helicity() const;

        // The muted-by-default warning logger shared with the built
        // WareHouse/reweightor (see WareHouse::verbose()). Call
        // verbose().set_verbose(true) to unmute.
        REX::verbosity& verbose();
        const REX::verbosity& verbose() const;
        // Plain-bool convenience for verbose().set_verbose(v)/is_verbose().
        Driver& set_verbose(bool v);
        bool is_verbose() const;

        // Maximum number of events per REX::process (see
        // WareHouse::set_max_batch()/REX::lhe::max_batch), applied by
        // load_events()/warehouse().build(). Left at REX::npos (unset) by
        // default. Setting this also makes every subprocess loaded via
        // load_process()/load_process_directory() reload its own
        // MatrixElementApi instance per batch instead of every max_batch
        // overflow clone sharing (and serializing calls through) the one
        // instance that was loaded for it -- see TupperWare::build() and
        // ReloadSpec.
        Driver& set_max_batch(std::size_t max_batch);
        std::size_t max_batch() const;

        // Worker threads and the minimum events per batch when splitting
        // the sample across them: see WareHouse::set_threads()/
        // set_min_batch()/batch_size_for(). Applied by load_events().
        Driver& set_threads(std::size_t threads);
        std::size_t threads() const;
        Driver& set_min_batch(std::size_t min_batch);
        std::size_t min_batch() const;

        // Passed straight through to WareHouse::set_max_deferred_iterations()
        // (see its own doc comment): caps how many outer reweighting
        // iterations' worth of deferred (eg GPU) results are allowed to
        // accumulate device-side, per process, before an automatic mid-run
        // flush back to the host. REX::npos (the default) means no cap --
        // deferred results are only ever copied back once, at the end of
        // the whole run, same as before this existed.
        Driver& set_max_deferred_iterations(std::size_t max_deferred_iterations);
        std::size_t max_deferred_iterations() const;

        // Enables Rex's streaming mode for the next load_events() call:
        // rather than loading the whole event file into memory up front,
        // load_events() opens a REX::eventSource for path and builds the
        // WareHouse's reweightor via REX::lhe::open_streaming(), so events
        // are pulled and processed batch_size at a time. Off by default.
        // Only supported for LHEF XML event sources (REX::open_lhe_source);
        // load_events() throws if streaming is enabled and path is detected
        // as MadGraph7's internal binary EventFile format. Once enabled,
        // run the reweighting via run_streaming() (or reweight()'s
        // end-to-end convenience) instead of warehouse().get().run() +
        // write_weights(): streamed output is written incrementally to a
        // sink as each batch finishes, not held for a later
        // write_weights() call.
        Driver& set_streaming(bool enable, std::size_t batch_size = 10000);
        bool streaming() const;
        std::size_t stream_batch_size() const;

        // Runs the streaming reweightor built by load_events() (see
        // set_streaming()) to completion, writing events incrementally to
        // output_path as an LHEF XML sink (REX::open_lhe_sink) unless
        // binary_output() is true (set automatically by load_events() when
        // it detected a binary source), in which case output_path is
        // instead used as a path_prefix for one streamed weight-only
        // binary file per launch name (open_weights_sink_binary()), the
        // same "<output_path>_<weight_id>.npy" convention
        // save_weights_binary()/write_weights() use -- this requires
        // warehouse().launch_names() to be non-empty (eg via
        // load_param_reweighting() or warehouse().set_launch_names()),
        // since binary weight files are named per weight id; throws
        // otherwise. Throws if streaming() is false, or the WareHouse
        // hasn't been built yet.
        Driver& run_streaming(const std::string& output_path, bool include_ids = false);

        // Parses subprocesses_json_path, loads each entry's matrix element
        // library (resolving {device} against what's actually on disk) via
        // the owned Context, and registers one TupperWare (api +
        // flavor/helicity-aware event checker) per entry into the owned
        // WareHouse. param_card defaults to
        // PROCESS_DIRECTORY/Cards/param_card.dat, where PROCESS_DIRECTORY is
        // taken to be the grandparent directory of subprocesses_json_path (ie
        // .../PROCESS_DIRECTORY/SubProcesses/subprocesses.json).
        Driver& load_process(const std::string& subprocesses_json_path, const std::string& param_card = "");
        // Convenience: process_directory/SubProcesses/subprocesses.json.
        Driver& load_process_directory(const std::string& process_directory, const std::string& param_card = "");

        // Configures UMAMI-direct parameter reweighting: builds (or
        // replaces) the owned ParamHandler over every matrix element api
        // registered so far (ie everything load_process()/
        // load_process_directory() has loaded up to this call -- call this
        // after those, not before), parses rwgt_path, and wires the
        // resulting iterators/launch_names onto the WareHouse via
        // warehouse().set_iterators()/set_launch_names(). Must be called
        // before load_events() (or warehouse().build()), since that is what
        // actually captures the WareHouse's current iterators/launch_names
        // into the built reweightor. Requires every registered api to
        // implement umami_set_parameter (see ParamHandler::add_api); throws
        // otherwise -- for libraries that don't, build the iterators from
        // REX::tea::param_rwgt's file-based approach instead and pass them to
        // warehouse().set_iterators() directly.
        Driver& load_param_reweighting(const std::string& rwgt_path);

        ParamHandler& param_handler();
        const ParamHandler& param_handler() const;

        // Loads the event source the reweighting will run against: sniffs
        // path via detect_lhe_format() to decide between MadGraph7's
        // internal binary EventFile format and LHEF XML, loads it
        // accordingly, and builds the WareHouse's REX::tea::reweightor
        // against the result (ie warehouse().build(*load_lhe(path))).
        // meta supplies init-block information (beam ids/energies, xsec,
        // PDF set) for the binary path, which -- unlike LHEF -- stores none
        // of that on disk; ignored when path is XML, whose own <init> block
        // is used instead. Also sets binary_output() to match the detected
        // input format, so write_weights() round-trips to the same
        // representation by default; call set_binary_output() afterwards to
        // override. May be called again to rebuild against a different
        // event source, discarding any previously-built reweightor (see
        // WareHouse::build()).
        //
        // If set_streaming(true) was called, this instead opens a streaming
        // REX::eventSource for path (which must be LHEF XML; see
        // set_streaming()) and builds against a REX::lhe::open_streaming()
        // lhe, without pulling any events yet -- use run_streaming() rather
        // than warehouse().get().run() + write_weights() to actually run
        // the reweighting in that case.
        Driver& load_events(const std::string& path, const madspace::LHEMeta& meta = {});

        // Which format write_weights() targets: true for
        // madtrex::save_weights_binary() (one weight-only binary EventFile
        // per weight id), false for a full LHEF XML rewrite via
        // REX::write_lhef() (events plus every recorded weight). Defaults to
        // whatever load_events() last detected; false if load_events() was
        // never called.
        bool binary_output() const;
        Driver& set_binary_output(bool binary);

        // Writes out whatever weights are currently recorded on the built
        // reweightor's events (ie warehouse().get(), typically after
        // REX::tea::reweightor::run()) via save_weights_binary(path) or
        // REX::write_lhef(path), according to binary_output(). Throws if
        // the WareHouse hasn't been built yet (see load_events()).
        Driver& write_weights(const std::string& path);

        // End-to-end convenience running the entire pipeline in one call:
        // load_process_directory(process_directory, param_card), then --
        // if rwgt_path is non-empty -- load_param_reweighting(rwgt_path),
        // then load_events(events_path, meta), then
        // warehouse().get().run() (REX::tea::reweightor's full setup/
        // run_all_iterations/finalise_reweighting loop), then
        // write_weights(output_path). rwgt_path may be left empty to skip
        // parameter reweighting (eg when every process's WareHouse entry
        // already fully determines its own reweighting via
        // add_api()/set_normaliser() and needs no per-iteration parameter
        // changes at all). Returns *this so eg driver.reweight(...).warehouse()
        // stays chainable for inspecting the result afterwards.
        Driver& reweight(
            const std::string& process_directory,
            const std::string& events_path,
            const std::string& output_path,
            const std::string& rwgt_path = "",
            const std::string& param_card = "",
            const madspace::LHEMeta& meta = {}
        );

        madspace::ContextPtr context() const;
        WareHouse& warehouse();
        const WareHouse& warehouse() const;
        const std::vector<SubProcessSpec>& specs() const;

    private:
        madspace::ContextPtr reload_context();

        madspace::ContextPtr _context;
        madspace::ContextPtr _reload_context;
        std::vector<std::string> _device_priority = {"cpp512z", "cpp512y", "cppavx2", "cppsse4", "cppnone"};
        std::vector<std::string> _fptype_priority = {"m", "d", "f"};
        // Unset (see set_normaliser_fptype_priority()'s own doc comment) by
        // default: _normaliser_fptype_priority_set distinguishes "never
        // called set_normaliser_fptype_priority()" from "called it with an
        // explicitly empty priority" (itself a legal, meaningful value --
        // "auto-use a lone match, throw on ambiguity", same as
        // fptype_priority()'s own empty-priority meaning).
        std::vector<std::string> _normaliser_fptype_priority;
        bool _normaliser_fptype_priority_set = false;
        WareHouse _warehouse;
        std::vector<SubProcessSpec> _specs;
        ParamHandler _param_handler;
        bool _binary_output = false;
        bool _helicity_reweighting = true;
        bool _use_stored_normalisation = true;
        std::string _normaliser_process_directory;
        std::string _normaliser_param_card;
        bool _streaming = false;
        std::size_t _stream_batch_size = 10000;
    };

}