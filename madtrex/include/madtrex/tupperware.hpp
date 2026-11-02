/* .---------------------------------------------------------. */
/* | _____                          __        __             | */
/* ||_   _|   _ _ __  _ __   ___ _ _\ \      / /_ _ _ __ ___ | */
/* |  | || | | | '_ \| '_ \ / _ \ '__\ \ /\ / / _` | '__/ _ \| */
/* |  | || |_| | |_) | |_) |  __/ |   \ V  V / (_| | | |  __/| */
/* |  |_| \__,_| .__/| .__/ \___|_|    \_/\_/ \__,_|_|  \___|| */
/* |           |_|   |_|                                     | */
/* '---------------------------------------------------------' */
/*       Tearex-Umami Pointer Passing Event Reweighting        */
/*           With Accelerated Repetitive Execution             */
//
// TupperWare is the "simple" generic reweighting bridge between MadGraph7 and
// the standalone Rex/teaRex libraries: it turns already-loaded, UMAMI-compliant
// madspace::MatrixElementApi objects into REX::tea::weightors, ie plain
// std::function<shared_ptr<vector<double>>(REX::process&)> callables, and combines
// them with a caller-supplied event checker into a REX::tea::procReweightor.
//

#pragma once

#include "teaRex.h"
#include "madspace/driver/context.hpp"

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <string>
#include <vector>

namespace madtrex {

// Identifies how to reload a fresh, independent instance of a
// MatrixElementApi's underlying UMAMI library from its owning Context --
// same file, same parameter card, but a distinct in-memory instance with its
// own global state (its own process_instance()s, its own set/get parameter
// state, etc). Attached to an add_api()/set_normaliser() registration so
// that TupperWare::build() can honor WareHouse::set_max_batch(): once
// REX::lhe::max_batch causes a process to be split across several
// REX::process batches (reweightor::extract_sorter()'s overflow cloning),
// each batch's weightor reloads its own instance via this instead of every
// batch sharing -- and having to serialize calls through, see
// detail::api_call_mutex -- the one MatrixElementApi that add_api()/
// set_normaliser() was originally given. Only consulted when the TupperWare
// is actually built with reload_per_batch=true (see TupperWare::build()),
// which WareHouse::build() does exactly when set_max_batch() is active.
//
// The instance is shared by the weightors of the batch (see
// detail::reload_matrix_element()): one whose normaliser uses the same
// library loads it once, while the normalisations are computed.
struct ReloadSpec {
    madspace::ContextPtr context;
    std::string me_path;
    std::string param_card;
};

namespace detail {
// Internal, not part of the public TupperWare/WareHouse API. api's
// process_instance() resolves to a per-thread instance via
// madspace::ThreadPool::thread_index(), a thread_local that defaults to 0 on
// any thread that isn't a madspace ThreadPool worker; REX::tea::threadPool
// (both for procReweightor evaluation and, potentially, madtrex::ParamHandler
// iterators run from within one) spawns its own plain std::threads, which all
// resolve to instance slot 0. Every call into api that touches that instance
// -- MatrixElementApi::call() in make_weightor()'s evaluate(), and
// set_parameter()/get_parameter() from madtrex::ParamHandler -- goes through
// this same per-api mutex so they can't race regardless of which thread pool
// (if any) ends up invoking them.
std::mutex& api_call_mutex(const madspace::MatrixElementApi& api);
// Prints message to stderr the first time key is seen in this process (see
// its definition); shared by TupperWare and Driver for conditions that change
// what is computed, such as falling back to helicity summation.
void warn_once(const std::string& key, const std::string& message);
// MadMatrix libraries keep their model parameters in a process-wide
// singleton, in the common library all subprocess libraries of a process
// directory link against: every MatrixElementApi of the process shares them
// (an instance reloaded from a ReloadSpec follows ParamHandler's launches
// like the api it was reloaded for), and umami_initialize() -- run by each
// load_matrix_element() -- resets them to the param_card. So:
// - set_api_parameter(): api.set_parameter(), with the value recorded per
//   library (file_name()) (madtrex::ParamHandler's launches). Callers hold
//   api_call_mutex(api), as for any other set_parameter() call;
// - reload_matrix_element(): the instance reloaded from reload for the batch
//   proc, loaded on its first use (and kept in proc's device_data, shared by
//   the batch's weightors) holding parameters_lock() exclusively, with the
//   recorded parameter values set again right after;
// - forget_api_parameters(): drops what is recorded for api's library, for a
//   Driver that has just loaded it (umami_initialize() reset its parameters
//   to the param_card): the values a previous Driver's launches set on the
//   same library (the normaliser of a later fptype or model group is the
//   original's library) must not follow its reloaded instances;
// - parameters_lock(): held shared by every matrix element evaluation.
std::shared_mutex& parameters_lock();
void set_api_parameter(
    const madspace::MatrixElementApi& api, const std::string& location, double real, double imag
);
void forget_api_parameters(const madspace::MatrixElementApi& api);
const madspace::MatrixElementApi&
reload_matrix_element(const ReloadSpec& reload, REX::process& proc);
} // namespace detail

// event::helicity_ of an event whose helicities are all fixed (LHE SPINUP -1,
// 0 or +1) but whose helicity configuration is not in the helicity table of
// its subprocess, as set by Driver's event checkers; -1 (REX's default) means
// the event has no fixed helicities. Either is an error with helicity
// reweighting, and irrelevant with helicity summation.
inline constexpr int helicity_not_in_table = -2;

// Whether an event lists its two incoming legs the other way round from the
// matrix element's leg order: an event of a beam-swapped ("mirrored") flavor
// channel, as MG7 writes them (LHE puts the +z beam first, while the matrix
// element was evaluated in its own leg order). Such events are evaluated with
// legs 0 and 1 swapped back -- momenta, and so everything derived from them --
// without touching the events themselves. Driver derives this from the flavor
// channels of subprocesses.json; empty means no event is swapped.
using IncomingSwapped = std::function<bool(const REX::event&)>;

// Converts an already-loaded madspace::MatrixElementApi into a REX::tea::weightor.
// Input keys are auto-derived from api.required_inputs()/supported_inputs(),
// restricted to what can actually be sourced from a REX::process: momenta
// (UMAMI_IN_MOMENTA), the strong coupling (UMAMI_IN_ALPHA_S), and, if the api
// supports channel pinning, the per-event flavor_/helicity_ indices
// (UMAMI_IN_FLAVOR_INDEX / UMAMI_IN_HELICITY_INDEX). Throws if the api requires an
// input that cannot be derived this way (eg a sampling-time random number), or if
// it does not report supporting UMAMI_OUT_MATRIX_ELEMENT. The returned weightor
// reads the process's momenta out of its flat SoA storage and reshapes them into
// UMAMI's batch-contiguous call convention; all matching events in a process are
// evaluated in a single UMAMI call.
std::shared_ptr<REX::tea::weightor> make_weightor(const madspace::MatrixElementApi& api);

// As above, but with explicit control over which UMAMI keys are used. input_keys
// may only contain keys derivable from a REX::process (UMAMI_IN_MOMENTA,
// UMAMI_IN_ALPHA_S, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_HELICITY_INDEX); output_keys
// may only contain UMAMI_OUT_MATRIX_ELEMENT and UMAMI_OUT_DIAGRAM_AMP2 (the latter
// is summed over diagrams before being folded into the result). result_keys picks
// which of the requested output_keys are summed together into the weight returned
// per event, and must be a subset of output_keys.
std::shared_ptr<REX::tea::weightor> make_weightor(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    IncomingSwapped incoming_swapped = {}
);

// As the four-argument make_weightor above, but the returned weightor never
// actually evaluates against api: the first time it is bound to a given
// REX::process it reloads its own MatrixElementApi instance via
// reload.context->load_matrix_element(reload.me_path, reload.param_card)
// and reuses that instance (not api) for every later call against the same
// process. api is only used, by the caller, to resolve input_keys/
// output_keys when those aren't given explicitly -- it plays no further
// role once the returned weightor actually runs. See ReloadSpec and
// WareHouse::set_max_batch().
std::shared_ptr<REX::tea::weightor> make_weightor(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    ReloadSpec reload,
    IncomingSwapped incoming_swapped = {}
);

// The auto-derived key sets used by the single-argument make_weightor overload,
// exposed so callers building a manual override can start from the defaults.
// Throws under the same conditions as the single-argument make_weightor overload.
//
// UMAMI_IN_HELICITY_INDEX decides between helicity reweighting (each event
// reweighted at its own helicity, the matrix element evaluated for that one
// helicity) and helicity summation. It is included when helicity_reweighting
// is true and api supports it -- with a one-time warning on stderr, and
// helicity summation, when api doesn't -- or whenever api requires it. Off by
// default for this low-level function; TupperWare and Driver default to
// helicity reweighting (see TupperWare::set_helicity_reweighting()).
// Reweighting with UMAMI_IN_HELICITY_INDEX then requires every event to
// carry helicity information (a SPINUP of -1, 0 or +1 for every particle):
// evaluation throws for a process with an event that doesn't.
std::vector<UmamiInputKey> default_input_keys(
    const madspace::MatrixElementApi& api, bool helicity_reweighting = false
);
std::vector<UmamiOutputKey> default_output_keys(const madspace::MatrixElementApi& api);

// As make_weightor, but returns a REX::tea::deferred_weightor: the api's
// umami_matrix_element call still runs once per procReweightor::evaluate(amp)
// call (ie once per reweighting iteration -- its result legitimately differs
// each time, eg because umami_set_parameter changed the couplings between
// iterations), but the result is left device-resident rather than copied
// back to the host immediately. The returned REX::tea::weight_puller does
// that copy, lazily -- WareHouse/procReweightor::pull_deferred_backlog()
// calls it exactly once per accumulated iteration, after the whole run's
// iteration loop finishes, not after each individual call.
//
// Momenta/alphaS/flavor/helicity inputs are staged onto the api's device
// exactly once for a given process (cached via REX::process::device_data,
// so it's automatically invalidated whenever this weightor is bound to a
// different process) and reused verbatim for the rest of the run, same as
// make_weightor already does for repeated calls against an unchanged
// process.
//
// n_iterations must equal (or exceed) the number of times the returned
// deferred_weightor will actually be invoked over the process's lifetime:
// each call claims its own slice of a per-output-key device buffer sized
// for exactly that many iterations up front, since deferring the copy-back
// means every iteration's result has to survive, un-overwritten, until
// it's pulled -- an undercount throws rather than silently clobbering an
// earlier iteration's not-yet-pulled result. WareHouse::build() derives
// this automatically from its own iterators (see TupperWare::build());
// pass it explicitly only when building a procReweightor by hand.
//
// Only meaningful for a cuda/hip-backed api (ie api.device_type() !=
// DeviceType::cpu) -- callers don't need to check this themselves;
// TupperWare::build() already routes cpu apis through make_weightor
// instead.
std::shared_ptr<REX::tea::deferred_weightor>
make_deferred_weightor(const madspace::MatrixElementApi& api, std::size_t n_iterations);

std::shared_ptr<REX::tea::deferred_weightor> make_deferred_weightor(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    std::size_t n_iterations,
    IncomingSwapped incoming_swapped = {}
);

// As the five-argument make_deferred_weightor above, but reloads its own
// MatrixElementApi instance per REX::process the same way the ReloadSpec
// overload of make_weightor does, rather than ever evaluating against api
// itself. See ReloadSpec and WareHouse::set_max_batch().
std::shared_ptr<REX::tea::deferred_weightor> make_deferred_weightor(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    std::size_t n_iterations,
    ReloadSpec reload,
    IncomingSwapped incoming_swapped = {}
);

// The REX::event::extra / REX::process::extra key under which load_lhe_binary()
// and open_lhe_source_binary() store each event's own generation-time squared
// matrix element (a double), when the binary event file carries one (the
// "matrix_element" column madspace writes on request, see EventRecord::
// f_matrix_element). See TupperWare::set_stored_normalisation().
inline constexpr const char* stored_matrix_element_key = "matrix_element";

// TupperWare bundles what's needed to build a REX::tea::procReweightor for a single
// physical process: one or more externally-owned MatrixElementApis (evaluated in
// registration order as procReweightor::reweight_functions), an optional separate
// normalisation MatrixElementApi, and an event checker defining which events belong
// to this process. The MatrixElementApis are expected to outlive the TupperWare
// (and the procReweightor/weightors it produces), matching how madspace::Context
// owns them.
class TupperWare {
public:
    TupperWare() = default;

    // Convenience constructors for the common case of a single MatrixElementApi and
    // event checker. The api is registered as the first reweight_function, and also
    // as the normaliser if no separate normaliser is set later.
    TupperWare(const madspace::MatrixElementApi& api, REX::eventBelongs checker);
    TupperWare(const madspace::MatrixElementApi& api, REX::event_bool_fn checker);

    // Registers a matrix element as the next reweight_functions amplitude
    // (registration order determines the amp index used by
    // procReweightor::evaluate(amp)), with auto-derived UMAMI keys.
    TupperWare& add_api(const madspace::MatrixElementApi& api);
    // As above, with an explicit override of the UMAMI input/output keys used for
    // this particular api instead of auto-derivation.
    TupperWare& add_api(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys
    );

    // As above, additionally registering how to reload an independent
    // instance of api for reweightor::extract_sorter()'s max_batch overflow
    // clones. Only consulted when this TupperWare is built with
    // reload_per_batch=true; see ReloadSpec and WareHouse::set_max_batch().
    TupperWare& add_api(const madspace::MatrixElementApi& api, ReloadSpec reload);
    TupperWare& add_api(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys,
        ReloadSpec reload
    );

    TupperWare& add_api(std::vector<const madspace::MatrixElementApi*> apis);
    TupperWare& add_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<UmamiInputKey> input_keys_list,
        std::vector<UmamiOutputKey> output_keys_list
    );
    TupperWare& add_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<std::vector<UmamiInputKey>> input_keys_list,
        std::vector<std::vector<UmamiOutputKey>> output_keys_list
    );

    TupperWare& set_api(std::vector<const madspace::MatrixElementApi*> apis);
    TupperWare& set_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<UmamiInputKey> input_keys_list,
        std::vector<UmamiOutputKey> output_keys_list
    );
    TupperWare& set_api(
        std::vector<const madspace::MatrixElementApi*> apis,
        std::vector<std::vector<UmamiInputKey>> input_keys_list,
        std::vector<std::vector<UmamiOutputKey>> output_keys_list
    );

    // Registers the matrix element used for normalisation
    // (procReweightor::normaliser). If never called, build() leaves the normaliser
    // unset, which makes procReweightor::initialise() default to the first
    // reweight function.
    TupperWare& set_normaliser(const madspace::MatrixElementApi& api);
    TupperWare& set_normaliser(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys
    );
    // As above, additionally registering how to reload an independent
    // instance of the normaliser for max_batch overflow clones; see
    // add_api(api, reload) and WareHouse::set_max_batch().
    TupperWare& set_normaliser(const madspace::MatrixElementApi& api, ReloadSpec reload);
    TupperWare& set_normaliser(
        const madspace::MatrixElementApi& api,
        std::vector<UmamiInputKey> input_keys,
        std::vector<UmamiOutputKey> output_keys,
        ReloadSpec reload
    );

    // Normalise against the squared matrix element each event was generated
    // with, stored under REX::process::extra[key] (see
    // stored_matrix_element_key), instead of evaluating the normaliser on it
    // again. Only where that is the same quantity: build() uses it only when
    // the normaliser is evaluated helicity-summed (no UMAMI_IN_HELICITY_INDEX
    // among its input keys -- the generation's matrix element is summed
    // over helicities, it only samples one for the event record), and each
    // process falls back to evaluating the normaliser whenever any of its
    // events has no stored value. An empty key (the default) always
    // evaluates the normaliser. Has no effect without a normaliser.
    TupperWare& set_stored_normalisation(std::string key);
    const std::string& stored_normalisation() const;

    // Which UMAMI output key(s) are summed into the weight returned by every
    // weightor generated by this TupperWare, including the normaliser's. Defaults
    // to just UMAMI_OUT_MATRIX_ELEMENT.
    TupperWare& set_result_keys(std::vector<UmamiOutputKey> keys);

    // Helicity reweighting (true, the default: each event is reweighted at
    // its own helicity, via UMAMI_IN_HELICITY_INDEX) or helicity summation
    // (false) for apis registered here without explicit input_keys (see
    // default_input_keys()); apis added with explicit input_keys use exactly
    // those. A matrix element without per-event helicity support falls back
    // to helicity summation with a one-time warning; events without helicity
    // information (any SPINUP other than -1, 0, +1) are an error with
    // helicity reweighting on.
    TupperWare& set_helicity_reweighting(bool helicity_reweighting);
    bool helicity_reweighting() const { return _helicity_reweighting; }
    // Former name of set_helicity_reweighting().
    TupperWare& set_use_helicity(bool use_helicity);

    // Which events list their incoming legs the other way round from the
    // matrix elements (see IncomingSwapped); passed to every weightor built
    // here, the normaliser's included.
    TupperWare& set_incoming_swapped(IncomingSwapped incoming_swapped);

    // Defines which events belong to this process. Setting one clears the other.
    TupperWare& set_event_checker(REX::eventBelongs checker);
    TupperWare& set_event_checker(std::shared_ptr<REX::eventBelongs> checker);
    TupperWare& set_event_checker(REX::event_bool_fn checker);

    // The MatrixElementApis registered via add_api()/set_api() (reweight
    // functions only; not the separate normaliser, if any -- it is almost
    // always one of these same apis anyway). Exposed so external code that
    // needs direct access to the underlying apis themselves rather than the
    // weightor::procReweightor wrapping them -- eg madtrex::ParamHandler,
    // for driving parameter reweighting straight through
    // MatrixElementApi::set_parameter() -- doesn't need its own separate
    // tracking of "which apis did this WareHouse load".
    const std::vector<const madspace::MatrixElementApi*>& apis() const { return _apis; }

    // Builds the procReweightor from the registered apis and event checker.
    // Throws if no apis or no event checker have been registered. Each
    // registered api is routed to make_weightor() (cpu) or
    // make_deferred_weightor() (cuda/hip) according to its own
    // api.device_type() -- registering a mix of cpu and off-cpu apis in the
    // same TupperWare works as expected, each amp slot serviced by whichever
    // is appropriate for that api. n_iterations is only used by (forwarded
    // to) any deferred apis; defaults to 1 for callers that build a
    // TupperWare directly rather than through WareHouse::build(), which
    // derives it from its own iterators automatically.
    // reload_per_batch, when true, routes every api (and the normaliser)
    // that was registered with a ReloadSpec through the reload-aware
    // make_weightor()/make_deferred_weightor() overloads instead of the
    // plain ones, so each of reweightor::extract_sorter()'s max_batch
    // overflow clones of the resulting procReweightor ends up bound to its
    // own freshly-loaded MatrixElementApi instance rather than sharing (and
    // serializing calls through) the one instance add_api()/
    // set_normaliser() were given. Apis registered without a ReloadSpec are
    // unaffected either way. WareHouse::build() passes true automatically
    // whenever WareHouse::set_max_batch() is active; false is the right
    // choice for any other caller, since it costs an extra, otherwise
    // pointless reload on every registered api's first evaluation.
    //
    // max_deferred_iterations is copied straight onto the built
    // procReweightor's own field of the same name (see
    // REX::tea::procReweightor::max_deferred_iterations): REX::npos (the
    // default) means no cap, matching the pre-existing behavior of only
    // ever copying deferred (eg GPU) results back to the host once, at the
    // end of the whole run; any other value caps how many outer
    // reweighting iterations' worth of them are allowed to accumulate
    // device-side before an automatic mid-run flush. WareHouse::build()
    // passes whatever WareHouse::set_max_deferred_iterations() was given.
    std::shared_ptr<REX::tea::procReweightor> build(
        std::size_t n_iterations = 1,
        bool reload_per_batch = false,
        std::size_t max_deferred_iterations = REX::npos
    );

private:
    const madspace::MatrixElementApi* _normaliser = nullptr;
    std::vector<UmamiInputKey> _normaliser_input_keys;
    std::vector<UmamiOutputKey> _normaliser_output_keys;
    std::optional<ReloadSpec> _normaliser_reload;
    std::string _stored_normalisation;

    std::vector<const madspace::MatrixElementApi*> _apis;
    std::vector<std::vector<UmamiInputKey>> _input_keys_list;
    std::vector<std::vector<UmamiOutputKey>> _output_keys_list;
    std::vector<std::optional<ReloadSpec>> _reload_specs;
    std::vector<UmamiOutputKey> _result_keys;
    bool _helicity_reweighting = true;
    IncomingSwapped _incoming_swapped;

    std::vector<std::shared_ptr<REX::tea::weightor>> _weightors;
    std::vector<std::shared_ptr<REX::tea::deferred_weightor>> _deferred_weightors;
    std::shared_ptr<REX::eventBelongs> _event_checker = nullptr;
    REX::event_bool_fn _event_checker_fn = nullptr;
};

// WareHouse aggregates one or more per-process TupperWare specifications together
// with the reweightor-level plumbing (initialise/finalise/iterators/launch_names)
// into a single REX::tea::reweightor, which it owns. Each registered TupperWare
// becomes one entry in the resulting reweightor::reweightors (ie one
// procReweightor per physical process, matching the "one TupperWare per process"
// model documented on TupperWare above); the REX::lhe passed to build() supplies
// the lhe base that reweightor::extract_sorter() reads events from and sorts into
// those per-process procReweightors via their event checkers.
//
// TupperWares are retained after build(), so the same registered apis/checkers can
// be rebuilt against a different REX::lhe by calling build() again.
class WareHouse {
public:
    WareHouse() = default;
    WareHouse(const WareHouse&) = delete;
    WareHouse& operator=(const WareHouse&) = delete;
    WareHouse(WareHouse&&) noexcept = default;
    WareHouse& operator=(WareHouse&&) noexcept = default;

    // Convenience: appends a new, empty TupperWare and returns a reference to it
    // for in-place configuration, eg house.add_process().add_api(api).set_event_checker(chk).
    TupperWare& add_process();

    // Reweighting-loop plumbing, mirroring REX::tea::reweightor's own setters:
    // per-iteration hooks (eg SLHA card rewriting between amplitude evaluations),
    // hooks run once before/after the iteration loop, and the weight-id names
    // recorded for each iteration's appended weights. Optional: if never called,
    // build() leaves the corresponding reweightor default in place (a single
    // no-op iteration, true_function initialise/finalise, no launch_names).
    WareHouse& set_iterators(std::vector<REX::tea::iterator> iters);
    WareHouse& add_iterator(REX::tea::iterator iter);
    WareHouse& set_initialise(REX::tea::iterator init);
    WareHouse& set_finalise(REX::tea::iterator fin);
    WareHouse& set_launch_names(std::vector<std::string> names);
    WareHouse& add_launch_name(const std::string& name);
    const std::vector<std::string>& launch_names() const;

    // Maximum number of events per REX::process (see REX::lhe::max_batch),
    // applied to the lhe passed to build(). Left at REX::npos (unset) by
    // default, meaning no splitting. Setting this to something else also
    // makes build() pass reload_per_batch=true to every registered
    // TupperWare's build() (see TupperWare::build()), so that apis
    // registered with a ReloadSpec get their own freshly-loaded instance per
    // batch instead of every overflow clone sharing (and serializing calls
    // through) the same one -- which is the whole point of splitting a
    // process into independently-runnable batches to begin with. Apis
    // registered without a ReloadSpec are unaffected.
    WareHouse& set_max_batch(std::size_t max_batch);
    std::size_t max_batch() const;

    // Worker threads of the built reweightor (REX::tea::reweightor::
    // pool_threads): 0, the default, uses std::thread::hardware_concurrency().
    // The reweightor runs one REX::process batch per thread at a time, so
    // threads only help with as many batches; see set_min_batch().
    WareHouse& set_threads(std::size_t threads);
    std::size_t threads() const;

    // Splitting the events across threads (see batch_size_for()): with
    // max_batch() unset, build() splits every process with more than
    // max(min_batch, events / threads) events into batches of that size, so
    // that a sample dominated by one subprocess still keeps every thread
    // busy, while no batch gets smaller than min_batch events (below which
    // a batch's fixed cost -- its reloaded MatrixElementApi instance, its
    // own sorting and setup -- outweighs what another thread gains).
    // REX::npos, the default, never splits.
    WareHouse& set_min_batch(std::size_t min_batch);
    std::size_t min_batch() const;

    // The REX::lhe::max_batch build() uses for a sample of n_events: max_batch()
    // when set; else max(min_batch(), ceil(n_events / threads)) when
    // min_batch() is set, more than one thread is used and the sample has
    // at least 2 * min_batch() events; else REX::npos (no splitting).
    std::size_t batch_size_for(std::size_t n_events) const;

    // Passed straight through to every registered TupperWare's build() (see
    // TupperWare::build()'s own doc comment) -- ie every built
    // procReweightor's max_deferred_iterations field. Left at REX::npos
    // (the default) meaning no cap: deferred (eg GPU) results only ever
    // get copied back to the host once, at the end of the whole run, same
    // as before this existed. Useful together with off-cpu apis (see
    // TupperWare::add_api()/make_deferred_weightor()) to bound how much
    // device-side memory a long run's worth of not-yet-pulled results can
    // occupy at once, at the cost of more frequent, smaller device->host
    // copies.
    WareHouse& set_max_deferred_iterations(std::size_t max_deferred_iterations);
    std::size_t max_deferred_iterations() const;

    // The muted-by-default warning logger shared with the built reweightor
    // (and, transitively, every procReweightor/process batch it runs --
    // see REX::tea::reweightor::extract_sorter(), which links each batch's
    // procReweightor::verbose to this same flag). Muted (false) from
    // construction, independent of whether build() has been called yet;
    // call verbose().set_verbose(true) to unmute. Re-linked into the
    // reweightor's own verbose on every build()/rebuild.
    REX::verbosity& verbose();
    const REX::verbosity& verbose() const;
    // Plain-bool convenience for verbose().set_verbose(v)/is_verbose(),
    // for callers (eg Python bindings) that would rather not need
    // REX::verbosity itself.
    WareHouse& set_verbose(bool v);
    bool is_verbose() const;

    // Builds (or rebuilds) the owned reweightor from the registered TupperWares'
    // procReweightors and the given REX::lhe, which becomes the reweightor's lhe
    // base. Throws if no processes have been registered. Repeated calls discard
    // any previously-built reweightor.
    REX::tea::reweightor& build(REX::lhe&& mother);
    REX::tea::reweightor& build(const REX::lhe& mother);

    // Access to the owned reweightor. Throws if build() has not been called yet.
    REX::tea::reweightor& get();
    const REX::tea::reweightor& get() const;
    bool built() const noexcept;

    // Direct access to TupperWare; getting, setting, and adding
    TupperWare& process(std::size_t index);
    const TupperWare& process(std::size_t index) const;
    std::deque<TupperWare>& processes();
    const std::deque<TupperWare>& processes() const;
    WareHouse& set_processes(std::vector<TupperWare> processes);
    WareHouse& add_process(TupperWare process);
    WareHouse& add_process(std::vector<TupperWare> processes);

private:
    // a deque, not a vector: add_process() hands out references to its
    // elements (also to Python), which adding further processes must not
    // invalidate
    std::deque<TupperWare> _processes;
    std::vector<REX::tea::iterator> _iterators;
    REX::tea::iterator _initialise = REX::tea::true_function;
    REX::tea::iterator _finalise = REX::tea::true_function;
    std::vector<std::string> _launch_names;
    std::size_t _max_batch = REX::npos;
    std::size_t _threads = 0;
    std::size_t _min_batch = REX::npos;
    std::size_t _max_deferred_iterations = REX::npos;
    REX::verbosity _verbose{false};

    std::unique_ptr<REX::tea::reweightor> _reweightor;
};

} // namespace madtrex
