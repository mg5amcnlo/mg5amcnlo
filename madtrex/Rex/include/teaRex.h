/***
 *     _            ______
 *    | |           | ___ \
 *    | |_ ___  __ _| |_/ /_____  __
 *    | __/ _ \/ _` |    // _ \ \/ /
 *    | ||  __/ (_| | |\ \  __/>  <
 *     \__\___|\__,_\_| \_\___/_/\_\
 *
 ***/
//
// *t*ensorial *e*vent *a*daption with *R*e*x* Version 1.1.0
// teaRex is an extension to the Rex library for the generic reweighting of parton-level events.
// It provides a flexible framework for applying weight modifications to events based on user-defined criteria,
// using the underlying Rex formats to sort, extract, and rewrite event-level information,
// and extending it to allow for generic reweighting using any information stored in an LHE file as input for a
// user-provided reweighting function acting on REX::process objects, which are SoA (Structure of Arrays)
// objects for storing event information. Users can either provide the REX::process objects themselves,
// or use the flexible Rex sorting architecture to extract the necessary information from an LHE file.
//
// Copyright © 2023-2025 CERN, CERN Author Zenny Wettersten.
// Copyright © 2025-2026 Zenny Wettersten.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// All rights not expressly granted are reserved.
//

#ifndef _TEAREX_H_
#define _TEAREX_H_

#include "Rex.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <filesystem>

namespace REX::tea
{
    using eventBelongs = REX::eventBelongs;
    using eventSorter = REX::eventSorter;
    using lhe = REX::lhe;
    using process = REX::process;
    using slha = REX::slha;
    using event_bool_fn = REX::event_bool_fn;
    using verbosity = REX::verbosity;
    using eventSource = REX::eventSource;
    using eventSink = REX::eventSink;
    using iterator = std::function<bool()>;
    using weightor = std::function<std::shared_ptr<std::vector<double>>(process &)>;

    // A puller is a zero-argument callable that materializes a previously
    // computed weight vector on demand. deferred_weightor is a
    // weightor-like callable that, rather than returning the weights
    // directly, returns a puller for them -- letting an implementation
    // whose actual result currently lives somewhere other than host memory
    // (eg process::device_data, see Rex.h) defer materializing it until
    // genuinely needed, instead of paying for that on every reweighting
    // iteration
    using weight_puller = std::function<std::shared_ptr<std::vector<double>>()>;
    using deferred_weightor = std::function<weight_puller(process &)>;

    bool true_function();

    // Convenience predicates for building eventBelongs/eventSorter selectors,
    // key off of the event-level helicity_/flavor_ indices in REX::event
    // and REX::process. A negative index matches "any" (ie unset selection),
    // mirroring the -1 "unset" sentinel used by those fields
    event_bool_fn helicity_selector(int helicity);
    event_bool_fn flavor_selector(int flavor);
    event_bool_fn helicity_flavor_selector(int helicity, int flavor);

    // Type for handling SLHA card modifictions using cards of the form
    // # launch rwgt_name=OPTIONAL_NAME
    // # set BLOCK_NAME PARAM_ID PARAM_VALUE
    // # set BLOCK_NAME PARAM_ID PARAM_VALUE
    // #
    // # launch rwgt_name=OPTIONAL_NAME2...
    struct rwgt_slha : public slha
    {
        // Default constructors
        rwgt_slha() = default;
        rwgt_slha(const rwgt_slha &) = default;
        rwgt_slha(rwgt_slha &&) = default;
        rwgt_slha &operator=(const rwgt_slha &) = default;
        rwgt_slha &operator=(rwgt_slha &&) = default;
        static rwgt_slha create(std::istream &slha_in, std::istream &rwgt_in)
        {
            rwgt_slha r;
            r.read(slha_in);
            r.parse_rwgt_card(rwgt_in);
            return r;
        }

        static rwgt_slha create(const std::string &slha_path, const std::string &rwgt_path)
        {
            rwgt_slha r;
            std::ifstream slha_in(slha_path);
            std::ifstream rwgt_in(rwgt_path);
            if (!slha_in || !rwgt_in)
                throw std::runtime_error("rwgt_slha::create: failed to open input files");
            r.card_path = slha_path;
            r.read(slha_in);
            r.parse_rwgt_card(rwgt_in);
            return r;
        }

        struct rwgt_block
        {
            std::string name;
            std::vector<std::pair<int, double>> params;
            // entries with several indices ("set NMIX 1 2 0.5")
            std::vector<std::pair<std::vector<int>, double>> multi_params = {};
        };

        struct rwgt_card
        {
            std::string launch_name;
            std::string rwgt_com = "";
            std::unordered_map<std::string, rwgt_block> blocks;
            void add_param(const std::string &block_name, std::pair<int, double> param);
            void add_param(const std::string &block_name, int param_id, double param_value);
            void add_param(const std::string &block_name, std::vector<int> param_ids, double param_value);
        };

        std::vector<rwgt_card> cards = {};

        std::string card_path;
        std::string mod_card_path;
        rwgt_slha &set_card_path(const std::string &path);
        rwgt_slha &set_mod_card_path(const std::string &path);

        rwgt_card orig_params;

        bool move_param_card(const std::string &new_path = "");
        bool remove_mod_card();

        void parse_rwgt_card(std::istream &is);
        bool write_rwgt_card(size_t idx);
        std::vector<std::function<bool()>> get_card_writers();
        std::vector<std::string> get_launch_names();
        std::vector<std::string> get_rwgt_commands();
    };

    class threadPool
    {
    public:
        using Task = std::function<void()>;

        explicit threadPool(unsigned nthreads);

        ~threadPool();

        void enqueue(Task t);
        void begin_batch();
        void wait_batch();

        bool cancel_requested() const noexcept;

    private:
        std::vector<std::thread> workers_;
        std::queue<Task> q_;
        std::mutex m_;
        std::condition_variable cv_;

        std::atomic<bool> cancel_{false};
        std::atomic<bool> stop_;
        size_t active_;
        std::condition_variable drained_;

        std::mutex err_m_;
        std::exception_ptr first_error_ = nullptr;
    };

    struct procReweightor
    {
        // Default constructors
        procReweightor() = default;
        procReweightor(const procReweightor &) = default;
        procReweightor(procReweightor &&) = default;
        procReweightor &operator=(const procReweightor &) = default;
        procReweightor &operator=(procReweightor &&) = default;
        // Explicit constructors wrt reweighting
        procReweightor(weightor reweight_function);
        procReweightor(weightor reweight_function, eventBelongs selector);
        procReweightor(weightor reweight_function, std::shared_ptr<eventBelongs> selector);
        procReweightor(std::vector<weightor> rwgts);
        procReweightor(std::vector<weightor> rwgts, eventBelongs selector);
        procReweightor(std::vector<weightor> rwgts, std::shared_ptr<eventBelongs> selector);
        procReweightor(std::vector<weightor> rwgts, eventBelongs selector, weightor normaliser);
        procReweightor(std::vector<weightor> rwgts, std::shared_ptr<eventBelongs> selector, weightor normaliser);
        // Mirror set of constructors for deferred_weightor
        procReweightor(deferred_weightor reweight_function);
        procReweightor(deferred_weightor reweight_function, eventBelongs selector);
        procReweightor(deferred_weightor reweight_function, std::shared_ptr<eventBelongs> selector);
        procReweightor(std::vector<deferred_weightor> rwgts);
        procReweightor(std::vector<deferred_weightor> rwgts, eventBelongs selector);
        procReweightor(std::vector<deferred_weightor> rwgts, std::shared_ptr<eventBelongs> selector);
        procReweightor(std::vector<deferred_weightor> rwgts, eventBelongs selector, weightor normaliser);
        procReweightor(std::vector<deferred_weightor> rwgts, std::shared_ptr<eventBelongs> selector, weightor normaliser);

        // A uniform reweighting slot. Registering a plain weightor wraps it
        // in a trivial deferred_weightor -- computed immediately, right
        // where the weightor would have been, with a puller that just
        // returns the already-computed value -- so eager and genuinely
        // deferred registrations share one list, one evaluate(amp)
        // dispatch, and one backlog/pull pipeline instead of two parallel
        // ones. `immediate` records which case a slot came from, since
        // that -- not its type -- is what determines whether pulling it is
        // safe/cheap right away: used both by append_backlog()'s
        // per-iteration flush and by initialise()'s implicit-normaliser
        // default below
        struct rwgt_slot
        {
            deferred_weightor fn = nullptr;
            bool immediate = false;
            explicit operator bool() const noexcept { return static_cast<bool>(fn); }
        };

        std::shared_ptr<eventBelongs> event_checker;
        REX::event_bool_fn event_checker_fn = nullptr;
        weightor normaliser = nullptr;
        std::vector<rwgt_slot> reweight_functions = {};
        std::vector<double> normalisation = {};
        std::shared_ptr<process> proc = nullptr;
        
        verbosity verbose;

        // Every evaluate(amp) call produces exactly one backlog_entry. An
        // immediate slot's result is already fully computed by the time
        // it's pushed here, so its puller is just a trivial wrapper around
        // that value
        struct backlog_entry
        {
            weight_puller pull;
            // true: value already computed (came from an immediate slot),
            // flushed every iteration by append_backlog() -- same timing
            // the old eager-only backlog always had.
            // false: actual result may not exist yet (eg still on device),
            // so it accumulates here and is only pulled once, in
            // pull_deferred_backlog() -- called from
            // reweightor::finalise_reweighting() after the whole run
            bool immediate;
        };
        std::vector<backlog_entry> backlog = {};

        // Caps how many outer reweighting iterations' worth of deferred
        // (eg GPU/device-resident) results are allowed to accumulate in
        // backlog before append_backlog() pulls and flushes them itself --
        // ie copies them back to the host and clears backlog -- rather
        // than letting every iteration's deferred results pile up
        // unflushed until the whole run (or, in streaming mode, the whole
        // batch) finishes, as REX::npos (the default) does. Lower values
        // trade some throughput (more, smaller device->host copies) for
        // bounded device-side memory: make_deferred_weightor()'s
        // n_iterations-wide device buffer only ever needs to hold
        // max_deferred_iterations iterations' worth of results at a time,
        // not the whole run's, once its caller actually resizes for that
        // (see madtrex::TupperWare -- this field alone only changes when
        // pull_deferred_backlog() gets called, not how large a buffer a
        // given deferred_weightor was built to assume). Immediate (eager,
        // eg cpu) results are unaffected either way: append_backlog()
        // already flushes those every iteration, independent of this
        size_t max_deferred_iterations = npos;

        procReweightor &set_event_checker(eventBelongs checker);
        procReweightor &set_event_checker(REX::event_bool_fn checker);
        procReweightor &set_normaliser(weightor normaliser);
        procReweightor &calc_normalisation(std::vector<double> wgts);
        procReweightor &set_initial_weights(std::vector<double> normalisation);
        procReweightor &set_normalisation(std::vector<double> normalisation);
        // set_reweight_functions/set_deferred_reweight_functions both
        // replace the whole of reweight_functions.
        // To build a slot-by-slot mix of eager and deferred
        // functions (eg one per amp), use add_reweight_function/
        // add_deferred_reweight_function in the desired order instead
        procReweightor &set_reweight_functions(weightor rwgt);
        procReweightor &set_reweight_functions(std::vector<weightor> rwgts);
        procReweightor &add_reweight_function(weightor rwgt);
        procReweightor &set_deferred_reweight_functions(deferred_weightor rwgt);
        procReweightor &set_deferred_reweight_functions(std::vector<deferred_weightor> rwgts);
        procReweightor &add_deferred_reweight_function(deferred_weightor rwgt);
        procReweightor &set_process(std::shared_ptr<process> p);

        // Member functions for handling reweighting
        void initialise();
        void initialise(std::shared_ptr<process> p);
        void evaluate();
        void evaluate(size_t amp);
        void append_zero_weights();
        // Pulls and appends every immediate (already-computed) entry in
        // backlog into proc->wgts_, in accumulation order, leaving any
        // non-immediate ones to pick up later. Called once per
        // iteration from reweightor::run_all_iterations(); also the one
        // place that advances/checks max_deferred_iterations, so it may
        // itself end up calling pull_deferred_backlog() before returning
        void append_backlog();
        // Pulls and appends every remaining entry in backlog into proc->wgts_.
        // A no-op if backlog is empty. Called automatically by
        // append_backlog() once max_deferred_iterations is reached; always
        // called at least once more, unconditionally, from
        // reweightor::finalise_reweighting() (or, in streaming mode, once
        // per batch) after the whole iteration loop, to drain whatever's
        // left
        void pull_deferred_backlog();

    private:
        static rwgt_slot wrap(weightor w);
        static rwgt_slot wrap(deferred_weightor w);
        bool explicitly_set_norm = false;
        // Outer iterations since backlog was last drained (by either
        // pull_deferred_backlog() itself, or its own reset at the top of
        // that function); see max_deferred_iterations
        size_t deferred_iterations_since_flush = 0;
    };

    // Running-reduction accumulator standing in for the cross section/error
    // calculations reweightor::calc_xSec_from_weights()/calc_norm()/
    // calc_xSecs()/calc_xErrs() otherwise compute in one pass over a fully
    // memory-resident events vector
    struct xSecAccumulator
    {
        size_t n_events = 0;
        double weight_sum = 0.0, weight_sumSq = 0.0;
        std::vector<double> wgt_sum = {};     // per weight column: sum(ev->wgts_[k])
        std::vector<double> ratio_sum = {};   // per weight column: sum(wgts_[k]/weight_)
        std::vector<double> ratio_sumSq = {}; // per weight column: sum((wgts_[k]/weight_)^2)

        void absorb(const std::vector<std::shared_ptr<REX::event>> &events);
        // Sets xSec_/xSecErr_ (only if no header cross section was already
        // present), rwgt_xSec, rwgt_xErr and norm_factor on r.
        void finalise_into(struct reweightor &r) const;
    };

    // The reweightor object is an extension to REX::lhe
    // with member functions for handling the details of reweighting
    struct reweightor : public lhe
    {
        // Default constructors
        reweightor() = default;
        reweightor(const reweightor &) = default;
        reweightor(reweightor &&) = default;
        reweightor &operator=(const reweightor &) = default;
        reweightor &operator=(reweightor &&) = default;
        reweightor(lhe &&lhe);
        reweightor(const lhe &lhe);
        reweightor(lhe &&mother, std::vector<std::shared_ptr<procReweightor>> rws);
        reweightor(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws);
        reweightor(lhe &&mother, std::vector<std::shared_ptr<procReweightor>> rws, std::vector<iterator> iters);
        reweightor(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws, std::vector<iterator> iters);
        reweightor(lhe &&mother, std::vector<procReweightor> rws);
        reweightor(const lhe &mother, std::vector<procReweightor> rws);
        reweightor(lhe &&mother, std::vector<procReweightor> rws, std::vector<iterator> iters);
        reweightor(const lhe &mother, std::vector<procReweightor> rws, std::vector<iterator> iters);

        std::vector<std::shared_ptr<procReweightor>> reweightors;
        iterator initialise = true_function;
        iterator finalise = true_function;
        std::vector<iterator> iterators = {true_function};
        std::vector<std::string> launch_names = {};

        size_t curr_iter = 0;
        size_t n_amps = 0;
        verbosity verbose = true;

        double norm_factor = 0.0;

        void calc_norm();
        void set_norm(double norm);
        void calc_xSec_from_weights();

        std::vector<double> rwgt_xSec = {};
        std::vector<double> rwgt_xErr = {};

        std::unique_ptr<threadPool> pool; // persistent worker pool
        unsigned long pool_threads = 0;
        void setup_pool();

        reweightor &set_reweightors(std::vector<std::shared_ptr<procReweightor>> rws);
        reweightor &set_reweightors(std::vector<procReweightor> rws);
        reweightor &add_reweightor(std::shared_ptr<procReweightor> rw);
        reweightor &add_reweightor(procReweightor &rw);
        reweightor &add_reweightor(procReweightor &&rw);
        reweightor &set_initialise(iterator init);
        reweightor &set_finalise(iterator fin);
        reweightor &set_iterators(const std::vector<iterator> &iters);
        reweightor &add_iterator(const iterator &iter);
        reweightor &add_iterator(iterator &&iter);
        reweightor &set_launch_names(const std::vector<std::string> &names);
        reweightor &add_launch_name(const std::string &name);

        void extract_sorter();
        void initialise_reweightors();
        void finalise_reweighting();
        // Recomputes n_amps (the max reweight_functions size across
        // this->reweightors) from their current contents, throwing if it
        // comes out to 0. Factored out of setup() since run_streaming() needs
        // recompute once per batch, not just once at the start of a run
        void sync_n_amps();
        void setup();
        void run_iteration();
        void run_all_iterations();
        void run();

        void calc_xSecs();
        void calc_xErrs();

        // Streaming counterpart to setup()/run_all_iterations()/
        // finalise_reweighting()/run(), for when this->streaming() (see
        // REX::lhe::source). Repeatedly pull_batch()es from the source, runs
        // every registered procReweightor's full iterator sequence against
        // just that batch's processes, writes the finished batch to sink,
        // and frees it -- so only O(batch_size) events are ever resident
        void run_streaming(eventSink &sink, bool include_ids = false);

        // Per-batch counterpart to extract_sorter(): rebuilds this->sorted_events/
        // processes/reweightors/process_group from the current batch in
        // this->events, matched against a caller-held, never-mutated `base`
        // (the procReweightors registered before streaming began). Unlike
        // extract_sorter() -- which is only safe to call once, since it
        // rebuilds the sorter from whatever this->reweightors currently
        // holds and may bind a base procReweightor directly into
        // this->reweightors -- this always clones a fresh procReweightor
        // per matched process, so `base` stays an indefinitely-reusable
        // template across batches and this is safe to call once per batch
        void extract_sorter_stream(const std::vector<std::shared_ptr<procReweightor>> &base);
    };

    struct param_rwgt : public reweightor
    {
        param_rwgt() = default;
        param_rwgt(const param_rwgt &) = default;
        param_rwgt(param_rwgt &&) = default;
        param_rwgt &operator=(const param_rwgt &) = default;
        param_rwgt &operator=(param_rwgt &&) = default;

        param_rwgt(const lhe &mother) : reweightor(mother) {};
        param_rwgt(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws) : reweightor(mother, rws) {};

        param_rwgt(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws, const std::string &slha_path, const std::string &rwgt_path);

        rwgt_slha card_iter;

        void read_slha_rwgt(std::istream &slha_in, std::istream &rwgt_in);
        void read_slha_rwgt(const std::string &slha_file, const std::string &rwgt_file);
    };

} // namespace REX::tea

#endif // _TEAREX_H_