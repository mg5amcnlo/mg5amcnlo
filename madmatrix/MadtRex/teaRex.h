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
// *t*ensorial *e*vent *a*daption with *R*e*x* Version 1.0.0
// teaRex is an extension to the Rex library for the generic reweighting of parton-level events.
// It provides a flexible framework for applying weight modifications to events based on user-defined criteria,
// using the underlying Rex formats to sort, extract, and rewrite event-level information,
// and extending it to allow for generic reweighting using any information stored in an LHE file as input for a
// user-provided reweighting function acting on REX::process objects, which are SoA (Structure of Arrays)
// objects for storing event information. Users can either provide the REX::process objects themselves,
// or use the flexible Rex sorting architecture to extract the necessary information from an LHE file.
//
// Copyright © 2023-2026 CERN, CERN Author Zenny Wettersten.
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
    using iterator = std::function<bool()>;
    using weightor = std::function<std::shared_ptr<std::vector<double>>(process &)>;

    bool true_function();

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
        };

        struct rwgt_card
        {
            std::string launch_name;
            std::string rwgt_com = "";
            std::unordered_map<std::string, rwgt_block> blocks;
            void add_param(const std::string &block_name, std::pair<int, double> param);
            void add_param(const std::string &block_name, int param_id, double param_value);
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

        std::shared_ptr<eventBelongs> event_checker;
        REX::event_bool_fn event_checker_fn = nullptr;
        weightor normaliser = nullptr;
        std::vector<weightor> reweight_functions = {};
        std::vector<double> normalisation = {};
        std::shared_ptr<process> proc = nullptr;
        std::vector<std::vector<double>> backlog = {};

        procReweightor &set_event_checker(eventBelongs checker);
        procReweightor &set_event_checker(REX::event_bool_fn checker);
        procReweightor &set_normaliser(weightor normaliser);
        procReweightor &set_reweight_functions(weightor rwgt);
        procReweightor &set_reweight_functions(std::vector<weightor> rwgts);
        procReweightor &add_reweight_function(weightor rwgt);
        procReweightor &set_process(std::shared_ptr<process> p);

        // Member functions for handling reweighting
        void initialise();
        void initialise(std::shared_ptr<process> p);
        void evaluate();
        void evaluate(size_t amp);
        void append_zero_weights();
        void append_backlog();
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

        double norm_factor = 0.0;

        void calc_norm();
        void set_norm(double norm);

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
        void setup();
        void run_iteration();
        void run_all_iterations();
        void run();

        void calc_xSecs();
        void calc_xErrs();
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
