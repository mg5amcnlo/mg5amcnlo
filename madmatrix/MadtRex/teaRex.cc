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

#ifndef _TEAREX_CPP_
#define _TEAREX_CPP_

#include "teaRex.h"

namespace REX::tea
{

    bool true_function()
    {
        return true;
    }

    void rwgt_slha::rwgt_card::add_param(const std::string &block_name, std::pair<int, double> param)
    {
        if (this->blocks.find(block_name) == this->blocks.end())
            this->blocks[block_name] = rwgt_block{block_name, {}};
        this->blocks[block_name].params.push_back(param);
    }

    void rwgt_slha::rwgt_card::add_param(const std::string &block_name, int param_id, double param_value)
    {
        this->add_param(block_name, std::make_pair(param_id, param_value));
    }

    void rwgt_slha::parse_rwgt_card(std::istream &is)
    {
        this->cards.clear();
        size_t curr_launch = 0;
        this->cards.emplace_back();
        std::string line;
        bool first_launch = false;
        while (std::getline(is, line))
        {
            if (line.empty() || line[0] == '#')
                continue; // Skip empty lines and comments
            std::string iss = line.c_str();
            if (iss.find("launch") != std::string::npos)
            {
                if (!first_launch)
                {
                    first_launch = true;
                }
                else
                {
                    this->cards.emplace_back();
                    curr_launch++;
                }
                size_t name_pos = iss.find("rwgt_name");
                if (name_pos != std::string::npos)
                {
                    std::string curr_name = iss.substr(name_pos + 10);
                    auto name_lines = REX::blank_splitter(curr_name);
                    this->cards.back().launch_name = std::string(name_lines[0]);
                }
                continue;
            }
            if (iss.find("set") != std::string::npos)
            {
                auto rwgt_line = REX::blank_splitter(iss);
                if (rwgt_line[0] != "set")
                    continue;
                if (rwgt_line.size() < 3)
                    continue;
                if (rwgt_line.size() == 3)
                    throw std::runtime_error("rwgt_slha::parse_rwgt_card: \"set\" line appears to use parameter names. This is not supported by teaRex.");
                if (!first_launch)
                {
                    REX::warning("rwgt_slha::parse_rwgt_card: \"set\" line appears before first launch command. Assuming launch command missed. May end up appending meaningless reweighting iterations.");
                    first_launch = true;
                }
                size_t curr_ind_of_line = 1;
                if (rwgt_line[curr_ind_of_line] == "param_card")
                    ++curr_ind_of_line;
                std::string curr_block = std::string(rwgt_line[curr_ind_of_line]);
                ++curr_ind_of_line;
                int curr_param = REX::ctoi(rwgt_line[curr_ind_of_line]);
                ++curr_ind_of_line;
                double curr_val = REX::ctod(rwgt_line[curr_ind_of_line]);
                this->cards[curr_launch].add_param(curr_block, curr_param, curr_val);
                this->cards[curr_launch].rwgt_com += line + "\n";
                continue;
            }
        }
    }

    rwgt_slha &rwgt_slha::set_card_path(const std::string &path)
    {
        this->card_path = path;
        return *this;
    }

    rwgt_slha &rwgt_slha::set_mod_card_path(const std::string &path)
    {
        this->mod_card_path = path;
        return *this;
    }

    bool rwgt_slha::move_param_card(const std::string &new_path)
    {
        if (this->card_path.empty())
            throw std::runtime_error("rwgt_slha::move_param_card: card_path not set");
        if (new_path.empty() && this->mod_card_path.empty())
        {
            this->mod_card_path = this->card_path + ".mod";
        }
        else if (!new_path.empty())
        {
            this->mod_card_path = new_path;
        }
        try
        {
            std::filesystem::rename(this->card_path, this->mod_card_path);
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            throw std::runtime_error("rwgt_slha::move_param_card: failed to rename original card to .mod backup: " + std::string(e.what()));
        }
        return true;
    }

    bool rwgt_slha::remove_mod_card()
    {
        if (this->card_path.empty())
            throw std::runtime_error("rwgt_slha::remove_mod_card: card_path not set");
        if (this->mod_card_path.empty())
            throw std::runtime_error("rwgt_slha::remove_mod_card: mod_card_path not set");
        try
        {
            std::filesystem::remove(this->card_path);
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            throw std::runtime_error("rwgt_slha::remove_mod_card: failed to remove mod card: " + std::string(e.what()));
        }
        try
        {
            std::filesystem::rename(this->mod_card_path, this->card_path);
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            throw std::runtime_error("rwgt_slha::remove_mod_card: failed to rename .mod backup to original card: " + std::string(e.what()));
        }
        return true;
    }

    bool rwgt_slha::write_rwgt_card(size_t idx)
    {
        if (idx >= this->cards.size())
            throw std::out_of_range("rwgt_slha::write_rwgt_card: index out of range");
        if (this->card_path.empty())
            throw std::runtime_error("rwgt_slha::write_rwgt_card: card_path not set");
        if (this->mod_card_path.empty())
            this->mod_card_path = this->card_path + ".mod";
        if (!std::filesystem::exists(this->mod_card_path))
        {
            this->move_param_card();
        }
        if (!this->orig_params.blocks.empty())
        {
            for (const auto &[key, val] : this->orig_params.blocks)
            {
                if (REX::to_upper(key) == "DECAY")
                {
                    for (const auto &[pid, width] : val.params)
                    {
                        this->REX::slha::set_decay(pid, width);
                    }
                }
                else
                {
                    for (auto [param_id, param_value] : val.params)
                    {
                        this->REX::slha::set(key, param_id, param_value);
                    }
                }
            }
            this->orig_params.blocks.clear();
        }
        for (const auto &[key, val] : this->cards[idx].blocks)
        {
            if (REX::to_upper(key) == "DECAY")
            {
                for (const auto &[pid, width] : val.params)
                {
                    auto curr_decay = this->REX::slha::get_decay(pid);
                    this->orig_params.add_param("DECAY", pid, curr_decay);
                    this->REX::slha::set_decay(pid, width);
                }
            }
            else
            {
                for (auto [param_id, param_value] : val.params)
                {
                    auto curr_param = this->REX::slha::get(key, param_id);
                    this->orig_params.add_param(key, param_id, curr_param);
                    this->REX::slha::set(key, param_id, param_value);
                }
            }
        }
        std::ofstream ofs(this->card_path);
        if (!ofs)
            throw std::runtime_error("rwgt_slha::write_rwgt_card: failed to open file");
        this->REX::slha::write(ofs);
        return true;
    }

    std::vector<std::function<bool()>> rwgt_slha::get_card_writers()
    {
        std::vector<std::function<bool()>> writers;
        for (size_t i = 0; i < this->cards.size(); ++i)
        {
            writers.push_back([this, i]()
                              { return this->write_rwgt_card(i); });
        }
        return writers;
    }

    std::vector<std::string> rwgt_slha::get_launch_names()
    {
        std::vector<std::string> names;
        for (const auto &card : this->cards)
        {
            names.push_back(card.launch_name);
        }
        return names;
    }

    std::vector<std::string> rwgt_slha::get_rwgt_commands()
    {
        std::vector<std::string> commands;
        for (const auto &card : this->cards)
        {
            commands.push_back(card.rwgt_com);
        }
        return commands;
    }

    threadPool::threadPool(unsigned nthreads)
        : stop_(false), active_(0)
    {
        workers_.reserve(nthreads);
        for (unsigned i = 0; i < nthreads; ++i)
        {
            workers_.emplace_back([this]
                                  {
                for (;;) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait(lk, [this]{ return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        task = std::move(q_.front());
                        q_.pop();
                        ++active_;
                    }
                    try {
                        task();
                    } catch (...) {
                        // Record first exception and signal cancellation
                        {
                            std::lock_guard<std::mutex> g(err_m_);
                            if (!first_error_) first_error_ = std::current_exception();
                        }
                        cancel_.store(true, std::memory_order_relaxed);
                    }
                    {
                        std::lock_guard<std::mutex> lk(m_);
                        --active_;
                        if (q_.empty() && active_ == 0) drained_.notify_all();
                    }
                } });
        }
    }

    threadPool::~threadPool()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &t : workers_)
            t.join();
    }

    void threadPool::enqueue(Task t)
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push(std::move(t));
        }
        cv_.notify_one();
    }

    void threadPool::begin_batch()
    {
        cancel_.store(false, std::memory_order_relaxed);
        std::lock_guard<std::mutex> g(err_m_);
        first_error_ = nullptr;
    }

    void threadPool::wait_batch()
    {
        std::unique_lock<std::mutex> lk(m_);
        drained_.wait(lk, [this]
                      { return (q_.empty() && active_ == 0) || first_error_; });
        lk.unlock();
        if (first_error_)
            std::rethrow_exception(first_error_);
    }

    bool threadPool::cancel_requested() const noexcept
    {
        return cancel_.load(std::memory_order_relaxed);
    }

    procReweightor::procReweightor(weightor reweight_function)
    {
        this->reweight_functions.push_back(reweight_function);
    }

    procReweightor::procReweightor(weightor reweight_function, eventBelongs selector)
    {
        this->reweight_functions.push_back(reweight_function);
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(weightor reweight_function, std::shared_ptr<eventBelongs> selector)
    {
        this->reweight_functions.push_back(reweight_function);
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts)
    {
        this->reweight_functions = rwgts;
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, std::shared_ptr<eventBelongs> selector)
    {
        this->reweight_functions = rwgts;
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, eventBelongs selector)
    {
        this->reweight_functions = rwgts;
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, eventBelongs selector, weightor normaliser)
    {
        this->reweight_functions = rwgts;
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
        this->normaliser = normaliser;
    }

    procReweightor &procReweightor::set_event_checker(eventBelongs checker)
    {
        this->event_checker = std::make_shared<eventBelongs>(std::move(checker));
        this->event_checker_fn = this->event_checker->get_event_bool();
        return *this;
    }

    procReweightor &procReweightor::set_event_checker(REX::event_bool_fn checker)
    {
        this->event_checker = nullptr;
        this->event_checker_fn = checker;
        return *this;
    }

    procReweightor &procReweightor::set_normaliser(weightor normaliser)
    {
        this->normaliser = normaliser;
        return *this;
    }

    procReweightor &procReweightor::set_reweight_functions(weightor rwgt)
    {
        this->reweight_functions = {rwgt};
        if (!this->normaliser)
            this->normaliser = rwgt;
        return *this;
    }

    procReweightor &procReweightor::set_reweight_functions(std::vector<weightor> rwgts)
    {
        this->reweight_functions = rwgts;
        return *this;
    }

    procReweightor &procReweightor::add_reweight_function(weightor rwgt)
    {
        this->reweight_functions.push_back(rwgt);
        return *this;
    }

    procReweightor &procReweightor::set_process(std::shared_ptr<process> p)
    {
        this->proc = p;
        return *this;
    }

    // Member functions for handling reweighting
    void procReweightor::initialise()
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::initialise: process not set before initialisation");
        if (!this->normaliser)
        {
            if (this->reweight_functions.empty())
            {
                warning("procReweightor::initialise: no reweight functions set, process will only yield zero weights.");
                this->normalisation = std::vector<double>(this->proc->weight_.size(), 0.0);
                return;
            }
            if (this->reweight_functions.size() != 1)
                warning("procReweightor::initialise: multiple reweight functions set, assuming first is default evaluator and using it for normalisation.");
            this->normaliser = this->reweight_functions[0];
        }
        auto normalised = this->normaliser(*this->proc);
        if (!normalised)
            throw std::runtime_error("procReweightor::initialise: normaliser function returned null pointer");
        if (normalised->size() != this->proc->weight_.size())
            throw std::runtime_error("procReweightor::initialise: normalisation vector size does not match number of original weights in process");
        this->normalisation = *normalised;
        std::transform(this->normalisation.begin(), this->normalisation.end(), this->normalisation.begin(),
                       [](double val)
                       { return (val == 0.0) ? 0.0 : 1.0 / val; });
        this->normalisation = *REX::vec_elem_mult<double>(this->normalisation, this->proc->weight_);
    }

    void procReweightor::initialise(std::shared_ptr<process> p)
    {
        this->proc = p;
        initialise();
    }

    void procReweightor::evaluate()
    {
        return this->evaluate(0);
    }

    void procReweightor::evaluate(size_t amp)
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::evaluate: process not set before evaluation");
        if (this->reweight_functions.size() <= amp)
            return this->append_zero_weights();
        if (this->normalisation.empty())
            this->initialise();
        auto newweights = this->reweight_functions[amp](*this->proc);
        if (!newweights)
            throw std::runtime_error("procReweightor::evaluate: reweight function returned null pointer");
        this->backlog.push_back(std::move(*newweights));
    }

    void procReweightor::append_zero_weights()
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::append_zero_weights: process not set before appending zero weights");
        this->backlog.push_back(std::vector<double>(this->proc->weight_.size(), 0.0));
    }

    void procReweightor::append_backlog()
    {
        if (this->normalisation.empty())
            throw std::runtime_error("procReweightor::append_backlog: normalisation is empty; call initialise() first");

        for (auto &weights : this->backlog)
        {
            if (weights.size() != this->normalisation.size())
                throw std::runtime_error("procReweightor::append_backlog: size mismatch between weights and normalisation");

            this->proc->append_wgts(*REX::vec_elem_mult<double>(weights, this->normalisation));
        }
        this->backlog.clear();
    }

    reweightor::reweightor(lhe &&mother) : lhe(std::move(mother)) {}

    reweightor::reweightor(const lhe &mother) : lhe(mother) {}

    reweightor::reweightor(lhe &&mother, std::vector<std::shared_ptr<procReweightor>> rws) : lhe(std::move(mother)), reweightors(rws) {}

    reweightor::reweightor(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws) : lhe(mother), reweightors(rws) {}

    reweightor::reweightor(lhe &&mother, std::vector<std::shared_ptr<procReweightor>> rws, std::vector<iterator> iters) : lhe(std::move(mother)), reweightors(rws), iterators(std::move(iters)) {}

    reweightor::reweightor(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws, std::vector<iterator> iters) : lhe(mother), reweightors(rws), iterators(std::move(iters)) {}

    reweightor::reweightor(lhe &&mother, std::vector<procReweightor> rws) : lhe(std::move(mother))
    {
        this->set_reweightors(rws);
    }

    reweightor::reweightor(const lhe &mother, std::vector<procReweightor> rws) : lhe(mother)
    {
        this->set_reweightors(rws);
    }

    reweightor::reweightor(lhe &&mother, std::vector<procReweightor> rws, std::vector<iterator> iters) : lhe(std::move(mother)), iterators(std::move(iters))
    {
        this->set_reweightors(rws);
    }

    reweightor::reweightor(const lhe &mother, std::vector<procReweightor> rws, std::vector<iterator> iters) : lhe(mother), iterators(std::move(iters))
    {
        this->set_reweightors(rws);
    }

    reweightor &reweightor::set_reweightors(std::vector<std::shared_ptr<procReweightor>> rws)
    {
        this->reweightors = rws;
        return *this;
    }

    reweightor &reweightor::set_reweightors(std::vector<procReweightor> rws)
    {
        this->reweightors.clear();
        for (auto &rw : rws)
        {
            this->reweightors.push_back(std::make_shared<procReweightor>(std::move(rw)));
        }
        return *this;
    }

    reweightor &reweightor::add_reweightor(procReweightor &rw)
    {
        this->reweightors.push_back(std::make_shared<procReweightor>(rw));
        return *this;
    }

    reweightor &reweightor::add_reweightor(procReweightor &&rw)
    {
        this->reweightors.push_back(std::make_shared<procReweightor>(std::move(rw)));
        return *this;
    }

    reweightor &reweightor::add_reweightor(std::shared_ptr<procReweightor> rw)
    {
        this->reweightors.push_back(rw);
        return *this;
    }

    reweightor &reweightor::set_initialise(iterator init)
    {
        this->initialise = init;
        return *this;
    }

    reweightor &reweightor::set_finalise(iterator fin)
    {
        this->finalise = fin;
        return *this;
    }

    reweightor &reweightor::set_iterators(const std::vector<iterator> &iters)
    {
        this->iterators = iters;
        return *this;
    }

    reweightor &reweightor::add_iterator(const iterator &iter)
    {
        this->iterators.push_back(iter);
        return *this;
    }

    reweightor &reweightor::add_iterator(iterator &&iter)
    {
        this->iterators.push_back(std::move(iter));
        return *this;
    }

    reweightor &reweightor::set_launch_names(const std::vector<std::string> &names)
    {
        this->launch_names = names;
        return *this;
    }

    reweightor &reweightor::add_launch_name(const std::string &name)
    {
        this->launch_names.push_back(name);
        return *this;
    }

    void reweightor::calc_norm()
    {
        if (this->events.empty())
            throw std::runtime_error("reweightor::calc_norm: no events loaded, cannot calculate norm");
        this->norm_factor = 1.0;
        if (std::abs(this->idWgt_) == 3)
        {
            this->norm_factor = std::accumulate(this->xSec_.begin(), this->xSec_.end(), 0.0);
            this->norm_factor /= this->events.size();
        }
        else if (std::abs(this->idWgt_) == 4)
        {
            this->norm_factor = 1. / this->events.size();
        }
        else
        {
            if (std::abs(this->idWgt_) > 2 || this->idWgt_ == 0)
                warning("reweightor::calc_norm: idWgt is not set to a value defined in the LHE standard. Assuming weighted events.");
            this->norm_factor = std::accumulate(this->xSec_.begin(), this->xSec_.end(), 0.0);
            double accumulated_wgts = 0.0;
            for (const auto &proc : this->processes)
            {
                accumulated_wgts += std::accumulate(proc->weight_.begin(), proc->weight_.end(), 0.0);
            }
            if (accumulated_wgts == 0.0)
            {
                for (auto ev : this->events)
                {
                    accumulated_wgts += ev->weight_;
                }
            }
            if (accumulated_wgts == 0.0)
                throw std::runtime_error("reweightor::calc_norm: total weight is zero, cannot calculate norm");
            this->norm_factor /= accumulated_wgts;
        }
    }

    void reweightor::set_norm(double norm)
    {
        this->norm_factor = norm;
    }

    void reweightor::setup_pool()
    {
        if (!pool)
        {
            unsigned hc = std::thread::hardware_concurrency();
            if (hc == 0)
                hc = 1;
            unsigned want = this->pool_threads ? this->pool_threads : hc;
            unsigned n = static_cast<unsigned>(
                std::max<size_t>(1, std::min<size_t>(reweightors.size(), want)));
            this->pool = std::make_unique<threadPool>(n);
        }
    }

    void reweightor::extract_sorter()
    {
        if (this->reweightors.empty())
            throw std::runtime_error("reweightor::extract_sorter: no procReweightors set in reweightor");

        std::vector<event_bool_fn> preds;
        preds.reserve(this->reweightors.size());
        for (const auto &rw : this->reweightors)
        {
            if (rw->event_checker_fn)
            {
                preds.push_back(rw->event_checker_fn);
            }
            else
            {
                preds.push_back(rw->event_checker->get_event_bool());
            }
        }

        this->set_sorter(eventSorter(std::move(preds)));
        this->sorted_events.clear();
        this->processes.clear();
        this->sort_events();
        this->events_to_processes();

        const size_t R = this->reweightors.size();
        const size_t B = this->processes.size();
        const bool has_unsorted = (B == R + 1);
        auto processes_full = this->processes;

        std::vector<size_t> keep;
        keep.reserve(R);
        for (size_t i = 0; i < R; ++i)
            if (!processes_full[i]->events.empty())
                keep.push_back(i);

        std::vector<std::shared_ptr<process>> procs;
        std::vector<std::shared_ptr<procReweightor>> rwgs;
        procs.reserve(keep.size() + (has_unsorted ? 1u : 0u));
        rwgs.reserve(keep.size() + (has_unsorted ? 1u : 0u));

        for (size_t i : keep)
        {
            procs.push_back(processes_full[i]);
            rwgs.push_back(this->reweightors[i]);
        }

        if (has_unsorted && !processes_full.back()->events.empty())
        {
            procs.push_back(processes_full.back());

            auto dummy = std::make_shared<procReweightor>();
            rwgs.push_back(std::move(dummy));
        }

        this->processes = std::move(procs);
        this->reweightors = std::move(rwgs);

        if (this->processes.size() != this->reweightors.size())
            throw std::runtime_error("reweightor::extract_sorter: number of processes does not match number of reweightors.");

        for (size_t i = 0; i < this->reweightors.size(); ++i)
        {
            auto &p = this->processes[i];
            p->validate();
            this->reweightors[i]->set_process(p);
        }
    }

    void reweightor::initialise_reweightors()
    {
        if (this->reweightors.size() != this->processes.size())
            throw std::runtime_error("initialise_reweightors: reweightors/processes size mismatch");

        for (size_t i = 0; i < this->reweightors.size(); ++i)
            this->reweightors[i]->initialise(this->processes[i]);
    }

    void reweightor::finalise_reweighting()
    {
        for (auto proc : this->processes)
        {
            proc->transpose_wgts();
            proc->validate();
        }
        if (!this->finalise())
            warning("reweightor::finalise_reweighting: finalise iterator returned false, something might have gone wrong. Validate output manually.");
        if (this->launch_names.size() > 0)
        {
            this->extract_weight_ids();
            size_t nWgts = this->weight_ids->size();
            for (size_t i = 0; i < this->launch_names.size(); ++i)
            {
                std::string curr_name = (this->launch_names[i].empty()) ? "rwgt_" + std::to_string(i + nWgts + 1) : this->launch_names[i];
                this->weight_ids->push_back(curr_name);
            }
        }
        this->calc_xSecs();
        this->calc_xErrs();
    }

    void reweightor::setup()
    {
        if (!this->initialise())
            throw std::runtime_error("reweightor::setup: initialise iterator returned false, something went wrong.");
        this->extract_sorter();
        this->initialise_reweightors();
        this->n_amps = 0;
        for (auto &rwgt : this->reweightors)
        {
            size_t amps = rwgt->reweight_functions.size();
            this->n_amps = std::max(this->n_amps, amps);
        }

        if (this->n_amps == 0)
        {
            throw std::runtime_error("reweightor::setup: no reweight functions found, something went wrong.");
        }
        this->setup_pool();
    }

    void reweightor::run_iteration()
    {
        if (!this->iterators[this->curr_iter]())
            throw std::runtime_error("reweightor::run_iteration: iterator returned false, something went wrong in iteration " + std::to_string(this->curr_iter) + ".");
        this->curr_iter++;

        // Nothing to do?
        const size_t N = this->reweightors.size();
        if (N == 0 || this->n_amps == 0)
            return;

        // Ensure pool exists (persisted across iterations)
        setup_pool();

        // Parallel "reweightor" phase
        pool->begin_batch();
        for (size_t i = 0; i < N; ++i)
        {
            pool->enqueue([this, i]
                          {
                    // Early cancel check (best-effort)
                    if (pool->cancel_requested()) return;
        
                    auto &rwgt = this->reweightors[i];
                    for (size_t amp = 0; amp < this->n_amps; ++amp) {
                        // Safe because each task owns a distinct rwgt
                        rwgt->evaluate(amp);
                        if (pool->cancel_requested()) return; // responsive cancellation
                    } });
        }
        // Wait for completion (or rethrow first error)
        pool->wait_batch();
    }

    void reweightor::run_all_iterations()
    {
        while (this->curr_iter < this->iterators.size())
        {
            this->run_iteration();
            for (auto &rwgt : this->reweightors)
            {
                rwgt->append_backlog();
            }
#pragma optimize("", off)
            std::cout << ".";
#pragma optimize("", on)
            std::cout.flush();
        }
    }

    void reweightor::run()
    {
        this->setup();
        this->run_all_iterations();
        this->finalise_reweighting();
    }

    void reweightor::calc_xSecs()
    {
        if (this->norm_factor == 0.0)
            this->calc_norm();
        this->rwgt_xSec = std::vector<double>(this->events[0]->wgts_.size(), 0.0);
        for (auto ev : this->events)
        {
            for (size_t i = 0; i < ev->wgts_.size(); ++i)
            {
                this->rwgt_xSec[i] += ev->wgts_[i];
            }
        }
        for (auto &x : this->rwgt_xSec)
            x *= this->norm_factor;
    }

    void reweightor::calc_xErrs()
    {
        if (this->rwgt_xSec.size() == 0)
            this->calc_xSecs();
        double loc_xSec = std::accumulate(this->xSec_.begin(), this->xSec_.end(), 0.0);
        double loc_xErr = std::sqrt(std::accumulate(this->xSecErr_.begin(), this->xSecErr_.end(), 0.0, [](double a, double b)
                                                    { return a + b * b; }));
        size_t nEvs = this->events.size();
        if (nEvs == 0)
        {
            this->transpose();
            nEvs = this->events.size();
            if (nEvs == 0)
                throw std::runtime_error("reweightor::calc_xErrs: no events found, cannot calculate errors");
        }
        this->rwgt_xErr = std::vector<double>(this->rwgt_xSec.size(), 0.0);
        auto omg = std::vector<double>(this->rwgt_xSec.size(), 0.0);
        auto omgSq = std::vector<double>(this->rwgt_xSec.size(), 0.0);
        for (auto ev : this->events)
        {
            for (size_t k = 0; k < ev->wgts_.size(); ++k)
            {
                double ratio = ev->wgts_[k] / ev->weight_;
                omg[k] += ratio;
                omgSq[k] += std::pow(ratio, 2);
            }
        }
        double invNoEvs = 1.0 / double(nEvs);
        double sqrtInvNoEvs = std::sqrt(invNoEvs);
        for (size_t k = 0; k < this->rwgt_xSec.size(); ++k)
        {
            double variance = (omgSq[k] - std::pow(omg[k], 2) * invNoEvs) * invNoEvs;
            variance = std::max(variance, 0.0);
            this->rwgt_xErr[k] = loc_xSec * std::sqrt(variance) * sqrtInvNoEvs + loc_xErr * omg[k] * invNoEvs;
            if (std::isnan(this->rwgt_xErr[k]) || std::isinf(this->rwgt_xErr[k]) || this->rwgt_xErr[k] <= 0.0)
            {
                warning("reweightor::calc_xErrs: Error propagation failed for weight " + std::to_string(k) + ". Approximating the error at the level of the cross section.");
                this->rwgt_xErr[k] = loc_xErr * std::max(loc_xSec / this->rwgt_xSec[k], this->rwgt_xSec[k] / loc_xSec);
            }
        }
    }

    param_rwgt::param_rwgt(const lhe &mother, std::vector<std::shared_ptr<procReweightor>> rws, const std::string &slha_path, const std::string &rwgt_path)
        : reweightor(mother, rws)
    {
        this->read_slha_rwgt(slha_path, rwgt_path);
    }

    void param_rwgt::read_slha_rwgt(std::istream &slha_in, std::istream &rwgt_in)
    {
        this->card_iter = rwgt_slha::create(slha_in, rwgt_in);
        this->initialise = [&]()
        { return this->card_iter.move_param_card(); };
        this->finalise = [&]()
        { return this->card_iter.remove_mod_card(); };
        this->iterators = this->card_iter.get_card_writers();
        this->launch_names = this->card_iter.get_launch_names();
        this->weight_context = this->card_iter.get_rwgt_commands();
    }

    void param_rwgt::read_slha_rwgt(const std::string &slha_file, const std::string &rwgt_file)
    {
        std::ifstream slha_in(slha_file);
        std::ifstream rwgt_in(rwgt_file);
        if (!slha_in || !rwgt_in)
            throw std::runtime_error("param_rwgt::read_slha_rwgt: failed to open input files");
        this->read_slha_rwgt(slha_in, rwgt_in);
        this->card_iter.set_card_path(slha_file);
    }

} // namespace REX::tea
#endif // _TEAREX_CPP_
