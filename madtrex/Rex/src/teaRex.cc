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

#ifndef _TEAREX_CPP_
#define _TEAREX_CPP_

#include "teaRex.h"

namespace REX::tea
{

    bool true_function()
    {
        return true;
    }

    event_bool_fn helicity_selector(int helicity)
    {
        return [helicity](REX::event &e) -> bool
        { return helicity < 0 || e.helicity_ == helicity; };
    }

    event_bool_fn flavor_selector(int flavor)
    {
        return [flavor](REX::event &e) -> bool
        { return flavor < 0 || e.flavor_ == flavor; };
    }

    event_bool_fn helicity_flavor_selector(int helicity, int flavor)
    {
        return [helicity, flavor](REX::event &e) -> bool
        { return (helicity < 0 || e.helicity_ == helicity) && (flavor < 0 || e.flavor_ == flavor); };
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

    void rwgt_slha::rwgt_card::add_param(const std::string &block_name, std::vector<int> param_ids, double param_value)
    {
        if (param_ids.size() == 1)
            return this->add_param(block_name, param_ids[0], param_value);
        if (this->blocks.find(block_name) == this->blocks.end())
            this->blocks[block_name] = rwgt_block{block_name, {}};
        this->blocks[block_name].multi_params.emplace_back(std::move(param_ids), param_value);
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
                // one index, or several for a multi-index block ("set NMIX 1 2
                // 0.5"): every word between the block and the value
                if (curr_ind_of_line + 1 >= rwgt_line.size())
                    throw std::runtime_error("rwgt_slha::parse_rwgt_card: \"set\" line without a parameter index: " + line);
                std::vector<int> curr_params;
                for (; curr_ind_of_line + 1 < rwgt_line.size(); ++curr_ind_of_line)
                    curr_params.push_back(REX::ctoi(rwgt_line[curr_ind_of_line]));
                double curr_val = REX::ctod(rwgt_line[curr_ind_of_line]);
                this->cards[curr_launch].add_param(curr_block, std::move(curr_params), curr_val);
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
                    for (const auto &[param_ids, param_value] : val.multi_params)
                    {
                        this->REX::slha::set(key, param_ids, param_value);
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
                for (const auto &[param_ids, param_value] : val.multi_params)
                {
                    auto curr_param = this->REX::slha::get(key, param_ids);
                    this->orig_params.add_param(key, param_ids, curr_param);
                    this->REX::slha::set(key, param_ids, param_value);
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

    procReweightor::rwgt_slot procReweightor::wrap(weightor w)
    {
        if (!w)
            return {};
        return {[w](process &p) -> weight_puller
                {
                    auto result = w(p);
                    return [result]()
                    { return result; };
                },
                true};
    }

    procReweightor::rwgt_slot procReweightor::wrap(deferred_weightor w)
    {
        if (!w)
            return {};
        return {w, false};
    }

    procReweightor::procReweightor(weightor reweight_function)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
    }

    procReweightor::procReweightor(weightor reweight_function, eventBelongs selector)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(weightor reweight_function, std::shared_ptr<eventBelongs> selector)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, std::shared_ptr<eventBelongs> selector)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, std::shared_ptr<eventBelongs> selector, weightor normaliser)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
        this->normaliser = normaliser;
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, eventBelongs selector)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<weightor> rwgts, eventBelongs selector, weightor normaliser)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
        this->normaliser = normaliser;
    }

    procReweightor::procReweightor(deferred_weightor reweight_function)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
    }

    procReweightor::procReweightor(deferred_weightor reweight_function, eventBelongs selector)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(deferred_weightor reweight_function, std::shared_ptr<eventBelongs> selector)
    {
        this->reweight_functions.push_back(wrap(reweight_function));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<deferred_weightor> rwgts)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
    }

    procReweightor::procReweightor(std::vector<deferred_weightor> rwgts, std::shared_ptr<eventBelongs> selector)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<deferred_weightor> rwgts, std::shared_ptr<eventBelongs> selector, weightor normaliser)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = selector;
        this->event_checker_fn = selector->get_event_bool();
        this->normaliser = normaliser;
    }

    procReweightor::procReweightor(std::vector<deferred_weightor> rwgts, eventBelongs selector)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
    }

    procReweightor::procReweightor(std::vector<deferred_weightor> rwgts, eventBelongs selector, weightor normaliser)
    {
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        this->event_checker = std::make_shared<eventBelongs>(std::move(selector));
        this->event_checker_fn = this->event_checker->get_event_bool();
        this->normaliser = normaliser;
    }

    procReweightor &procReweightor::calc_normalisation(std::vector<double> wgts)
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::calc_normalisation: process not set before calculating normalisation");
        if (wgts.size() != this->proc->weight_.size())
            throw std::runtime_error("procReweightor::calc_normalisation: weights vector size does not match number of original weights in process");
        this->normalisation = wgts;
        std::transform(this->normalisation.begin(), this->normalisation.end(), this->normalisation.begin(),
                       [](double val)
                       { return (val == 0.0) ? 0.0 : 1.0 / val; });
        this->normalisation = *REX::vec_elem_mult<double>(this->normalisation, this->proc->weight_);
        return *this;
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

    procReweightor &procReweightor::set_normalisation(std::vector<double> normalisation)
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::set_normalisation: process not set before setting normalisation");
        if(normalisation.size() != this->proc->weight_.size() && this->proc->weight_.size() > 0)
            throw std::runtime_error("procReweightor::set_normalisation: normalisation vector size does not match number of original weights in process");
        this->normalisation = normalisation;
        this->explicitly_set_norm = true;
        return *this;
    }

    procReweightor &procReweightor::set_reweight_functions(weightor rwgt)
    {
        this->reweight_functions = {wrap(rwgt)};
        if (!this->normaliser)
            this->normaliser = rwgt;
        return *this;
    }

    procReweightor &procReweightor::set_reweight_functions(std::vector<weightor> rwgts)
    {
        this->reweight_functions.clear();
        this->reweight_functions.reserve(rwgts.size());
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        return *this;
    }

    procReweightor &procReweightor::add_reweight_function(weightor rwgt)
    {
        this->reweight_functions.push_back(wrap(rwgt));
        return *this;
    }

    procReweightor &procReweightor::set_deferred_reweight_functions(deferred_weightor rwgt)
    {
        this->reweight_functions = {wrap(rwgt)};
        return *this;
    }

    procReweightor &procReweightor::set_deferred_reweight_functions(std::vector<deferred_weightor> rwgts)
    {
        this->reweight_functions.clear();
        this->reweight_functions.reserve(rwgts.size());
        for (auto &w : rwgts)
            this->reweight_functions.push_back(wrap(w));
        return *this;
    }

    procReweightor &procReweightor::add_deferred_reweight_function(deferred_weightor rwgt)
    {
        this->reweight_functions.push_back(wrap(rwgt));
        return *this;
    }

    procReweightor &procReweightor::set_process(std::shared_ptr<process> p)
    {
        this->proc = p;
        return *this;
    }

    procReweightor &procReweightor::set_initial_weights(std::vector<double> weights)
    {
        this->explicitly_set_norm = true;
        return this->calc_normalisation(weights);
    }

    // Member functions for handling reweighting
    void procReweightor::initialise()
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::initialise: process not set before initialisation");
        if (this->explicitly_set_norm) return; // If normalisation is explicitly set, skip the rest of the initialisation
        if (!this->normaliser)
        {
            // reweight_functions may hold a mix of immediate and genuinely
            // deferred slots -- scan for the first immediate one, since
            // only its result is guaranteed to already exist right now.
            size_t first_immediate = this->reweight_functions.size();
            size_t immediate_count = 0;
            bool any_slot = false;
            for (size_t i = 0; i < this->reweight_functions.size(); ++i)
            {
                if (!this->reweight_functions[i])
                    continue;
                any_slot = true;
                if (!this->reweight_functions[i].immediate)
                    continue;
                if (first_immediate == this->reweight_functions.size())
                    first_immediate = i;
                ++immediate_count;
            }
            if (first_immediate == this->reweight_functions.size())
            {
                if (!any_slot)
                {
                    this->verbose.warn("procReweightor::initialise: no reweight functions set, process will only yield zero weights.");
                    this->normalisation = std::vector<double>(this->proc->weight_.size(), 0.0);
                    return;
                }
                throw std::runtime_error(
                    "procReweightor::initialise: only deferred reweight functions "
                    "are registered and no normaliser was set; call "
                    "set_normaliser() explicitly");
            }
            if (immediate_count != 1)
                this->verbose.warn("procReweightor::initialise: multiple reweight functions set, assuming first is default evaluator and using it for normalisation.");
            auto slot = this->reweight_functions[first_immediate];
            this->normaliser = [slot](process &p)
            { return slot.fn(p)(); };
        }
        auto normalised = this->normaliser(*this->proc);
        this->calc_normalisation(*normalised);
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
        if (amp >= this->reweight_functions.size() || !this->reweight_functions[amp])
            return this->append_zero_weights();
        if (this->normalisation.empty())
            this->initialise();
        auto &slot = this->reweight_functions[amp];
        auto puller = slot.fn(*this->proc);
        if (!puller)
            throw std::runtime_error("procReweightor::evaluate: reweight function returned an empty puller");
        this->backlog.push_back({std::move(puller), slot.immediate});
    }

    void procReweightor::append_zero_weights()
    {
        if (!this->proc)
            throw std::runtime_error("procReweightor::append_zero_weights: process not set before appending zero weights");
        auto zeros = std::make_shared<std::vector<double>>(this->proc->weight_.size(), 0.0);
        this->backlog.push_back({[zeros]()
                                  { return zeros; },
                                  true});
    }

    void procReweightor::append_backlog()
    {
        if (this->normalisation.empty())
            throw std::runtime_error("procReweightor::append_backlog: normalisation is empty; call initialise() first");

        std::vector<backlog_entry> remaining;
        remaining.reserve(this->backlog.size());
        for (auto &entry : this->backlog)
        {
            if (!entry.immediate)
            {
                remaining.push_back(std::move(entry));
                continue;
            }
            auto weights = entry.pull();
            if (!weights)
                throw std::runtime_error("procReweightor::append_backlog: reweight function returned null pointer");
            if (weights->size() != this->normalisation.size())
                throw std::runtime_error("procReweightor::append_backlog: size mismatch between weights and normalisation");

            this->proc->append_wgts(*REX::vec_elem_mult<double>(*weights, this->normalisation));
        }
        this->backlog = std::move(remaining);

        // One outer reweighting iteration just finished (this is called
        // exactly once per iteration, from reweightor::run_all_iterations()/
        // run_streaming()); once max_deferred_iterations of them have gone
        // by without a flush, pull and clear whatever deferred results are
        // still pending -- see max_deferred_iterations' own doc comment
        ++this->deferred_iterations_since_flush;
        if (this->max_deferred_iterations != npos &&
            this->deferred_iterations_since_flush >= this->max_deferred_iterations)
        {
            this->pull_deferred_backlog();
        }
    }

    void procReweightor::pull_deferred_backlog()
    {
        this->deferred_iterations_since_flush = 0;
        if (this->backlog.empty())
            return;
        if (this->normalisation.empty())
            throw std::runtime_error("procReweightor::pull_deferred_backlog: normalisation is empty; call initialise() first");

        // append_backlog() flushes every immediate entry each iteration, so
        // whatever remains here, whether this runs mid-run (because
        // max_deferred_iterations fired) or once at the very end, must be a
        // genuinely deferred entry
        for (auto &entry : this->backlog)
        {
            auto weights = entry.pull();
            if (!weights)
                throw std::runtime_error("procReweightor::pull_deferred_backlog: weight puller returned null pointer");
            if (weights->size() != this->normalisation.size())
                throw std::runtime_error("procReweightor::pull_deferred_backlog: size mismatch between pulled weights and normalisation");

            this->proc->append_wgts(*REX::vec_elem_mult<double>(*weights, this->normalisation));
        }
        this->backlog.clear();
    }

    void xSecAccumulator::absorb(const std::vector<std::shared_ptr<REX::event>> &events)
    {
        for (auto &ev : events)
        {
            if (!ev)
                continue;
            ++this->n_events;
            this->weight_sum += ev->weight_;
            this->weight_sumSq += ev->weight_ * ev->weight_;
            if (this->wgt_sum.size() < ev->wgts_.size())
            {
                this->wgt_sum.resize(ev->wgts_.size(), 0.0);
                this->ratio_sum.resize(ev->wgts_.size(), 0.0);
                this->ratio_sumSq.resize(ev->wgts_.size(), 0.0);
            }
            for (size_t k = 0; k < ev->wgts_.size(); ++k)
            {
                this->wgt_sum[k] += ev->wgts_[k];
                double ratio = (ev->weight_ != 0.0) ? ev->wgts_[k] / ev->weight_ : 0.0;
                this->ratio_sum[k] += ratio;
                this->ratio_sumSq[k] += ratio * ratio;
            }
        }
    }

    void xSecAccumulator::finalise_into(reweightor &r) const
    {
        if (this->n_events == 0)
            throw std::runtime_error("xSecAccumulator::finalise_into: no events were streamed, cannot calculate cross sections");

        // Mirrors reweightor::calc_xSec_from_weights(): only estimate
        // xSec_/xSecErr_ from event weights if no header cross section (eg a
        // real <init> block) is already available
        if (std::accumulate(r.xSec_.begin(), r.xSec_.end(), 0.0) == 0.0)
        {
            double mean = this->weight_sum / double(this->n_events);
            double variance = std::max(this->weight_sumSq / double(this->n_events) - std::pow(mean, 2), 0.0);
            r.xSec_ = {mean};
            r.xSecErr_ = {std::sqrt(variance / double(this->n_events))};
        }

        // Mirrors reweightor::calc_norm()
        r.norm_factor = 1.0;
        if (std::abs(r.idWgt_) == 3)
        {
            r.norm_factor = std::accumulate(r.xSec_.begin(), r.xSec_.end(), 0.0) / double(this->n_events);
        }
        else if (std::abs(r.idWgt_) == 4)
        {
            r.norm_factor = 1.0 / double(this->n_events);
        }
        else
        {
            if (std::abs(r.idWgt_) > 2 || r.idWgt_ == 0)
                r.verbose.warn("reweightor::run_streaming: idWgt is not set to a value defined in the LHE standard. Assuming weighted events.");
            double total_xSec = std::accumulate(r.xSec_.begin(), r.xSec_.end(), 0.0);
            if (this->weight_sum == 0.0)
                throw std::runtime_error("xSecAccumulator::finalise_into: total weight is zero, cannot calculate norm");
            r.norm_factor = total_xSec / this->weight_sum;
        }

        // Mirrors reweightor::calc_xSecs()
        r.rwgt_xSec = this->wgt_sum;
        for (auto &x : r.rwgt_xSec)
            x *= r.norm_factor;

        // Mirrors reweightor::calc_xErrs()
        double loc_xSec = std::accumulate(r.xSec_.begin(), r.xSec_.end(), 0.0);
        double loc_xErr = std::sqrt(std::accumulate(r.xSecErr_.begin(), r.xSecErr_.end(), 0.0, [](double a, double b)
                                                     { return a + b * b; }));
        double invNoEvs = 1.0 / double(this->n_events);
        double sqrtInvNoEvs = std::sqrt(invNoEvs);
        r.rwgt_xErr.assign(r.rwgt_xSec.size(), 0.0);
        for (size_t k = 0; k < r.rwgt_xSec.size(); ++k)
        {
            double variance = (this->ratio_sumSq[k] - std::pow(this->ratio_sum[k], 2) * invNoEvs) * invNoEvs;
            variance = std::max(variance, 0.0);
            r.rwgt_xErr[k] = loc_xSec * std::sqrt(variance) * sqrtInvNoEvs + loc_xErr * this->ratio_sum[k] * invNoEvs;
            if (std::isnan(r.rwgt_xErr[k]) || std::isinf(r.rwgt_xErr[k]) || r.rwgt_xErr[k] <= 0.0)
            {
                if (loc_xSec <= 0.0 || r.rwgt_xSec[k] == 0.0)
                {
                    r.verbose.warn("reweightor::run_streaming: Error propagation failed for weight " + std::to_string(k) + " and no baseline cross section is available to approximate it from; setting the error to 0.", true);
                    r.rwgt_xErr[k] = 0.0;
                }
                else
                {
                    r.verbose.warn("reweightor::run_streaming: Error propagation failed for weight " + std::to_string(k) + ". Approximating the error at the level of the cross section.");
                    r.rwgt_xErr[k] = loc_xErr * std::max(loc_xSec / r.rwgt_xSec[k], r.rwgt_xSec[k] / loc_xSec);
                }
            }
        }
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

    void reweightor::calc_xSec_from_weights()
    {
        if (std::accumulate(this->xSec_.begin(), this->xSec_.end(), 0.0) != 0.0)
            return; // a header cross section (e.g. from a real <init> block) is already available
        if (this->events.empty())
            this->transpose();
        if (this->events.empty())
            throw std::runtime_error("reweightor::calc_xSec_from_weights: no events found, cannot estimate a cross section");
        double sum = 0.0, sumSq = 0.0;
        for (auto ev : this->events)
        {
            sum += ev->weight_;
            sumSq += std::pow(ev->weight_, 2);
        }
        double nEvs = double(this->events.size());
        double mean = sum / nEvs;
        double variance = std::max(sumSq / nEvs - std::pow(mean, 2), 0.0);
        this->xSec_ = {mean};
        this->xSecErr_ = {std::sqrt(variance / nEvs)};
    }

    void reweightor::calc_norm()
    {
        if (this->events.empty())
            throw std::runtime_error("reweightor::calc_norm: no events loaded, cannot calculate norm");
        this->calc_xSec_from_weights();
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
                this->verbose.warn("reweightor::calc_norm: idWgt is not set to a value defined in the LHE standard. Assuming weighted events.");
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
        // classify the events on as many threads as the reweighting uses
        unsigned hc = std::thread::hardware_concurrency();
        this->sort_threads = this->pool_threads ? this->pool_threads : std::max(hc, 1u);
        this->events_to_processes(); // sorts the events; also (re)populates process_group, 1:1 with processes

        const size_t R = this->reweightors.size();
        if (this->processes.size() != this->process_group.size())
            throw std::runtime_error("reweightor::extract_sorter: processes/process_group size mismatch.");

        auto processes_full = std::move(this->processes);
        auto groups_full = std::move(this->process_group);
        auto base_reweightors = std::move(this->reweightors);

        // Tracks, per registered procReweightor (ie per sorter class), whether its
        // first batch has already been assigned. That first batch reuses the
        // original procReweightor object; any further max_batch overflow batches
        // of the same class each get a fresh clone of it -- identical reweight
        // functions, normaliser and event checker, bound (via set_process below)
        // to their own process/event subsample -- mirroring lhe::process_group's
        // "additional index per overflow batch" on the reweightor side
        std::vector<bool> group_claimed(R, false);

        std::vector<std::shared_ptr<process>> procs;
        std::vector<std::shared_ptr<procReweightor>> rwgs;
        std::vector<size_t> groups_kept;
        procs.reserve(processes_full.size());
        rwgs.reserve(processes_full.size());
        groups_kept.reserve(processes_full.size());

        for (size_t i = 0; i < processes_full.size(); ++i)
        {
            if (processes_full[i]->events.empty())
                continue;

            size_t group = groups_full[i];
            std::shared_ptr<procReweightor> rw;
            if (group == npos)
            {
                // Catch-all "matched no registered procReweightor" bucket: every
                // batch of it gets its own independent, unconfigured
                // procReweightor (which yields zero weights, same as before)
                rw = std::make_shared<procReweightor>();
            }
            else
            {
                if (group >= R)
                    throw std::runtime_error("reweightor::extract_sorter: process_group index out of range.");
                if (!group_claimed[group])
                {
                    rw = base_reweightors[group];
                    group_claimed[group] = true;
                }
                else
                {
                    rw = std::make_shared<procReweightor>(*base_reweightors[group]);
                    // Clone only the config; runtime state (proc/normalisation/
                    // backlog) belongs to the batch it will be bound to below,
                    // not whatever the base procReweightor may have accumulated
                    rw->proc = nullptr;
                    rw->normalisation.clear();
                    rw->backlog.clear();
                }
            }

            procs.push_back(processes_full[i]);
            rwgs.push_back(std::move(rw));
            groups_kept.push_back(group);
        }

        this->processes = std::move(procs);
        this->reweightors = std::move(rwgs);
        this->process_group = std::move(groups_kept);

        if (this->processes.size() != this->reweightors.size())
            throw std::runtime_error("reweightor::extract_sorter: number of processes does not match number of reweightors.");

        for (size_t i = 0; i < this->reweightors.size(); ++i)
        {
            auto &p = this->processes[i];
            p->validate();
            this->reweightors[i]->set_process(p);
            // Mute this batch's procReweightor whenever this reweightor is
            // muted, without touching its own default (a procReweightor
            // used standalone, outside of any reweightor, stays verbose)
            this->reweightors[i]->verbose.link(this->verbose);
        }
    }

    void reweightor::extract_sorter_stream(const std::vector<std::shared_ptr<procReweightor>> &base)
    {
        if (base.empty())
            throw std::runtime_error("reweightor::extract_sorter_stream: no procReweightors registered");

        std::vector<event_bool_fn> preds;
        preds.reserve(base.size());
        for (const auto &rw : base)
        {
            if (rw->event_checker_fn)
                preds.push_back(rw->event_checker_fn);
            else
                preds.push_back(rw->event_checker->get_event_bool());
        }

        this->set_sorter(eventSorter(std::move(preds)));
        this->sorted_events.clear();
        this->processes.clear();
        this->events_to_processes(); // sorts the events; also (re)populates process_group, 1:1 with processes

        const size_t R = base.size();
        if (this->processes.size() != this->process_group.size())
            throw std::runtime_error("reweightor::extract_sorter_stream: processes/process_group size mismatch.");

        std::vector<std::shared_ptr<process>> procs;
        std::vector<std::shared_ptr<procReweightor>> rwgs;
        std::vector<size_t> groups_kept;
        procs.reserve(this->processes.size());
        rwgs.reserve(this->processes.size());
        groups_kept.reserve(this->processes.size());

        for (size_t i = 0; i < this->processes.size(); ++i)
        {
            if (this->processes[i]->events.empty())
                continue;

            size_t group = this->process_group[i];
            std::shared_ptr<procReweightor> rw;
            if (group == npos)
            {
                // Catch-all "matched no registered procReweightor" bucket
                rw = std::make_shared<procReweightor>();
            }
            else
            {
                if (group >= R)
                    throw std::runtime_error("reweightor::extract_sorter_stream: process_group index out of range.");
                // Unlike extract_sorter(), always clone: `base` must stay an
                // untouched, reusable template across every batch, so no
                // batch's runtime state (proc/normalisation/backlog) is ever
                // written into it
                rw = std::make_shared<procReweightor>(*base[group]);
                rw->proc = nullptr;
                rw->normalisation.clear();
                rw->backlog.clear();
            }

            procs.push_back(this->processes[i]);
            rwgs.push_back(std::move(rw));
            groups_kept.push_back(group);
        }

        this->processes = std::move(procs);
        this->reweightors = std::move(rwgs);
        this->process_group = std::move(groups_kept);

        if (this->processes.size() != this->reweightors.size())
            throw std::runtime_error("reweightor::extract_sorter_stream: number of processes does not match number of reweightors.");

        for (size_t i = 0; i < this->reweightors.size(); ++i)
        {
            auto &p = this->processes[i];
            p->validate();
            this->reweightors[i]->set_process(p);
            this->reweightors[i]->verbose.link(this->verbose);
        }
    }

    void reweightor::initialise_reweightors()
    {
        if (this->reweightors.size() != this->processes.size())
            throw std::runtime_error("initialise_reweightors: reweightors/processes size mismatch");

        // The normalisations, one task per batch on the worker pool, as the
        // reweighting iterations (run_iteration())
        setup_pool();
        pool->begin_batch();
        for (size_t i = 0; i < this->reweightors.size(); ++i)
        {
            pool->enqueue([this, i]
                          {
                    if (pool->cancel_requested()) return;
                    this->reweightors[i]->initialise(this->processes[i]); });
        }
        pool->wait_batch();
    }

    void reweightor::finalise_reweighting()
    {
        // Materializes every genuinely-deferred result accumulated across
        // the whole iteration loop, one pull per procReweightor rather than
        // one per iteration
        for (auto &rwgt : this->reweightors)
        {
            rwgt->pull_deferred_backlog();
        }
        for (auto proc : this->processes)
        {
            proc->transpose_wgts();
            proc->validate();
        }
        if (!this->finalise())
            this->verbose.warn("reweightor::finalise_reweighting: finalise iterator returned false, something might have gone wrong. Validate output manually.");
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

    void reweightor::sync_n_amps()
    {
        this->n_amps = 0;
        for (auto &rwgt : this->reweightors)
            this->n_amps = std::max(this->n_amps, rwgt->reweight_functions.size());
        if (this->n_amps == 0)
            throw std::runtime_error("reweightor::sync_n_amps: no reweight functions found, something went wrong.");
    }

    void reweightor::setup()
    {
        if (!this->initialise())
            throw std::runtime_error("reweightor::setup: initialise iterator returned false, something went wrong.");
        this->extract_sorter();
        this->initialise_reweightors();
        this->sync_n_amps();
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
            this->verbose.print(".");
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

    void reweightor::run_streaming(eventSink &sink, bool include_ids)
    {
        if (!this->streaming())
            throw std::runtime_error("reweightor::run_streaming: lhe is not in streaming mode (no source set); use run() instead");
        if (!this->initialise())
            throw std::runtime_error("reweightor::run_streaming: initialise iterator returned false, something went wrong.");
        if (this->reweightors.empty())
            throw std::runtime_error("reweightor::run_streaming: no procReweightors set in reweightor");

        // The procReweightors as originally registered, before any batch is
        // pulled: extract_sorter_stream() only ever reads from this (cloning
        // fresh instances per batch), never writes into it, so it stays a
        // valid template for every batch of the run
        auto base_reweightors = this->reweightors;

        // Weight IDs (pre-existing <initrwgt> ids plus this run's
        // launch_names) are entirely header/config-derived -- see
        // lhe::extract_weight_ids() and finalise_reweighting() -- so they
        // can be resolved once, up front, instead of once per batch. That
        // way every batch's events already carry the final weight_ids as
        // they're written
        this->extract_weight_ids();
        if (this->launch_names.size() > 0)
        {
            size_t nWgts = this->weight_ids->size();
            for (size_t i = 0; i < this->launch_names.size(); ++i)
            {
                std::string curr_name = (this->launch_names[i].empty()) ? "rwgt_" + std::to_string(i + nWgts + 1) : this->launch_names[i];
                this->weight_ids->push_back(curr_name);
            }
        }
        this->append_weight_ids(include_ids);
        sink.write_open(static_cast<const initNode &>(*this), this->header);

        this->setup_pool();

        xSecAccumulator acc;
        while (this->pull_batch() > 0)
        {
            this->extract_sorter_stream(base_reweightors);
            this->initialise_reweightors();
            this->sync_n_amps();

            this->curr_iter = 0;
            this->run_all_iterations();

            // Materialize every genuinely-deferred result for this batch
            // now (rather than once at the very end, as finalise_reweighting()
            // does for the non-streaming path): this batch's procReweightor
            // instances are clones, about to be discarded, and the next
            // batch's clones start from a fresh copy of `base_reweightors`
            // regardless -- but draining now is also what makes it safe for
            // extract_sorter_stream() to matter-of-factly clone `base` again
            // next iteration without inheriting stale backlog state
            for (auto &rwgt : this->reweightors)
                rwgt->pull_deferred_backlog();
            for (auto &proc : this->processes)
            {
                proc->transpose_wgts();
                proc->validate();
            }

            acc.absorb(this->events);
            for (auto &ev : this->events)
                sink.write_event(ev);

            this->verbose.print(".");
            std::cout.flush();
        }

        if (!this->finalise())
            this->verbose.warn("reweightor::run_streaming: finalise iterator returned false, something might have gone wrong. Validate output manually.");

        acc.finalise_into(*this);
        sink.write_close();
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
        this->calc_xSec_from_weights();
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
                if (loc_xSec <= 0.0 || this->rwgt_xSec[k] == 0.0)
                {
                    this->verbose.warn("reweightor::calc_xErrs: Error propagation failed for weight " + std::to_string(k) + " and no baseline cross section is available to approximate it from; setting the error to 0.", true);
                    this->rwgt_xErr[k] = 0.0;
                }
                else
                {
                    this->verbose.warn("reweightor::calc_xErrs: Error propagation failed for weight " + std::to_string(k) + ". Approximating the error at the level of the cross section.");
                    this->rwgt_xErr[k] = loc_xErr * std::max(loc_xSec / this->rwgt_xSec[k], this->rwgt_xSec[k] / loc_xSec);
                }
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