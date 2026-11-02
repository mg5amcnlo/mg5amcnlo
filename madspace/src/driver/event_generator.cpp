#include "madspace/driver/event_generator.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include "madspace/driver/logger.hpp"
#include "madspace/driver/random.hpp"
#include "madspace/util.hpp"

using namespace madspace;

const GeneratorConfig EventGenerator::default_config = {};

EventGenerator::EventGenerator(
    const std::vector<ContextPtr>& contexts,
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    std::uint64_t seed,
    std::shared_ptr<StatusFile> status_file,
    const GeneratorConfig& config
) :
    _config(config),
    _status{
        .subprocess = 0,
        .name = "",
        .mean = 0.,
        .error = 0.,
        .mean_abs = 0.,
        .error_abs = 0.,
        .rel_std_dev = 0.,
        .count = 0,
        .count_opt = 0,
        .count_after_cuts = 0,
        .count_after_cuts_opt = 0,
        .count_unweighted = 0.,
        .count_target = config.target_count,
        .optimized = false,
        .done = false
    },
    _channels(channels),
    _contexts(contexts),
    _job_id(0),
    _channel_job_counts(channels.size()),
    _channel_optimizing(channels.size()),
    _channel_integral_fractions(channels.size(), 1.),
    _context_job_counts(contexts.size()),
    _channel_batch_pending(channels.size()),
    _channel_batch_dispatch_done(channels.size()),
    _channel_gen_order(channels.size()),
    _channel_ready_gen(channels.size()),
    _channel_unweight_order(channels.size()),
    _channel_unweight_ready(channels.size()),
    _context_unweight_queue(contexts.size()),
    _seed(seed),
    _status_file(status_file) {}

// Commits a job's generate stage in per-channel dispatch order. Doesn't write
// events itself -- if job.unweight, that's deferred to commit_unweight_job().
void EventGenerator::commit_generate_job(GeneratorBatchJob& job) {
    auto& channel = _channels.at(job.channel_index);
    auto& channel_job_count = _channel_job_counts.at(job.channel_index);
    if (job.is_vegas_batch && channel_job_count == job.split_job_count) {
        channel->clear_events();
    }
    channel->integrate(job);
    update_integral_status();
    channel->update_max_weight(job.weights);
    --channel_job_count;
    --_context_job_counts.at(job.context_index);
    if (job.unweight) {
        // Snapshot max_weight here, at this job's fixed commit position.
        channel->prepare_unweight_job(job);
        _context_unweight_queue.at(job.context_index).push_back(job.job_id);
        _channel_unweight_order.at(job.channel_index).push_back(job.job_id);
    }
    finish_channel_job(job);
    print_gen_update(false);
}

void EventGenerator::commit_unweight_job(GeneratorBatchJob& job) {
    auto& channel = _channels.at(job.channel_index);
    channel->write_events(job.unweighted_events, job.max_weight);
    update_counts();
    --_channel_job_counts.at(job.channel_index);
    --_context_job_counts.at(job.context_index);
    finish_channel_job(job);
    print_gen_update(false);
}

void EventGenerator::finish_channel_job(const GeneratorBatchJob& job) {
    auto& channel_job_count = _channel_job_counts.at(job.channel_index);
    if (channel_job_count != 0) {
        return;
    }
    if (job.is_vegas_batch) {
        _channels.at(job.channel_index)->optimize_vegas(job);
        _channel_optimizing.at(job.channel_index) = false;
    } else if (_channel_batch_dispatch_done.at(job.channel_index)) {
        _channel_batch_pending.at(job.channel_index) = false;
    }
}

// Jobs run on the pool, but commit strictly in per-channel dispatch order, so
// results are a pure function of the seed.
void EventGenerator::generate() { run_generate(GenerateMode::events); }

void EventGenerator::optimize() { run_generate(GenerateMode::optimize_only); }

void EventGenerator::fix_max_weights() { run_generate(GenerateMode::fix_max_weights); }

void EventGenerator::run_generate(GenerateMode mode) {
    _generate_mode = mode;
    _survey_job = false;
    reset_start_time();
    print_gen_init();

    // update_max_weight() scales its truncation budget with count_target, but
    // set_target_count() only runs once a round has committed. Seed a generous target
    // so round one truncates at all; _max_weight only rises, and set_target_count()
    // tightens it when the real (smaller) target arrives.
    for (auto& channel : _channels) {
        if (channel->status().count_opt == 0 || mode == GenerateMode::fix_max_weights) {
            channel->set_target_count(
                mode == GenerateMode::fix_max_weights
                    ? _config.freeze_max_weight_after
                    : _config.target_count
            );
        }
    }

    // channels loaded with a known integral can share the events from the start
    if (std::any_of(_channels.begin(), _channels.end(), [](auto& channel) {
            return channel->has_abs_integral_prior();
        })) {
        update_integral_fractions();
    }

    while (true) {
        _abort_check_function();

        for (std::size_t channel_index = 0; channel_index < _channels.size();
             ++channel_index) {
            auto& channel = _channels.at(channel_index);
            double integral_frac = _channel_integral_fractions.at(channel_index);
            if (mode == GenerateMode::optimize_only) {
                if (!channel->needs_optimization()) {
                    continue;
                }
            } else if (integral_frac > 0 &&
                       channel->status().count_unweighted >=
                           channel->status().count_target) {
                continue;
            }
            if (channel->needs_optimization()) {
                if (!_channel_optimizing.at(channel_index)) {
                    _channel_optimizing.at(channel_index) = true;
                    _ready_jobs.push_back({
                        .channel_index = channel_index,
                        // optimize_only keeps no events
                        .unweight = mode != GenerateMode::optimize_only,
                        .batch_event_count = channel->next_vegas_batch_size(),
                        .is_vegas_batch = true,
                    });
                }
            } else if (!_channel_batch_pending.at(channel_index)) {
                _channel_batch_pending.at(channel_index) = true;
                _channel_batch_dispatch_done.at(channel_index) = false;
                _ready_jobs.push_back({
                    .channel_index = channel_index,
                    .unweight = true,
                    .batch_event_count = next_batch_event_count(channel_index),
                    .is_vegas_batch = false,
                });
            }
        }

        std::size_t job_id_before = _job_id;
        std::size_t unweight_dispatched = start_jobs();
        std::size_t round_in_flight = (_job_id - job_id_before) + unweight_dispatched;

        // Wait for the round to fully commit before resyncing cross-channel fractions.
        while (round_in_flight > 0) {
            std::size_t job_id = wait_for_result(round_in_flight);
            auto& job = _running_jobs.at(job_id);
            if (job.unweighted_events.size() == 0) {
                auto& ready = _channel_ready_gen.at(job.channel_index);
                auto& order = _channel_gen_order.at(job.channel_index);
                ready.insert(job_id);
                while (!order.empty() && ready.erase(order.front()) > 0) {
                    std::size_t commit_id = order.front();
                    order.pop_front();
                    auto& gen_job = _running_jobs.at(commit_id);
                    commit_generate_job(gen_job);
                    if (!gen_job.unweight) {
                        _running_jobs.erase(commit_id);
                    }
                }
            } else {
                auto& ready = _channel_unweight_ready.at(job.channel_index);
                auto& order = _channel_unweight_order.at(job.channel_index);
                ready.insert(job_id);
                while (!order.empty() && ready.erase(order.front()) > 0) {
                    std::size_t commit_id = order.front();
                    order.pop_front();
                    commit_unweight_job(_running_jobs.at(commit_id));
                    _running_jobs.erase(commit_id);
                }
            }
            std::size_t job_id_refill = _job_id;
            std::size_t refill_unweight_dispatched = start_jobs();
            round_in_flight += (_job_id - job_id_refill) + refill_unweight_dispatched;
        }

        if (mode == GenerateMode::optimize_only) {
            if (std::none_of(_channels.begin(), _channels.end(), [](auto& channel) {
                    return channel->needs_optimization();
                })) {
                break;
            }
            continue;
        }

        update_integral_fractions();
        update_counts();

        if (_status.done) {
            unweight_all();
        }
        if (_status.done) {
            break;
        }
    }
    print_gen_update(true);
}

// Same commit-ordering approach as generate().
void EventGenerator::survey(std::size_t survey_pass) {
    _survey_job = true;
    _survey_pass = survey_pass;
    reset_start_time();
    _commit_cursor = _job_id;
    _ready_gen.clear();
    bool done = false;
    std::size_t min_iters = _config.survey_min_iters;
    std::size_t max_iters = std::max(min_iters, _config.survey_max_iters);
    double target_precision = _config.survey_target_precision;

    std::size_t total_event_count = 0;
    std::size_t done_event_count = 0;
    for (auto& channel : _channels) {
        std::size_t chan_batch_size = channel->batch_size();
        for (std::size_t iter = channel->status().iterations; iter < min_iters;
             ++iter) {
            total_event_count += chan_batch_size;
            chan_batch_size = std::min(chan_batch_size * 2, _config.max_batch_size);
        }
    }
    print_survey_init();

    std::size_t iter = 0;
    for (; !done && iter < max_iters; ++iter) {
        for (std::size_t i = 0; auto channel : _channels) {
            if (channel->status().iterations > iter) {
                ++i;
                continue;
            }
            if (iter >= min_iters &&
                channel->abs_cross_section().rel_error() < target_precision) {
                ++i;
                continue;
            }
            std::size_t vegas_batch_size = channel->next_vegas_batch_size();
            _ready_jobs.push_back({
                .channel_index = i,
                .unweight = iter >= min_iters - 1,
                .batch_event_count = vegas_batch_size,
                .is_vegas_batch = true,
            });
            if (iter >= min_iters) {
                total_event_count += vegas_batch_size;
            }
            ++i;
        }
        std::size_t job_id_before = _job_id;
        std::size_t unweight_dispatched = start_jobs();
        std::size_t in_flight = (_job_id - job_id_before) + unweight_dispatched;
        done = true;
        while (in_flight > 0) {
            std::size_t job_id = wait_for_result(in_flight);
            auto& job = _running_jobs.at(job_id);
            if (job.unweighted_events.size() == 0) {
                // Generate-stage completion, globally ordered via
                // _ready_gen/_commit_cursor.
                _ready_gen.insert(job_id);
                while (_ready_gen.erase(_commit_cursor) > 0) {
                    auto& gen_job = _running_jobs.at(_commit_cursor);
                    auto& channel = _channels.at(gen_job.channel_index);
                    auto& channel_job_count =
                        _channel_job_counts.at(gen_job.channel_index);
                    if (channel_job_count == gen_job.split_job_count) {
                        channel->clear_events();
                    }
                    channel->integrate(gen_job);
                    update_integral();
                    channel->update_max_weight(gen_job.weights);
                    --channel_job_count;
                    --_context_job_counts.at(gen_job.context_index);
                    // Global order is ascending job id and so is the per-channel one,
                    // so this job is necessarily at the front of its channel's deque.
                    _channel_gen_order.at(gen_job.channel_index).pop_front();
                    if (gen_job.unweight) {
                        // Snapshot max_weight at this fixed commit position.
                        channel->prepare_unweight_job(gen_job);
                        _context_unweight_queue.at(gen_job.context_index)
                            .push_back(gen_job.job_id);
                        _channel_unweight_order.at(gen_job.channel_index)
                            .push_back(gen_job.job_id);
                    } else {
                        done = false;
                        if (channel_job_count == 0) {
                            channel->optimize_vegas(gen_job);
                            done_event_count += gen_job.batch_event_count;
                        }
                        _running_jobs.erase(_commit_cursor);
                    }
                    ++_commit_cursor;
                }
            } else {
                // Unweight-stage completion, ordered per channel only.
                auto& ready = _channel_unweight_ready.at(job.channel_index);
                auto& order = _channel_unweight_order.at(job.channel_index);
                ready.insert(job_id);
                while (!order.empty() && ready.erase(order.front()) > 0) {
                    std::size_t commit_id = order.front();
                    order.pop_front();
                    auto& uw_job = _running_jobs.at(commit_id);
                    auto& channel = _channels.at(uw_job.channel_index);
                    channel->write_events(uw_job.unweighted_events, uw_job.max_weight);
                    update_counts();
                    auto& channel_job_count =
                        _channel_job_counts.at(uw_job.channel_index);
                    --channel_job_count;
                    --_context_job_counts.at(uw_job.context_index);
                    if (channel_job_count == 0 &&
                        channel->abs_cross_section().rel_error() < target_precision) {
                        done = false;
                    }
                    if (channel_job_count == 0) {
                        channel->optimize_vegas(uw_job);
                        done_event_count += uw_job.batch_event_count;
                    }
                    _running_jobs.erase(commit_id);
                }
            }
            std::size_t job_id_refill = _job_id;
            std::size_t refill_unweight_dispatched = start_jobs();
            in_flight += (_job_id - job_id_refill) + refill_unweight_dispatched;
            print_survey_update(false, done_event_count, total_event_count, iter);
        }
    }
    print_survey_update(true, done_event_count, total_event_count, iter - 1);
}

// Every job counted in *in_flight* posts exactly one result (see
// ResultQueue::submit), so this can't block on a job that died. If a job threw, or
// the abort check does while waiting, the other jobs still in flight write into
// _running_jobs: wait them out and drop the bookkeeping of the aborted run before
// rethrowing, so the exception unwinds with no job left behind. The running jobs
// stop at their next cancellation check, and an abort requested while they do
// takes precedence over the job's own error.
std::size_t EventGenerator::wait_for_result(std::size_t& in_flight) {
    std::exception_ptr exception;
    try {
        auto result = _result_queue.wait(_abort_check_function);
        --in_flight;
        if (!result.exception) {
            return result.id;
        }
        exception = result.exception;
    } catch (...) {
        exception = std::current_exception();
    }
    auto aborted = _result_queue.discard(in_flight, _abort_check_function);
    in_flight = 0;
    reset_jobs();
    std::rethrow_exception(aborted ? aborted : exception);
}

void EventGenerator::reset_jobs() {
    _running_jobs.clear();
    _ready_jobs.clear();
    _ready_job_rr_cursor = 0;
    _ready_gen.clear();
    _commit_cursor = _job_id;
    std::fill(_channel_job_counts.begin(), _channel_job_counts.end(), 0);
    std::fill(_context_job_counts.begin(), _context_job_counts.end(), 0);
    std::fill(_channel_optimizing.begin(), _channel_optimizing.end(), false);
    std::fill(_channel_batch_pending.begin(), _channel_batch_pending.end(), false);
    std::fill(
        _channel_batch_dispatch_done.begin(), _channel_batch_dispatch_done.end(), false
    );
    for (auto& queue : _channel_gen_order) {
        queue.clear();
    }
    for (auto& ready : _channel_ready_gen) {
        ready.clear();
    }
    for (auto& queue : _channel_unweight_order) {
        queue.clear();
    }
    for (auto& ready : _channel_unweight_ready) {
        ready.clear();
    }
    for (auto& queue : _context_unweight_queue) {
        queue.clear();
    }
}

// Reads only committed data (never jobs still in flight), so the estimate is a
// pure function of the seed. Sizing itself is in
// compute_generation_batch_event_count().
std::size_t EventGenerator::next_batch_event_count(std::size_t channel_index) const {
    auto& channel = _channels.at(channel_index);
    auto& status = channel->status();
    auto& abs_cross_section = channel->abs_cross_section();
    return compute_generation_batch_event_count(
        status.count_target,
        status.count_unweighted,
        // Not count_opt: start_jobs() splits batches by nominal size, and a job
        // repeating batches to pass cuts yields more raw points than that.
        channel->count_requested_opt(),
        abs_cross_section.count(),
        abs_cross_section.rel_error(),
        _config
    );
}

std::size_t EventGenerator::start_jobs() {
    std::size_t context_index = 0;
    std::size_t unweight_dispatched = 0;
    for (auto [context, job_count] : zip(_contexts, _context_job_counts)) {
        // fill the queue to twice the thread count to keep the worker threads busy
        std::size_t target_count = 2 * context->thread_pool().thread_count();
        std::size_t batch_size = context->device()->device_type() == DeviceType::cpu
            ? _config.cpu_batch_size
            : _config.gpu_batch_size;

        // give priority to unweighting jobs to reduce memory usage
        auto& unweight_queue = _context_unweight_queue.at(context_index);
        std::size_t unweight_index = 0;
        for (; job_count < target_count && unweight_index < unweight_queue.size();
             ++unweight_index) {
            auto& job = _running_jobs.at(unweight_queue.at(unweight_index));
            _channels.at(job.channel_index)->submit_unweight_job(job, _result_queue);
            ++job_count;
            ++unweight_dispatched;
        }
        unweight_queue.erase(
            unweight_queue.begin(), unweight_queue.begin() + unweight_index
        );

        // Round-robin through _ready_jobs, one device batch per generation entry per
        // visit, so multiple channels' batches interleave instead of one channel's
        // batch draining before the next is even touched -- see _ready_job_rr_cursor.
        // A VEGAS batch is still dispatched atomically in a single visit: VEGAS needs
        // next_vegas_batch_size()'s exact progression.
        while (job_count < target_count && !_ready_jobs.empty()) {
            if (_ready_job_rr_cursor >= _ready_jobs.size()) {
                _ready_job_rr_cursor = 0;
            }
            auto& ready_job = _ready_jobs.at(_ready_job_rr_cursor);
            bool atomic_dispatch =
                ready_job.is_vegas_batch || ready_job.batch_event_count == 0;
            std::size_t remaining_sub_jobs = ready_job.batch_event_count != 0
                ? (ready_job.batch_event_count + batch_size - 1) / batch_size
                : 1;
            std::size_t to_dispatch = atomic_dispatch ? remaining_sub_jobs : 1;

            for (std::size_t i = 0; i < to_dispatch; ++i) {
                GeneratorBatchJob new_job{
                    .channel_index = ready_job.channel_index,
                    .unweight = ready_job.unweight,
                    .batch_event_count = ready_job.batch_event_count,
                    .split_job_count = to_dispatch * (1 + ready_job.unweight),
                    .context_index = context_index,
                    .job_id = _job_id,
                    .is_vegas_batch = ready_job.is_vegas_batch,
                };
                auto& job =
                    std::get<0>(_running_jobs.emplace(_job_id, std::move(new_job)))
                        ->second;
                // Records this channel's commit order; drained by survey()/generate()
                // -- see _channel_gen_order.
                _channel_gen_order.at(job.channel_index).push_back(_job_id);
                _channels.at(job.channel_index)
                    ->start_job(job, _result_queue, _seed, _survey_job, _survey_pass);
                _channel_job_counts.at(job.channel_index) += 1 + job.unweight;
                ++_job_id;
                ++job_count;
            }

            if (!atomic_dispatch) {
                ready_job.batch_event_count -=
                    std::min(to_dispatch * batch_size, ready_job.batch_event_count);
            }
            bool fully_dispatched = atomic_dispatch || ready_job.batch_event_count == 0;
            if (fully_dispatched) {
                if (!ready_job.is_vegas_batch) {
                    // Only meaningful for generate(); harmlessly unused by the other
                    // callers of start_jobs().
                    _channel_batch_dispatch_done.at(ready_job.channel_index) = true;
                }
                _ready_jobs.erase(_ready_jobs.begin() + _ready_job_rr_cursor);
                // The next entry has shifted into this index -- don't advance past it.
            } else {
                ++_ready_job_rr_cursor;
            }
        }
        ++context_index;
    }
    return unweight_dispatched;
}

void EventGenerator::update_integral() {
    update_integral_status();
    update_integral_fractions();
}

void EventGenerator::update_integral_status() {
    double total_mean = 0., total_var = 0.;
    double total_mean_abs = 0., total_var_abs = 0.;
    std::size_t total_count = 0, total_count_opt = 0;
    std::size_t total_count_after_cuts = 0, total_count_after_cuts_opt = 0;
    std::size_t total_integ_count = 0;
    std::size_t iterations = 0;
    bool optimized = true;
    const ChannelEventGenerator* bad_channel = nullptr;
    for (auto& channel : _channels) {
        auto& status = channel->status();
        auto& cross_section = channel->cross_section();
        auto& abs_cross_section = channel->abs_cross_section();
        // special case for channels with 0 samples, as they have nan variance
        if (bad_channel == nullptr && cross_section.count() > 0 &&
            (!std::isfinite(cross_section.mean()) ||
             !std::isfinite(cross_section.variance()) ||
             !std::isfinite(abs_cross_section.mean()) ||
             !std::isfinite(abs_cross_section.variance()))) {
            bad_channel = channel.get();
        }
        if (channel->uses_integral_prior()) {
            // not sampled in this run (no events assigned): use the earlier estimate
            auto [mean, error] = channel->integral_prior();
            auto [mean_abs, error_abs] = channel->abs_integral_prior();
            total_mean += mean;
            total_var += error * error;
            total_mean_abs += mean_abs;
            total_var_abs += error_abs * error_abs;
        } else {
            total_mean += cross_section.mean();
            total_var += cross_section.variance() / cross_section.count();
            total_mean_abs += abs_cross_section.mean();
            total_var_abs += abs_cross_section.variance() / abs_cross_section.count();
        }
        total_count += status.count;
        total_count_opt += status.count_opt;
        total_count_after_cuts += status.count_after_cuts;
        total_count_after_cuts_opt += status.count_after_cuts_opt;
        total_integ_count += cross_section.count();
        iterations = std::max(status.iterations, iterations);
        if (!channel->status().optimized) {
            optimized = false;
        }
    }
    if (bad_channel != nullptr || !std::isfinite(total_mean) ||
        !std::isfinite(total_mean_abs)) {
        std::string where = bad_channel != nullptr
            ? std::format(
                  "channel '{}' after {} samples (mean={}, variance={})",
                  bad_channel->status().name,
                  bad_channel->cross_section().count(),
                  bad_channel->cross_section().mean(),
                  bad_channel->cross_section().variance()
              )
            : std::format("the sum over channels (mean={})", total_mean);
        throw std::runtime_error(
            std::format(
                "non-finite integral in {}. A non-finite weight cannot be recovered "
                "from by sampling further, so the integration is aborted. This usually "
                "means the matrix element or the phase-space mapping returned nan/inf "
                "for "
                "some phase-space point -- check the process for a zero or negative "
                "width, "
                "a parameter point outside the model's validity, or a kinematic "
                "configuration at a threshold.",
                where
            )
        );
    }

    _status.mean = total_mean;
    _status.error = std::sqrt(total_var);
    _status.mean_abs = total_mean_abs;
    _status.error_abs = std::sqrt(total_var_abs);
    _status.rel_std_dev = std::sqrt(total_var_abs * total_integ_count) / total_mean_abs;
    _status.count = total_count;
    _status.count_opt = total_count_opt;
    _status.count_after_cuts = total_count_after_cuts;
    _status.count_after_cuts_opt = total_count_after_cuts_opt;
    _status.iterations = iterations;
    _status.optimized = optimized;
}

void EventGenerator::update_integral_fractions() {
    // With an integral known from an earlier run, the sum of the channel estimates
    // normalizes the fractions; without, this is _status.mean_abs.
    double total_estimate = _status.mean_abs;
    if (std::any_of(_channels.begin(), _channels.end(), [](auto& channel) {
            return channel->has_abs_integral_prior();
        })) {
        total_estimate = 0.;
        for (auto& channel : _channels) {
            total_estimate += channel->abs_integral_estimate();
        }
    }
    for (auto [channel, integral_fraction] :
         zip(_channels, _channel_integral_fractions)) {
        integral_fraction = channel->abs_integral_estimate() / total_estimate;
    }

    // Distribute events between channels, ensure sum is exactly target_count
    std::vector<std::size_t> counts(_channels.size());
    std::vector<double> remainders(_channels.size());
    std::size_t allocated = 0;
    for (auto [count, remainder, fraction] :
         zip(counts, remainders, _channel_integral_fractions)) {
        double raw = fraction * _config.target_count;
        double floored = std::floor(std::isfinite(raw) && raw > 0. ? raw : 0.);
        count = static_cast<std::size_t>(floored);
        remainder = raw - floored;
        allocated += count;
    }
    std::vector<std::size_t> order(_channels.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        double rem_a = remainders.at(a), rem_b = remainders.at(b);
        return rem_a == rem_b ? a < b : rem_a > rem_b;
    });
    std::size_t leftover =
        allocated <= _config.target_count ? _config.target_count - allocated : 0;
    leftover = std::min(leftover, order.size());
    for (std::size_t index : order | std::views::take(leftover)) {
        ++counts.at(index);
    }
    for (auto [channel, count, fraction] :
         zip(_channels, counts, _channel_integral_fractions)) {
        if (_generate_mode == GenerateMode::fix_max_weights) {
            count = fraction > 0 ? _config.freeze_max_weight_after : 0;
        }
        channel->set_target_count(count);
    }
}

void EventGenerator::update_counts() {
    double total_eff_count = 0.;
    bool done = true;
    for (auto& channel : _channels) {
        std::size_t chan_target = channel->status().count_target;
        if (channel->status().count_unweighted < chan_target) {
            total_eff_count += channel->status().count_unweighted;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status.count_unweighted = total_eff_count;
    _status.done = done;
    if (_generate_mode == GenerateMode::fix_max_weights) {
        // every channel runs to its own target, whatever the config says
        _status.count_target = 0;
        for (auto& channel : _channels) {
            _status.count_target += channel->status().count_target;
        }
    }
}

void EventGenerator::combine_to_compact_npy(
    const std::string& file_name,
    SystematicsCalculator* systematics,
    EventHistograms* histograms
) {
    reset_start_time();
    _systematics = systematics;
    _event_histograms = histograms;
    MixMaxRandom select_rng(DerivedSeed(_seed, DerivedSeed::combine_select));
    auto [channel_data, particle_count, norm_factor] = init_combine();
    DataLayout layout = combined_layout();
    DataLayout out_layout = output_layout(
        EventRecord::f_weight | EventRecord::f_subproc_index |
            EventRecord::f_event_data,
        ParticleRecord::f_particle_data
    );
    EventBuffer buffer(0, particle_count, layout);
    EventBuffer buffer_out(0, particle_count, out_layout);
    EventFile event_file(file_name, out_layout, particle_count, EventFile::create);
    std::vector<double> syst_weights;
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    print_combine_init();
    while (true) {
        _abort_check_function();
        read_and_combine(channel_data, buffer, norm_factor, select_rng);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        process_combined_batch(buffer, syst_weights);
        buffer_out.resize(buffer.event_count());
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            copy_event_fields(buffer, i, buffer_out, i, syst_weights);
        }
        if (event_count - last_update_count > 10000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
        event_file.write(buffer_out);
    }
    print_combine_update(_config.target_count);
    _systematics = nullptr;
    _event_histograms = nullptr;
}

void EventGenerator::combine_to_lhe_npy(
    const std::string& file_name,
    LHECompleter& lhe_completer,
    SystematicsCalculator* systematics,
    EventHistograms* histograms
) {
    reset_start_time();
    _systematics = systematics;
    _event_histograms = histograms;
    std::uint64_t seed = _seed;
    MixMaxRandom select_rng(DerivedSeed(seed, DerivedSeed::combine_select));
    MixMaxRandom rand_gen(DerivedSeed(seed, DerivedSeed::lhe_complete));
    auto [channel_data, particle_count, norm_factor] = init_combine();
    DataLayout in_layout = combined_layout();
    DataLayout out_layout =
        output_layout(EventRecord::f_lhe_event, ParticleRecord::f_lhe_particle);
    EventBuffer buffer(0, particle_count, in_layout);
    EventBuffer buffer_out(0, lhe_completer.max_particle_count(), out_layout);
    EventFile event_file(
        file_name, out_layout, lhe_completer.max_particle_count(), EventFile::create
    );
    std::vector<double> syst_weights;
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    LHEEvent lhe_event;
    print_combine_init();
    while (true) {
        _abort_check_function();
        read_and_combine(channel_data, buffer, norm_factor, select_rng);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        process_combined_batch(buffer, syst_weights);
        buffer_out.resize(buffer.event_count());
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            fill_lhe_event(lhe_completer, lhe_event, buffer, i, rand_gen);
            buffer_out.event(i).from_lhe_event(lhe_event);
            copy_event_fields(buffer, i, buffer_out, i, syst_weights);
            std::size_t j = 0;
            for (; j < lhe_event.particles.size(); ++j) {
                buffer_out.particle(i, j).from_lhe_particle(
                    lhe_event.particles[j], lhe_event.has_cluster_scales
                );
            }
            for (; j < lhe_completer.max_particle_count(); ++j) {
                buffer_out.particle(i, j).from_lhe_particle(
                    LHEParticle{}, lhe_event.has_cluster_scales
                );
            }
        }
        event_file.write(buffer_out);
        if (event_count - last_update_count > 10000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
    }
    print_combine_update(_config.target_count);
    _systematics = nullptr;
    _event_histograms = nullptr;
}

void EventGenerator::combine_to_lhe(
    const std::string& file_name,
    LHECompleter& lhe_completer,
    const LHEMeta& meta,
    SystematicsCalculator* systematics,
    EventHistograms* histograms
) {
    reset_start_time();
    _systematics = systematics;
    _event_histograms = histograms;
    std::uint64_t seed = _seed;
    MixMaxRandom select_rng(DerivedSeed(seed, DerivedSeed::combine_select));
    ThreadPool pool(_config.combine_thread_count);
    auto [channel_data, particle_count, norm_factor] = init_combine();
    struct CombineBuffers {
        EventBuffer in_buffer;
        std::string out_buffer;
        std::vector<double> syst_weights;
    };
    std::vector<CombineBuffers> buffers;
    std::vector<std::size_t> idle_buffers;
    DataLayout layout = combined_layout();
    for (std::size_t i = 0; i < 2 * pool.thread_count(); ++i) {
        buffers.push_back({{0, particle_count, layout}, {}, {}});
        idle_buffers.push_back(i);
    }
    std::vector<int> syst_ids;
    LHEMeta full_meta = meta;
    if (_systematics) {
        syst_ids = _systematics->weight_ids();
        bool has_initrwgt = false;
        for (auto& header : full_meta.headers) {
            if (header.name == "initrwgt") {
                has_initrwgt = true;
            }
        }
        if (!has_initrwgt && !syst_ids.empty()) {
            full_meta.headers.push_back({"initrwgt", _systematics->initrwgt(), false});
        }
    }
    LHEFileWriter event_file(file_name, full_meta);
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    bool done = false;
    // Batches are seeded by submission order and written in that same order
    // (buffered in ready_slots until their turn), independent of completion timing.
    std::size_t next_batch_seq = 0;
    std::size_t write_cursor = 0;
    std::vector<std::size_t> slot_batch_seq(buffers.size());
    std::unordered_map<std::size_t, std::size_t> ready_slots;
    print_combine_init();
    while (true) {
        _abort_check_function();
        while (idle_buffers.size() > 0 && !done) {
            std::size_t slot = idle_buffers.back();
            auto& [in_buffer, out_buffer, syst_weights] = buffers.at(slot);
            read_and_combine(channel_data, in_buffer, norm_factor, select_rng);
            if (in_buffer.event_count() == 0) {
                done = true;
                break;
            }
            idle_buffers.pop_back();
            std::size_t batch_seq = next_batch_seq++;
            slot_batch_seq.at(slot) = batch_seq;
            pool.submit(
                [slot,
                 batch_seq,
                 seed,
                 this,
                 &in_buffer,
                 &out_buffer,
                 &syst_weights,
                 &lhe_completer,
                 &syst_ids] {
                    MixMaxRandom rand_gen(
                        DerivedSeed(seed, DerivedSeed::lhe_complete, batch_seq)
                    );
                    LHEEvent lhe_event;
                    out_buffer.clear();
                    process_combined_batch(in_buffer, syst_weights);
                    for (std::size_t i = 0; i < in_buffer.event_count(); ++i) {
                        fill_lhe_event(
                            lhe_completer, lhe_event, in_buffer, i, rand_gen
                        );
                        fill_systematics(
                            lhe_event, in_buffer, i, syst_weights, syst_ids
                        );
                        lhe_event.format_to(out_buffer);
                    }
                    return slot;
                }
            );
        }

        auto done_slots = pool.wait_multiple();
        for (std::size_t slot : done_slots) {
            ready_slots.emplace(slot_batch_seq.at(slot), slot);
        }
        for (auto it = ready_slots.find(write_cursor); it != ready_slots.end();
             it = ready_slots.find(write_cursor)) {
            std::size_t slot = it->second;
            auto& [in_buffer, out_buffer, syst_weights] = buffers.at(slot);
            event_file.write_string(out_buffer);
            event_count += in_buffer.event_count();
            idle_buffers.push_back(slot);
            ready_slots.erase(it);
            ++write_cursor;
            if (event_count - last_update_count > 10000) {
                print_combine_update(event_count);
                last_update_count = event_count;
            }
        }
        if (done_slots.size() == 0 && done) {
            break;
        }
    }
    print_combine_update(_config.target_count);
    _systematics = nullptr;
    _event_histograms = nullptr;
}

namespace {

// The MLM alpha_s vertex scales and pdf ladder of any channel: a merged run
// mixes a 2 -> 1 sample, which has no vertex, with the jet samples, which do,
// and channel 0 may be either.
template <typename Channels>
int any_alphas_scales(const Channels& channels) {
    int flags = 0;
    for (auto& channel : channels) {
        flags |= channel->particle_layout_extra_flags() &
            (ParticleRecord::f_alphas_scales | ParticleRecord::f_pdf_ladder);
    }
    return flags;
}

// the pdf ladder fields of one particle, or zeros (an inactive slot)
void copy_ladder(ParticleRecord in, ParticleRecord out, bool from_in) {
    out.ladder_x() = from_in ? in.ladder_x().value() : 0.;
    out.ladder_pdg() = from_in ? in.ladder_pdg().value() : 0.;
    out.ladder_q() = from_in ? in.ladder_q().value() : 0.;
    out.ladder_info() = from_in ? in.ladder_info().value() : 0.;
    out.ladder_lead_q() = from_in ? in.ladder_lead_q().value() : 0.;
}

} // namespace

DataLayout EventGenerator::combined_layout() const {
    return DataLayout(
        EventRecord::layout(
            EventRecord::f_weight | EventRecord::f_subproc_index |
            EventRecord::f_event_data |
            (_channels.at(0)->event_layout_extra_flags() & ~EventRecord::f_bias)
        ),
        ParticleRecord::layout(
            ParticleRecord::f_particle_data |
            _channels.at(0)->particle_layout_extra_flags() | any_alphas_scales(_channels)
        )
    );
}

DataLayout
EventGenerator::output_layout(int event_flags, std::size_t particle_flags) const {
    int extra_flags =
        _channels.at(0)->event_layout_extra_flags() & ~EventRecord::f_bias;
    std::vector<int> syst_ids;
    if (_systematics) {
        if (!_systematics->config().write_inputs) {
            extra_flags &=
                ~(EventRecord::f_beam1 | EventRecord::f_beam2 |
                  EventRecord::f_partial_weights);
        }
        syst_ids = _systematics->weight_ids();
        if (!syst_ids.empty()) {
            extra_flags |= EventRecord::f_syst_weights;
        }
    }
    // the alpha_s vertex scales and the pdf ladder only serve the systematics,
    // which run on the combined buffer: no event file carries them
    return DataLayout(
        EventRecord::layout(event_flags | extra_flags, syst_ids),
        ParticleRecord::layout(
            particle_flags |
            (_channels.at(0)->particle_layout_extra_flags() &
             ~(ParticleRecord::f_alphas_scales | ParticleRecord::f_pdf_ladder))
        )
    );
}

void EventGenerator::copy_event_fields(
    EventBuffer& in_buffer,
    std::size_t in_index,
    EventBuffer& out_buffer,
    std::size_t out_index,
    const std::vector<double>& syst_weights
) {
    // the output layouts are fixed: weight/subprocess/event data and the raw
    // momenta are present in the compact layout, absent in the LHE layout
    auto& out_fields = out_buffer.layout().event_layout();
    auto contains = [&](const std::string& name) {
        for (auto& field : out_fields) {
            if (field.name == name) {
                return true;
            }
        }
        return false;
    };
    auto event_in = in_buffer.event(in_index);
    auto event_out = out_buffer.event(out_index);
    if (contains("diagram_index")) {
        event_out.weight() = event_in.weight();
        event_out.subprocess_index() = event_in.subprocess_index();
        event_out.diagram_index() = event_in.diagram_index();
        event_out.color_index() = event_in.color_index();
        event_out.flavor_index() = event_in.flavor_index();
        event_out.helicity_index() = event_in.helicity_index();
        event_out.ren_scale() = event_in.ren_scale();
        event_out.alpha_qcd() = event_in.alpha_qcd();
        std::size_t i = 0;
        for (; i < in_buffer.particle_count(); ++i) {
            auto particle_in = in_buffer.particle(in_index, i);
            auto particle_out = out_buffer.particle(out_index, i);
            particle_out.energy() = particle_in.energy();
            particle_out.px() = particle_in.px();
            particle_out.py() = particle_in.py();
            particle_out.pz() = particle_in.pz();
        }
        for (; i < out_buffer.particle_count(); ++i) {
            auto particle_out = out_buffer.particle(out_index, i);
            particle_out.energy() = 0.;
            particle_out.px() = 0.;
            particle_out.py() = 0.;
            particle_out.pz() = 0.;
        }
    }
    if (contains("fact_scale1")) {
        event_out.x1() = event_in.x1();
        event_out.fact_scale1() = event_in.fact_scale1();
    }
    if (contains("fact_scale2")) {
        event_out.x2() = event_in.x2();
        event_out.fact_scale2() = event_in.fact_scale2();
    }
    if (contains("partial_weight_product")) {
        event_out.partial_weight_product() = event_in.partial_weight_product();
    }
    if (contains("matrix_element")) {
        event_out.matrix_element() = event_in.matrix_element();
    }
    if (_systematics) {
        std::size_t count = _systematics->weight_count();
        for (std::size_t k = 0; k < count; ++k) {
            event_out.syst_weight(k) = syst_weights.at(in_index * count + k);
        }
    }
}

void EventGenerator::fill_systematics(
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    const std::vector<double>& syst_weights,
    const std::vector<int>& syst_ids
) {
    lhe_event.rwgt_ids.clear();
    lhe_event.rwgt.clear();
    lhe_event.lo_info.reset();
    if (!_systematics) {
        return;
    }
    std::size_t count = _systematics->weight_count();
    lhe_event.rwgt_ids = syst_ids;
    lhe_event.rwgt.assign(
        syst_weights.begin() + event_index * count,
        syst_weights.begin() + (event_index + 1) * count
    );
    if (_systematics->config().write_inputs) {
        lhe_event.lo_info = _systematics->reweight_info(buffer, event_index);
    }
}

void EventGenerator::process_combined_batch(
    EventBuffer& buffer, std::vector<double>& syst_weights
) {
    std::size_t weight_count = 0;
    if (_systematics) {
        _systematics->compute(buffer, syst_weights);
        _systematics->accumulate(buffer, syst_weights);
        weight_count = _systematics->weight_count();
    } else {
        syst_weights.clear();
    }
    if (_event_histograms) {
        _event_histograms->fill(buffer, syst_weights, weight_count);
    }
}

void EventGenerator::reset_start_time() {
    _start_time = std::chrono::steady_clock::now();
    _start_cpu_microsec = cpu_time_microsec();
}

void EventGenerator::add_timing_data(const std::string& key) {
    using namespace std::chrono_literals;
    std::size_t diff = cpu_time_microsec() - _start_cpu_microsec;
    _timing_data[key] = EventGenerator::TimingData{
        .wall_time_sec = static_cast<double>(
            (std::chrono::steady_clock::now() - _start_time) / 1.0s
        ),
        .cpu_time_sec = diff / 1e6,
    };
}

void EventGenerator::unweight_all() {
    std::size_t call_index = _unweight_call_index++;
    bool done = true;
    double total_eff_count = 0.;
    for (std::size_t channel_index = 0; auto& channel : _channels) {
        MixMaxRandom rand_gen(
            DerivedSeed(_seed, DerivedSeed::unweight_pass, call_index, channel_index)
        );
        channel->unweight_file(rand_gen);
        ++channel_index;

        std::size_t chan_target = channel->status().count_target;
        if (channel->status().count_unweighted < chan_target) {
            total_eff_count += channel->status().count_unweighted;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status.count_unweighted = total_eff_count;
    _status.done = done;
}

std::unordered_set<std::string> EventGenerator::used_globals() const {
    std::unordered_set<std::string> ret;
    for (auto& channel : _channels) {
        ret.insert(channel->used_globals().begin(), channel->used_globals().end());
    }
    return ret;
}

std::vector<GeneratorStatus> EventGenerator::channel_status() const {
    std::vector<GeneratorStatus> status;
    for (auto& channel : _channels) {
        // double target_count = channel.integral_fraction * _config.target_count;
        status.push_back(channel->status());
    }
    return status;
}

std::vector<Histogram> EventGenerator::histograms() const {
    std::vector<Histogram> hists = _channels.at(0)->histograms();
    for (auto& hist : hists) {
        std::fill(hist.bin_values.begin(), hist.bin_values.end(), 0.);
        std::fill(hist.bin_errors.begin(), hist.bin_errors.end(), 0.);
    }
    for (auto& channel : _channels) {
        for (auto [chan_hist, out_hist] : zip(channel->histograms(), hists)) {
            for (auto [chan_w, chan_w2, val, err] :
                 zip(chan_hist.bin_values,
                     chan_hist.bin_errors,
                     out_hist.bin_values,
                     out_hist.bin_errors)) {
                auto n = channel->status().count_opt;
                val += chan_w / n;
                err += (chan_w2 - chan_w * chan_w / n) / (n * n);
            }
        }
    }
    for (auto& hist : hists) {
        for (double& err : hist.bin_errors) {
            err = std::sqrt(err);
        }
    }
    return hists;
}

std::tuple<std::vector<EventGenerator::CombineChannelData>, std::size_t, double>
EventGenerator::init_combine() {
    std::vector<EventGenerator::CombineChannelData> channel_data;
    std::size_t count_sum = 0;
    std::size_t particle_count = 0;
    double weight_sum = 0.;
    for (auto& channel : _channels) {
        particle_count =
            std::max(particle_count, channel->event_file().particle_count());
        // Exact apportioned target (see update_integral_fractions()), so counts
        // sum to exactly _config.target_count.
        std::size_t count = channel->status().count_target;
        count_sum += count;
        channel->event_file().seek(0);
        weight_sum += channel->channel_weight_sum(count);
        channel->weight_file().seek(0);
        channel_data.push_back({
            .cum_count = count_sum,
            .event_buffer = EventBuffer(
                0, channel->event_file().particle_count(), channel->event_file_layout()
            ),
            .weight_buffer = EventBuffer(0, 0, weight_file_layout),
            .buffer_index = 0,
        });
    }
    return {channel_data, particle_count, _status.mean_abs * count_sum / weight_sum};
}

void EventGenerator::read_and_combine(
    std::vector<EventGenerator::CombineChannelData>& channel_data,
    EventBuffer& buffer,
    double norm_factor,
    MixMaxRandom& rand_gen
) {
    std::size_t batch_size = 1000;
    std::size_t event_count = std::min(batch_size, channel_data.back().cum_count);
    buffer.resize(event_count);

    bool has_beam1 = _channels.at(0)->event_layout_extra_flags() & EventRecord::f_beam1;
    bool has_beam2 = _channels.at(0)->event_layout_extra_flags() & EventRecord::f_beam2;
    bool has_partial =
        _channels.at(0)->event_layout_extra_flags() & EventRecord::f_partial_weights;
    bool has_clustering =
        _channels.at(0)->particle_layout_extra_flags() & ParticleRecord::f_clustering;
    int mlm_flags = any_alphas_scales(_channels);
    bool has_alphas_scales = mlm_flags & ParticleRecord::f_alphas_scales;
    bool has_ladder = mlm_flags & ParticleRecord::f_pdf_ladder;
    bool has_subproc_index =
        _channels.at(0)->event_layout_extra_flags() & EventRecord::f_subproc_index;
    bool has_bias = _channels.at(0)->event_layout_extra_flags() & EventRecord::f_bias;
    bool has_matrix_element =
        _channels.at(0)->event_layout_extra_flags() & EventRecord::f_matrix_element;

    for (std::size_t event_index = 0; event_index < event_count; ++event_index) {
        std::size_t random_index = rand_gen.generate_int(channel_data.back().cum_count);
        auto sampled_chan = select_combine_channel(
            channel_data.begin(),
            channel_data.end(),
            random_index,
            [](const CombineChannelData& chan) { return chan.cum_count; }
        );
        std::for_each(sampled_chan, channel_data.end(), [](auto& chan) {
            --chan.cum_count;
        });
        auto& channel = _channels.at(sampled_chan - channel_data.begin());
        bool channel_has_alphas_scales =
            channel->particle_layout_extra_flags() & ParticleRecord::f_alphas_scales;
        bool channel_has_ladder =
            channel->particle_layout_extra_flags() & ParticleRecord::f_pdf_ladder;

        double weight = 0.;
        while (true) {
            if (sampled_chan->buffer_index ==
                sampled_chan->event_buffer.event_count()) {
                channel->event_file().read(sampled_chan->event_buffer, batch_size);
                channel->weight_file().read(sampled_chan->weight_buffer, batch_size);
                sampled_chan->buffer_index = 0;
            }
            weight =
                sampled_chan->weight_buffer.event(sampled_chan->buffer_index).weight();
            if (weight != 0.) {
                break;
            }
            ++sampled_chan->buffer_index;
        }

        auto event_in = sampled_chan->event_buffer.event(sampled_chan->buffer_index);
        auto event_out = buffer.event(event_index);
        event_out.weight() =
            std::copysign(
                std::max(1., std::abs(weight) / channel->max_weight()), weight
            ) *
            norm_factor;
        if (has_bias) {
            // unweighted with the bias b, so the event carries 1 / b
            event_out.weight() = event_out.weight() / event_in.bias().value();
        }
        event_out.subprocess_index() = has_subproc_index
            ? event_in.subprocess_index().value()
            : static_cast<int>(channel->status().subprocess);
        event_out.diagram_index() = event_in.diagram_index();
        event_out.color_index() = event_in.color_index();
        event_out.flavor_index() = event_in.flavor_index();
        event_out.helicity_index() = event_in.helicity_index();
        event_out.ren_scale() = event_in.ren_scale();
        event_out.alpha_qcd() = event_in.alpha_qcd();

        if (has_beam1) {
            event_out.x1() = event_in.x1();
            event_out.fact_scale1() = event_in.fact_scale1();
        }
        if (has_beam2) {
            event_out.x2() = event_in.x2();
            event_out.fact_scale2() = event_in.fact_scale2();
        }
        if (has_partial) {
            event_out.partial_weight_product() = event_in.partial_weight_product();
        }
        if (has_matrix_element) {
            event_out.matrix_element() = event_in.matrix_element();
        }

        std::size_t i = 0;
        for (; i < sampled_chan->event_buffer.particle_count(); ++i) {
            auto particle_in =
                sampled_chan->event_buffer.particle(sampled_chan->buffer_index, i);
            auto particle_out = buffer.particle(event_index, i);
            particle_out.energy() = particle_in.energy();
            particle_out.px() = particle_in.px();
            particle_out.py() = particle_in.py();
            particle_out.pz() = particle_in.pz();
            if (has_clustering) {
                particle_out.cluster_scale() = particle_in.cluster_scale();
            }
            if (has_alphas_scales) {
                // zero from a channel without vertices (2 -> 1)
                particle_out.alphas_scale() = channel_has_alphas_scales
                    ? particle_in.alphas_scale().value()
                    : 0.;
            }
            if (has_ladder) {
                copy_ladder(particle_in, particle_out, channel_has_ladder);
            }
        }
        for (; i < buffer.particle_count(); ++i) {
            auto particle_out = buffer.particle(event_index, i);
            particle_out.energy() = 0.;
            particle_out.px() = 0.;
            particle_out.py() = 0.;
            particle_out.pz() = 0.;
            // the buffer is reused: a padding slot must not keep the scales of
            // an earlier, higher-multiplicity event
            if (has_clustering) {
                particle_out.cluster_scale() = 0.;
            }
            if (has_alphas_scales) {
                particle_out.alphas_scale() = 0.;
            }
            if (has_ladder) {
                copy_ladder(particle_out, particle_out, false);
            }
        }
        ++sampled_chan->buffer_index;
    }
}

void EventGenerator::fill_lhe_event(
    LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    MixMaxRandom& rand_gen
) {
    EventRecord event_in = buffer.event(event_index);
    lhe_event.weight = event_in.weight();
    lhe_event.process_id = 0;
    lhe_event.scale = event_in.ren_scale();
    lhe_event.alpha_qed = 0; // TODO: populate this
    lhe_event.alpha_qcd = event_in.alpha_qcd();
    // Both scales, as LHEF3 <scales> attributes: SCALUP holds only one number
    // and the codes do not agree on which, so write them out explicitly.
    int scale_flags = _channels.at(0)->event_layout_extra_flags();
    lhe_event.mu_r = event_in.ren_scale();
    lhe_event.mu_f1 = (scale_flags & EventRecord::f_beam1)
        ? double(event_in.fact_scale1())
        : double(event_in.ren_scale());
    lhe_event.mu_f2 = (scale_flags & EventRecord::f_beam2)
        ? double(event_in.fact_scale2())
        : lhe_event.mu_f1;
    lhe_event.has_scales = true;
    lhe_event.particles.clear();
    for (std::size_t i = 0; i < buffer.particle_count(); ++i) {
        auto particle_in = buffer.particle(event_index, i);
        if (particle_in.energy() == 0.) {
            break;
        }
        lhe_event.particles.push_back(
            LHEParticle{
                .px = particle_in.px(),
                .py = particle_in.py(),
                .pz = particle_in.pz(),
                .energy = particle_in.energy(),
            }
        );
    }
    bool has_clustering =
        _channels.at(0)->particle_layout_extra_flags() & ParticleRecord::f_clustering;
    lhe_event.has_cluster_scales = has_clustering;
    if (has_clustering) {
        for (std::size_t i = 0; i < lhe_event.particles.size(); ++i) {
            lhe_event.particles.at(i).cluster_scale =
                buffer.particle(event_index, i).cluster_scale();
        }
    }
    lhe_completer.complete_event_data(
        lhe_event,
        event_in.subprocess_index(),
        event_in.diagram_index(),
        event_in.color_index(),
        event_in.flavor_index(),
        event_in.helicity_index(),
        rand_gen
    );
}

void EventGenerator::init_status(const std::string& status) {
    write_status(status, true);
}

void EventGenerator::write_status(const std::string& status, bool force_write) {
    if (!_status_file) {
        return;
    }
    nlohmann::json j{
        {"status", status},
        {"seed", _seed},
        {"process", _status},
        {"channels", channel_status()},
        {"run_times", _timing_data},
        {"histograms", histograms()},
    };
    if (_systematics) {
        j["systematics"] = _systematics->summary();
    }
    if (_event_histograms) {
        j["event_histograms"] = _event_histograms->to_json(_systematics);
    }
    _status_file->write(j, force_write);
}

void EventGenerator::print_survey_init() {
    init_status("survey");
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != Verbosity::pretty) {
        Logger::info("survey started");
        return;
    }
    _pretty_box_upper = PrettyBox("Survey", 4, {18, 0});
    _pretty_box_upper.set_column(
        0, {"Iteration:", "Result:", "Number of events:", "Run time:"}
    );
    _pretty_box_upper.print_first();
}

void EventGenerator::print_survey_update(
    bool done,
    std::size_t done_event_count,
    std::size_t total_event_count,
    std::size_t iter
) {
    if (done) {
        add_timing_data("survey");
    }
    write_status("survey", done);
    if (_config.verbosity == Verbosity::pretty) {
        print_survey_update_pretty(done, done_event_count, total_event_count, iter);
    } else if (_config.verbosity == Verbosity::log) {
        print_survey_update_log(done, done_event_count, total_event_count, iter);
    }
}

void EventGenerator::print_survey_update_pretty(
    bool done,
    std::size_t done_event_count,
    std::size_t total_event_count,
    std::size_t iter
) {
    std::string int_str = format_with_error(_status.mean, _status.error);
    std::string count_str = std::format(
        "{} before cuts, {} after",
        format_si_prefix(_status.count),
        format_si_prefix(_status.count_after_cuts)
    );
    if (done) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("survey");
        _pretty_box_upper.set_column(
            1,
            {std::format("{}", iter + 1),
             int_str,
             count_str,
             format_run_time(wall_time_sec, cpu_time_sec)}
        );
    } else {
        auto now = std::chrono::steady_clock::now();
        using namespace std::chrono_literals;
        if (now - _last_print_time < 0.1s) {
            return;
        }
        _last_print_time = now;

        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:<15} {}",
                 iter + 1,
                 format_progress(
                     static_cast<double>(done_event_count) / total_event_count, 52
                 )
             ),
             int_str,
             count_str,
             std::format(
                 "{:%H:%M:%S}",
                 std::chrono::round<std::chrono::seconds>(now - _start_time)
             )}
        );
    }
    _pretty_box_upper.print_update();
}

void EventGenerator::print_survey_update_log(
    bool done,
    std::size_t done_event_count,
    std::size_t total_event_count,
    std::size_t iter
) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && !done) {
        return;
    }
    _last_print_time = now;

    Logger::info(
        std::format(
            "survey, iter: {}, integral: {}, samps: {}, samps. after cuts: {}, "
            "time: {:%H:%M:%S}",
            iter + 1,
            format_with_error(_status.mean, _status.error),
            format_si_prefix(_status.count),
            format_si_prefix(_status.count_after_cuts),
            std::chrono::round<std::chrono::seconds>(now - _start_time)
        )
    );

    if (done) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("survey");
        Logger::info(
            std::format("survey done, {}", format_run_time(wall_time_sec, cpu_time_sec))
        );
    }
}

std::pair<std::size_t, std::size_t> EventGenerator::optimized_channel_count() const {
    std::size_t optimizable = 0, optimized = 0;
    for (auto& channel : _channels) {
        // a channel without an optimizer is never marked optimized
        if (channel->status().optimized) {
            ++optimizable;
            ++optimized;
        } else if (channel->needs_optimization()) {
            ++optimizable;
        }
    }
    return {optimized, optimizable};
}

void EventGenerator::print_gen_init() {
    init_status("generate");
    _last_print_time = std::chrono::steady_clock::now();
    bool optimize_only = _generate_mode == GenerateMode::optimize_only;
    if (_config.verbosity != Verbosity::pretty) {
        Logger::info(optimize_only ? "optimizing started" : "generating started");
        return;
    }

    std::size_t offset = 0;
    if (_channels.size() > 1) {
        _pretty_box_lower = PrettyBox(
            "Individual channels",
            _channels.size() < 21 ? _channels.size() + 1 : 22,
            optimize_only ? std::vector<std::size_t>{6, 16, 9, 7, 6, 0}
                          : std::vector<std::size_t>{6, 16, 9, 9, 7, 6, 0}
        );
        _pretty_box_lower.set_row(
            0,
            optimize_only
                ? std::vector<
                      std::string>{"#", "integral ↓", "RSD", "N", "iter", "patience"}
                : std::vector<std::string>{
                      "#", "integral ↓", "RSD", "uweff", "N", "opt", "unweighted"
                  }
        );
        if (_channels.size() > 20) {
            _pretty_box_lower.set_cell(21, 0, "..");
        }
        offset = _pretty_box_lower.line_count();
    }
    if (optimize_only) {
        _pretty_box_upper =
            PrettyBox("Integration and optimization", 6, {19, 0}, offset);
        _pretty_box_upper.set_column(
            0,
            {"Result:",
             "Rel. error:",
             "Rel. stddev:",
             "Number of events:",
             "Optimized channels:",
             "Run time:"}
        );
    } else {
        _pretty_box_upper = PrettyBox(
            _generate_mode == GenerateMode::fix_max_weights
                ? "Integration and max. weight estimation"
                : "Integration and unweighting",
            7,
            {19, 0},
            offset
        );
        _pretty_box_upper.set_column(
            0,
            {"Result:",
             "Rel. error:",
             "Rel. stddev:",
             "Number of events:",
             "Unweighting eff.:",
             "Unweighted events:",
             "Run time:"}
        );
    }
    _pretty_box_upper.print_first();
    if (_channels.size() > 1) {
        _pretty_box_lower.print_first();
    }
}

void EventGenerator::print_gen_update(bool done) {
    if (done) {
        add_timing_data("generate");
    }
    write_status("generate", done);
    if (_config.verbosity == Verbosity::pretty) {
        print_gen_update_pretty(done);
    } else if (_config.verbosity == Verbosity::log) {
        print_gen_update_log(done);
    }
}

void EventGenerator::print_gen_update_pretty(bool done) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 0.1s && !done) {
        return;
    }
    _last_print_time = now;
    bool optimize_only = _generate_mode == GenerateMode::optimize_only;

    std::string int_str, rel_str, rsd_str, uweff_str, count_str;
    count_str = std::format(
        "{} before cuts, {} after",
        format_si_prefix(_status.count),
        format_si_prefix(_status.count_after_cuts)
    );

    if (!std::isnan(_status.error)) {
        int_str = format_with_error(_status.mean, _status.error);
        if (std::abs(_status.error) < std::abs(_status.mean)) {
            rel_str =
                std::format("{:.4f} %", _status.error / std::abs(_status.mean) * 100);
        }
        rsd_str = std::format("{:.3f}", _status.rel_std_dev);
        uweff_str = std::format(
            "{:.5f} before cuts, {:.5f} after",
            _status.count_unweighted / _status.count_opt,
            _status.count_unweighted / _status.count_after_cuts_opt
        );
    }
    // the progress of the run: unweighted events, or optimized channels
    std::string progress_str;
    double progress_fraction;
    if (optimize_only) {
        auto [optimized, optimizable] = optimized_channel_count();
        progress_str = std::format("{} / {}", optimized, optimizable);
        progress_fraction =
            optimizable > 0 ? static_cast<double>(optimized) / optimizable : 1.;
    } else {
        progress_str = std::format(
            "{} / {}",
            format_si_prefix(_status.count_unweighted),
            format_si_prefix(_status.count_target)
        );
        progress_fraction = _status.count_unweighted / _status.count_target;
    }
    std::string time_str;
    if (done) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("generate");
        time_str = format_run_time(wall_time_sec, cpu_time_sec);
    } else {
        progress_str = std::format(
            "{:<15} {}", progress_str, format_progress(progress_fraction, 52)
        );
        time_str = std::format(
            "{:%H:%M:%S}", std::chrono::round<std::chrono::seconds>(now - _start_time)
        );
    }
    if (optimize_only) {
        _pretty_box_upper.set_column(
            1, {int_str, rel_str, rsd_str, count_str, progress_str, time_str}
        );
    } else {
        _pretty_box_upper.set_column(
            1, {int_str, rel_str, rsd_str, count_str, uweff_str, progress_str, time_str}
        );
    }
    _pretty_box_upper.print_update();

    if (_channels.size() > 1) {
        auto channels = channel_status();
        std::sort(channels.begin(), channels.end(), [](auto& chan1, auto& chan2) {
            return chan1.mean_abs > chan2.mean_abs;
        });

        for (std::size_t row = 1; auto& channel : channels | std::views::take(20)) {
            std::string index_str = std::format("{}", channel.name);
            std::string int_str, rsd_str, count_str, unw_str, opt_str, patience_str;
            if (!std::isnan(channel.error)) {
                int_str = format_with_error(channel.mean, channel.error);
                rsd_str = std::format("{:.3f}", channel.rel_std_dev);
                uweff_str = std::format(
                    "{:.5f}", channel.count_unweighted / channel.count_after_cuts_opt
                );
                count_str = format_si_prefix(channel.count_after_cuts);
                opt_str = std::format(
                    "{} {}",
                    channel.iterations,
                    channel.optimized || channel.done ? "✓" : ""
                );
                std::string unw_count_str = std::format(
                    "{:>5} / {:>5}",
                    format_si_prefix(channel.count_unweighted),
                    format_si_prefix(channel.count_target)
                );
                std::string progress;
                if (!_status.done) {
                    progress = format_progress(
                        channel.count_unweighted / channel.count_target, 19
                    );
                }
                unw_str = std::format("{:<14} {:<19}", unw_count_str, progress);
                std::size_t patience = _config.optimization_patience;
                // an optimized channel stays so, even if its counter was reset later
                std::size_t waited =
                    channel.optimized ? patience : channel.iters_without_improvement;
                patience_str = std::format(
                    "{:>3} / {:<3} {}",
                    waited,
                    patience,
                    format_progress(
                        static_cast<double>(waited) /
                            std::max<std::size_t>(patience, 1),
                        19
                    )
                );
            }
            if (optimize_only) {
                _pretty_box_lower.set_row(
                    row, {index_str, int_str, rsd_str, count_str, opt_str, patience_str}
                );
            } else {
                _pretty_box_lower.set_row(
                    row,
                    {index_str,
                     int_str,
                     rsd_str,
                     uweff_str,
                     count_str,
                     opt_str,
                     unw_str}
                );
            }
            ++row;
        }
        _pretty_box_lower.print_update();
    }
}

void EventGenerator::print_gen_update_log(bool done) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && !done) {
        return;
    }
    _last_print_time = now;
    bool optimize_only = _generate_mode == GenerateMode::optimize_only;

    std::string rel_str = std::abs(_status.error) < std::abs(_status.mean)
        ? std::format("{:.4f} %", _status.error / std::abs(_status.mean) * 100)
        : "";
    if (optimize_only) {
        auto [optimized, optimizable] = optimized_channel_count();
        Logger::info(
            std::format(
                "optimizing, optimized channels: {} / {}, integral: {}, "
                "rel. error: {}, RSD: {:.3f}, samps: {}, samps. after cuts: {}, "
                "time: {:%H:%M:%S}",
                optimized,
                optimizable,
                format_with_error(_status.mean, _status.error),
                rel_str,
                _status.rel_std_dev,
                format_si_prefix(_status.count),
                format_si_prefix(_status.count_after_cuts),
                std::chrono::round<std::chrono::seconds>(now - _start_time)
            )
        );
    } else {
        Logger::info(
            std::format(
                "generating, events: {} / {}, integral: {}, rel. error: {}, "
                "RSD: {:.3f}, samps: {}, samps. after cuts: {}, "
                "unw. eff.: {:.5f}, unw. eff. after cuts: {:.5f}, time: {:%H:%M:%S}",
                format_si_prefix(_status.count_unweighted),
                format_si_prefix(_status.count_target),
                format_with_error(_status.mean, _status.error),
                rel_str,
                _status.rel_std_dev,
                format_si_prefix(_status.count),
                format_si_prefix(_status.count_after_cuts),
                _status.count_unweighted / _status.count_opt,
                _status.count_unweighted / _status.count_after_cuts_opt,
                std::chrono::round<std::chrono::seconds>(now - _start_time)
            )
        );
    }

    if (done) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("generate");
        Logger::info(
            std::format(
                "{} done, {}",
                optimize_only ? "optimizing" : "generating",
                format_run_time(wall_time_sec, cpu_time_sec)
            )
        );
    }
}

void EventGenerator::print_combine_init() {
    init_status("combine");
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != Verbosity::pretty) {
        Logger::info("combining started");
        return;
    }
    _pretty_box_upper = PrettyBox("Writing final output", 2, {10, 0});
    _pretty_box_upper.set_column(0, {"Events:", "Run time:"});
    _pretty_box_upper.print_first();
}

void EventGenerator::print_combine_update(std::size_t count) {
    if (count == _config.target_count) {
        add_timing_data("combine");
        write_status("done", true);
    } else {
        write_status("combine", false);
    }
    if (_config.verbosity == Verbosity::pretty) {
        print_combine_update_pretty(count);
    } else if (_config.verbosity == Verbosity::log) {
        print_combine_update_log(count);
    }
}

void EventGenerator::print_combine_update_pretty(std::size_t count) {
    if (count == _config.target_count) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("combine");
        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:>5} / {:>5}",
                 format_si_prefix(count),
                 format_si_prefix(_config.target_count)
             ),
             format_run_time(wall_time_sec, cpu_time_sec)}
        );
    } else {
        auto now = std::chrono::steady_clock::now();
        using namespace std::chrono_literals;
        if (now - _last_print_time < 0.1s) {
            return;
        }
        _last_print_time = now;

        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:>5} / {:>5}   {}",
                 format_si_prefix(count),
                 format_si_prefix(_config.target_count),
                 format_progress(static_cast<double>(count) / _config.target_count, 60)
             ),
             std::format(
                 "{:%H:%M:%S}",
                 std::chrono::round<std::chrono::seconds>(now - _start_time)
             )}
        );
    }
    _pretty_box_upper.print_update();
}

void EventGenerator::print_combine_update_log(std::size_t count) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && count != _config.target_count) {
        return;
    }
    _last_print_time = now;

    Logger::info(
        std::format(
            "combining, events: {} / {}, time: {:%H:%M:%S}",
            format_si_prefix(count),
            format_si_prefix(_config.target_count),
            std::chrono::round<std::chrono::seconds>(now - _start_time)
        )
    );

    if (count == _config.target_count) {
        auto [wall_time_sec, cpu_time_sec] = _timing_data.at("combine");
        Logger::info(
            std::format(
                "combining done, {}", format_run_time(wall_time_sec, cpu_time_sec)
            )
        );
    }
}

void madspace::to_json(
    nlohmann::json& j, const EventGenerator::TimingData& timing_data
) {
    j = nlohmann::json{
        {"wall_time_sec", timing_data.wall_time_sec},
        {"cpu_time_sec", timing_data.cpu_time_sec},
    };
}
