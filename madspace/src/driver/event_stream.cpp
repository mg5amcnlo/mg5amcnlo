#include "madspace/driver/event_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "madspace/driver/event_utils.hpp"
#include "madspace/util.hpp"

using namespace madspace;
using namespace std::chrono_literals;

namespace {

// separates the stream's random streams from those of the combine pass
constexpr std::size_t stream_rng_index = 1;

} // namespace

EventStream::EventStream(
    ContextPtr context,
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    std::uint64_t seed,
    const LHECompleter& lhe_completer,
    const GeneratorConfig& config,
    const std::vector<WeightingGroup>& groups
) :
    _context(context),
    _channels(channels),
    _seed(seed),
    _lhe_completer(lhe_completer),
    _config(config),
    _combined_layout([&] {
        if (channels.empty()) {
            throw std::invalid_argument("EventStream needs at least one channel");
        }
        return combined_event_layout(
            channels.at(0)->event_layout_extra_flags(),
            channels.at(0)->particle_layout_extra_flags() | mlm_particle_flags(channels)
        );
    }()),
    _select_rng(DerivedSeed(seed, DerivedSeed::combine_select, 0, 0, stream_rng_index)),
    _lhe_rng(DerivedSeed(seed, DerivedSeed::lhe_complete, 0, 0, stream_rng_index)),
    _channel_states(channels.size()),
    _lookahead(
        context->device()->device_type() == DeviceType::cpu
            ? config.cpu_batch_size
            : config.gpu_batch_size
    ) {
    if (!groups.empty()) {
        _pipeline = std::make_unique<WeightingPipeline>(
            std::vector<ContextPtr>{context},
            channels,
            groups,
            config,
            [this](StepEvents& step_events) { close_step_events(step_events); }
        );
    }
    bool any_active = false;
    for (auto [channel, state] : zip(_channels, _channel_states)) {
        state.active = channel->max_weight_fixed() && channel->step_max_weights_fixed();
        any_active |= state.active;
    }
    if (!any_active) {
        throw std::invalid_argument(
            "EventStream needs channels with a fixed maximum weight"
        );
    }
    _coordinator = std::thread([this] { run_coordinator(); });
}

EventStream::~EventStream() { close(); }

void EventStream::close() {
    {
        std::unique_lock lock(_mutex);
        _stopping = true;
    }
    _coordinator_cv.notify_all();
    _consumer_cv.notify_all();
    if (_coordinator.joinable()) {
        _coordinator.join();
    }
}

void EventStream::run_coordinator() {
    try {
        // dispatches new jobs when the consumer asked for them, stops on close()
        auto poll = [this] {
            std::unique_lock lock(_mutex);
            if (_stopping) {
                throw StopRequest{};
            }
            if (_dispatch_requested) {
                _dispatch_requested = false;
                dispatch_jobs();
            }
        };
        while (true) {
            {
                std::unique_lock lock(_mutex);
                if (_stopping) {
                    break;
                }
                _dispatch_requested = false;
                dispatch_jobs();
                if (_pipeline) {
                    // partial batches only once no events are coming anymore
                    _results_pending +=
                        _pipeline->start_jobs(_result_queue, _running_jobs.empty());
                }
                if (_results_pending == 0) {
                    _coordinator_cv.wait(lock, [this] {
                        return _stopping || _dispatch_requested;
                    });
                    continue;
                }
            }
            auto result = _result_queue.wait(poll, 10ms);
            --_results_pending;
            if (result.exception) {
                std::rethrow_exception(result.exception);
            }
            handle_result(result.id);
        }
    } catch (StopRequest&) {
    } catch (...) {
        std::unique_lock lock(_mutex);
        _error = std::current_exception();
    }
    _consumer_cv.notify_all();
    _result_queue.discard(_results_pending, nullptr);
    _results_pending = 0;
    _running_jobs.clear();
    if (_pipeline) {
        _pipeline->reset();
    }
    _step_jobs.clear();
}

double EventStream::expected_job_events(std::size_t channel_index) const {
    auto& state = _channel_states.at(channel_index);
    if (state.completed_jobs > 0) {
        return std::max<double>(state.completed_events, 1.) / state.completed_jobs;
    }
    // before the first job, assume a single event to stay on the safe side
    return 1.;
}

// Called with _mutex held. Which jobs are started depends on timing, but not
// what they contain: that is fixed by the channel's job sequence number.
void EventStream::dispatch_jobs() {
    std::size_t max_jobs = 2 * _context->thread_pool().thread_count();
    if (_probabilities_dirty) {
        update_probabilities();
    }
    while (_running_jobs.size() < max_jobs) {
        // the channel furthest below its share of the lookahead
        std::size_t best_index = _channels.size();
        double best_ratio = 1.;
        double prev_prob = 0.;
        for (std::size_t i = 0; i < _channels.size(); ++i) {
            double prob = _cum_probabilities.at(i) - prev_prob;
            prev_prob = _cum_probabilities.at(i);
            auto& state = _channel_states.at(i);
            if (!state.active) {
                continue;
            }
            // bounds the memory of channels that rarely yield events
            if (state.in_flight + state.unopened_jobs >= max_jobs) {
                continue;
            }
            double available =
                state.buffered_events + state.in_flight * expected_job_events(i);
            double target = prob * _lookahead;
            double ratio;
            if (state.buffered_events == 0 && state.in_flight == 0 &&
                state.unopened_jobs == 0) {
                // every channel keeps at least one job buffered or running
                ratio = -1.;
            } else if (available < target) {
                ratio = available / target;
            } else {
                continue;
            }
            if (ratio < best_ratio) {
                best_ratio = ratio;
                best_index = i;
            }
        }
        if (best_index == _channels.size()) {
            break;
        }

        std::size_t job_id = _job_id++;
        auto& job =
            _running_jobs
                .emplace(
                    job_id,
                    GeneratorBatchJob{
                        .channel_index = best_index,
                        .unweight = true,
                        .batch_event_count = 0,
                        .split_job_count = 1,
                        .context_index = 0,
                        .job_id = job_id,
                        .is_vegas_batch = false,
                    }
                )
                .first->second;
        auto& state = _channel_states.at(best_index);
        state.launch_order.push_back(job_id);
        ++state.in_flight;
        _channels.at(best_index)->start_job(job, _result_queue, _seed, false, 0);
        ++_results_pending;
    }
}

void EventStream::handle_result(std::size_t job_id) {
    if (WeightingPipeline::owns_job(job_id)) {
        _pipeline->commit_job(job_id);
        return;
    }
    auto& job = _running_jobs.at(job_id);
    auto& channel = _channels.at(job.channel_index);
    if (job.unweighted_events.empty()) {
        // generation stage done, the max weight is fixed so unweight right away
        std::unique_lock lock(_mutex);
        channel->start_unweight_job(job, _result_queue);
        ++_results_pending;
        return;
    }
    if (_pipeline && _pipeline->contains(job.channel_index)) {
        // keep what open_job() needs until the events are through the steps
        auto& step_job = _step_jobs[job_id];
        step_job.channel_index = job.channel_index;
        step_job.weights = job.weights;
        step_job.biases = job.biases;
        step_job.requested_event_count = job.requested_event_count;
        _pipeline->add_events(job);
        _running_jobs.erase(job_id);
        return;
    }

    ClosedJob closed_job{
        .job = {},
        .events =
            EventBuffer(0, channel->particle_count(), channel->event_file_layout()),
        .weights = EventBuffer(0, 0, weight_file_layout),
    };
    channel->fill_event_buffers(
        job.unweighted_events, closed_job.events, closed_job.weights
    );
    // only the weights are needed to integrate the job later
    closed_job.job.channel_index = job.channel_index;
    closed_job.job.weights = job.weights;
    closed_job.job.biases = job.biases;
    closed_job.job.requested_event_count = job.requested_event_count;
    std::size_t channel_index = job.channel_index;
    _running_jobs.erase(job_id);
    close_job(job_id, channel_index, std::move(closed_job));
}

void EventStream::close_step_events(StepEvents& step_events) {
    auto& channel = _channels.at(step_events.channel_index);
    auto node = _step_jobs.extract(step_events.job_id);
    ClosedJob closed_job{
        .job = std::move(node.mapped()),
        .events =
            EventBuffer(0, channel->particle_count(), channel->event_file_layout()),
        .weights =
            EventBuffer(0, channel->weight_record_count(), channel->weight_layout()),
        .step_integral = channel->step_integral(step_events),
    };
    channel->fill_step_buffers(step_events, closed_job.events, closed_job.weights);
    for (std::size_t i = 0; i < closed_job.weights.event_count(); ++i) {
        closed_job.abs_weight_sum +=
            std::abs(channel->unweighted_weight(closed_job.weights, i));
    }
    close_job(step_events.job_id, step_events.channel_index, std::move(closed_job));
}

void EventStream::close_job(
    std::size_t job_id, std::size_t channel_index, ClosedJob&& closed_job
) {
    std::unique_lock lock(_mutex);
    auto& state = _channel_states.at(channel_index);
    ++state.completed_jobs;
    state.completed_events += closed_job.events.event_count();
    state.finished.emplace(job_id, std::move(closed_job));
    // close the jobs in launch order
    bool closed_any = false;
    while (!state.launch_order.empty()) {
        auto node = state.finished.extract(state.launch_order.front());
        if (node.empty()) {
            break;
        }
        state.launch_order.pop_front();
        --state.in_flight;
        state.buffered_events += node.mapped().events.event_count();
        ++state.unopened_jobs;
        state.closed.push_back(std::move(node.mapped()));
        closed_any = true;
    }
    if (closed_any) {
        _consumer_cv.notify_all();
    }
}

// Called with _mutex held.
void EventStream::update_probabilities() {
    _cum_probabilities.resize(_channels.size());
    double total = 0.;
    for (std::size_t i = 0; i < _channels.size(); ++i) {
        auto& state = _channel_states.at(i);
        double prob = 0.;
        if (state.active) {
            // divided by the mean weight of the unweighted events (> 1 with
            // overweight events), so that the weight sums are proportional to
            // the cross sections
            double mean_weight = state.opened_capped_sum > 0
                ? state.opened_abs_sum / state.opened_capped_sum
                : 1.;
            prob = _channels.at(i)->abs_integral_estimate() / mean_weight;
            if (!std::isfinite(prob) || prob < 0) {
                prob = 0.;
            }
        }
        total += prob;
        _cum_probabilities.at(i) = total;
    }
    if (total > 0) {
        for (auto& prob : _cum_probabilities) {
            prob /= total;
        }
    }
    _probabilities_dirty = false;
}

std::size_t EventStream::select_channel() {
    if (_probabilities_dirty) {
        update_probabilities();
    }
    if (_cum_probabilities.back() <= 0) {
        throw std::runtime_error("EventStream: all channels have a zero cross section");
    }
    // the first channel whose cumulative probability exceeds r, so channels
    // with zero probability are never picked
    double r = _select_rng.generate_double() * _cum_probabilities.back();
    auto it = std::upper_bound(_cum_probabilities.begin(), _cum_probabilities.end(), r);
    if (it == _cum_probabilities.end()) {
        it = std::lower_bound(
            _cum_probabilities.begin(),
            _cum_probabilities.end(),
            _cum_probabilities.back()
        );
    }
    return it - _cum_probabilities.begin();
}

// Called with _mutex held.
void EventStream::open_job(std::size_t channel_index, ClosedJob& closed_job) {
    auto& channel = _channels.at(channel_index);
    auto& state = _channel_states.at(channel_index);
    channel->integrate(closed_job.job);
    if (!channel->weighting_steps().empty()) {
        channel->add_step_integral(closed_job.step_integral);
        // the mean weight of the events after the steps
        state.opened_abs_sum += closed_job.abs_weight_sum;
        state.opened_capped_sum += closed_job.events.event_count();
    } else {
        double max_weight = channel->max_weight();
        auto w_view = closed_job.job.weights.view<double, 1>();
        for (std::size_t i = 0; i < w_view.size(); ++i) {
            double w = std::abs(w_view[i]);
            state.opened_abs_sum += w;
            state.opened_capped_sum += std::min(w, max_weight);
        }
    }
    ++state.opened_jobs;
    --state.unopened_jobs;
    closed_job.opened = true;
    _probabilities_dirty = true;
}

void EventStream::wait_for_events(
    std::size_t channel_index, std::unique_lock<std::mutex>& lock
) {
    _dispatch_requested = true;
    _coordinator_cv.notify_all();
    auto& state = _channel_states.at(channel_index);
    while (state.closed.empty()) {
        if (_error) {
            std::rethrow_exception(_error);
        }
        if (_stopping) {
            throw std::runtime_error("EventStream was closed");
        }
        if (!_consumer_cv.wait_for(lock, 100ms, [&] {
                return !state.closed.empty() || _error || _stopping;
            })) {
            lock.unlock();
            _abort_check_function();
            lock.lock();
        }
    }
}

// The first job of the channel's buffer that still has events, opened.
// Called with _mutex held.
EventStream::ClosedJob&
EventStream::next_job(std::size_t channel_index, std::unique_lock<std::mutex>& lock) {
    auto& state = _channel_states.at(channel_index);
    while (true) {
        if (state.closed.empty()) {
            wait_for_events(channel_index, lock);
        }
        auto& front = state.closed.front();
        if (!front.opened) {
            open_job(channel_index, front);
        }
        if (front.next_event < front.events.event_count()) {
            return front;
        }
        state.closed.pop_front();
    }
}

std::vector<LHEEvent> EventStream::next_events(std::size_t count) {
    std::unique_lock consumer_lock(_consumer_mutex);
    EventBuffer buffer(count, max_particle_count(), _combined_layout);
    {
        std::unique_lock lock(_mutex);
        if (_error) {
            std::rethrow_exception(_error);
        }
        if (_stopping) {
            throw std::runtime_error("EventStream was closed");
        }
        _lookahead = std::max(_lookahead, 2 * count);
        if (!_started) {
            // every channel's estimate starts from one job (also those
            // without an integral prior), so the first events are already
            // drawn with all channels known
            for (std::size_t i = 0; i < _channels.size(); ++i) {
                if (_channel_states.at(i).active) {
                    next_job(i, lock);
                }
            }
            _started = true;
        }
        for (std::size_t i = 0; i < count; ++i) {
            std::size_t channel_index = select_channel();
            auto& channel = _channels.at(channel_index);
            auto& state = _channel_states.at(channel_index);
            auto& job = next_job(channel_index, lock);
            std::size_t event_index = job.next_event++;
            --state.buffered_events;
            ++state.returned_events;
            copy_channel_event(
                job.events,
                event_index,
                buffer,
                i,
                channel->unweighted_weight(job.weights, event_index),
                static_cast<int>(channel->status().subprocess),
                channel->event_layout_extra_flags(),
                channel->particle_layout_extra_flags(),
                _channels.at(0)->particle_layout_extra_flags() |
                    mlm_particle_flags(_channels)
            );
            // keep the buffers filled while the batch is assembled
            if (state.buffered_events == 0) {
                _dispatch_requested = true;
                _coordinator_cv.notify_all();
            }
        }
        _event_count += count;
        _dispatch_requested = true;
    }
    _coordinator_cv.notify_all();

    std::vector<LHEEvent> events(count);
    for (std::size_t i = 0; i < count; ++i) {
        fill_lhe_event(
            _lhe_completer,
            events.at(i),
            buffer,
            i,
            _lhe_rng,
            _channels.at(0)->event_layout_extra_flags(),
            _channels.at(0)->particle_layout_extra_flags()
        );
    }
    return events;
}

GeneratorStatus EventStream::status() const {
    std::unique_lock lock(_mutex);
    GeneratorStatus status{};
    // the inactive channels return no events
    std::vector<std::shared_ptr<ChannelEventGenerator>> active_channels;
    for (auto [channel, state] : zip(_channels, _channel_states)) {
        if (state.active) {
            active_channels.push_back(channel);
        }
    }
    sum_channel_status(active_channels, status);
    status.count_unweighted = _event_count;
    status.count_target = 0;
    status.optimized = true;
    status.done = false;
    return status;
}

std::vector<GeneratorStatus> EventStream::channel_status() const {
    std::unique_lock lock(_mutex);
    std::vector<GeneratorStatus> ret;
    for (auto [channel, state] : zip(_channels, _channel_states)) {
        auto status = channel->status();
        status.count_unweighted = state.returned_events;
        status.count_target = 0;
        ret.push_back(status);
    }
    return ret;
}

std::vector<double> EventStream::channel_probabilities() const {
    std::unique_lock lock(_mutex);
    // const: compute from scratch instead of updating the cache
    auto& self = const_cast<EventStream&>(*this);
    if (self._probabilities_dirty) {
        self.update_probabilities();
    }
    std::vector<double> probs;
    double prev = 0.;
    for (double cum_prob : _cum_probabilities) {
        probs.push_back(cum_prob - prev);
        prev = cum_prob;
    }
    return probs;
}

std::size_t EventStream::event_count() const {
    std::unique_lock lock(_mutex);
    return _event_count;
}
