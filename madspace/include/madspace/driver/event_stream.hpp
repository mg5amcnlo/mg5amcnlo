#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"
#include "madspace/driver/random.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "madspace/driver/weighting_pipeline.hpp"

namespace madspace {

/**
 * Endless stream of unweighted events, generated in memory.
 *
 * Every channel needs a fixed maximum weight (see @ref
 * ChannelEventGenerator::set_fixed_max_weight), so that an event is final as
 * soon as it is unweighted; channels without one are skipped. With re- and
 * unweighting steps (see @ref WeightingStep), the maximum weights of the steps
 * have to be fixed as well, and an event is final after the last step. A coordinator
 * thread keeps generation jobs running on the context's thread pool and
 * buffers each channel's unweighted events, in launch order, until they are
 * consumed. @ref next_events picks the channel of every event at random,
 * proportional to its current cross-section estimate, and completes the
 * events with an @ref LHECompleter.
 *
 * The integral estimates only include the jobs whose events were requested
 * so far, so the event sequence is a pure function of the seed: it depends
 * neither on thread timing nor on how the requests are split into batches.
 *
 * Event weights are in units of the maximum weight: +-1, except for
 * overweight events. After reweighting steps, they also contain the ratios of
 * the reweighting steps.
 */
class EventStream {
public:
    /// Install `func`, called periodically while @ref next_events waits for
    /// events, to check for a requested abort.
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    /**
     * Start the coordinator thread and the first generation jobs.
     *
     * @param context        Context to generate on.
     * @param channels       The channels, each created with `context` as its
     *                       only context.
     * @param seed           Run seed every random stream is derived from.
     * @param lhe_completer  Completes the events into LHE events.
     * @param config         Generator configuration; the batch sizes and cut
     *                       settings are used.
     * @param groups         Groups of channels with re- and unweighting steps,
     *                       see @ref EventGenerator.
     */
    EventStream(
        ContextPtr context,
        const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
        std::uint64_t seed,
        const LHECompleter& lhe_completer,
        const GeneratorConfig& config,
        const std::vector<WeightingGroup>& groups = {}
    );
    ~EventStream();
    EventStream(const EventStream&) = delete;
    EventStream& operator=(const EventStream&) = delete;

    /// The next `count` events of the stream. Blocks until they are available.
    std::vector<LHEEvent> next_events(std::size_t count);
    /// Stop the coordinator thread and discard the running jobs. Called by
    /// the destructor.
    void close();
    /// Combined status over the channels with fixed maximum weights, the others
    /// return no events. `count_unweighted` is the number of events returned
    /// so far.
    GeneratorStatus status() const;
    /// Per-channel status; `count_unweighted` is the number of events of the
    /// channel returned so far.
    std::vector<GeneratorStatus> channel_status() const;
    /// Current probability of each channel to be picked for the next event.
    std::vector<double> channel_probabilities() const;
    /// Number of events returned so far.
    std::size_t event_count() const;
    /// Largest particle count of the returned LHE events.
    std::size_t max_particle_count() const {
        return _lhe_completer.max_particle_count();
    }

private:
    // A job whose events are buffered, waiting to be consumed.
    struct ClosedJob {
        GeneratorBatchJob job;
        EventBuffer events;
        EventBuffer weights;
        bool opened = false;
        std::size_t next_event = 0;
        // with steps: the integral after the steps and the sum of the
        // absolute event weights
        StepIntegral step_integral;
        double abs_weight_sum = 0.;
    };
    struct ChannelState {
        bool active = false;
        // job ids in launch order, waiting to be closed
        std::deque<std::size_t> launch_order;
        std::map<std::size_t, ClosedJob> finished;
        std::deque<ClosedJob> closed;
        std::size_t in_flight = 0;
        std::size_t buffered_events = 0;
        std::size_t unopened_jobs = 0;
        // statistics of the completed and of the opened jobs
        std::size_t completed_jobs = 0;
        std::size_t completed_events = 0;
        std::size_t opened_jobs = 0;
        double opened_abs_sum = 0.;
        double opened_capped_sum = 0.;
        std::size_t returned_events = 0;
    };
    // thrown inside the coordinator to stop it
    struct StopRequest {};

    inline static std::function<void(void)> _abort_check_function = [] {};

    ContextPtr _context;
    std::vector<std::shared_ptr<ChannelEventGenerator>> _channels;
    std::uint64_t _seed;
    LHECompleter _lhe_completer;
    GeneratorConfig _config;
    DataLayout _combined_layout;
    MixMaxRandom _select_rng;
    MixMaxRandom _lhe_rng;

    // guards everything below, shared by the consumer and the coordinator
    mutable std::mutex _mutex;
    std::condition_variable _coordinator_cv;
    std::condition_variable _consumer_cv;
    std::vector<ChannelState> _channel_states;
    std::vector<double> _cum_probabilities;
    bool _probabilities_dirty = true;
    bool _dispatch_requested = false;
    bool _stopping = false;
    std::exception_ptr _error;
    std::size_t _lookahead;
    std::size_t _event_count = 0;
    bool _started = false;

    // only used by the consumer, which next_events() serializes
    std::mutex _consumer_mutex;
    // only used by the coordinator thread
    std::unordered_map<std::size_t, GeneratorBatchJob> _running_jobs;
    std::size_t _results_pending = 0;
    std::size_t _job_id = 0;
    ResultQueue _result_queue;
    std::unique_ptr<WeightingPipeline> _pipeline;
    // jobs whose events are in the steps
    std::unordered_map<std::size_t, GeneratorBatchJob> _step_jobs;
    std::thread _coordinator;

    void run_coordinator();
    void dispatch_jobs();
    void handle_result(std::size_t job_id);
    void
    close_job(std::size_t job_id, std::size_t channel_index, ClosedJob&& closed_job);
    void close_step_events(StepEvents& step_events);
    double expected_job_events(std::size_t channel_index) const;
    void update_probabilities();
    std::size_t select_channel();
    ClosedJob& next_job(std::size_t channel_index, std::unique_lock<std::mutex>& lock);
    void open_job(std::size_t channel_index, ClosedJob& closed_job);
    void wait_for_events(std::size_t channel_index, std::unique_lock<std::mutex>& lock);
};

} // namespace madspace
