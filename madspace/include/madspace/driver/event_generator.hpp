#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <set>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/discrete_optimizer.hpp"
#include "madspace/driver/event_histograms.hpp"
#include "madspace/driver/format.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"
#include "madspace/driver/status_file.hpp"
#include "madspace/driver/systematics.hpp"
#include "madspace/driver/vegas_optimizer.hpp"
#include "madspace/driver/weighting_pipeline.hpp"
#include "madspace/phasespace.hpp"

namespace madspace {

/**
 * Drives event generation across every channel of a subprocess.
 *
 * Coordinates a group of @ref ChannelEventGenerator instances that share the
 * same list of @ref Context devices: runs their survey, dispatches and
 * collects generation batches so multiple channels and devices stay busy
 * concurrently, and combines the resulting per-channel weighted events into
 * unweighted output (@ref combine_to_compact_npy, @ref combine_to_lhe_npy,
 * @ref combine_to_lhe).
 */
class EventGenerator {
public:
    /// Default-constructed @ref GeneratorConfig.
    static const GeneratorConfig default_config;
    /// Install `func`, called periodically during @ref generate to check for
    /// a requested abort.
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    /**
     * @param contexts    One context per device to run on, shared by every
     *                    channel.
     * @param channels    The channels to generate, e.g. one per @ref
     *                    Topology channel of the subprocess.
     * @param seed        Top-level run seed every channel's RNG streams are
     *                    derived from.
     * @param status_file Optional status file progress is periodically
     *                    written to.
     * @param config      Generator configuration.
     * @param groups      Groups of channels with re- and unweighting steps
     *                    after their first unweighting, see @ref WeightingStep.
     *                    The integrals and event counts are then those after
     *                    the last step.
     */
    EventGenerator(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
        std::uint64_t seed,
        std::shared_ptr<StatusFile> status_file = nullptr,
        const GeneratorConfig& config = default_config,
        const std::vector<WeightingGroup>& groups = {}
    );
    EventGenerator(EventGenerator&&) = default;
    EventGenerator& operator=(EventGenerator&&) = default;
    EventGenerator(const EventGenerator&) = delete;
    EventGenerator& operator=(const EventGenerator&) = delete;
    /// Run the survey/optimization phase of every channel. @p survey_pass
    /// salts job seeds so repeated calls on the same channels (e.g.
    /// re-surveying after simplification) don't share a seed stream.
    void survey(std::size_t survey_pass = 0);
    /// Generate unweighted events until every channel reaches its target
    /// count.
    void generate();
    /// Run only the VEGAS and channel-weight optimization of every channel
    /// that still needs it, until all are optimized. No events are kept.
    void optimize();
    /// Like @ref generate, but every channel with a nonzero cross section runs
    /// until it has `freeze_max_weight_after` unweighted events, regardless of
    /// its share of the cross section, so that its maximum weight is final
    /// (see @ref ChannelEventGenerator::max_weight). The events are not meant
    /// to be combined.
    void fix_max_weights();
    /**
     * Combine every channel's generated events into a single unweighted
     * `.npy` file.
     *
     * @param file_name    Output file base name.
     * @param systematics  Optional; adds the scale/PDF variation weights to
     *                     the written events (`.npy` columns `rwgt_<id>`).
     * @param histograms   Optional; filled with the written events and all
     *                     their weights (recorded in `info.json`'s
     *                     `"event_histograms"`).
     */
    void combine_to_compact_npy(
        const std::string& file_name,
        SystematicsCalculator* systematics = nullptr,
        EventHistograms* histograms = nullptr
    );
    /// Like @ref combine_to_compact_npy, additionally completing every event
    /// with `lhe_completer` and writing its LHE-ready particle records to the
    /// `.npy` file.
    void combine_to_lhe_npy(
        const std::string& file_name,
        LHECompleter& lhe_completer,
        SystematicsCalculator* systematics = nullptr,
        EventHistograms* histograms = nullptr
    );
    /// Like @ref combine_to_lhe_npy, writing a complete LHE file with @p meta
    /// as its `<init>` header instead of a `.npy` file.
    void combine_to_lhe(
        const std::string& file_name,
        LHECompleter& lhe_completer,
        const LHEMeta& meta = {},
        SystematicsCalculator* systematics = nullptr,
        EventHistograms* histograms = nullptr
    );
    /// Combined progress snapshot over every channel.
    GeneratorStatus status() const { return _status; }
    /// Per-channel progress snapshots.
    std::vector<GeneratorStatus> channel_status() const;
    /// Combined observable histograms over every channel.
    std::vector<Histogram> histograms() const;
    /// Names of the compute-graph globals used by any channel's integrand.
    std::unordered_set<std::string> used_globals() const;
    /// The channels passed to the constructor.
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels() const {
        return _channels;
    };

private:
    struct CombineChannelData {
        std::size_t cum_count;
        EventBuffer event_buffer;
        EventBuffer weight_buffer;
        std::size_t buffer_index;
    };
    struct TimingData {
        double wall_time_sec;
        double cpu_time_sec;
    };
    inline static std::function<void(void)> _abort_check_function = [] {};

    GeneratorConfig _config;
    std::vector<std::shared_ptr<ChannelEventGenerator>> _channels;
    GeneratorStatus _status;
    std::vector<ContextPtr> _contexts;
    std::unordered_map<std::size_t, GeneratorBatchJob> _running_jobs;
    std::vector<ReadyJob> _ready_jobs;
    std::size_t _job_id;
    std::vector<std::size_t> _channel_job_counts;
    std::vector<bool> _channel_optimizing;
    std::vector<double> _channel_integral_fractions;
    std::vector<std::size_t> _context_job_counts;
    // invalid step values per channel that were already warned about
    std::vector<std::size_t> _warned_invalid_step_values;
    // True while a channel has a steady-state batch dispatched but not yet fully
    // committed; keeps next_batch_event_count() from double-counting in-flight work.
    std::vector<bool> _channel_batch_pending;
    // True once a channel's current generation ReadyJob has had its full event count
    // dispatched (batch_event_count reached zero). Needed alongside
    // channel_job_count == 0 before finish_channel_job() clears _channel_batch_pending
    // -- dispatch now happens incrementally, so channel_job_count can transiently hit
    // zero mid-batch, between one sub-job's commit and the next one being dispatched.
    std::vector<bool> _channel_batch_dispatch_done;
    // Round-robin position into _ready_jobs for generation-batch dispatch, persisted
    // across start_jobs() calls so multiple channels' batches interleave (one device
    // batch at a time) instead of one channel's batch draining before the next is
    // touched.
    std::size_t _ready_job_rr_cursor = 0;
    // Per-channel commit ordering, analogous to _ready_gen/_commit_cursor but ordered
    // per channel instead of globally. _channel_gen_order holds a channel's dispatched
    // job ids in dispatch order and _channel_ready_gen the ones that have completed;
    // the front of the order deque is the next commit due. An explicit deque rather
    // than a "next id" counter because a channel's job ids are not contiguous: dispatch
    // round-robins between channels, so consecutive ids belong to different channels.
    std::vector<std::deque<std::size_t>> _channel_gen_order;
    std::vector<std::set<std::size_t>> _channel_ready_gen;
    // Same, for a job's unweight-stage completion (tracked separately since it's a
    // distinct completion event). Order is appended at generate-commit time, which is
    // also when the unweight stage is queued.
    std::vector<std::deque<std::size_t>> _channel_unweight_order;
    std::vector<std::set<std::size_t>> _channel_unweight_ready;
    // generate() only: per-context queue of job ids awaiting unweight-stage
    // dispatch, drained with priority by start_jobs().
    std::vector<std::vector<std::size_t>> _context_unweight_queue;
    ResultQueue _result_queue;
    std::unique_ptr<WeightingPipeline> _pipeline;

    // Base seed for reproducible event generation.
    std::uint64_t _seed;

    // unweight_all() may run more than once per generate() (a channel's target can
    // grow after it looked done, un-finishing it and triggering another round).
    // Salted by this counter so repeated calls don't replay the same stream.
    std::size_t _unweight_call_index = 0;

    // What a run of the generate loop does; see run_generate().
    enum class GenerateMode { events, optimize_only, fix_max_weights };
    void run_generate(GenerateMode mode);
    // Channels with an optimizer, and how many of them are optimized.
    std::pair<std::size_t, std::size_t> optimized_channel_count() const;
    GenerateMode _generate_mode = GenerateMode::events;

    // Scheduling context for the running survey()/generate() call, read by
    // start_jobs() to derive job seeds.
    bool _survey_job = false;
    std::size_t _survey_pass = 0;

    // Generate completions are committed in ascending job id, with
    // _commit_cursor as the next id due.
    std::set<std::size_t> _ready_gen;
    std::size_t _commit_cursor = 0;

    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::size_t _start_cpu_microsec;
    std::chrono::time_point<std::chrono::steady_clock> _last_print_time;
    PrettyBox _pretty_box_upper;
    PrettyBox _pretty_box_lower;
    std::shared_ptr<StatusFile> _status_file;
    std::unordered_map<std::string, TimingData> _timing_data;
    SystematicsCalculator* _systematics = nullptr;
    EventHistograms* _event_histograms = nullptr;

    void commit_generate_job(GeneratorBatchJob& job);
    void commit_unweight_job(GeneratorBatchJob& job);
    void finish_channel_job(const GeneratorBatchJob& job);
    std::size_t next_batch_event_count(std::size_t channel_index) const;
    std::size_t start_jobs();
    std::size_t wait_for_result(std::size_t& in_flight);
    void reset_jobs();
    void update_integral();
    void update_integral_status();
    // With `final_step`, the integrals after the weighting steps are used. They
    // only change in a fixed order once the steps are drained.
    void update_integral_fractions(bool final_step = true);
    void update_counts();
    void reset_start_time();
    void add_timing_data(const std::string& key);
    void unweight_all();
    std::tuple<std::vector<CombineChannelData>, std::size_t, std::vector<double>>
    init_combine();
    void read_and_combine(
        std::vector<CombineChannelData>& channel_data,
        EventBuffer& buffer,
        const std::vector<double>& norm_factors,
        MixMaxRandom& rand_gen
    );
    // run the re- and unweighting steps of all collected events
    void drain_pipeline();
    // unweighting efficiency of each pass, the first one before and after cuts
    std::tuple<double, double, std::vector<double>> pass_efficiencies() const;
    // start the jobs of the steps whose batches are full, and the partial ones
    // once the generation jobs leave threads idle
    std::size_t start_pipeline_jobs();
    // Layout of the combined events as read from the channel files
    DataLayout combined_layout() const;
    // Layout of the written events: the reweighting inputs are dropped unless
    // requested, and the systematic weight columns are appended
    DataLayout output_layout(int event_flags, std::size_t particle_flags) const;
    void copy_event_fields(
        EventBuffer& in_buffer,
        std::size_t in_index,
        EventBuffer& out_buffer,
        std::size_t out_index,
        const std::vector<double>& syst_weights
    );
    void fill_systematics(
        LHEEvent& lhe_event,
        EventBuffer& buffer,
        std::size_t event_index,
        const std::vector<double>& syst_weights,
        const std::vector<int>& syst_ids
    );
    // systematics weights + event histograms for one combined batch
    void process_combined_batch(EventBuffer& buffer, std::vector<double>& syst_weights);

    void init_status(const std::string& status);
    void write_status(const std::string& status, bool force_write);

    void print_survey_init();
    void print_survey_update(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
        std::size_t iter
    );
    void print_survey_update_pretty(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
        std::size_t iter
    );
    void print_survey_update_log(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
        std::size_t iter
    );

    void warn_invalid_step_values();
    void print_gen_init();
    void print_gen_update(bool done);
    void print_gen_update_pretty(bool done);
    void print_gen_update_log(bool done);

    void print_combine_init();
    void print_combine_update(std::size_t count);
    void print_combine_update_pretty(std::size_t count);
    void print_combine_update_log(std::size_t count);

    friend void
    to_json(nlohmann::json& j, const EventGenerator::TimingData& timing_data);
};

void to_json(nlohmann::json& j, const EventGenerator::TimingData& timing_data);

} // namespace madspace
