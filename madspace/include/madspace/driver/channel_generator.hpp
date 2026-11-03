#pragma once

#include <array>
#include <optional>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/generator_data.hpp"

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/discrete_optimizer.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/random.hpp"
#include "madspace/driver/vegas_optimizer.hpp"
#include "madspace/phasespace.hpp"

namespace madspace {

/// Events of one job that passed all re- and unweighting steps, see
/// @ref WeightingPipeline.
struct StepEvents {
    /// Index of the channel.
    std::size_t channel_index;
    /// Id of the generation job of the events.
    std::size_t job_id;
    /// The events on the CPU, as returned by the unweighter.
    TensorVec events;
    /// The weights of the first step.
    std::vector<double> weights;
    /// The maximum weight of the first step they were unweighted with.
    double max_weight;
    /// Per step, the ratios of the matrix elements.
    std::vector<std::vector<double>> ratios;
    /// Per step, the maximum weight they were unweighted with.
    std::vector<double> max_weights;
    /// The biases of the events, empty without.
    std::vector<double> biases;
    /// Number of points generated for these events.
    std::size_t generated_count;
};

/// Sums of the final weights over the generated points.
struct StepIntegral {
    std::size_t count = 0;
    double sum = 0., sum2 = 0., abs_sum = 0., abs_sum2 = 0.;
};

/**
 * Event generation and integration for one integration channel.
 *
 * Owns one channel's compiled integrand, unweighter and (optional)
 * histogram runtimes, one set per @ref Context passed to the constructor, and
 * runs its survey, VEGAS/channel-weight optimization and unweighting.
 * @ref EventGenerator coordinates one `ChannelEventGenerator` per channel of a
 * subprocess.
 */
class ChannelEventGenerator {
public:
    /// Reconstruct a generator previously written with @ref save, reopening
    /// `event_file` and `weight_file`.
    static ChannelEventGenerator load(
        const std::string& channel_file,
        const std::vector<ContextPtr>& contexts,
        const std::string& event_file,
        const std::string& weight_file,
        const GeneratorConfig& config
    );
    /// Like @ref load, from the JSON text produced by @ref to_json instead of
    /// a file.
    static ChannelEventGenerator load_json(
        const std::string& channel_json,
        const std::vector<ContextPtr>& contexts,
        const std::string& event_file,
        const std::string& weight_file,
        const GeneratorConfig& config
    );

    /**
     * @param contexts          One context per device to run on.
     * @param integrand         The channel's compiled integrand.
     * @param event_file        Path of the weighted event file; empty for no
     *                          file (events are only kept in memory).
     * @param weight_file       Path of the max-weight tracking file; empty
     *                          for no file.
     * @param config            Generator configuration.
     * @param subprocess_index  Index of the owning subprocess.
     * @param name              Human-readable channel name.
     * @param histograms        Optional observable histograms filled
     *                          alongside integration.
     */
    ChannelEventGenerator(
        const std::vector<ContextPtr>& contexts,
        const Integrand& integrand,
        const std::string& event_file,
        const std::string& weight_file,
        const GeneratorConfig& config,
        std::size_t subprocess_index,
        const std::string& name,
        const std::optional<ObservableHistograms>& histograms
    );
    ChannelEventGenerator(ChannelEventGenerator&&) = default;
    ChannelEventGenerator& operator=(ChannelEventGenerator&&) = default;
    ChannelEventGenerator(const ChannelEventGenerator&) = delete;
    ChannelEventGenerator& operator=(const ChannelEventGenerator&) = delete;

    /// Current progress snapshot.
    const GeneratorStatus& status() const { return _status; }
    /// Running cross-section estimate.
    const RunningIntegral& cross_section() const { return _cross_section; }
    /// Running estimate of the mean absolute weight.
    const RunningIntegral& abs_cross_section() const { return _abs_cross_section; }
    /// Weighted events requested since the last reset of `count_opt`, counting
    /// each job at its nominal batch size rather than its raw point count.
    std::size_t count_requested_opt() const { return _count_requested_opt; }
    /// The filled observable histograms, if any were requested.
    const std::vector<Histogram>& histograms() const { return _histograms; }
    /// The weighted event file. Throws if the channel was created without one.
    EventFile& event_file();
    /// The max-weight tracking file. Throws if the channel was created without
    /// one.
    EventFile& weight_file();
    /// Number of particles per event.
    std::size_t particle_count() const { return _particle_count; }
    /// Current maximum-weight estimate, used for unweighting.
    double max_weight() const { return _max_weight; }
    /// Whether the maximum weight was fixed with @ref set_fixed_max_weight.
    bool max_weight_fixed() const { return _max_weight_fixed; }
    /// Use `max_weight` for unweighting from now on, instead of estimating it
    /// from the generated events. Must be positive.
    void set_fixed_max_weight(double max_weight);
    /// Set the integral of the absolute weights known from an earlier run, with
    /// its standard error. Both must be positive.
    void set_abs_integral_prior(double mean, double error);
    /// Whether @ref set_abs_integral_prior was called.
    bool has_abs_integral_prior() const { return _abs_integral_prior.has_value(); }
    /// Set the signed integral known from an earlier run, with its standard error.
    void set_integral_prior(double mean, double error);
    /// Whether this run has too few samples for an integral estimate of its own, so
    /// that @ref EventGenerator uses the priors instead.
    bool uses_integral_prior(bool final_step = true) const;
    /// The prior of the signed integral as (mean, standard error), or of the
    /// absolute one if no signed prior was set.
    std::pair<double, double> integral_prior() const;
    /// The prior of the integral of the absolute weights as (mean, standard error).
    std::pair<double, double> abs_integral_prior() const {
        return *_abs_integral_prior;
    }
    /// The integral of the absolute weights: the mean of this run, combined
    /// with the prior (if any) weighted by the inverse variances. Used to share
    /// the events between channels. With `final_step`, the integral after the
    /// weighting steps, see @ref result_integral, otherwise of the first step.
    double abs_integral_estimate(bool final_step = true) const;
    /// Current generation batch size.
    std::size_t batch_size() const { return _batch_size; }
    /// Whether a VEGAS grid or channel-weight optimization pass is still
    /// pending.
    bool needs_optimization() const {
        return (_vegas_optimizer || _discrete_optimizer) && !_status.optimized;
    }
    /// Set the target unweighted event count; a smaller target tightens the
    /// maximum-weight estimate to the smaller truncation budget.
    void set_target_count(std::size_t target_count);
    /// Names of the compute-graph globals this channel's integrand reads.
    const std::unordered_set<std::string>& used_globals() const {
        return _used_globals;
    }
    /// Extra `EventRecord` layout flags this channel's events carry.
    int event_layout_extra_flags() const { return _event_layout_extra_flags; }
    /// Extra `ParticleRecord` layout flags this channel's events carry.
    int particle_layout_extra_flags() const { return _particle_layout_extra_flags; }
    /// Layout of @ref event_file.
    const DataLayout& event_file_layout() const { return _event_file_layout; }
    /// Layout of @ref weight_file.
    const DataLayout& weight_layout() const {
        return _step_settings.empty() ? weight_file_layout : step_weight_file_layout;
    }
    /// Records per event in @ref weight_file, none without steps.
    std::size_t weight_record_count() const {
        return _step_settings.empty() ? 0 : _step_settings.size() + 1;
    }
    /// Types of the unweighted events, the integrand outputs before `random`.
    const NamedVector<Type>& event_types() const { return _event_types; }
    /// Whether the matrix element of the integrand returns the color, helicity
    /// and diagram index.
    const std::array<bool, 3>& provided_indices() const { return _provided_indices; }

    /// Integral of the final weights: after the last re- or unweighting step,
    /// or the weights of the integrand without steps.
    struct IntegralEstimate {
        double mean;
        double variance;
        double mean_abs;
        double variance_abs;
        std::size_t count;
    };
    /// See @ref IntegralEstimate. With steps, but no events through all of
    /// them yet, the integral of the first step.
    IntegralEstimate result_integral() const;
    /// The integral of the first step, see @ref IntegralEstimate.
    IntegralEstimate first_step_integral() const;

    /// Set the re- and unweighting steps after the unweighting of this channel,
    /// see @ref WeightingStep. Their events are generated by the
    /// @ref EventGenerator, before any events of this channel.
    void set_weighting_steps(const std::vector<StepSettings>& steps);
    /// The re- and unweighting steps.
    const std::vector<StepSettings>& weighting_steps() const { return _step_settings; }
    /// Replace @ref GeneratorConfig::max_overweight_truncation and
    /// @ref GeneratorConfig::freeze_max_weight_after of the first unweighting,
    /// where set.
    void set_first_pass_settings(
        std::optional<double> max_overweight_truncation,
        std::optional<std::size_t> freeze_max_weight_after
    );
    /// The truncation of the first unweighting, see
    /// @ref GeneratorConfig::max_overweight_truncation.
    double max_overweight_truncation() const {
        return _config.max_overweight_truncation;
    }
    /// The freeze count of the first unweighting, see
    /// @ref GeneratorConfig::freeze_max_weight_after.
    std::size_t freeze_max_weight_after() const {
        return _config.freeze_max_weight_after;
    }
    /// The freeze count set with @ref set_first_pass_settings, if any.
    std::optional<std::size_t> first_pass_freeze_max_weight_after() const {
        return _first_pass_freeze_max_weight_after;
    }
    /// Maximum weight of step `step`, counted from 1.
    double step_max_weight(std::size_t step) const {
        return _step_states.at(step - 1).max_weight;
    }
    /// Whether the maximum weights of all unweighting steps are fixed.
    bool step_max_weights_fixed() const {
        for (auto [state, settings] : zip(_step_states, _step_settings)) {
            if (settings.unweight && !state.max_weight_fixed) {
                return false;
            }
        }
        return true;
    }
    /// Fix the maximum weight of step `step`, like @ref set_fixed_max_weight.
    void set_fixed_step_max_weight(std::size_t step, double max_weight);
    /// Update the maximum weight of step `step` from the values it unweights,
    /// with the truncation budget for `count_target` events.
    void update_step_max_weight(
        std::size_t step, const std::vector<double>& values, std::size_t count_target
    );
    /// Count `count` events given the weight 0 by a step, see
    /// @ref GeneratorStatus::count_invalid_step_values.
    void add_invalid_step_values(std::size_t count) {
        _status.count_invalid_step_values += count;
    }
    /// Apply the truncation budget of the current target count to the maxima
    /// of the steps, which are updated with the target count of each job. Like
    /// @ref set_target_count for the first maximum.
    void apply_step_truncation_budgets();
    /// Count `count` events accepted by step `step`, which passed the earlier
    /// unweightings with the maxima `max_weight` (first step) and
    /// `max_weights` (the steps before).
    void add_step_accepted(
        std::size_t step,
        double count,
        double max_weight,
        const std::vector<double>& max_weights
    );
    /// Estimated number of events accepted by the first unweighting and each
    /// step so far; without steps, just @ref GeneratorStatus::count_unweighted.
    std::vector<double> step_accepted_counts() const {
        if (_step_settings.empty()) {
            return {_status.count_unweighted};
        }
        std::vector<double> counts{_first_step_count};
        for (auto& state : _step_states) {
            counts.push_back(state.count_accepted);
        }
        return counts;
    }
    /// Count `count` events accepted by the first unweighting with
    /// `job_max_weight`, with steps.
    void add_first_step_events(std::size_t count, double job_max_weight);
    /// Incremented whenever the events are cleared, so that events still in the
    /// steps can be discarded.
    std::size_t event_epoch() const { return _event_epoch; }
    /// Write events that passed all steps and add them to the integral.
    void write_step_events(const StepEvents& step_events);
    /// The contribution of `step_events` to the integral of the final weights.
    StepIntegral step_integral(const StepEvents& step_events) const;
    /// Add to the integral of the final weights.
    void add_step_integral(const StepIntegral& integral);
    /// Pack `step_events` into an event buffer and a weight buffer (layouts
    /// @ref event_file_layout and @ref weight_layout).
    void fill_step_buffers(
        const StepEvents& step_events,
        EventBuffer& event_buffer,
        EventBuffer& weight_buffer
    ) const;
    /// Signed weight of event `index` of a buffer read from @ref weight_file,
    /// in units of the maximum weights, 0 for a rejected event.
    double unweighted_weight(EventBuffer& weight_buffer, std::size_t index) const;

    /// Run a final unweighting pass over @ref weight_file into @ref
    /// event_file, using `rand_gen`.
    void unweight_file(MixMaxRandom& rand_gen);
    /// Integrate `job`'s events, accumulating @ref cross_section.
    void integrate(const GeneratorBatchJob& job);
    /// Accumulate `job`'s events into the VEGAS grid and channel-weight
    /// optimizers.
    void optimize_vegas(const GeneratorBatchJob& job);
    /// Sum of this channel's weight over `event_count` freshly sampled events,
    /// used for multi-channel weight optimization.
    double channel_weight_sum(std::size_t event_count);
    /// Dispatch `job` (a survey or generation batch) onto the thread pool,
    /// seeded from `seed`.
    void start_job(
        GeneratorBatchJob& job,
        ResultQueue& result_queue,
        std::optional<std::uint64_t> seed,
        bool is_survey,
        std::size_t survey_pass
    );
    /// Snapshot @ref max_weight into `job.max_weight`, at a fixed point
    /// independent of when @ref submit_unweight_job actually dispatches it.
    void prepare_unweight_job(GeneratorBatchJob& job) const;
    /// Submit `job`'s unweighting to the thread pool of its own generation
    /// context (required on GPU: unweighting does a device-to-host copy).
    void submit_unweight_job(GeneratorBatchJob& job, ResultQueue& result_queue);
    /// @ref prepare_unweight_job followed by @ref submit_unweight_job.
    void start_unweight_job(GeneratorBatchJob& job, ResultQueue& result_queue);
    /// Size of the next VEGAS/channel-weight optimization batch.
    std::size_t next_vegas_batch_size();
    /// Discard the accumulated event and weight files.
    void clear_events();
    /// Update @ref max_weight from a batch of `weights`, which are unweighted
    /// later if `unweight` is true.
    void update_max_weight(Tensor weights, bool unweight);
    /// Append `unweighted_events` to @ref event_file, recording
    /// `job_max_weight`.
    void write_events(const TensorVec& unweighted_events, double job_max_weight);
    /// Pack `unweighted_events` into `event_buffer` (layout @ref
    /// event_file_layout) and their weights into `weight_buffer` (layout
    /// `weight_file_layout`), resizing both.
    void fill_event_buffers(
        const TensorVec& unweighted_events,
        EventBuffer& event_buffer,
        EventBuffer& weight_buffer
    ) const;
    /// Serialize this generator's state to `file_name`; see @ref load.
    void save(const std::string& file_name) const;
    /// The JSON text that @ref save writes; see @ref load_json. With
    /// `include_estimates`, the current maximum weight and the integral of the
    /// absolute weights are stored as well. @ref load_json then keeps the
    /// maximum weight fixed and uses the integral as a prior, see
    /// @ref abs_integral_estimate.
    std::string to_json(bool include_estimates = false) const;

private:
    struct StepState {
        double max_weight = 0.;
        bool max_weight_fixed = false;
        std::vector<double> large_weights;
        double count_accepted = 0.;
        // like _freeze_count, the expected accepted events of this step alone
        double freeze_count = 0.;
    };

    void apply_truncation_budget();
    void apply_step_truncation_budget(std::size_t step, std::size_t count_target);
    void update_status_integrals();
    void unweight_step_file(MixMaxRandom& rand_gen);
    StepChainResult step_chain(EventBuffer& weight_buffer, std::size_t index) const;
    ChannelEventGenerator(
        const std::vector<ContextPtr>& contexts,
        std::size_t particle_count,
        const Function& integrand_channel_function,
        const Function& integrand_common_function,
        const Function& integrand_concat_function,
        const Function& unweighter_function,
        const std::optional<Function>& histogram_function,
        const std::string& event_file,
        const std::string& weight_file,
        std::size_t subprocess_index,
        const std::string& name,
        const GeneratorConfig& config,
        const std::vector<Histogram>& histograms
    );
    void init_used_globals();
    void init_runtimes();
    void init_field_indices();

    struct ContextRuntimes {
        RuntimePtr integrand_channel = nullptr;
        RuntimePtr integrand_common = nullptr;
        RuntimePtr integrand_concat = nullptr;
        RuntimePtr unweighter = nullptr;
        RuntimePtr vegas_histogram = nullptr;
        RuntimePtr discrete_histogram = nullptr;
        RuntimePtr observable_histograms = nullptr;
        RuntimePtr step_random = nullptr;
    };

    struct FieldIndices {
        int weight, momenta;
        int color_index, helicity_index, diagram_index, flavor_index;
        int ren_scale, alpha_qcd;
        int x1, fact_scale1, x2, fact_scale2, partial_weight_product;
        int cluster_scales;
        int alphas_scales;
        // x, pdg, q, info, lead_q of the pdf ladder
        std::array<int, 5> ladder;
        int subprocess_index;
        int bias;
        int matrix_element;
        int random, rest;
    };

    GeneratorStatus _status;
    GeneratorConfig _config;
    std::vector<ContextPtr> _contexts;
    std::vector<ContextRuntimes> _runtimes;
    int _event_layout_extra_flags;
    int _particle_layout_extra_flags;
    DataLayout _event_file_layout;
    std::optional<EventFile> _event_file;
    std::optional<EventFile> _weight_file;
    std::string _weight_file_path;
    std::optional<VegasGridOptimizer> _vegas_optimizer;
    std::optional<DiscreteOptimizer> _discrete_optimizer;
    std::size_t _batch_size;
    std::size_t _particle_count;
    Function _integrand_channel_function;
    Function _integrand_common_function;
    Function _integrand_concat_function;
    Function _unweighter_function;
    std::optional<Function> _histogram_function;
    RunningIntegral _cross_section;
    RunningIntegral _abs_cross_section;
    double _max_weight = 0.;
    bool _max_weight_fixed = false;
    // mean and standard error
    std::optional<std::pair<double, double>> _abs_integral_prior;
    std::optional<std::pair<double, double>> _integral_prior;
    std::size_t _count_requested_opt = 0;
    // Monotonic per-job counters keying each job's deterministic random stream;
    // never reset. Separate for survey/generate so neither depends on the other.
    std::size_t _survey_rng_seq = 0;
    std::size_t _generate_rng_seq = 0;
    // Progress of unweight_file()'s final pass over _weight_file: _unweighted_count
    // is the index up to which it's been scanned under the *current* _max_weight,
    // and _unweighted_accept_count the number of those accepted. Both reset to 0
    // whenever _max_weight changes (forcing a full rescan), since a changed
    // max_weight invalidates every prior accept/reject decision.
    std::size_t _unweighted_count = 0;
    std::size_t _unweighted_accept_count = 0;
    std::size_t _iters_without_improvement = 0;
    double _best_rsd = std::numeric_limits<double>::max();
    std::vector<double> _large_weights;
    std::vector<Histogram> _histograms;
    std::unordered_set<std::string> _used_globals;
    FieldIndices _field_indices;
    NamedVector<Type> _event_types;
    std::array<bool, 3> _provided_indices{true, true, true};
    std::vector<StepSettings> _step_settings;
    std::vector<StepState> _step_states;
    std::vector<double> _loaded_step_max_weights;
    double _first_step_count = 0.;
    // expected number of unweighted events, counted in the commit order of the
    // jobs to freeze the maximum weight independently of the thread timing
    double _freeze_count = 0.;
    StepIntegral _final_integral;
    std::size_t _event_epoch = 0;
    std::optional<std::size_t> _first_pass_freeze_max_weight_after;

    friend void to_json(nlohmann::json& j, const ChannelEventGenerator& channel);
};

void to_json(nlohmann::json& j, const ChannelEventGenerator& channel);

} // namespace madspace
