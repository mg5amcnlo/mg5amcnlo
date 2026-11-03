#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/logger.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

/// Online mean, variance and standard error of a stream of values (Welford's
/// algorithm), used to track the integral estimate during event generation.
class RunningIntegral {
public:
    /// Empty accumulator.
    RunningIntegral() : _mean(0), _var_sum(0), _count(0) {}
    /// Running mean of the pushed values.
    double mean() const { return _mean; }
    /// Running variance of the pushed values.
    double variance() const { return _count > 1 ? _var_sum / (_count - 1) : 0; }
    /// Standard error of @ref mean.
    double error() const { return std::sqrt(variance() / _count); }
    /// @ref error relative to @ref mean.
    double rel_error() const { return error() / mean(); }
    /// Standard deviation of the pushed values relative to @ref mean.
    double rel_std_dev() const { return std::sqrt(variance()) / _mean; }
    /// Number of pushed values.
    std::size_t count() const { return _count; }
    /// Discard every pushed value.
    void reset() {
        _mean = 0;
        _var_sum = 0;
        _count = 0;
    }
    /// Add one more value to the running statistics.
    void push(double value) {
        ++_count;
        if (_count == 1) {
            _mean = value;
            _var_sum = 0;
        } else {
            double mean_diff = value - _mean;
            _mean += mean_diff / _count;
            _var_sum += mean_diff * (value - _mean);
        }
    }

private:
    double _mean;
    double _var_sum;
    std::size_t _count;
};

/// Tunable parameters of @ref EventGenerator.
struct GeneratorConfig {
    /// Number of unweighted events to generate.
    std::size_t target_count = 10000;
    /// Damping factor of the VEGAS grid updates; see @ref VegasGridOptimizer.
    double vegas_damping = 0.2;
    /// Fraction of the weight distribution's tail truncated when estimating
    /// the unweighting efficiency.
    double max_overweight_truncation = 0.01;
    /// Number of survey events after which the maximum weight estimate is
    /// frozen.
    std::size_t freeze_max_weight_after = 10000;
    /// Initial number of events per generation batch.
    std::size_t start_batch_size = 1000;
    /// Largest allowed number of events per generation batch.
    std::size_t max_batch_size = 64000;
    /// Minimum number of VEGAS/channel-weight survey iterations.
    std::size_t survey_min_iters = 3;
    /// Maximum number of survey iterations.
    std::size_t survey_max_iters = 4;
    /// Target relative precision of the survey's integral estimate.
    double survey_target_precision = 0.1;
    /// Number of consecutive non-improving iterations tolerated before the
    /// survey stops optimizing.
    std::size_t optimization_patience = 3;
    /// Minimum relative improvement counted as progress by
    /// `optimization_patience`.
    double optimization_threshold = 0.99;
    /// Batch size used on a CPU device.
    std::size_t cpu_batch_size = 1000;
    /// Batch size used on a GPU device.
    std::size_t gpu_batch_size = 64000;
    /// Progress-reporting verbosity.
    Verbosity verbosity = Verbosity::silent;
    /// If true, periodically write live progress data to disk.
    bool write_live_data = false;
    /// Thread count used to combine per-channel event files; `-1` for the
    /// hardware concurrency.
    int combine_thread_count = -1;
    /// Minimum acceptable fraction of events surviving fiducial cuts before a
    /// warning is raised.
    double cut_efficiency_threshold = 0.7;
    /// Maximum number of times a batch is resampled to reach
    /// `cut_efficiency_threshold`.
    std::size_t max_cut_repetitions = 100;
    /// Fraction of the target count below which the generator switches to
    /// finishing up the remaining events.
    double finish_remaining_fraction = 0.05;
    /// Largest fraction of the target count schedulable in a single batch.
    double max_batch_fraction = 0.6;
    /// Number of standard deviations of headroom added when sizing a batch,
    /// to avoid overshooting the target count.
    double batch_overshoot_sigma = 1.0;
};

/// Settings of a re- or unweighting step after the first unweighting, as far
/// as the channels need them; see @ref WeightingStep.
struct StepSettings {
    /// Unweighting step if true, reweighting step otherwise.
    bool unweight = true;
    /// See @ref GeneratorConfig::max_overweight_truncation.
    double max_overweight_truncation = 0.01;
    /// See @ref GeneratorConfig::freeze_max_weight_after.
    std::size_t freeze_max_weight_after = 10000;
    /// Whether the step returns the ratio instead of the matrix element.
    bool ratio = false;
};

/// Result of @ref evaluate_step_chain.
struct StepChainResult {
    /// Probability that the event is accepted by all steps.
    double acceptance;
    /// Signed weight of the event after all steps, in units of the product of
    /// the maximum weights of the unweighting steps.
    double weight;
};

/**
 * Acceptance probability and weight of an event after the first unweighting and
 * the re- and unweighting steps that follow.
 *
 * The first step accepts the event with probability
 * @f$\min(1, |w_0| / m_0)@f$ and carries the overweight factor
 * @f$o_0 = \max(1, |w_0| / m_0)@f$. Each reweighting step multiplies the weight
 * by its ratio @f$r_k@f$. Each unweighting step unweights
 * @f$u_k = o \prod r@f$, the overweight factor of the last unweighting step
 * times the ratios since, with probability @f$\min(1, u_k / m_k)@f$, and carries
 * @f$o_k = \max(1, u_k / m_k)@f$. Raising any maximum weight only lowers the
 * acceptance, so the events can be unweighted again with larger maxima.
 *
 * @param weight0      The stored weight of the first step, @f$\max(|w_0|, m)@f$
 *                     with the sign of @f$w_0@f$, for a maximum @f$m \le m_0@f$.
 * @param max_weight0  The maximum weight @f$m_0@f$ of the first step.
 * @param ratios       The ratios @f$r_k@f$ of the steps.
 * @param max_weights  The maximum weights @f$m_k@f$ of the steps, ignored for
 *                     reweighting steps.
 * @param steps        The steps.
 */
StepChainResult evaluate_step_chain(
    double weight0,
    double max_weight0,
    const double* ratios,
    const double* max_weights,
    const std::vector<StepSettings>& steps
);

/**
 * Number of events to schedule in the next generation batch.
 *
 * If close to finishing, returns the remaining count divided by the
 * unweighting efficiency. Otherwise selects a count safely below the target
 * using the integration uncertainty, capped by @p config's
 * `max_batch_fraction` of the target.
 *
 * @param count_target                 Total unweighted events wanted.
 * @param count_unweighted             Unweighted events generated so far.
 * @param count_requested              Weighted events requested so far, counting
 *                                     each job at its nominal batch size.
 * @param abs_cross_section_count      Events behind the current cross-section
 *                                     estimate.
 * @param abs_cross_section_rel_error  Relative error of that estimate.
 * @param config                       Generator configuration.
 */
std::size_t compute_generation_batch_event_count(
    std::size_t count_target,
    double count_unweighted,
    std::size_t count_requested,
    std::size_t abs_cross_section_count,
    double abs_cross_section_rel_error,
    const GeneratorConfig& config
);

/// Picks the channel owning `random_index` from per-channel cumulative event
/// counts: the first whose cumulative count is strictly greater. Lower-bound
/// (`>=`) semantics would hand `random_index == 0` to the first channel even
/// with its share spent, and the caller's decrement of a zero share wraps.
template <typename It, typename Proj>
It select_combine_channel(It first, It last, std::size_t random_index, Proj cum_count) {
    return std::upper_bound(
        first,
        last,
        random_index,
        [&cum_count](std::size_t index, const auto& channel) {
            return index < cum_count(channel);
        }
    );
}

/// `select_combine_channel()` over a plain cumulative-count vector, for
/// testing. Returns `cum_counts.size()` if `random_index` is out of range,
/// i.e. not below the total.
std::size_t select_combine_channel_index(
    const std::vector<std::size_t>& cum_counts, std::size_t random_index
);

/// Progress snapshot of one subprocess's @ref EventGenerator run.
struct GeneratorStatus {
    /// Index of the subprocess.
    std::size_t subprocess;
    /// Name of the subprocess.
    std::string name;
    /// Running mean weight, i.e. the cross-section estimate.
    double mean;
    /// Standard error of @ref mean.
    double error;
    /// Running mean of `|weight|`.
    double mean_abs;
    /// Standard error of @ref mean_abs.
    double error_abs;
    /// Relative standard deviation of the weights.
    double rel_std_dev;
    /// Weighted events generated so far.
    std::size_t count;
    /// Weighted events generated so far, restricted to the optimized phase.
    std::size_t count_opt;
    /// Weighted events surviving fiducial cuts.
    std::size_t count_after_cuts;
    /// Weighted events surviving fiducial cuts, restricted to the optimized
    /// phase.
    std::size_t count_after_cuts_opt;
    /// Unweighted events generated so far.
    double count_unweighted;
    /// Target unweighted event count; see @ref GeneratorConfig::target_count.
    std::size_t count_target;
    /// Number of survey iterations run.
    std::size_t iterations;
    /// Number of optimization iterations in a row without enough improvement;
    /// the channel counts as optimized when this reaches
    /// @ref GeneratorConfig::optimization_patience.
    std::size_t iters_without_improvement;
    /// Whether the survey/optimization phase has finished.
    bool optimized;
    /// Whether generation of this subprocess is complete.
    bool done;
    /// Events given the weight 0 by a weighting step, because their matrix
    /// element ratio was not finite or the earlier matrix element was 0.
    std::size_t count_invalid_step_values = 0;
};

/// One binned observable, accumulated during event generation; see @ref
/// ObservableHistograms.
struct Histogram {
    /// Observable name.
    std::string name;
    /// Lower edge of the first bin.
    double min;
    /// Upper edge of the last bin.
    double max;
    /// Summed weight per bin, including the under/overflow bins.
    std::vector<double> bin_values;
    /// Standard error per bin.
    std::vector<double> bin_errors;
};

// Lightweight pending-work entry for EventGenerator::_ready_jobs, before start_jobs()
// has decided which context/how many sub-jobs to create from it. Kept separate from
// GeneratorBatchJob so a queue of pending batches doesn't carry the weight of every
// dispatched job's tensors and RNG bookkeeping.
struct ReadyJob {
    std::size_t channel_index;
    bool unweight;
    // VEGAS batch: fixed size, start_jobs() dispatches it atomically in one go.
    // Generation batch: events not yet dispatched -- start_jobs() decrements this in
    // place as it creates sub-jobs, one device batch at a time, round-robining with
    // other channels' ReadyJobs over however many calls it takes to reach zero.
    std::size_t batch_event_count;
    bool is_vegas_batch = false;
};

struct GeneratorBatchJob {
    std::size_t channel_index;
    bool unweight;
    // Copied from the originating ReadyJob at dispatch time. For a VEGAS batch, this
    // is the batch's fixed total, read by start_job()'s shrink-to-fit and by the
    // done_event_count accounting in survey()/survey_deterministic(). For a
    // generation batch it isn't read after dispatch -- generation sub-jobs always
    // request a full device batch (see start_job()).
    std::size_t batch_event_count;
    // Total sub-jobs the batch was split into; only meaningful for VEGAS batches
    // (see commit_generate_job()'s clear_events trigger). Generation batches are
    // dispatched one sub-job per start_jobs() visit, so this isn't a full-batch count
    // for them and isn't read.
    std::size_t split_job_count;
    Tensor weights;
    // per-event bias, empty without a custom bias
    Tensor biases;
    TensorVec events;
    TensorVec unweighted_events;
    TensorVec hists;
    TensorVec vegas_hist;
    TensorVec discrete_hist;
    std::size_t context_index;
    std::size_t job_id;
    double max_weight;
    // target count of the channel when max_weight was taken
    std::size_t count_target = 0;
    // Top-level seed plus job identity, used to derive this job's DerivedSeed(s)
    // independently for generate vs unweight (see start_job()/submit_unweight_job()).
    std::optional<std::uint64_t> rng_seed;
    // Per-channel dispatch sequence, assigned once at start_job() time.
    std::size_t rng_job_index = 0;
    bool rng_is_survey = false;
    std::size_t rng_survey_pass = 0;
    // True for a VEGAS-grid-optimization batch, dispatched atomically by start_jobs()
    // and shrunk to fit by start_job(). False for a steady-state generation batch,
    // dispatched incrementally as device-sized sub-jobs (see ReadyJob).
    bool is_vegas_batch = false;
    // Nominal batch size start_job() requested; the job generates more raw points
    // when it repeats batches to get enough of them past the cuts.
    std::size_t requested_event_count = 0;
    // With re- and unweighting steps: unweighted_events stay on the device and
    // these are their weights, matrix elements and biases on the CPU
    TensorVec step_host;
};

void to_json(nlohmann::json& j, const GeneratorStatus& status);
void to_json(nlohmann::json& j, const Histogram& hist);

} // namespace madspace
