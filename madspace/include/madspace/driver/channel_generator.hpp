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
     * @param event_file        Path of the weighted event file.
     * @param weight_file       Path of the max-weight tracking file.
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
    /// The weighted event file.
    EventFile& event_file() { return _event_file; }
    /// The max-weight tracking file.
    EventFile& weight_file() { return _weight_file; }
    /// Current maximum-weight estimate, used for unweighting.
    double max_weight() const { return _max_weight; }
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
    bool uses_integral_prior() const;
    /// The prior of the signed integral as (mean, standard error), or of the
    /// absolute one if no signed prior was set.
    std::pair<double, double> integral_prior() const;
    /// The prior of the integral of the absolute weights as (mean, standard error).
    std::pair<double, double> abs_integral_prior() const {
        return *_abs_integral_prior;
    }
    /// The integral of the absolute weights: the mean of this run, combined
    /// with the prior (if any) weighted by the inverse variances. Used to share
    /// the events between channels.
    double abs_integral_estimate() const;
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
    /// Update @ref max_weight from a batch of `weights`.
    void update_max_weight(Tensor weights);
    /// Append `unweighted_events` to @ref event_file, recording
    /// `job_max_weight`.
    void write_events(const TensorVec& unweighted_events, double job_max_weight);
    /// Serialize this generator's state to `file_name`; see @ref load.
    void save(const std::string& file_name) const;
    /// The JSON text that @ref save writes; see @ref load_json. With
    /// `include_estimates`, the current maximum weight and the integral of the
    /// absolute weights are stored as well. @ref load_json then keeps the
    /// maximum weight fixed and uses the integral as a prior, see
    /// @ref abs_integral_estimate.
    std::string to_json(bool include_estimates = false) const;

private:
    void apply_truncation_budget();
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
    EventFile _event_file;
    EventFile _weight_file;
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
    // Expected number of accepted events, sum of min(|w| / max_weight, 1) over the
    // committed generate stages, rescaled like count_unweighted when the max
    // weight rises. Decides when the max weight is frozen: unlike count_unweighted
    // it only depends on the generate stages, which commit in a fixed order.
    double _expected_unweighted = 0.;
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

    friend void to_json(nlohmann::json& j, const ChannelEventGenerator& channel);
};

void to_json(nlohmann::json& j, const ChannelEventGenerator& channel);

} // namespace madspace
