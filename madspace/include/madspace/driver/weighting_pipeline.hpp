#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "madspace/phasespace/weighting_step.hpp"

namespace madspace {

/// A group of channels with the same re- and unweighting steps after their
/// first unweighting; see @ref WeightingStep.
struct WeightingGroup {
    /// Indices of the channels in the @ref EventGenerator.
    std::vector<std::size_t> channel_indices;
    /// The steps, in order.
    std::vector<WeightingStep> steps;
    /// If set, replaces @ref GeneratorConfig::max_overweight_truncation for
    /// the first unweighting of the channels of the group.
    std::optional<double> first_pass_max_overweight_truncation;
    /// If set, replaces @ref GeneratorConfig::freeze_max_weight_after for the
    /// first unweighting of the channels of the group.
    std::optional<std::size_t> first_pass_freeze_max_weight_after;
};

/// JSON text of `groups`, for example to store them in a gridpack. The matrix
/// elements are stored by their index in the context.
std::string weighting_groups_to_json(const std::vector<WeightingGroup>& groups);
/// The groups stored with @ref weighting_groups_to_json. The matrix elements
/// must be loaded into the contexts in the same order as before.
std::vector<WeightingGroup> weighting_groups_from_json(const std::string& text);

/**
 * Runs the re- and unweighting steps of the channel groups of an
 * @ref EventGenerator.
 *
 * The events that pass the first unweighting stay on the device. They are
 * collected per group and context until a device batch is full, then the
 * matrix element of the next step is evaluated on them. The decisions of the
 * steps are made on the scheduling thread, in the order of the jobs of each
 * channel, and the random numbers they use are drawn per event when the
 * events are first unweighted. The events of the channels sharing a batch
 * therefore don't influence each other, and the results don't depend on how
 * the events are batched.
 */
class WeightingPipeline {
public:
    /// Job ids of the pipeline have this bit set.
    static constexpr std::size_t job_id_flag = std::size_t(1) << 62;

    /// Receives the events after the last step, in the job order of each
    /// channel.
    using Sink = std::function<void(StepEvents&)>;

    /// Without `sink`, the events are written to the channel files with
    /// @ref ChannelEventGenerator::write_step_events.
    WeightingPipeline(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
        const std::vector<WeightingGroup>& groups,
        const GeneratorConfig& config,
        Sink sink = {}
    );
    /// Whether channel `channel_index` has steps.
    bool contains(std::size_t channel_index) const {
        return _channel_groups.at(channel_index) != -1;
    }
    /// Whether `job_id` belongs to the pipeline.
    static bool owns_job(std::size_t job_id) { return job_id & job_id_flag; }
    /// Take the unweighted events of `job`, in the job order of its channel.
    void add_events(GeneratorBatchJob& job);
    /// Start the jobs of full batches. With `flush`, all collected events are
    /// split over the free threads of each context, where `busy_counts` are
    /// the other jobs running on it. Returns the number of started jobs.
    std::size_t start_jobs(
        ResultQueue& result_queue,
        bool flush,
        const std::vector<std::size_t>& busy_counts = {}
    );
    /// Process the result of the finished job `job_id`.
    void commit_job(std::size_t job_id);
    /// Whether events are still collected or in jobs.
    bool busy() const { return _chunk_count > 0; }
    /// Drop all events, after a failed job.
    void reset();

private:
    // the unweighted events of one job of a channel
    struct Chunk {
        std::size_t channel_index;
        std::size_t job_id;
        std::size_t sequence;
        std::size_t epoch;
        std::size_t context_index;
        std::size_t generated_count;
        std::size_t count_target;
        // on the device: rows offset to offset + size() of source, before the
        // last decision, which keeps the rows in keep
        std::shared_ptr<TensorVec> source;
        std::size_t offset = 0;
        std::optional<std::vector<me_int_t>> keep;
        TensorVec events;
        // per event on the CPU
        std::vector<double> weights, matrix_elements, biases;
        std::vector<double> overweights, ratio_products;
        std::vector<std::vector<double>> ratios;
        double max_weight;
        std::vector<double> max_weights;
        // results of the last job
        std::vector<double> step_matrix_elements, step_random;
        TensorVec cpu_events;

        std::size_t size() const { return weights.size(); }
        void clear();
        void compact(const std::vector<me_int_t>& indices);
    };
    using ChunkPtr = std::shared_ptr<Chunk>;

    struct ContextRuntimes {
        RuntimePtr gather;
        // by the number of concatenated parts, built when first needed
        std::map<std::size_t, RuntimePtr> concats;
        std::vector<RuntimePtr> steps;
    };
    struct Group {
        NamedVector<Type> types;
        std::vector<StepSettings> settings;
        std::vector<ContextRuntimes> runtimes;
        // [context][stage - 1], stage step_count + 1 copies the events to the CPU
        std::vector<std::vector<std::vector<ChunkPtr>>> buffers;
        std::vector<std::vector<std::size_t>> buffer_sizes;
        std::vector<bool> multiple_matrix_elements;
        int momenta, alpha_s, flavor, me_index, color, helicity, diagram, random;
    };
    struct Job {
        std::size_t group;
        std::size_t stage;
        std::size_t context_index;
        std::vector<ChunkPtr> chunks;
    };

    void advance(ChunkPtr chunk, std::size_t stage);
    void set_ready(ChunkPtr chunk, std::size_t stage);
    void commit_stage(ChunkPtr chunk, std::size_t stage);
    void decide(Chunk& chunk, std::size_t step);
    void start_job(
        ResultQueue& result_queue,
        std::size_t group_index,
        std::size_t context_index,
        std::size_t stage,
        std::vector<ChunkPtr> chunks
    );
    void run_job(Job& job);
    void gather_events(Chunk& chunk, ContextRuntimes& runtimes);
    // the rows of `chunks` from their shared `source`, concatenated
    TensorVec gather_rows(
        const TensorVec& source,
        const std::vector<Chunk*>& chunks,
        std::size_t context_index,
        ContextRuntimes& runtimes
    );
    RuntimePtr& concat_runtime(
        std::size_t group_index, std::size_t context_index, std::size_t part_count
    );
    std::size_t batch_size(std::size_t context_index) const;

    std::vector<ContextPtr> _contexts;
    std::vector<std::shared_ptr<ChannelEventGenerator>> _channels;
    GeneratorConfig _config;
    std::vector<Group> _groups;
    std::vector<int> _channel_groups;
    std::vector<std::size_t> _channel_sequences;
    // per channel and stage: the next sequence to commit, and the finished
    // chunks waiting for their turn
    std::vector<std::vector<std::size_t>> _next_sequences;
    std::vector<std::vector<std::map<std::size_t, ChunkPtr>>> _ready;
    std::unordered_map<std::size_t, Job> _jobs;
    std::vector<std::size_t> _context_job_counts;
    std::mutex _concat_mutex;
    std::size_t _job_counter = 0;
    std::size_t _chunk_count = 0;
    Sink _sink;
};

} // namespace madspace
