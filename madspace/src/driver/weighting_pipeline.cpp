#include "madspace/driver/weighting_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <tuple>

#include "madspace/util.hpp"

#include <nlohmann/json.hpp>

using namespace madspace;

namespace {

std::vector<double> to_vector(const Tensor& tensor) {
    auto view = tensor.view<double, 1>();
    std::vector<double> ret(view.size());
    for (std::size_t i = 0; i < ret.size(); ++i) {
        ret[i] = view[i];
    }
    return ret;
}

template <typename T>
void compact_vector(std::vector<T>& vec, const std::vector<me_int_t>& indices) {
    if (vec.empty()) {
        return;
    }
    std::vector<T> ret;
    ret.reserve(indices.size());
    for (auto index : indices) {
        ret.push_back(vec.at(index));
    }
    vec = std::move(ret);
}

Function gather_function(const NamedVector<Type>& types) {
    NamedVector<Type> arg_types{{"indices", batch_int}};
    arg_types.insert_back(types);
    FunctionBuilder fb(arg_types, types);
    for (std::size_t i = 0; i < types.size(); ++i) {
        fb.output(i, fb.batch_gather(fb.input(0), fb.input(i + 1)));
    }
    return fb.function();
}

Function concat_function(const NamedVector<Type>& types, std::size_t part_count) {
    NamedVector<Type> arg_types;
    for (std::size_t part = 1; part <= part_count; ++part) {
        for (auto [key, type] : zip(types.keys(), types)) {
            arg_types.push_back(std::format("arg{}_{}", part, key), type);
        }
    }
    FunctionBuilder fb(arg_types, types);
    int count = static_cast<int>(types.size());
    for (int i = 0; i < count; ++i) {
        ValueVec parts;
        for (std::size_t part = 0; part < part_count; ++part) {
            parts.push_back(fb.input(i + static_cast<int>(part) * count));
        }
        auto [cat, sizes] = fb.batch_cat(parts);
        fb.output(i, cat);
    }
    return fb.function();
}

using json = nlohmann::json;

json matrix_element_to_json(const MatrixElement& matrix_element) {
    std::vector<me_int_t> me_frame;
    for (std::size_t i = 0; auto flag : matrix_element.frame_mask()) {
        if (flag) {
            me_frame.push_back(static_cast<me_int_t>(i + 1));
        }
        ++i;
    }
    // see the MatrixElement constructor: an empty reference mask with a frame
    // means that the frame is the incoming system or there are two incoming
    // particles
    auto& reference_mask = matrix_element.reference_mask();
    auto& incoming_mask =
        reference_mask.empty() ? matrix_element.frame_mask() : reference_mask;
    std::size_t incoming_count = incoming_mask.empty()
        ? 2
        : std::count(incoming_mask.begin(), incoming_mask.end(), 1);
    json inputs = json::array(), outputs = json::array();
    for (auto input : matrix_element.inputs()) {
        inputs.push_back(static_cast<int>(input));
    }
    for (auto output : matrix_element.outputs()) {
        outputs.push_back(static_cast<int>(output));
    }
    return {
        {"index", matrix_element.matrix_element_index()},
        {"particle_count", matrix_element.particle_count()},
        {"inputs", inputs},
        {"outputs", outputs},
        {"diagram_count", matrix_element.diagram_count()},
        {"sample_random_inputs", matrix_element.sample_random_inputs()},
        {"me_frame", me_frame},
        {"incoming_count", incoming_count},
        {"channel", matrix_element.channel()},
    };
}

MatrixElement matrix_element_from_json(const json& j) {
    std::vector<MatrixElement::MatrixElementInput> inputs;
    std::vector<MatrixElement::MatrixElementOutput> outputs;
    for (int input : j.at("inputs")) {
        inputs.push_back(static_cast<MatrixElement::MatrixElementInput>(input));
    }
    for (int output : j.at("outputs")) {
        outputs.push_back(static_cast<MatrixElement::MatrixElementOutput>(output));
    }
    return MatrixElement(
        j.at("index").get<std::size_t>(),
        j.at("particle_count").get<std::size_t>(),
        inputs,
        outputs,
        j.at("diagram_count").get<std::size_t>(),
        j.at("sample_random_inputs").get<bool>(),
        j.at("me_frame").get<std::vector<me_int_t>>(),
        j.at("incoming_count").get<std::size_t>(),
        j.at("channel").get<me_int_t>()
    );
}

} // namespace

std::string
madspace::weighting_groups_to_json(const std::vector<WeightingGroup>& groups) {
    json j = json::array();
    for (auto& group : groups) {
        json steps = json::array();
        for (auto& step : group.steps) {
            json matrix_elements = json::array(), override_outputs = json::array();
            for (auto& matrix_element : step.matrix_elements) {
                matrix_elements.push_back(matrix_element_to_json(matrix_element));
            }
            for (auto output : step.override_outputs) {
                override_outputs.push_back(static_cast<int>(output));
            }
            steps.push_back({
                {"matrix_elements", matrix_elements},
                {"mode",
                 step.mode == WeightingStep::unweight ? "unweight" : "reweight"},
                {"max_overweight_truncation", step.max_overweight_truncation},
                {"freeze_max_weight_after", step.freeze_max_weight_after},
                {"override_outputs", override_outputs},
            });
        }
        json group_json{{"channel_indices", group.channel_indices}, {"steps", steps}};
        if (group.first_pass_max_overweight_truncation) {
            group_json["first_pass_max_overweight_truncation"] =
                *group.first_pass_max_overweight_truncation;
        }
        if (group.first_pass_freeze_max_weight_after) {
            group_json["first_pass_freeze_max_weight_after"] =
                *group.first_pass_freeze_max_weight_after;
        }
        j.push_back(group_json);
    }
    return j.dump();
}

std::vector<WeightingGroup>
madspace::weighting_groups_from_json(const std::string& text) {
    std::vector<WeightingGroup> groups;
    for (auto& group_json : json::parse(text)) {
        WeightingGroup group{
            .channel_indices =
                group_json.at("channel_indices").get<std::vector<std::size_t>>()
        };
        if (group_json.contains("first_pass_max_overweight_truncation")) {
            group.first_pass_max_overweight_truncation =
                group_json.at("first_pass_max_overweight_truncation").get<double>();
        }
        if (group_json.contains("first_pass_freeze_max_weight_after")) {
            group.first_pass_freeze_max_weight_after =
                group_json.at("first_pass_freeze_max_weight_after").get<std::size_t>();
        }
        for (auto& step_json : group_json.at("steps")) {
            WeightingStep step{
                .mode = step_json.at("mode").get<std::string>() == "unweight"
                    ? WeightingStep::unweight
                    : WeightingStep::reweight,
                .max_overweight_truncation =
                    step_json.at("max_overweight_truncation").get<double>(),
                .freeze_max_weight_after =
                    step_json.at("freeze_max_weight_after").get<std::size_t>(),
            };
            for (auto& me_json : step_json.at("matrix_elements")) {
                step.matrix_elements.push_back(matrix_element_from_json(me_json));
            }
            for (int output : step_json.at("override_outputs")) {
                step.override_outputs.push_back(
                    static_cast<MatrixElement::MatrixElementOutput>(output)
                );
            }
            group.steps.push_back(std::move(step));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

void WeightingPipeline::Chunk::clear() {
    source.reset();
    keep.reset();
    events.clear();
    cpu_events.clear();
    weights.clear();
    matrix_elements.clear();
    biases.clear();
    overweights.clear();
    ratio_products.clear();
    for (auto& step_ratios : ratios) {
        step_ratios.clear();
    }
}

void WeightingPipeline::Chunk::compact(const std::vector<me_int_t>& indices) {
    compact_vector(weights, indices);
    compact_vector(matrix_elements, indices);
    compact_vector(biases, indices);
    compact_vector(overweights, indices);
    compact_vector(ratio_products, indices);
    for (auto& step_ratios : ratios) {
        compact_vector(step_ratios, indices);
    }
}

WeightingPipeline::WeightingPipeline(
    const std::vector<ContextPtr>& contexts,
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    const std::vector<WeightingGroup>& groups,
    const GeneratorConfig& config,
    Sink sink
) :
    _contexts(contexts),
    _channels(channels),
    _config(config),
    _channel_groups(channels.size(), -1),
    _channel_sequences(channels.size()),
    _next_sequences(channels.size()),
    _ready(channels.size()),
    _context_job_counts(contexts.size()),
    _sink(sink) {
    if (!_sink) {
        _sink = [this](StepEvents& step_events) {
            _channels.at(step_events.channel_index)->write_step_events(step_events);
        };
    }
    const std::array<MatrixElement::MatrixElementOutput, 3> index_outputs{
        MatrixElement::color_index_out,
        MatrixElement::helicity_index_out,
        MatrixElement::diagram_index_out,
    };
    for (std::size_t group_index = 0; auto& group_config : groups) {
        if (group_config.steps.empty() || group_config.channel_indices.empty()) {
            throw std::invalid_argument(
                "a weighting group needs at least one channel and one step"
            );
        }
        for (auto channel_index : group_config.channel_indices) {
            if (channel_index >= channels.size()) {
                throw std::invalid_argument(
                    "channel index of weighting group too large"
                );
            }
            if (_channel_groups.at(channel_index) != -1) {
                throw std::invalid_argument(
                    "a channel can only belong to one weighting group"
                );
            }
            _channel_groups.at(channel_index) = static_cast<int>(group_index);
        }

        auto& first_channel = channels.at(group_config.channel_indices.at(0));
        NamedVector<Type> types = first_channel->event_types();
        for (auto channel_index : group_config.channel_indices) {
            auto& channel = channels.at(channel_index);
            auto& channel_types = channel->event_types();
            if (channel_types.keys() != types.keys() ||
                !std::equal(
                    channel_types.begin(), channel_types.end(), types.begin()
                ) ||
                channel->provided_indices() != first_channel->provided_indices()) {
                throw std::invalid_argument(
                    "the channels of a weighting group need the same events"
                );
            }
        }
        auto& index_map = types.index_map();
        if (!index_map.contains("matrix_element")) {
            throw std::invalid_argument(
                "a weighting group needs integrands that return the matrix element"
            );
        }
        std::size_t step_count = group_config.steps.size();
        types.push_back(
            "step_random",
            batch_float_array(WeightingStepFunction::random_count * step_count)
        );

        Group group{
            .momenta = static_cast<int>(index_map.at("momenta")),
            .alpha_s = static_cast<int>(index_map.at("alpha_qcd")),
            .flavor = static_cast<int>(index_map.at("matrix_element_flavor")),
            .me_index = index_map.contains("matrix_element_index")
                ? static_cast<int>(index_map.at("matrix_element_index"))
                : -1,
            .color = static_cast<int>(index_map.at("color_index")),
            .helicity = static_cast<int>(index_map.at("helicity_index")),
            .diagram = static_cast<int>(index_map.at("diagram_index")),
            .random = static_cast<int>(types.size() - 1),
        };

        std::vector<Function> step_functions;
        auto provided = first_channel->provided_indices();
        for (std::size_t step_index = 0; auto& step : group_config.steps) {
            if (step.matrix_elements.size() > 1 && group.me_index == -1) {
                throw std::invalid_argument(
                    "several matrix elements in a weighting step, but the integrand "
                    "has only one"
                );
            }
            group.multiple_matrix_elements.push_back(step.matrix_elements.size() > 1);
            group.settings.push_back({
                .unweight = step.mode == WeightingStep::unweight,
                .max_overweight_truncation = step.max_overweight_truncation,
                .freeze_max_weight_after = step.freeze_max_weight_after,
                .ratio = step.returns_ratio(),
            });
            std::array<bool, 3> write_indices{};
            for (std::size_t i = 0; i < index_outputs.size(); ++i) {
                auto output = index_outputs.at(i);
                auto& outputs = step.matrix_elements.at(0).outputs();
                bool returns =
                    std::find(outputs.begin(), outputs.end(), output) != outputs.end();
                bool overrides =
                    std::find(
                        step.override_outputs.begin(),
                        step.override_outputs.end(),
                        output
                    ) != step.override_outputs.end();
                write_indices.at(i) = returns && (!provided.at(i) || overrides);
                provided.at(i) = provided.at(i) || returns;
            }
            step_functions.push_back(
                WeightingStepFunction(
                    step,
                    first_channel->particle_count(),
                    write_indices,
                    step_index,
                    step_count
                )
                    .function()
            );
            ++step_index;
        }

        Function gather = gather_function(types);
        for (auto& context : contexts) {
            ContextRuntimes runtimes{
                .gather = build_runtime(gather, context, false),
            };
            for (auto& function : step_functions) {
                runtimes.steps.push_back(build_runtime(function, context, false));
            }
            group.runtimes.push_back(std::move(runtimes));
        }
        group.types = types;
        group.buffers.assign(
            contexts.size(), std::vector<std::vector<ChunkPtr>>(step_count + 1)
        );
        group.buffer_sizes.assign(
            contexts.size(), std::vector<std::size_t>(step_count + 1)
        );
        for (auto channel_index : group_config.channel_indices) {
            channels.at(channel_index)->set_weighting_steps(group.settings);
            channels.at(channel_index)
                ->set_first_pass_settings(
                    group_config.first_pass_max_overweight_truncation,
                    group_config.first_pass_freeze_max_weight_after
                );
            _next_sequences.at(channel_index).assign(step_count + 1, 0);
            _ready.at(channel_index).resize(step_count + 1);
        }
        _groups.push_back(std::move(group));
        ++group_index;
    }
}

std::size_t WeightingPipeline::batch_size(std::size_t context_index) const {
    return _contexts.at(context_index)->device()->device_type() == DeviceType::cpu
        ? _config.cpu_batch_size
        : _config.gpu_batch_size;
}

void WeightingPipeline::add_events(GeneratorBatchJob& job) {
    auto& channel = _channels.at(job.channel_index);
    auto chunk = std::make_shared<Chunk>();
    chunk->channel_index = job.channel_index;
    chunk->job_id = job.job_id;
    chunk->sequence = _channel_sequences.at(job.channel_index)++;
    chunk->epoch = channel->event_epoch();
    chunk->context_index = job.context_index;
    chunk->generated_count = job.weights.size(0);
    chunk->count_target = job.count_target;
    chunk->source = std::make_shared<TensorVec>(std::move(job.unweighted_events));
    chunk->weights = to_vector(job.step_host.at(0));
    chunk->matrix_elements = to_vector(job.step_host.at(1));
    if (job.step_host.size() > 2) {
        chunk->biases = to_vector(job.step_host.at(2));
    }
    chunk->max_weight = job.max_weight;
    for (double weight : chunk->weights) {
        chunk->overweights.push_back(std::max(1., std::abs(weight) / job.max_weight));
        chunk->ratio_products.push_back(std::copysign(1., weight));
    }
    channel->add_first_step_events(chunk->size(), job.max_weight);
    job.unweighted_events.clear();
    job.step_host.clear();
    ++_chunk_count;
    advance(chunk, 1);
}

void WeightingPipeline::advance(ChunkPtr chunk, std::size_t stage) {
    if (chunk->size() == 0) {
        set_ready(chunk, stage);
        return;
    }
    auto& group = _groups.at(_channel_groups.at(chunk->channel_index));
    group.buffers.at(chunk->context_index).at(stage - 1).push_back(chunk);
    group.buffer_sizes.at(chunk->context_index).at(stage - 1) += chunk->size();
}

void WeightingPipeline::set_ready(ChunkPtr chunk, std::size_t stage) {
    std::size_t channel_index = chunk->channel_index;
    auto& ready = _ready.at(channel_index).at(stage - 1);
    auto& next_sequence = _next_sequences.at(channel_index).at(stage - 1);
    ready.emplace(chunk->sequence, chunk);
    while (!ready.empty() && ready.begin()->first == next_sequence) {
        ChunkPtr next_chunk = ready.begin()->second;
        ready.erase(ready.begin());
        ++next_sequence;
        commit_stage(next_chunk, stage);
    }
}

void WeightingPipeline::commit_stage(ChunkPtr chunk, std::size_t stage) {
    auto& channel = _channels.at(chunk->channel_index);
    std::size_t step_count = channel->weighting_steps().size();
    // the events of the channel were cleared since
    bool stale = chunk->epoch != channel->event_epoch();
    if (stale) {
        chunk->clear();
    }
    if (stage <= step_count) {
        if (!stale && chunk->size() > 0) {
            decide(*chunk, stage);
        } else {
            chunk->ratios.emplace_back();
            chunk->max_weights.push_back(0.);
        }
        advance(chunk, stage + 1);
        return;
    }
    --_chunk_count;
    if (!stale) {
        StepEvents step_events{
            .channel_index = chunk->channel_index,
            .job_id = chunk->job_id,
            .events = std::move(chunk->cpu_events),
            .weights = std::move(chunk->weights),
            .max_weight = chunk->max_weight,
            .ratios = std::move(chunk->ratios),
            .max_weights = std::move(chunk->max_weights),
            .biases = std::move(chunk->biases),
            .generated_count = chunk->generated_count,
        };
        _sink(step_events);
    }
}

void WeightingPipeline::decide(Chunk& chunk, std::size_t step) {
    auto& channel = _channels.at(chunk.channel_index);
    auto& settings = channel->weighting_steps().at(step - 1);
    std::size_t count = chunk.size();
    std::vector<double> ratios(count);
    std::size_t invalid_count = 0;
    for (std::size_t i = 0; i < count; ++i) {
        double previous = chunk.matrix_elements.at(i);
        double& value = chunk.step_matrix_elements.at(i);
        double ratio = value;
        if (!settings.ratio) {
            ratio = previous != 0. ? value / previous
                : value != 0.
                ? std::numeric_limits<double>::quiet_NaN()
                : 0.;
        }
        // the event gets the weight 0, so that it cannot set the maximum
        if (!std::isfinite(ratio)) {
            ratio = 0.;
            ++invalid_count;
        }
        if (settings.ratio) {
            // the implied matrix element is the reference of the next step
            value = ratio * previous;
        }
        ratios.at(i) = ratio;
        chunk.ratio_products.at(i) *= ratio;
    }
    if (invalid_count > 0) {
        channel->add_invalid_step_values(invalid_count);
    }
    std::vector<me_int_t> keep;
    if (settings.unweight) {
        // the overweight of the last unweighting times the ratios since, see
        // evaluate_step_chain()
        std::vector<double> values(count);
        for (std::size_t i = 0; i < count; ++i) {
            values.at(i) =
                chunk.overweights.at(i) * std::abs(chunk.ratio_products.at(i));
        }
        channel->update_step_max_weight(step, values, chunk.count_target);
        double max_weight = channel->step_max_weight(step);
        for (std::size_t i = 0; i < count; ++i) {
            double value = values.at(i);
            if (value > 0. && value > chunk.step_random.at(i) * max_weight) {
                keep.push_back(static_cast<me_int_t>(i));
            }
            chunk.overweights.at(i) = std::max(1., value / max_weight);
            chunk.ratio_products.at(i) = std::copysign(1., chunk.ratio_products.at(i));
        }
        channel->add_step_accepted(
            step, static_cast<double>(keep.size()), chunk.max_weight, chunk.max_weights
        );
        chunk.max_weights.push_back(max_weight);
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            if (ratios.at(i) != 0.) {
                keep.push_back(static_cast<me_int_t>(i));
            }
        }
        channel->add_step_accepted(
            step, static_cast<double>(keep.size()), chunk.max_weight, chunk.max_weights
        );
        chunk.max_weights.push_back(0.);
    }
    chunk.ratios.push_back(std::move(ratios));
    chunk.matrix_elements = std::move(chunk.step_matrix_elements);
    chunk.step_matrix_elements.clear();
    chunk.step_random.clear();
    if (keep.size() < count) {
        chunk.compact(keep);
        chunk.keep = std::move(keep);
    }
    if (chunk.size() == 0) {
        chunk.source.reset();
        chunk.keep.reset();
    }
}

std::size_t WeightingPipeline::start_jobs(
    ResultQueue& result_queue, bool flush, const std::vector<std::size_t>& busy_counts
) {
    std::size_t started = 0;
    for (std::size_t context_index = 0; context_index < _contexts.size();
         ++context_index) {
        std::vector<std::tuple<std::size_t, std::size_t>> stages;
        for (std::size_t group_index = 0; auto& group : _groups) {
            for (std::size_t stage = 1; stage <= group.settings.size() + 1; ++stage) {
                if (!group.buffers.at(context_index).at(stage - 1).empty()) {
                    stages.emplace_back(group_index, stage);
                }
            }
            ++group_index;
        }
        std::size_t free = 0, share = 1;
        if (flush && !stages.empty()) {
            // with flush, the buffers share the free threads
            std::size_t thread_count =
                _contexts.at(context_index)->thread_pool().thread_count();
            std::size_t busy = _context_job_counts.at(context_index) +
                (busy_counts.empty() ? 0 : busy_counts.at(context_index));
            free = thread_count > busy ? thread_count - busy : 0;
            share = std::max<std::size_t>(1, free / stages.size());
        }
        for (auto [group_index, stage] : stages) {
            auto& group = _groups.at(group_index);
            auto& buffer = group.buffers.at(context_index).at(stage - 1);
            auto& size = group.buffer_sizes.at(context_index).at(stage - 1);
            bool full = size >= batch_size(context_index);
            if (!full && free == 0) {
                continue;
            }
            std::size_t job_count =
                std::clamp<std::size_t>(std::min(share, free), 1, buffer.size());
            free -= std::min(free, job_count);
            // split at the chunk boundaries into jobs of similar size
            std::vector<ChunkPtr> chunks;
            std::size_t job_events = 0, done_events = 0;
            for (std::size_t job_index = 0; auto& chunk : buffer) {
                chunks.push_back(chunk);
                job_events += chunk->size();
                if (job_events * (job_count - job_index) >= size - done_events ||
                    &chunk == &buffer.back()) {
                    done_events += job_events;
                    job_events = 0;
                    start_job(
                        result_queue,
                        group_index,
                        context_index,
                        stage,
                        std::move(chunks)
                    );
                    chunks.clear();
                    ++job_index;
                    ++started;
                }
            }
            buffer.clear();
            size = 0;
        }
    }
    return started;
}

void WeightingPipeline::start_job(
    ResultQueue& result_queue,
    std::size_t group_index,
    std::size_t context_index,
    std::size_t stage,
    std::vector<ChunkPtr> chunks
) {
    std::size_t job_id = job_id_flag | _job_counter++;
    auto& job =
        _jobs
            .emplace(
                job_id,
                Job{
                    .group = group_index,
                    .stage = stage,
                    .context_index = context_index,
                    .chunks = std::move(chunks),
                }
            )
            .first->second;
    ++_context_job_counts.at(context_index);
    result_queue.submit(
        _contexts.at(context_index)->thread_pool(), job_id, [this, &job] {
            run_job(job);
        }
    );
}

void WeightingPipeline::gather_events(Chunk& chunk, ContextRuntimes& runtimes) {
    auto& source = *chunk.source;
    std::size_t count = chunk.size();
    if (!chunk.keep && chunk.offset == 0 && source.at(0).size(0) == count) {
        chunk.events = source;
    } else {
        chunk.events = gather_rows(source, {&chunk}, chunk.context_index, runtimes);
    }
    chunk.source.reset();
    chunk.keep.reset();
    chunk.offset = 0;
}

TensorVec WeightingPipeline::gather_rows(
    const TensorVec& source,
    const std::vector<Chunk*>& chunks,
    std::size_t context_index,
    ContextRuntimes& runtimes
) {
    std::size_t count = 0;
    for (auto chunk : chunks) {
        count += chunk->size();
    }
    Tensor indices(DataType::dt_int32, {count});
    auto view = indices.view<me_int_t, 1>();
    for (std::size_t row = 0; auto chunk : chunks) {
        for (std::size_t i = 0; i < chunk->size(); ++i, ++row) {
            view[row] = static_cast<me_int_t>(
                chunk->offset + (chunk->keep ? chunk->keep->at(i) : i)
            );
        }
    }
    auto device = _contexts.at(context_index)->device();
    if (device != cpu_device()) {
        Tensor device_indices(DataType::dt_int32, {count}, device);
        device_indices.copy_from(indices);
        indices = device_indices;
    }
    TensorVec args{indices};
    args.insert(args.end(), source.begin(), source.end());
    return runtimes.gather->run(args);
}

RuntimePtr& WeightingPipeline::concat_runtime(
    std::size_t group_index, std::size_t context_index, std::size_t part_count
) {
    std::lock_guard lock(_concat_mutex);
    auto& group = _groups.at(group_index);
    auto& runtime = group.runtimes.at(context_index).concats[part_count];
    if (!runtime) {
        runtime = build_runtime(
            concat_function(group.types, part_count), _contexts.at(context_index), false
        );
    }
    return runtime;
}

void WeightingPipeline::run_job(Job& job) {
    auto& group = _groups.at(job.group);
    auto& runtimes = group.runtimes.at(job.context_index);
    if (job.stage > group.settings.size()) {
        // after the last step, the events go to the CPU to be written
        for (auto& chunk : job.chunks) {
            gather_events(*chunk, runtimes);
            for (auto& tensor : chunk->events) {
                chunk->cpu_events.push_back(tensor.cpu());
            }
            chunk->events.clear();
        }
        return;
    }

    // one gather per source, then one concatenation of the sources
    std::vector<std::vector<ChunkPtr>> source_chunks;
    for (auto& chunk : job.chunks) {
        auto found =
            std::find_if(source_chunks.begin(), source_chunks.end(), [&](auto& chunks) {
                return chunks.at(0)->source == chunk->source;
            });
        if (found == source_chunks.end()) {
            source_chunks.push_back({chunk});
        } else {
            found->push_back(chunk);
        }
    }
    job.chunks.clear();
    TensorVec events, concat_args;
    for (auto& chunks : source_chunks) {
        auto& first = *chunks.at(0);
        auto& source = *first.source;
        TensorVec part;
        if (chunks.size() == 1 && !first.keep && first.offset == 0 &&
            source.at(0).size(0) == first.size()) {
            part = source;
        } else {
            std::vector<Chunk*> chunk_ptrs;
            for (auto& chunk : chunks) {
                chunk_ptrs.push_back(chunk.get());
            }
            part = gather_rows(source, chunk_ptrs, job.context_index, runtimes);
        }
        concat_args.insert(concat_args.end(), part.begin(), part.end());
        for (auto& chunk : chunks) {
            chunk->source.reset();
            chunk->keep.reset();
            chunk->offset = 0;
            // the chunks in the order of the events
            job.chunks.push_back(chunk);
        }
    }
    events = source_chunks.size() == 1
        ? std::move(concat_args)
        : concat_runtime(job.group, job.context_index, source_chunks.size())
              ->run(concat_args);

    TensorVec step_args{
        events.at(group.momenta), events.at(group.alpha_s), events.at(group.flavor)
    };
    if (group.multiple_matrix_elements.at(job.stage - 1)) {
        step_args.push_back(events.at(group.me_index));
    }
    step_args.push_back(events.at(group.color));
    step_args.push_back(events.at(group.helicity));
    step_args.push_back(events.at(group.diagram));
    step_args.push_back(events.at(group.random));
    TensorVec outputs = runtimes.steps.at(job.stage - 1)->run(step_args);
    auto matrix_elements = to_vector(outputs.at(0).cpu());
    auto random = to_vector(outputs.at(4).cpu());
    events.at(group.color) = outputs.at(1);
    events.at(group.helicity) = outputs.at(2);
    events.at(group.diagram) = outputs.at(3);

    auto shared_events = std::make_shared<TensorVec>(std::move(events));
    std::size_t offset = 0;
    for (auto& chunk : job.chunks) {
        std::size_t count = chunk->size();
        chunk->source = shared_events;
        chunk->offset = offset;
        chunk->step_matrix_elements.assign(
            matrix_elements.begin() + offset, matrix_elements.begin() + offset + count
        );
        chunk->step_random.assign(
            random.begin() + offset, random.begin() + offset + count
        );
        offset += count;
    }
}

void WeightingPipeline::commit_job(std::size_t job_id) {
    auto node = _jobs.extract(job_id);
    auto& job = node.mapped();
    --_context_job_counts.at(job.context_index);
    for (auto& chunk : job.chunks) {
        set_ready(chunk, job.stage);
    }
}

void WeightingPipeline::reset() {
    _jobs.clear();
    std::fill(_context_job_counts.begin(), _context_job_counts.end(), 0);
    for (auto& group : _groups) {
        for (auto& buffers : group.buffers) {
            for (auto& buffer : buffers) {
                buffer.clear();
            }
        }
        for (auto& sizes : group.buffer_sizes) {
            std::fill(sizes.begin(), sizes.end(), 0);
        }
    }
    for (auto& channel_ready : _ready) {
        for (auto& ready : channel_ready) {
            ready.clear();
        }
    }
    // the sequences continue, as the dropped chunks are never committed
    for (std::size_t channel_index = 0; channel_index < _channels.size();
         ++channel_index) {
        for (auto& next_sequence : _next_sequences.at(channel_index)) {
            next_sequence = _channel_sequences.at(channel_index);
        }
    }
    _chunk_count = 0;
}
