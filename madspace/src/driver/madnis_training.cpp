#include "madspace/driver/madnis_training.hpp"

#include "madspace/phasespace/batch_sampler.hpp"

using namespace madspace;

MadnisTraining::MadnisTraining(
    ContextPtr generator_context,
    ContextPtr optimizer_context,
    const Config& config,
    const std::vector<std::shared_ptr<Integrand>>& integrands,
    const std::optional<ChannelWeightNetwork>& cwnet
) :
    _generator_context(generator_context),
    _optimizer_context(optimizer_context),
    _config(config),
    _channels(integrands.size()),
    _cwnet(cwnet) {
    for (std::size_t index = 0;
         auto [integrand, channel] : zip(integrands, _channels)) {
        channel.index = index;
        channel.integrand = integrand;
        channel.integrand_prob = std::make_shared<IntegrandProbability>(*integrand);
        if (_arg_permutation.size() == 0) {
            auto& integ_args = channel.integrand->return_types().index_map();
            _arg_permutation.push_back(integ_args.at("weight"));
            _arg_permutation.push_back(integ_args.at("adaptive_prob"));
            if (cwnet) {
                _arg_permutation.push_back(integ_args.at("cwnet_input"));
                _arg_permutation.push_back(integ_args.at("channel_weights"));
                _arg_permutation.push_back(integ_args.at("channel_index"));
            }
            for (auto key : channel.integrand_prob->arg_types().keys()) {
                _arg_permutation.push_back(integ_args.at(key));
            }
        }
        auto& active_flavors = integrand->active_flavors();
        for (std::size_t flav_index : active_flavors) {
            if (_active_flavors_count.size() <= flav_index) {
                _active_flavors_count.resize(flav_index + 1);
            }
            ++_active_flavors_count.at(flav_index);
        }
        ++index;
    }
    build_runtimes_and_optimizer();
}

void MadnisTraining::train_step(std::size_t batch_index) {
    auto& gen_thread_pool = _generator_context->thread_pool();
    _abort_check_function();
    std::vector<std::size_t> channel_sizes = compute_channel_sizes();
    bool try_buffered =
        _config.buffer_capacity > 0 && batch_index % (_config.buffered_steps + 1) != 0;
    TensorVec training_batch;
    while (true) {
        start_generator_jobs(channel_sizes);
        if (try_buffered) {
            if (check_buffered_training_batch(channel_sizes)) {
                training_batch = build_buffered_training_batch(channel_sizes);
                break;
            }
            try_buffered = false;
        } else if (check_online_training_batch(channel_sizes)) {
            training_batch = build_online_training_batch(channel_sizes);
            break;
        }
        process_job_results(gen_thread_pool.wait_multiple());
    }
    TensorVec results = _optimizer->step(training_batch);
    update_history(results, channel_sizes);
    if (_channels.size() > 0 && _cwnet &&
        (batch_index + 1) % _config.channel_dropping_interval == 0) {
        std::vector<std::size_t> job_ids;
        while ((job_ids = gen_thread_pool.wait_multiple()).size() != 0) {
            process_job_results(job_ids);
        }
        drop_channels();
    }
    if (batch_index ==
        static_cast<std::size_t>(
            (1 - _config.fixed_cwnet_fraction) * _config.batches
        )) {
        std::vector<std::size_t> job_ids;
        while ((job_ids = gen_thread_pool.wait_multiple()).size() != 0) {
            process_job_results(job_ids);
        }
        freeze_cwnet();
    }
}

std::vector<std::size_t> MadnisTraining::active_channels() const {
    std::vector<std::size_t> indices(_channels.size());
    for (auto [channel, index] : zip(_channels, indices)) {
        index = channel.index;
    }
    return indices;
}

double MadnisTraining::average_loss() const {
    double loss_sum = 0;
    std::size_t loss_count = 0, nan_count = 0;
    for (double loss : _loss_history) {
        if (!std::isnan(loss)) {
            loss_sum += loss;
            ++loss_count;
        } else {
            ++nan_count;
        }
    }
    return loss_sum / loss_count;
}

void MadnisTraining::build_runtimes_and_optimizer() {
    std::vector<std::shared_ptr<FunctionGenerator>> functions;
    functions.reserve(_channels.size());
    _multi_channel_generator.reset();
    for (auto& channel : _channels) {
        functions.push_back(channel.integrand_prob);
        channel.generator_runtime.reset();
    }
    Function madnis_func =
        MadnisLoss(functions, _cwnet, _config.softclip_threshold).function();
    if (_optimizer) {
        _optimizer->replace_function(madnis_func);
    } else {
        _optimizer.emplace(
            madnis_func,
            _optimizer_context,
            _config.learning_rate,
            _config.lr_schedule,
            _config.batches,
            _config.adam_beta1,
            _config.adam_beta2,
            _config.adam_eps
        );
    }
    _generator_params =
        _generator_context->reallocate_globals_contiguously(_optimizer->param_names());
    if (_generator_context->device()->device_type() == DeviceType::cpu) {
        for (auto& channel : _channels) {
            channel.generator_runtime =
                build_runtime(channel.integrand->function(), _generator_context, false);
            if (_config.buffer_capacity > 0) {
                channel.unweighter_runtime = build_runtime(
                    BufferUnweighter(
                        channel.integrand->return_types(),
                        _config.buffer_unweighting_quantile
                    )
                        .function(),
                    _generator_context,
                    false
                );
            }
        }
    } else {
        std::vector<std::shared_ptr<Integrand>> integrands;
        integrands.reserve(_channels.size());
        for (auto& channel : _channels) {
            integrands.push_back(channel.integrand);
        }
        _multi_channel_generator = build_runtime(
            MultiChannelIntegrand(integrands, true).function(),
            _generator_context,
            false
        );
        if (_config.buffer_capacity > 0) {
            std::vector<std::shared_ptr<FunctionGenerator>> buf_unw_funcs;
            for (auto& channel : _channels) {
                buf_unw_funcs.push_back(
                    std::make_shared<BufferUnweighter>(
                        channel.integrand->return_types(),
                        _config.buffer_unweighting_quantile
                    )
                );
            }
            _multi_channel_unweighter = build_runtime(
                MultiChannelFunction(buf_unw_funcs, true).function(),
                _generator_context,
                false
            );
        }
    }
    if (_config.buffer_capacity > 0) {
        std::vector<NamedVector<Type>> types;
        for (auto& channel : _channels) {
            auto& chan_types = types.emplace_back();
            auto& ret_types_named = channel.integrand->return_types();
            auto ret_keys = ret_types_named.keys();
            auto& ret_types = ret_types_named.values();
            bool buffer_needs_init = channel.buffer.tensors.size() == 0;
            for (std::size_t index : _arg_permutation) {
                auto& type = ret_types.at(index);
                chan_types.push_back(ret_keys.at(index), type);
                if (buffer_needs_init) {
                    Sizes full_shape(type.shape.size() + 1);
                    full_shape[0] = _config.buffer_capacity;
                    std::copy(
                        type.shape.begin(), type.shape.end(), full_shape.begin() + 1
                    );
                    channel.buffer.tensors.emplace_back(
                        type.dtype, full_shape, _optimizer_context->device()
                    );
                }
            }
        }
        _multi_channel_sampler =
            build_runtime(BatchSampler(types).function(), _optimizer_context, false);
    }
}

std::vector<std::size_t> MadnisTraining::compute_channel_sizes() {
    std::size_t chan_count = _channels.size();
    std::size_t batch_size =
        _config.batch_size_per_channel * chan_count + _config.batch_size_offset;
    std::vector<std::size_t> sizes;
    sizes.reserve(chan_count);
    if (_channels.at(0).integration_history.size() <
        _config.integration_history_length) {
        sizes.assign(
            chan_count, std::ceil(static_cast<double>(batch_size) / chan_count)
        );
        return sizes;
    }

    std::vector<double> fractions;
    fractions.reserve(chan_count);
    double stddev_sum = 0.;
    for (auto& channel : _channels) {
        std::size_t total_count = 0;
        double total_var = 0.0;
        for (auto [count, abs_mean, var] : channel.integration_history) {
            if (!std::isnan(var)) {
                total_count += count;
                total_var += count * var;
            }
        }
        double stddev = std::sqrt(total_var / total_count);
        stddev_sum += stddev;
        fractions.push_back(stddev);
    }

    double frac_sum = 0.0;
    double uniform_per_chan = _config.uniform_channel_ratio / chan_count;
    for (double& fraction : fractions) {
        fraction = std::max(fraction / stddev_sum - uniform_per_chan, 0.0);
        frac_sum += fraction;
    }

    double weighted_part = 1.0 - _config.uniform_channel_ratio;
    for (double& fraction : fractions) {
        sizes.push_back(
            std::ceil(
                batch_size * (uniform_per_chan + weighted_part * fraction / frac_sum)
            )
        );
    }
    return sizes;
}

void MadnisTraining::start_generator_jobs(const std::vector<std::size_t>& counts) {
    if (_running_jobs.size() > 0) {
        return;
    }
    _generator_params.copy_from(_optimizer->parameters());
    std::size_t chan_count = counts.size();
    bool is_gpu = _generator_context->device()->device_type() != DeviceType::cpu;
    std::size_t batch_size = is_gpu
        ? _config.gpu_generator_batch_granularity
        : _config.cpu_generator_batch_size;
    std::vector<std::size_t> missing_batch_counts(chan_count);
    std::vector<std::size_t> target_batch_counts(chan_count);
    for (auto [channel, count, missing_batch_count, target_batch_count] :
         zip(_channels, counts, missing_batch_counts, target_batch_counts)) {
        std::size_t target_count = count * _config.generator_target_size_factor;
        missing_batch_count = count > channel.sample_count
            ? (count - channel.sample_count + batch_size - 1) / batch_size
            : 0;
        target_batch_count = target_count > channel.sample_count
            ? (target_count - channel.sample_count + batch_size - 1) / batch_size
            : 0;
    }
    std::size_t available_jobs = _generator_context->thread_pool().thread_count();
    std::vector<std::size_t> channel_sizes;
    std::size_t gpu_subbatches =
        (_config.gpu_generator_batch_size + _config.gpu_generator_batch_granularity -
         1) /
        _config.gpu_generator_batch_granularity;
    if (is_gpu) {
        available_jobs *= gpu_subbatches;
        channel_sizes.resize(chan_count, 0);
    }

    for (std::size_t chan_index = 0;
         auto [missing_batch_count, target_batch_count] :
         zip(missing_batch_counts, target_batch_counts)) {
        while (missing_batch_count > 0 && available_jobs > 0) {
            --available_jobs;
            if (is_gpu) {
                channel_sizes.at(chan_index) += batch_size;
                if (available_jobs % gpu_subbatches == 0) {
                    start_multi_job(channel_sizes);
                    channel_sizes.assign(chan_count, 0);
                }
            } else {
                start_single_job(chan_index, batch_size);
            }
            --missing_batch_count;
            --target_batch_count;
        }
        ++chan_index;
        if (available_jobs == 0) {
            break;
        }
    }
    std::vector<std::size_t> chan_indices(counts.size());
    std::iota(chan_indices.begin(), chan_indices.end(), 0);
    std::sort(chan_indices.begin(), chan_indices.end(), [&](auto i, auto j) {
        return target_batch_counts.at(i) > target_batch_counts.at(j);
    });
    for (std::size_t index = 0; available_jobs > 0;) {
        std::size_t chan_index = chan_indices.at(index);
        std::size_t& count = target_batch_counts.at(chan_index);
        if (count == 0) {
            break;
        }
        --available_jobs;
        if (is_gpu) {
            channel_sizes.at(chan_index) += batch_size;
            if (available_jobs % gpu_subbatches == 0) {
                start_multi_job(channel_sizes);
                channel_sizes.assign(chan_count, 0);
            }
        } else {
            start_single_job(chan_index, batch_size);
        }
        --count;
        if (chan_index == chan_count - 1) {
            index = 0;
        } else {
            std::size_t next_count = target_batch_counts.at(chan_indices.at(index + 1));
            if (count < next_count) {
                ++index;
            }
        }
    }
    if (available_jobs > 0 && is_gpu) {
        start_multi_job(channel_sizes);
    }
}

TensorVec MadnisTraining::permute_tensors(const TensorVec& tensors) const {
    TensorVec ret;
    ret.reserve(tensors.size());
    for (std::size_t index : _arg_permutation) {
        ret.push_back(tensors.at(index));
    }
    return ret;
}

void MadnisTraining::start_single_job(
    std::size_t channel_index, std::size_t batch_size
) {
    std::size_t job_id = _job_id;
    ++_job_id;
    auto& job = std::get<0>(_running_jobs.emplace(job_id, SampleJob{}))->second;
    _generator_context->thread_pool().submit(
        [this, channel_index, batch_size, job_id, &job]() {
            auto& channel = _channels.at(channel_index);
            auto samples = channel.generator_runtime->run({Tensor({batch_size})});
            job.samples.tensors = permute_tensors(samples);
            job.samples.size = samples.at(0).size(0);
            job.samples.channel_index = channel_index;
            if (channel.unweighter_runtime) {
                auto unw_samples = channel.unweighter_runtime->run(samples);
                job.unweighted_samples.tensors = permute_tensors(unw_samples);
                job.unweighted_samples.size = unw_samples.at(0).size(0);
                job.unweighted_samples.channel_index = channel_index;
            }
            return job_id;
        }
    );
}

void MadnisTraining::start_multi_job(const std::vector<std::size_t> batch_sizes) {
    std::size_t job_id = _job_id;
    ++_job_id;
    auto& job = std::get<0>(_running_jobs.emplace(job_id, SampleJob{}))->second;
    _generator_context->thread_pool().submit([this, batch_sizes, job_id, &job]() {
        auto samples = _multi_channel_generator->run({Tensor(batch_sizes)});
        job.samples.tensors = permute_tensors(samples);
        job.samples.channel_sizes = samples.back().batch_sizes();
        if (_multi_channel_unweighter) {
            auto unw_samples = _multi_channel_unweighter->run(samples);
            job.unweighted_samples.tensors = permute_tensors(unw_samples);
            job.unweighted_samples.channel_sizes = unw_samples.back().batch_sizes();
        }
        return job_id;
    });
}

bool MadnisTraining::check_online_training_batch(
    const std::vector<std::size_t>& channel_sizes
) {
    for (auto [channel, count] : zip(_channels, channel_sizes)) {
        if (count > channel.sample_count) {
            return false;
        }
    }
    return true;
}

bool MadnisTraining::check_buffered_training_batch(
    const std::vector<std::size_t>& channel_sizes
) {
    for (auto [channel, count] : zip(_channels, channel_sizes)) {
        if (count > channel.buffer.size ||
            channel.buffer.size < _config.minimum_buffer_size) {
            return false;
        }
    }
    return true;
}

TensorVec
MadnisTraining::build_online_training_batch(const std::vector<size_t>& counts) {
    TensorVec training_batch;
    for (auto [channel, count] : zip(_channels, counts)) {
        auto& first_batch = channel.sample_batches.at(0);
        std::size_t consumed_batches = 0;
        if (first_batch.size - first_batch.consumed_count >= count) {
            for (auto& tensor : first_batch.tensors) {
                training_batch.push_back(tensor.slice(
                    0, first_batch.consumed_count, first_batch.consumed_count + count
                ));
            }
            if (first_batch.size - first_batch.consumed_count == count) {
                consumed_batches = 1;
            } else {
                first_batch.consumed_count += count;
            }
            channel.sample_count -= count;
        } else {
            for (auto& tensor : first_batch.tensors) {
                Sizes shape = tensor.shape();
                shape[0] = count;
                training_batch.emplace_back(tensor.dtype(), shape, tensor.device());
            }
            std::size_t remaining_count = count;
            for (auto& batch : channel.sample_batches) {
                std::size_t batch_size = batch.size - batch.consumed_count;
                std::size_t offset = count - remaining_count;
                if (batch_size >= remaining_count) {
                    for (auto [tensor_out, tensor_in] :
                         zip(std::span(
                                 training_batch.end() - batch.tensors.size(),
                                 training_batch.end()
                             ),
                             batch.tensors)) {
                        tensor_out.slice(0, offset, offset + remaining_count)
                            .copy_from(tensor_in.slice(
                                0,
                                batch.consumed_count,
                                batch.consumed_count + remaining_count
                            ));
                    }
                    batch.consumed_count += remaining_count;
                    channel.sample_count -= remaining_count;
                    break;
                } else {
                    for (auto [tensor_out, tensor_in] :
                         zip(std::span(
                                 training_batch.end() - batch.tensors.size(),
                                 training_batch.end()
                             ),
                             batch.tensors)) {
                        tensor_out.slice(0, offset, offset + batch_size)
                            .copy_from(
                                tensor_in.slice(0, batch.consumed_count, batch.size)
                            );
                    }
                    remaining_count -= batch_size;
                    channel.sample_count -= batch_size;
                    ++consumed_batches;
                }
            }
        }
        channel.sample_batches.erase(
            channel.sample_batches.begin(),
            channel.sample_batches.begin() + consumed_batches
        );
    }
    return training_batch;
}

TensorVec
MadnisTraining::build_buffered_training_batch(const std::vector<size_t>& counts) {
    TensorVec args;
    for (auto& channel : _channels) {
        for (auto& tensor : channel.buffer.tensors) {
            args.push_back(tensor.slice(0, 0, channel.buffer.size));
        }
    }
    args.emplace_back(counts);
    return _multi_channel_sampler->run(args);
}

void MadnisTraining::process_job_results(const std::vector<std::size_t>& job_ids) {
    for (auto job_id : job_ids) {
        auto job = std::move(_running_jobs.extract(job_id).mapped());
        if (job.samples.channel_sizes.size() == 0) {
            auto& channel = _channels.at(job.samples.channel_index);
            channel.sample_count += job.samples.size;
            channel.sample_batches.push_back(std::move(job.samples));
            if (job.unweighted_samples.size > 0) {
                buffer_store(channel, job.unweighted_samples);
            }
        } else {
            std::size_t offset = 0, unw_offset = 0, chan_index = 0;
            SampleBatch chan_unweighted_samples;
            for (auto [channel, chan_size] :
                 zip(_channels, job.samples.channel_sizes)) {
                if (chan_size == 0) {
                    ++chan_index;
                    continue;
                }
                channel.sample_count += chan_size;
                channel.sample_batches.emplace_back();
                auto& batch = channel.sample_batches.back();
                batch.tensors.reserve(job.samples.tensors.size());
                for (auto& tensor : job.samples.tensors) {
                    batch.tensors.push_back(
                        tensor.slice(0, offset, offset + chan_size)
                    );
                }
                if (job.unweighted_samples.channel_sizes.size() > 0) {
                    std::size_t unw_chan_size =
                        job.unweighted_samples.channel_sizes.at(chan_index);
                    chan_unweighted_samples.tensors.clear();
                    chan_unweighted_samples.size = unw_chan_size;
                    for (auto& tensor : job.unweighted_samples.tensors) {
                        chan_unweighted_samples.tensors.push_back(
                            tensor.slice(0, unw_offset, unw_offset + unw_chan_size)
                        );
                    }
                    buffer_store(channel, chan_unweighted_samples);
                    unw_offset += unw_chan_size;
                }
                batch.size = chan_size;
                offset += chan_size;
                ++chan_index;
            }
        }
    }
}

void MadnisTraining::buffer_store(ChannelData& channel, SampleBatch& samples) {
    std::size_t empty_count = _config.buffer_capacity - channel.buffer.size;

    std::size_t size = std::min(samples.size, _config.buffer_capacity);
    std::size_t end_index = channel.buffer.consumed_count + size;
    std::size_t store_begin1, store_end1, load_begin1, load_end1;
    std::size_t store_begin2 = 0, store_end2 = 0, load_begin2 = 0, load_end2 = 0;
    if (end_index < _config.buffer_capacity) {
        store_begin1 = channel.buffer.consumed_count;
        store_end1 = end_index;
        load_begin1 = 0;
        load_end1 = size;
    } else {
        store_begin1 = channel.buffer.consumed_count;
        store_end1 = _config.buffer_capacity;
        load_begin1 = 0;
        load_end1 = _config.buffer_capacity - channel.buffer.consumed_count;
        store_begin2 = 0;
        store_end2 = end_index - _config.buffer_capacity;
        load_begin2 = _config.buffer_capacity - channel.buffer.consumed_count;
        load_end2 = size;
    }
    channel.buffer.consumed_count = end_index % _config.buffer_capacity;
    channel.buffer.size = std::min(channel.buffer.size + size, _config.buffer_capacity);
    for (auto [buf, tensor] : zip(channel.buffer.tensors, samples.tensors)) {
        buf.slice(0, store_begin1, store_end1)
            .copy_from(tensor.slice(0, load_begin1, load_end1));
        if (load_end2 - load_begin2 > 0) {
            buf.slice(0, store_begin2, store_end2)
                .copy_from(tensor.slice(0, load_begin2, load_end2));
        }
    }
}

void MadnisTraining::update_history(
    const TensorVec& results, const std::vector<std::size_t>& counts
) {
    Tensor loss_cpu = results.at(0).cpu();
    Tensor abs_means_cpu = results.at(1).cpu();
    Tensor variances_cpu = results.at(2).cpu();
    double loss = loss_cpu.view<double, 1>()[0];
    if (_loss_history.size() < _config.log_interval) {
        _loss_history.push_back(loss);
    } else {
        _loss_history.at(_loss_history_index) = loss;
    }
    if (++_loss_history_index == _config.log_interval) {
        _loss_history_index = 0;
    }

    auto abs_means = abs_means_cpu.view<double, 2>()[0];
    auto variances = variances_cpu.view<double, 2>()[0];
    if (abs_means.size() != _channels.size() || variances.size() != _channels.size()) {
        throw std::logic_error("wrong channel count returned");
    }
    for (std::size_t i = 0; auto [channel, count] : zip(_channels, counts)) {
        if (channel.integration_history.size() < _config.integration_history_length) {
            channel.integration_history.push_back({count, abs_means[i], variances[i]});
        } else {
            channel.integration_history.at(channel.history_index) = {
                count, abs_means[i], variances[i]
            };
        }
        if (++channel.history_index == _config.log_interval) {
            channel.history_index = 0;
        }
        ++i;
    }
}

void MadnisTraining::drop_channels() {
    std::vector<double> abs_means;
    abs_means.reserve(_channels.size());
    double abs_mean_sum = 0.;
    for (auto& channel : _channels) {
        std::size_t chan_count = 0;
        double chan_abs_mean = 0.0;
        for (auto [count, abs_mean, var] : channel.integration_history) {
            if (!std::isnan(abs_mean)) {
                chan_count += count;
                chan_abs_mean += count * abs_mean;
            }
        }
        abs_mean_sum += chan_abs_mean;
        abs_means.push_back(chan_abs_mean);
    }
    std::vector<std::size_t> indices(_channels.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto i, auto j) {
        return abs_means.at(i) < abs_means.at(j);
    });

    Tensor active_mask_glob = _optimizer_context->global(_cwnet.value().mask_name());
    Tensor active_mask = active_mask_glob.cpu();
    auto mask_view = active_mask.view<double, 2>()[0];

    double drop_sum = 0.;
    std::size_t drop_count = 0;
    for (std::size_t chan_index : indices) {
        drop_sum += abs_means.at(chan_index);
        if (drop_sum / abs_mean_sum > _config.channel_dropping_threshold) {
            break;
        }
        auto& channel = _channels.at(chan_index);

        if (_active_flavors_count.size() > 0) {
            auto& active_flavors = channel.integrand->active_flavors();
            if (std::any_of(
                    active_flavors.begin(),
                    active_flavors.end(),
                    [&](std::size_t flav_index) {
                        return _active_flavors_count.at(flav_index) == 0;
                    }
                )) {
                // cannot drop this channel because one of its flavors is not
                // available in any other channel
                continue;
            }
            for (std::size_t flav_index : active_flavors) {
                --_active_flavors_count.at(flav_index);
            }
        }

        for (me_int_t index : channel.integrand->channel_indices()) {
            if (index < 0 || index > mask_view.size()) {
                throw std::out_of_range("channel index out of bounds");
            }
            mask_view[index] = 0;
        }
        channel.integrand = nullptr;
        ++drop_count;
    }
    if (drop_count == 0) {
        return;
    }
    _channels.erase(
        std::remove_if(
            _channels.begin(),
            _channels.end(),
            [&](auto& channel) { return !channel.integrand; }
        ),
        _channels.end()
    );
    if (_optimizer_context->device()->device_type() != DeviceType::cpu) {
        active_mask_glob.copy_from(active_mask);
    }
    _generator_context->global(_cwnet.value().mask_name()).copy_from(active_mask);
    build_runtimes_and_optimizer();
}

void MadnisTraining::freeze_cwnet() {
    if (!_cwnet) {
        return;
    }
    for (auto& name : _cwnet->mlp().global_names()) {
        _optimizer_context->set_global_requires_grad(name, false);
    }
    build_runtimes_and_optimizer();
}

MultiMadnisTraining::MultiMadnisTraining(
    ContextPtr generator_context,
    ContextPtr optimizer_context,
    const MadnisTraining::Config& config,
    const nested_vector2<std::shared_ptr<Integrand>>& integrands,
    const std::vector<std::optional<ChannelWeightNetwork>>& cwnets
) :
    _config(config) {
    _subprocesses.reserve(integrands.size());
    for (auto [integ, cwnet] : zip(integrands, cwnets)) {
        _subprocesses.emplace_back(
            generator_context, optimizer_context, config, integ, cwnet
        );
    }
}

void MultiMadnisTraining::train() {
    print_progress_init();
    _start_time = std::chrono::steady_clock::now();
    _start_cpu_microsec = cpu_time_microsec();
    for (std::size_t subproc_index = 0; auto& subproc : _subprocesses) {
        for (std::size_t batch_index = 0; batch_index < _config.batches;
             ++batch_index) {
            subproc.train_step(batch_index);
            double loss = subproc.average_loss();
            std::size_t chan_count = subproc.active_channel_count();
            print_progress_update(subproc_index, batch_index, loss, chan_count);
        }
        ++subproc_index;
    }
}

nested_vector2<std::size_t> MultiMadnisTraining::active_channels() const {
    nested_vector2<std::size_t> ret;
    ret.reserve(_subprocesses.size());
    for (auto& subproc : _subprocesses) {
        ret.push_back(subproc.active_channels());
    }
    return ret;
}

void MultiMadnisTraining::print_progress_init() {
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != Verbosity::pretty) {
        Logger::info("training started");
        return;
    }

    if (_subprocesses.size() > 1) {
        _pretty_box_lower =
            PrettyBox("Subprocesses", _subprocesses.size() + 1, {12, 12, 12, 0});
        _pretty_box_lower.set_row(0, {"Subprocess", "Loss", "Channels", "Batch"});
        _pretty_box_upper =
            PrettyBox("MadNIS training", 2, {18, 0}, _pretty_box_lower.line_count());
        _pretty_box_upper.set_column(0, {"Subprocesses:", "Run time:"});
        _pretty_box_upper.print_first();
        _pretty_box_lower.print_first();
    } else {
        _pretty_box_upper = PrettyBox("MadNIS training", 4, {18, 0});
        _pretty_box_upper.set_column(0, {"Batch:", "Loss:", "Channels:", "Run time:"});
        _pretty_box_upper.print_first();
    }
}

void MultiMadnisTraining::print_progress_update(
    std::size_t subproc_index,
    std::size_t batch_index,
    double loss,
    std::size_t chan_count
) {
    using namespace std::chrono_literals;
    if (_config.verbosity == Verbosity::log) {
        if ((batch_index + 1) % _config.log_interval != 0) {
            return;
        }
        auto now = std::chrono::steady_clock::now();
        Logger::info(
            std::format(
                "training, subproc: {} / {}, batch: {} / {}, loss: {:.4f}, channels: "
                "{}, time: {:%H:%M:%S}",
                subproc_index + 1,
                _subprocesses.size(),
                batch_index + 1,
                _config.batches,
                loss,
                chan_count,
                std::chrono::round<std::chrono::seconds>(now - _start_time)
            )
        );
        if (batch_index + 1 == _config.batches) {
            double wall_time_sec = (now - _start_time) / 1.0s;
            double cpu_time_sec = (cpu_time_microsec() - _start_cpu_microsec) / 1e6;
            Logger::info(
                std::format(
                    "training done, {}", format_run_time(wall_time_sec, cpu_time_sec)
                )
            );
        }
    } else if (_config.verbosity == Verbosity::pretty) {
        auto now = std::chrono::steady_clock::now();
        if (now - _last_print_time < 0.1s && batch_index + 1 != _config.batches) {
            return;
        }
        _last_print_time = now;

        std::size_t subproc_count = _subprocesses.size();
        bool is_last_batch = batch_index + 1 == _config.batches;
        bool is_done = is_last_batch && subproc_index == subproc_count - 1;

        std::string time_str;
        if (is_done) {
            double wall_time_sec = (now - _start_time) / 1.0s;
            double cpu_time_sec = (cpu_time_microsec() - _start_cpu_microsec) / 1e6;
            time_str = format_run_time(wall_time_sec, cpu_time_sec);
        } else {
            time_str = std::format(
                "{:%H:%M:%S}",
                std::chrono::round<std::chrono::seconds>(now - _start_time)
            );
        }
        std::string batch_str =
            std::format("{} / {}", batch_index + 1, _config.batches);

        if (subproc_count == 1) {
            std::string progress_bar = is_done
                ? ""
                : format_progress(
                      static_cast<double>(batch_index + 1) / _config.batches, 52
                  );
            _pretty_box_upper.set_column(
                1,
                {std::format("{:<15} {}", batch_str, progress_bar),
                 std::format("{:>.4f}", loss),
                 std::format("{}", chan_count),
                 time_str}
            );
            _pretty_box_upper.print_update();
        } else {
            std::string progress_bar, progress_bar_all;
            std::string subproc_str =
                std::format("{} / {}", subproc_index, subproc_count);
            if (!is_last_batch) {
                progress_bar = format_progress(
                    static_cast<double>(batch_index + 1) / _config.batches, 34
                );
                progress_bar_all = format_progress(
                    static_cast<double>(
                        subproc_index * _config.batches + batch_index + 1
                    ) / (subproc_count * _config.batches),
                    52
                );
            } else if (!is_done) {
                progress_bar_all = format_progress(
                    static_cast<double>((subproc_index + 1) * _config.batches + 1) /
                        (subproc_count * _config.batches),
                    52
                );
            }
            _pretty_box_upper.set_column(
                1, {std::format("{:<15} {}", subproc_str, progress_bar_all), time_str}
            );
            _pretty_box_lower.set_row(
                subproc_index + 1,
                {std::format("{}", subproc_index),
                 std::format("{:>.4f}", loss),
                 std::format("{}", chan_count),
                 std::format("{:<15} {}", batch_str, progress_bar)}
            );
            _pretty_box_upper.print_update();
            _pretty_box_lower.print_update();
        }
    }
}
