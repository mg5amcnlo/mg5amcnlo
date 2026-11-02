#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/random.hpp"
#include "madspace/util.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {

int event_extra_flags(const std::unordered_map<std::string, std::size_t>& index_map) {
    int flags = 0;
    if (index_map.contains("fact_scale1")) {
        flags |= EventRecord::f_beam1;
    }
    if (index_map.contains("fact_scale2")) {
        flags |= EventRecord::f_beam2;
    }
    if (index_map.contains("partial_weight_product")) {
        flags |= EventRecord::f_partial_weights;
    }
    if (index_map.contains("subprocess_index")) {
        flags |= EventRecord::f_subproc_index;
    }
    if (index_map.contains("bias")) {
        flags |= EventRecord::f_bias;
    }
    if (index_map.contains("matrix_element")) {
        flags |= EventRecord::f_matrix_element;
    }
    return flags;
}

int particle_extra_flags(
    const std::unordered_map<std::string, std::size_t>& index_map
) {
    int flags = 0;
    if (index_map.contains("cluster_scales")) {
        flags |= ParticleRecord::f_clustering;
    }
    if (index_map.contains("alphas_scales")) {
        flags |= ParticleRecord::f_alphas_scales;
    }
    if (index_map.contains("ladder_x")) {
        flags |= ParticleRecord::f_pdf_ladder;
    }
    return flags;
}

DerivedSeed::SeedType generate_seed_type(bool is_survey, std::size_t survey_pass) {
    if (!is_survey) {
        return DerivedSeed::generator_generate;
    }
    return survey_pass == 0
        ? DerivedSeed::first_survey_generate
        : DerivedSeed::second_survey_generate;
}

DerivedSeed::SeedType unweight_seed_type(bool is_survey, std::size_t survey_pass) {
    if (!is_survey) {
        return DerivedSeed::generator_unweight;
    }
    return survey_pass == 0
        ? DerivedSeed::first_survey_unweight
        : DerivedSeed::second_survey_unweight;
}

// Role folded into job_index so integrand_channel and integrand_common get
// independent streams despite sharing seed_type/channel_index.
enum RngRole { rng_role_channel = 0, rng_role_common = 1, rng_role_count = 2 };

DerivedSeed generate_phase_seed(
    std::optional<std::uint64_t> seed,
    bool is_survey,
    std::size_t survey_pass,
    std::size_t job_index,
    std::size_t channel_index,
    RngRole role
) {
    return DerivedSeed(
        seed,
        generate_seed_type(is_survey, survey_pass),
        job_index * rng_role_count + role,
        channel_index
    );
}

} // namespace

ChannelEventGenerator::ChannelEventGenerator(
    const std::vector<ContextPtr>& contexts,
    const Integrand& integrand,
    const std::string& event_file,
    const std::string& weight_file,
    const GeneratorConfig& config,
    std::size_t subprocess_index,
    const std::string& name,
    const std::optional<ObservableHistograms>& histograms
) :
    _contexts(contexts),
    _status{
        .subprocess = subprocess_index,
        .name = name,
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
        .count_target = 1,
        .optimized = false,
        .done = false
    },
    _config(config),
    _event_layout_extra_flags(event_extra_flags(integrand.return_types().index_map())),
    _particle_layout_extra_flags(
        particle_extra_flags(integrand.return_types().index_map())
    ),
    _event_file_layout(
        EventRecord::layout(EventRecord::f_event_data | _event_layout_extra_flags),
        ParticleRecord::layout(
            ParticleRecord::f_particle_data | _particle_layout_extra_flags
        )
    ),
    _event_file(
        event_file,
        _event_file_layout,
        integrand.particle_count(),
        EventFile::create,
        true
    ),
    _weight_file(weight_file, weight_file_layout, 0, EventFile::create, true),
    _batch_size(config.start_batch_size),
    _particle_count(integrand.particle_count()),
    _integrand_channel_function(IntegrandChannelPart(integrand).function()),
    _integrand_common_function(IntegrandCommonPart(integrand).function()),
    _integrand_concat_function(IntegrandConcatenator(integrand).function()),
    _unweighter_function(
        Unweighter([&] {
            auto& ret_types = integrand.return_types();
            auto keys = ret_types.keys();
            std::size_t vegas_index = ret_types.index_map().at("random");
            return NamedVector<Type>(
                {keys.begin(), keys.begin() + vegas_index},
                {ret_types.begin(), ret_types.begin() + vegas_index}
            );
        }())
            .function()
    ) {
    if (integrand.madnis_training()) {
        throw std::invalid_argument(
            "Integrand must not be in madnis_training mode for event generation"
        );
    }
    init_used_globals();
    init_runtimes();
    init_field_indices();
    if (const auto& grid_name = integrand.vegas_grid_name(); grid_name) {
        _vegas_optimizer =
            VegasGridOptimizer(contexts, grid_name.value(), config.vegas_damping);
        VegasHistogram hist(integrand.vegas_dimension(), integrand.vegas_bin_count());
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.vegas_histogram = build_runtime(hist.function(), context, false);
        }
    }
    std::vector<std::string> prob_names;
    std::vector<std::size_t> option_counts;
    auto add_names = Overloaded{
        [&](const DiscreteSampler& sampler) {
            auto& names = sampler.prob_names();
            prob_names.insert(prob_names.end(), names.begin(), names.end());
            auto& opts = sampler.option_counts();
            option_counts.insert(option_counts.end(), opts.begin(), opts.end());
        },
        [](auto sampler) {}
    };
    std::visit(add_names, integrand.discrete_sym());
    std::visit(add_names, integrand.discrete_flavor());
    std::optional<DiscreteOptimizer> discrete_optimizer;
    RuntimePtr discrete_histogram = nullptr;
    if (prob_names.size() > 0) {
        discrete_optimizer = DiscreteOptimizer(contexts, prob_names);
        DiscreteHistogram hist(option_counts);
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.discrete_histogram =
                build_runtime(hist.function(), context, false);
        }
    }
    if (histograms) {
        _histogram_function = histograms.value().function();
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.observable_histograms =
                build_runtime(_histogram_function.value(), context, false);
        }
        for (auto& item : histograms.value().observables()) {
            _histograms.push_back({
                .name = item.observable.name(),
                .min = item.min,
                .max = item.max,
                .bin_values = std::vector<double>(item.bin_count + 2),
                .bin_errors = std::vector<double>(item.bin_count + 2),
            });
        }
    }
}

ChannelEventGenerator::ChannelEventGenerator(
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
) :
    _contexts(contexts),
    _status{
        .subprocess = subprocess_index,
        .name = name,
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
        .count_target = 1,
        .optimized = false,
        .done = false
    },
    _config(config),
    _event_layout_extra_flags(
        event_extra_flags(integrand_common_function.outputs().index_map())
    ),
    _particle_layout_extra_flags(
        particle_extra_flags(integrand_common_function.outputs().index_map())
    ),
    _event_file_layout(
        EventRecord::layout(EventRecord::f_event_data | _event_layout_extra_flags),
        ParticleRecord::layout(
            ParticleRecord::f_particle_data | _particle_layout_extra_flags
        )
    ),
    _event_file(
        event_file, _event_file_layout, particle_count, EventFile::create, true
    ),
    _weight_file(weight_file, weight_file_layout, 0, EventFile::create, true),
    _batch_size(config.start_batch_size),
    _particle_count(particle_count),
    _integrand_channel_function(integrand_channel_function),
    _integrand_common_function(integrand_common_function),
    _integrand_concat_function(integrand_concat_function),
    _unweighter_function(unweighter_function),
    _histograms(histograms) {
    init_used_globals();
    init_runtimes();
    init_field_indices();
    if (histogram_function) {
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.observable_histograms =
                build_runtime(_histogram_function.value(), context, false);
        }
    }
}

void ChannelEventGenerator::init_used_globals() {
    for (auto& item : _integrand_channel_function.globals()) {
        _used_globals.insert(item.first);
    }
    for (auto& item : _integrand_common_function.globals()) {
        _used_globals.insert(item.first);
    }
    for (auto& item : _integrand_concat_function.globals()) {
        _used_globals.insert(item.first);
    }
}

void ChannelEventGenerator::init_runtimes() {
    for (auto& context : _contexts) {
        _runtimes.push_back(
            {.integrand_channel =
                 build_runtime(_integrand_channel_function, context, false),
             .integrand_common =
                 build_runtime(_integrand_common_function, context, false),
             .integrand_concat =
                 build_runtime(_integrand_concat_function, context, false),
             .unweighter = build_runtime(_unweighter_function, context, false)}
        );
    }
}

void ChannelEventGenerator::init_field_indices() {
    auto index_map = _integrand_common_function.outputs().index_map();
    _field_indices.weight = index_map.at("weight");
    _field_indices.momenta = index_map.at("momenta");
    _field_indices.color_index = index_map.at("color_index");
    _field_indices.helicity_index = index_map.at("helicity_index");
    _field_indices.diagram_index = index_map.at("diagram_index");
    _field_indices.flavor_index = index_map.at("flavor_index");
    _field_indices.ren_scale = index_map.at("ren_scale");
    _field_indices.alpha_qcd = index_map.at("alpha_qcd");
    if (index_map.contains("fact_scale1")) {
        _field_indices.x1 = index_map.at("x1");
        _field_indices.fact_scale1 = index_map.at("fact_scale1");
    } else {
        _field_indices.x1 = -1;
        _field_indices.fact_scale1 = -1;
    }
    if (index_map.contains("fact_scale2")) {
        _field_indices.x2 = index_map.at("x2");
        _field_indices.fact_scale2 = index_map.at("fact_scale2");
    } else {
        _field_indices.x2 = -1;
        _field_indices.fact_scale2 = -1;
    }
    if (index_map.contains("partial_weight_product")) {
        _field_indices.partial_weight_product = index_map.at("partial_weight_product");
    } else {
        _field_indices.partial_weight_product = -1;
    }
    if (index_map.contains("cluster_scales")) {
        _field_indices.cluster_scales = index_map.at("cluster_scales");
    } else {
        _field_indices.cluster_scales = -1;
    }
    if (index_map.contains("alphas_scales")) {
        _field_indices.alphas_scales = index_map.at("alphas_scales");
    } else {
        _field_indices.alphas_scales = -1;
    }
    if (index_map.contains("ladder_x")) {
        const char* names[] = {
            "ladder_x",
            "ladder_pdg",
            "ladder_q",
            "ladder_info",
            "ladder_lead_q",
        };
        for (std::size_t k = 0; k < _field_indices.ladder.size(); ++k) {
            _field_indices.ladder[k] = index_map.at(names[k]);
        }
    } else {
        _field_indices.ladder.fill(-1);
    }
    if (index_map.contains("subprocess_index")) {
        _field_indices.subprocess_index = index_map.at("subprocess_index");
    } else {
        _field_indices.subprocess_index = -1;
    }
    _field_indices.bias = index_map.contains("bias") ? index_map.at("bias") : -1;
    _field_indices.matrix_element =
        index_map.contains("matrix_element") ? index_map.at("matrix_element") : -1;
    _field_indices.random = index_map.at("random");
    _field_indices.rest = _field_indices.random + 1;
}

void ChannelEventGenerator::unweight_file(MixMaxRandom& rand_gen) {
    std::size_t buf_size = 1000000;
    EventBuffer buffer(0, 0, weight_file_layout);
    std::size_t accept_count = _unweighted_accept_count;
    for (std::size_t i = _unweighted_count; i < _weight_file.event_count();
         i += buf_size) {
        _weight_file.seek(i);
        _weight_file.read(buffer, buf_size);
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            auto weight = buffer.event(j).weight();
            if (std::abs(weight.value()) / _max_weight < rand_gen.generate_double()) {
                weight = 0;
            } else {
                weight = std::copysign(
                    std::max(std::abs(weight.value()), _max_weight), weight.value()
                );
                ++accept_count;
            }
        }
        _weight_file.seek(i);
        _weight_file.write(buffer);
    }
    _unweighted_count = _weight_file.event_count();
    _unweighted_accept_count = accept_count;
    _status.count_unweighted = accept_count;
    _expected_unweighted = accept_count;
}

void ChannelEventGenerator::integrate(const GeneratorBatchJob& job) {
    auto w_view = job.weights.view<double, 1>();
    std::size_t sample_count_after_cuts = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) {
            ++sample_count_after_cuts;
        }
        // With a bias b the weights are w b: the absolute integral, which drives
        // the adaptation, unweighting and channel allocation, keeps the bias,
        // while the reported cross section divides it out.
        double weight = w_view[i];
        if (_field_indices.bias != -1 && weight != 0.) {
            weight /= job.biases.view<double, 1>()[i];
        }
        _cross_section.push(weight);
        _abs_cross_section.push(std::abs(w_view[i]));
    }
    _status.mean = _cross_section.mean();
    _status.error = _cross_section.error();
    _status.mean_abs = _abs_cross_section.mean();
    _status.error_abs = _abs_cross_section.error();
    _status.rel_std_dev = _abs_cross_section.rel_std_dev();
    _status.count += w_view.size();
    _status.count_opt += w_view.size();
    _count_requested_opt += job.requested_event_count;
    _status.count_after_cuts += sample_count_after_cuts;
    _status.count_after_cuts_opt += sample_count_after_cuts;

    if (job.hists.size() > 0) {
        for (std::size_t i = 0; i < job.hists.size() / 2; ++i) {
            auto hist_view = job.hists.at(2 * i).view<double, 2>()[0];
            auto hist2_view = job.hists.at(2 * i + 1).view<double, 2>()[0];
            auto& chan_hist = _histograms.at(i);
            for (std::size_t j = 0; j < hist_view.size(); ++j) {
                // note: we don't assign the actuals means and errors here. There are
                // still some normalization steps necessary that are performed later
                chan_hist.bin_values.at(j) += hist_view[j];
                chan_hist.bin_errors.at(j) += hist2_view[j];
            }
        }
    }

    if (job.vegas_hist.size() > 0) {
        _vegas_optimizer->add_data(job.vegas_hist.at(0), job.vegas_hist.at(1));
    }
    if (job.discrete_hist.size() > 0) {
        _discrete_optimizer->add_data(job.discrete_hist);
    }
}

void ChannelEventGenerator::optimize_vegas(const GeneratorBatchJob& job) {
    if (_vegas_optimizer) {
        _vegas_optimizer->optimize();
    }
    if (_discrete_optimizer) {
        _discrete_optimizer->optimize();
    }
    double rsd = _abs_cross_section.rel_std_dev();
    if (rsd < _config.optimization_threshold * _best_rsd) {
        _iters_without_improvement = 0;
    } else {
        ++_iters_without_improvement;
        if (_iters_without_improvement >= _config.optimization_patience) {
            _status.optimized = true;
        }
    }
    _best_rsd = std::min(rsd, _best_rsd);
    _status.iters_without_improvement = _iters_without_improvement;
    ++_status.iterations;
}

double ChannelEventGenerator::channel_weight_sum(std::size_t event_count) {
    std::size_t buf_size = 1000000;
    EventBuffer buffer(0, 0, weight_file_layout);
    double weight_sum = 0;
    _weight_file.seek(0);
    std::size_t unweighted_count = 0;
    for (std::size_t i = 0; i < _weight_file.event_count(); i += buf_size) {
        _weight_file.read(buffer, buf_size);
        bool done = false;
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            if (unweighted_count == event_count) {
                done = true;
                break;
            }
            double weight = buffer.event(j).weight();
            if (weight == 0.) {
                continue;
            }
            weight_sum += std::abs(weight) / _max_weight;
            ++unweighted_count;
        }
        if (done) {
            break;
        }
    }
    return weight_sum;
}

// Assigns job's base seed on the scheduling thread, from its logical identity
// only (channel, kind, sequence) -- never from context_index -- so it doesn't
// depend on worker/context assignment.
void ChannelEventGenerator::start_job(
    GeneratorBatchJob& job,
    ResultQueue& result_queue,
    std::optional<std::uint64_t> seed,
    bool is_survey,
    std::size_t survey_pass
) {
    job.rng_seed = seed;
    job.rng_is_survey = is_survey;
    job.rng_survey_pass = survey_pass;
    job.rng_job_index = is_survey ? _survey_rng_seq++ : _generate_rng_seq++;
    // Through result_queue, so a job that throws still posts its result and the
    // exception reaches the waiting thread instead of leaving it blocked forever.
    result_queue.submit(
        _contexts.at(job.context_index)->thread_pool(),
        job.job_id,
        [this, &job, &result_queue]() {
            auto& runtimes = _runtimes.at(job.context_index);
            auto& context = _contexts.at(job.context_index);
            if (job.rng_seed) {
                runtimes.integrand_channel->set_seed(generate_phase_seed(
                    job.rng_seed,
                    job.rng_is_survey,
                    job.rng_survey_pass,
                    job.rng_job_index,
                    job.channel_index,
                    rng_role_channel
                ));
            }
            std::size_t max_batch_size =
                context->device()->device_type() == DeviceType::cpu
                ? _config.cpu_batch_size
                : _config.gpu_batch_size;
            std::size_t batch_size = max_batch_size;
            // Only VEGAS batches shrink to fit -- they need next_vegas_batch_size()'s
            // exact geometric progression for correct grid-adaptation statistics.
            // Generation batches always submit full device-sized jobs; a little
            // overshoot there is expected and cheaper than a partial job.
            if (job.is_vegas_batch && job.batch_event_count > 0 &&
                batch_size > job.batch_event_count) {
                batch_size = job.batch_event_count;
            }
            std::size_t target_count = batch_size;
            job.requested_event_count = batch_size;

            std::size_t total_count = 0, repetitions = 0;
            TensorVec all_ps_points;
            while (true) {
                // Successive calls draw further from integrand_channel's own
                // persistent stream (set once above) -- no per-call re-seeding.
                auto ps_points =
                    runtimes.integrand_channel->run({Tensor({batch_size})});
                std::size_t acc_count =
                    ps_points
                        .at(_integrand_channel_function.outputs().index_map().at(
                            "indices_acc"
                        ))
                        .size(0);
                if (all_ps_points.size() > 0 && acc_count > 0) {
                    all_ps_points.insert(
                        all_ps_points.end(), ps_points.begin(), ps_points.end()
                    );
                    all_ps_points = runtimes.integrand_concat->run(all_ps_points);
                } else {
                    all_ps_points = ps_points;
                }
                total_count += acc_count;
                ++repetitions;
                if (total_count >= _config.cut_efficiency_threshold * target_count) {
                    break;
                }
                if (result_queue.cancelled()) {
                    throw std::runtime_error("job cancelled");
                }
                if (repetitions == _config.max_cut_repetitions) {
                    throw std::runtime_error(
                        std::format(
                            "not enough points passing cuts were found after {} "
                            "batches",
                            repetitions
                        )
                    );
                }
                double cut_eff = static_cast<double>(acc_count) / batch_size;
                batch_size = std::min(
                    static_cast<double>(max_batch_size),
                    (target_count - total_count) / cut_eff
                );
            }
            if (result_queue.cancelled()) {
                throw std::runtime_error("job cancelled");
            }
            if (job.rng_seed) {
                runtimes.integrand_common->set_seed(generate_phase_seed(
                    job.rng_seed,
                    job.rng_is_survey,
                    job.rng_survey_pass,
                    job.rng_job_index,
                    job.channel_index,
                    rng_role_common
                ));
            }
            job.events = runtimes.integrand_common->run(all_ps_points);

            if (result_queue.cancelled()) {
                throw std::runtime_error("job cancelled");
            }
            job.weights = job.events.at(_field_indices.weight).cpu();
            if (_field_indices.bias != -1) {
                job.biases = job.events.at(_field_indices.bias).cpu();
            }
            // observable_histograms/vegas_histogram/discrete_histogram don't consume
            // random numbers, so they're never seeded.
            if (runtimes.observable_histograms) {
                auto hists = runtimes.observable_histograms->run(
                    {job.events.at(_field_indices.weight),
                     job.events.at(_field_indices.momenta)}
                );
                for (auto& item : hists) {
                    job.hists.push_back(item.cpu());
                }
            }
            if (job.is_vegas_batch) {
                if (_vegas_optimizer) {
                    auto hist = runtimes.vegas_histogram->run(
                        {job.events.at(_field_indices.random),
                         job.events.at(_field_indices.weight)}
                    );
                    for (auto& item : hist) {
                        job.vegas_hist.push_back(item.cpu());
                    }
                }
                if (_discrete_optimizer) {
                    TensorVec args{
                        job.events.begin() + _field_indices.rest, job.events.end()
                    };
                    args.push_back(job.events.at(_field_indices.weight));
                    auto hist = runtimes.discrete_histogram->run(args);
                    for (auto& item : hist) {
                        job.discrete_hist.push_back(item.cpu());
                    }
                }
            }
        }
    );
}

void ChannelEventGenerator::prepare_unweight_job(GeneratorBatchJob& job) const {
    job.max_weight = _max_weight;
}

void ChannelEventGenerator::submit_unweight_job(
    GeneratorBatchJob& job, ResultQueue& result_queue
) {
    result_queue.submit(
        _contexts.at(job.context_index)->thread_pool(), job.job_id, [this, &job]() {
            auto& runtimes = _runtimes.at(job.context_index);
            auto& context = _contexts.at(job.context_index);
            if (job.rng_seed) {
                runtimes.unweighter->set_seed(DerivedSeed(
                    job.rng_seed,
                    unweight_seed_type(job.rng_is_survey, job.rng_survey_pass),
                    job.rng_job_index,
                    job.channel_index
                ));
            }
            std::vector<Tensor> unweighter_args(
                job.events.begin(), job.events.begin() + _field_indices.random
            );
            unweighter_args.push_back(Tensor(job.max_weight, context->device()));
            TensorVec unw_events = runtimes.unweighter->run(unweighter_args);
            for (auto& item : unw_events) {
                job.unweighted_events.push_back(item.cpu());
            }
        }
    );
}

void ChannelEventGenerator::start_unweight_job(
    GeneratorBatchJob& job, ResultQueue& result_queue
) {
    prepare_unweight_job(job);
    submit_unweight_job(job, result_queue);
}

std::size_t ChannelEventGenerator::next_vegas_batch_size() {
    std::size_t batch_size = _batch_size;
    _batch_size = std::min(_batch_size * 2, _config.max_batch_size);
    return batch_size;
}

void ChannelEventGenerator::clear_events() {
    _status.count_unweighted = 0;
    _expected_unweighted = 0;
    if (!_max_weight_fixed) {
        _max_weight = 0;
    }
    _unweighted_count = 0;
    _unweighted_accept_count = 0;
    _status.count_opt = 0;
    _count_requested_opt = 0;
    _status.count_after_cuts_opt = 0;
    _event_file.clear();
    _weight_file.clear();
    _cross_section.reset();
    _abs_cross_section.reset();
    _large_weights.clear();
    for (auto& hist : _histograms) {
        std::fill(hist.bin_values.begin(), hist.bin_values.end(), 0.);
        std::fill(hist.bin_errors.begin(), hist.bin_errors.end(), 0.);
    }
}

void ChannelEventGenerator::update_max_weight(Tensor weights) {
    // Freeze on the expected unweighted count of the committed generate stages
    // rather than on count_unweighted: the latter is advanced by the unweight
    // stages, whose commits interleave with the generate stages depending on
    // thread timing, which made the freeze point (and the run) timing-dependent.
    if (_max_weight_fixed || _expected_unweighted > _config.freeze_max_weight_after) {
        return;
    }

    auto w_view = weights.view<double, 1>();
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) {
            w_min_nonzero = w;
        }
        if (w > _max_weight) {
            _large_weights.push_back(w);
        }
    }
    if (_max_weight == 0) {
        _max_weight = w_min_nonzero;
    }
    std::sort(_large_weights.begin(), _large_weights.end(), std::greater{});
    apply_truncation_budget();
    // expected accepted events of this job, against the max weight it will be
    // unweighted with (see prepare_unweight_job)
    if (_max_weight > 0) {
        for (std::size_t i = 0; i < w_view.size(); ++i) {
            _expected_unweighted += std::min(std::abs(w_view[i]) / _max_weight, 1.);
        }
    }
}

void ChannelEventGenerator::set_target_count(std::size_t target_count) {
    std::size_t old_target = _status.count_target;
    _status.count_target = target_count;
    // The truncation budget scales with count_target, and _max_weight only
    // rises. A cap chosen under a larger target than the channel ends up with
    // (e.g. generate()'s round-one seed) would otherwise never be corrected: a
    // channel that has met its smaller target is not revisited. Tighten the cap
    // now; unweight_all() re-unweights the whole file against it at the end.
    if (target_count < old_target && _max_weight != 0 && !_max_weight_fixed &&
        _expected_unweighted <= _config.freeze_max_weight_after) {
        apply_truncation_budget();
    }
}

void ChannelEventGenerator::apply_truncation_budget() {
    double w_sum = 0;
    double max_truncation = _config.max_overweight_truncation *
        static_cast<double>(std::min(
            _status.count_target, _config.freeze_max_weight_after
        ));
    std::size_t count = 0;
    for (auto w : _large_weights) {
        if (w < _max_weight) {
            break;
        }
        w_sum += w;
        ++count;
        if (w_sum / w > max_truncation) {
            if (_max_weight < w) {
                _status.count_unweighted *= _max_weight / w;
                _expected_unweighted *= _max_weight / w;
                _max_weight = w;
                _unweighted_count = 0;
                _unweighted_accept_count = 0;
            }
            break;
        }
    }
    _large_weights.erase(_large_weights.begin() + count, _large_weights.end());
}

void ChannelEventGenerator::write_events(
    const std::vector<Tensor>& unweighted_events, double job_max_weight
) {
    auto w_view = unweighted_events.at(_field_indices.weight).view<double, 1>();
    auto mom_view = unweighted_events.at(_field_indices.momenta).view<double, 3>();
    auto colors_view =
        unweighted_events.at(_field_indices.color_index).view<me_int_t, 1>();
    auto helicities_view =
        unweighted_events.at(_field_indices.helicity_index).view<me_int_t, 1>();
    auto diagrams_view =
        unweighted_events.at(_field_indices.diagram_index).view<me_int_t, 1>();
    auto flavors_view =
        unweighted_events.at(_field_indices.flavor_index).view<me_int_t, 1>();
    auto ren_scale_view =
        unweighted_events.at(_field_indices.ren_scale).view<double, 1>();
    auto alphas_view = unweighted_events.at(_field_indices.alpha_qcd).view<double, 1>();

    EventBuffer event_buffer(
        w_view.size(), _event_file.particle_count(), _event_file_layout
    );
    EventBuffer weight_buffer(w_view.size(), 0, weight_file_layout);
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        weight_buffer.event(i).weight() = w_view[i];
        auto event = event_buffer.event(i);
        event.diagram_index() = diagrams_view[i];
        event.color_index() = colors_view[i];
        event.flavor_index() = flavors_view[i];
        event.helicity_index() = helicities_view[i];
        event.ren_scale() = ren_scale_view[i];
        event.alpha_qcd() = alphas_view[i];
        auto event_mom = mom_view[i];
        for (std::size_t j = 0; j < event_mom.size(); ++j) {
            auto particle_mom = event_mom[j];
            auto particle = event_buffer.particle(i, j);
            particle.energy() = particle_mom[0];
            particle.px() = particle_mom[1];
            particle.py() = particle_mom[2];
            particle.pz() = particle_mom[3];
        }
    }

    if (_field_indices.fact_scale1 != -1) {
        auto x1_view = unweighted_events.at(_field_indices.x1).view<double, 1>();
        auto fact1_view =
            unweighted_events.at(_field_indices.fact_scale1).view<double, 1>();
        for (std::size_t i = 0; i < x1_view.size(); ++i) {
            auto event = event_buffer.event(i);
            event.x1() = x1_view[i];
            event.fact_scale1() = fact1_view[i];
        }
    }

    if (_field_indices.fact_scale2 != -1) {
        auto x2_view = unweighted_events.at(_field_indices.x2).view<double, 1>();
        auto fact2_view =
            unweighted_events.at(_field_indices.fact_scale2).view<double, 1>();
        for (std::size_t i = 0; i < x2_view.size(); ++i) {
            auto event = event_buffer.event(i);
            event.x2() = x2_view[i];
            event.fact_scale2() = fact2_view[i];
        }
    }

    if (_field_indices.partial_weight_product != -1) {
        auto pw_view =
            unweighted_events.at(_field_indices.partial_weight_product)
                .view<double, 1>();
        for (std::size_t i = 0; i < pw_view.size(); ++i) {
            auto event = event_buffer.event(i);
            event.partial_weight_product() = pw_view[i];
        }
    }

    if (_field_indices.cluster_scales != -1) {
        auto cluster_view =
            unweighted_events.at(_field_indices.cluster_scales).view<double, 2>();
        for (std::size_t i = 0; i < w_view.size(); ++i) {
            auto scales = cluster_view[i];
            for (std::size_t j = 0; j < _particle_count; ++j) {
                auto particle = event_buffer.particle(i, j);
                particle.cluster_scale() = j >= 2 ? scales[j - 2] : 0.;
            }
        }
    }

    if (_field_indices.alphas_scales != -1) {
        // vertex i on outgoing particle 2 + i; there is one vertex less than
        // there are outgoing particles, the remaining slots hold zero
        auto alphas_view =
            unweighted_events.at(_field_indices.alphas_scales).view<double, 2>();
        for (std::size_t i = 0; i < w_view.size(); ++i) {
            auto scales = alphas_view[i];
            std::size_t vertex_count = scales.size();
            for (std::size_t j = 0; j < _particle_count; ++j) {
                auto particle = event_buffer.particle(i, j);
                particle.alphas_scale() =
                    j >= 2 && j - 2 < vertex_count ? scales[j - 2] : 0.;
            }
        }
    }

    if (_field_indices.ladder[0] != -1) {
        auto slot_view = [&](std::size_t k) {
            return unweighted_events.at(_field_indices.ladder[k]).view<double, 2>();
        };
        auto x_view = slot_view(0), pdg_view = slot_view(1), q_view = slot_view(2),
             info_view = slot_view(3), lead_q_view = slot_view(4);
        for (std::size_t i = 0; i < w_view.size(); ++i) {
            std::size_t slot_count = x_view[i].size();
            for (std::size_t j = 0; j < _particle_count; ++j) {
                auto particle = event_buffer.particle(i, j);
                bool has_slot = j < slot_count;
                particle.ladder_x() = has_slot ? x_view[i][j] : 0.;
                particle.ladder_pdg() = has_slot ? pdg_view[i][j] : 0.;
                particle.ladder_q() = has_slot ? q_view[i][j] : 0.;
                particle.ladder_info() = has_slot ? info_view[i][j] : 0.;
                particle.ladder_lead_q() = j < 2 ? lead_q_view[i][j] : 0.;
            }
        }
    }

    if (_field_indices.bias != -1) {
        auto bias_view = unweighted_events.at(_field_indices.bias).view<double, 1>();
        for (std::size_t i = 0; i < bias_view.size(); ++i) {
            event_buffer.event(i).bias() = bias_view[i];
        }
    }

    if (_field_indices.matrix_element != -1) {
        auto me_view =
            unweighted_events.at(_field_indices.matrix_element).view<double, 1>();
        for (std::size_t i = 0; i < me_view.size(); ++i) {
            event_buffer.event(i).matrix_element() = me_view[i];
        }
    }

    if (_field_indices.subprocess_index != -1) {
        auto subproc_view =
            unweighted_events.at(_field_indices.subprocess_index).view<me_int_t, 1>();
        for (std::size_t i = 0; i < subproc_view.size(); ++i) {
            auto event = event_buffer.event(i);
            event.subprocess_index() = subproc_view[i];
        }
    }

    _event_file.write(event_buffer);
    _weight_file.write(weight_buffer);
    _status.count_unweighted +=
        w_view.size() * (job_max_weight > 0 ? job_max_weight / _max_weight : 1);
    _status.done = _status.count_unweighted >= _status.count_target;
}

void ChannelEventGenerator::save(const std::string& file_name) const {
    std::ofstream f(file_name);
    f << to_json();
}

std::string ChannelEventGenerator::to_json(bool include_estimates) const {
    json j;
    j = *this;
    if (include_estimates) {
        j["max_weight"] = _max_weight;
        double error = _abs_cross_section.error();
        if (_abs_cross_section.count() > 1 && std::isfinite(error) && error > 0) {
            j["abs_integral"] = {{"mean", _abs_cross_section.mean()}, {"error", error}};
            j["integral"] = {
                {"mean", _cross_section.mean()}, {"error", _cross_section.error()}
            };
        }
    }
    return j.dump();
}

ChannelEventGenerator ChannelEventGenerator::load(
    const std::string& channel_file,
    const std::vector<ContextPtr>& contexts,
    const std::string& event_file,
    const std::string& weight_file,
    const GeneratorConfig& config
) {
    std::ifstream f(channel_file);
    if (!f) {
        throw std::runtime_error(std::format("Could not open file '{}'", channel_file));
    }
    std::string text(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()
    );
    return load_json(text, contexts, event_file, weight_file, config);
}

ChannelEventGenerator ChannelEventGenerator::load_json(
    const std::string& channel_json,
    const std::vector<ContextPtr>& contexts,
    const std::string& event_file,
    const std::string& weight_file,
    const GeneratorConfig& config
) {
    json channel = json::parse(channel_json);
    std::optional<Function> hist_function;
    std::vector<Histogram> histograms;
    if (!channel.at("histogram_function").is_null()) {
        for (json hist : channel.at("histograms")) {
            std::size_t bin_count = hist.at("bin_count").get<std::size_t>();
            histograms.push_back({
                .name = hist.at("name").get<std::string>(),
                .min = hist.at("min").get<double>(),
                .max = hist.at("max").get<double>(),
                .bin_values = std::vector<double>(bin_count + 2),
                .bin_errors = std::vector<double>(bin_count + 2),
            });
        }
    }

    ChannelEventGenerator generator(
        contexts,
        channel.at("particle_count").get<std::size_t>(),
        channel.at("integrand_channel_function").get<Function>(),
        channel.at("integrand_common_function").get<Function>(),
        channel.at("integrand_concat_function").get<Function>(),
        channel.at("unweighter_function").get<Function>(),
        hist_function,
        event_file,
        weight_file,
        channel.at("subprocess_index").get<std::size_t>(),
        channel.at("name").get<std::string>(),
        config,
        histograms
    );
    if (channel.contains("max_weight") && channel.at("max_weight").get<double>() > 0) {
        generator.set_fixed_max_weight(channel.at("max_weight").get<double>());
    }
    if (channel.contains("abs_integral")) {
        generator.set_abs_integral_prior(
            channel.at("abs_integral").at("mean").get<double>(),
            channel.at("abs_integral").at("error").get<double>()
        );
    }
    if (channel.contains("integral")) {
        generator.set_integral_prior(
            channel.at("integral").at("mean").get<double>(),
            channel.at("integral").at("error").get<double>()
        );
    }
    return generator;
}

void ChannelEventGenerator::set_fixed_max_weight(double max_weight) {
    if (!(max_weight > 0)) {
        throw std::invalid_argument("The fixed maximum weight must be positive");
    }
    _max_weight = max_weight;
    _max_weight_fixed = true;
    _unweighted_count = 0;
    _unweighted_accept_count = 0;
}

void ChannelEventGenerator::set_abs_integral_prior(double mean, double error) {
    if (!(mean > 0) || !(error > 0) || !std::isfinite(mean) || !std::isfinite(error)) {
        throw std::invalid_argument("The integral prior must be positive and finite");
    }
    _abs_integral_prior = {mean, error};
}

void ChannelEventGenerator::set_integral_prior(double mean, double error) {
    if (!std::isfinite(mean) || !std::isfinite(error) || error < 0) {
        throw std::invalid_argument("The integral prior must be finite");
    }
    _integral_prior = {mean, error};
}

bool ChannelEventGenerator::uses_integral_prior() const {
    double error = _abs_cross_section.error();
    return _abs_integral_prior &&
        (_abs_cross_section.count() < 2 || !std::isfinite(error) || !(error > 0));
}

std::pair<double, double> ChannelEventGenerator::integral_prior() const {
    return _integral_prior ? *_integral_prior : *_abs_integral_prior;
}

double ChannelEventGenerator::abs_integral_estimate() const {
    double mean = _abs_cross_section.mean();
    if (!_abs_integral_prior) {
        return mean;
    }
    auto [prior_mean, prior_error] = *_abs_integral_prior;
    double error = _abs_cross_section.error();
    if (uses_integral_prior()) {
        return prior_mean;
    }
    double prior_weight = 1. / (prior_error * prior_error);
    double weight = 1. / (error * error);
    return (prior_mean * prior_weight + mean * weight) / (prior_weight + weight);
}

void madspace::to_json(nlohmann::json& j, const ChannelEventGenerator& channel) {
    json histograms = json::array();
    json histogram_function;
    if (channel._histogram_function) {
        histogram_function = json(channel._histogram_function.value());
        for (auto& hist : channel._histograms) {
            histograms.push_back(
                json{
                    {"name", hist.name},
                    {"min", hist.min},
                    {"max", hist.max},
                    {"bin_count", hist.bin_values.size() - 2},
                }
            );
        }
    }
    j = json{
        {"particle_count", channel._particle_count},
        {"integrand_channel_function", json(channel._integrand_channel_function)},
        {"integrand_common_function", json(channel._integrand_common_function)},
        {"integrand_concat_function", json(channel._integrand_concat_function)},
        {"unweighter_function", json(channel._unweighter_function)},
        {"histogram_function", histogram_function},
        {"subprocess_index", channel._status.subprocess},
        {"name", channel._status.name},
        {"histograms", histograms},
    };
}
