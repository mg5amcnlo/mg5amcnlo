#include "madspace/driver/channel_generator.hpp"

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
    return flags;
}

int particle_extra_flags(
    const std::unordered_map<std::string, std::size_t>& index_map
) {
    int flags = 0;
    if (index_map.contains("cluster_scales")) {
        flags |= ParticleRecord::f_clustering;
    }
    return flags;
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
        .rel_std_dev = 0.,
        .count = 0,
        .count_opt = 0,
        .count_after_cuts = 0,
        .count_after_cuts_opt = 0,
        .count_unweighted = 0.,
        .count_target = 1.,
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
    std::visit(add_names, integrand.discrete_before());
    std::visit(add_names, integrand.discrete_after());
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
        .rel_std_dev = 0.,
        .count = 0,
        .count_opt = 0,
        .count_after_cuts = 0,
        .count_after_cuts_opt = 0,
        .count_unweighted = 0.,
        .count_target = 1.,
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
    _field_indices.random = index_map.at("random");
    _field_indices.rest = _field_indices.random + 1;
}

void ChannelEventGenerator::unweight_file(std::mt19937& rand_gen) {
    std::size_t buf_size = 1000000;
    std::uniform_real_distribution<double> rand_dist;
    EventBuffer buffer(0, 0, weight_file_layout);
    std::size_t accept_count = _unweighted_count;
    for (std::size_t i = _unweighted_count; i < _weight_file.event_count();
         i += buf_size) {
        _weight_file.seek(i);
        _weight_file.read(buffer, buf_size);
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            auto weight = buffer.event(j).weight();
            if (weight / _max_weight < rand_dist(rand_gen)) {
                weight = 0;
            } else {
                weight = std::max(weight.value(), _max_weight);
                ++accept_count;
            }
        }
        _weight_file.seek(i);
        _weight_file.write(buffer);
    }
    _status.count_unweighted = accept_count;
}

void ChannelEventGenerator::integrate(const GeneratorBatchJob& job) {
    auto w_view = job.weights.view<double, 1>();
    std::size_t sample_count_after_cuts = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) {
            ++sample_count_after_cuts;
        }
        _cross_section.push(w_view[i]);
    }
    _status.mean = _cross_section.mean();
    _status.error = _cross_section.error();
    _status.rel_std_dev = _cross_section.rel_std_dev();
    _status.count += w_view.size();
    _status.count_opt += w_view.size();
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
    double rsd = _cross_section.rel_std_dev();
    if (rsd < _config.optimization_threshold * _best_rsd) {
        _iters_without_improvement = 0;
    } else {
        ++_iters_without_improvement;
        if (_iters_without_improvement >= _config.optimization_patience) {
            _status.optimized = true;
        }
    }
    _best_rsd = std::min(rsd, _best_rsd);
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
            weight_sum += weight / _max_weight;
            _unweighted_count = 0;
            ++unweighted_count;
        }
        if (done) {
            break;
        }
    }
    return weight_sum;
}

void ChannelEventGenerator::start_job(
    GeneratorBatchJob& job, ResultQueue& result_queue
) {
    _contexts.at(job.context_index)
        ->thread_pool()
        .submit([this, &job, &result_queue]() {
            auto& runtimes = _runtimes.at(job.context_index);
            auto& context = _contexts.at(job.context_index);
            std::size_t max_batch_size =
                context->device()->device_type() == DeviceType::cpu
                ? _config.cpu_batch_size
                : _config.gpu_batch_size;
            std::size_t batch_size = max_batch_size;
            if (job.vegas_batch_size > 0 && batch_size > job.vegas_batch_size) {
                batch_size = job.vegas_batch_size;
            }
            std::size_t target_count = batch_size;

            std::size_t total_count = 0, repetitions = 0;
            TensorVec all_ps_points;
            while (true) {
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
            job.events = runtimes.integrand_common->run(all_ps_points);

            job.weights = job.events.at(_field_indices.weight).cpu();
            if (runtimes.observable_histograms) {
                auto hists = runtimes.observable_histograms->run(
                    {job.events.at(_field_indices.weight),
                     job.events.at(_field_indices.momenta)}
                );
                for (auto& item : hists) {
                    job.hists.push_back(item.cpu());
                }
            }
            if (job.vegas_batch_size != 0) {
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
            result_queue.push(job.job_id);
            return std::nullopt;
        });
}

void ChannelEventGenerator::start_unweight_job(
    GeneratorBatchJob& job, ResultQueue& result_queue
) {
    job.max_weight = _max_weight;
    _contexts.at(job.context_index)
        ->thread_pool()
        .submit([this, &job, &result_queue]() {
            auto& runtimes = _runtimes.at(job.context_index);
            auto& context = _contexts.at(job.context_index);
            std::vector<Tensor> unweighter_args(
                job.events.begin(), job.events.begin() + _field_indices.random
            );
            unweighter_args.push_back(Tensor(job.max_weight, context->device()));
            TensorVec unw_events = runtimes.unweighter->run(unweighter_args);
            for (auto& item : unw_events) {
                job.unweighted_events.push_back(item.cpu());
            }
            result_queue.push(job.job_id);
            return std::nullopt;
        });
}

std::size_t ChannelEventGenerator::next_vegas_batch_size() {
    std::size_t batch_size = _batch_size;
    _batch_size = std::min(_batch_size * 2, _config.max_batch_size);
    return batch_size;
}

void ChannelEventGenerator::clear_events() {
    _status.count_unweighted = 0;
    _max_weight = 0;
    _unweighted_count = 0;
    _status.count_opt = 0;
    _status.count_after_cuts_opt = 0;
    _event_file.clear();
    _weight_file.clear();
    _cross_section.reset();
    _large_weights.clear();
    for (auto& hist : _histograms) {
        std::fill(hist.bin_values.begin(), hist.bin_values.end(), 0.);
        std::fill(hist.bin_errors.begin(), hist.bin_errors.end(), 0.);
    }
}

void ChannelEventGenerator::update_max_weight(Tensor weights) {
    if (_status.count_unweighted > _config.freeze_max_weight_after) {
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

    double w_sum = 0, w_prev = 0;
    double max_truncation = _config.max_overweight_truncation *
        std::min(_status.count_target,
                 static_cast<double>(_config.freeze_max_weight_after));
    std::size_t count = 0;
    for (auto w : _large_weights) {
        if (w < _max_weight) {
            break;
        }
        w_sum += w;
        ++count;
        if (w_sum / w - count > max_truncation) {
            if (_max_weight < w) {
                _status.count_unweighted *= _max_weight / w_prev;
                _max_weight = w_prev;
                _unweighted_count = 0;
            }
            break;
        }
        w_prev = w;
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

    _event_file.write(event_buffer);
    _weight_file.write(weight_buffer);
    _status.count_unweighted +=
        w_view.size() * (job_max_weight > 0 ? job_max_weight / _max_weight : 1);
    _status.done = _status.count_unweighted >= _status.count_target;
}

void ChannelEventGenerator::save(const std::string& file_name) const {
    std::ofstream f(file_name);
    json j;
    j = *this;
    f << j.dump();
}

ChannelEventGenerator ChannelEventGenerator::load(
    const std::string& channel_file,
    const std::vector<ContextPtr>& contexts,
    const std::string& event_file,
    const std::string& weight_file,
    const GeneratorConfig& config
) {
    std::ifstream f(channel_file);
    json channel = json::parse(f);
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

    return ChannelEventGenerator(
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
