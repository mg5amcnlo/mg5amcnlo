#pragma once

#include <random>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/generator_data.hpp"

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/discrete_optimizer.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/vegas_optimizer.hpp"
#include "madspace/phasespace.hpp"

namespace madspace {

class ChannelEventGenerator {
public:
    static ChannelEventGenerator load(
        const std::string& channel_file,
        const std::vector<ContextPtr>& contexts,
        const std::string& event_file,
        const std::string& weight_file,
        const GeneratorConfig& config
    );

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

    const GeneratorStatus& status() const { return _status; }
    const RunningIntegral& cross_section() const { return _cross_section; }
    const std::vector<Histogram>& histograms() const { return _histograms; }
    EventFile& event_file() { return _event_file; }
    EventFile& weight_file() { return _weight_file; }
    double max_weight() const { return _max_weight; }
    std::size_t batch_size() const { return _batch_size; }
    bool needs_optimization() const {
        return (_vegas_optimizer || _discrete_optimizer) && !_status.optimized;
    }
    void set_target_count(double target_count) { _status.count_target = target_count; }
    const std::unordered_set<std::string>& used_globals() const {
        return _used_globals;
    }
    int event_layout_extra_flags() const { return _event_layout_extra_flags; }
    int particle_layout_extra_flags() const { return _particle_layout_extra_flags; }
    const DataLayout& event_file_layout() const { return _event_file_layout; }

    void unweight_file(std::mt19937& rand_gen);
    void integrate(const GeneratorBatchJob& job);
    void optimize_vegas(const GeneratorBatchJob& job);
    double channel_weight_sum(std::size_t event_count);
    void start_job(GeneratorBatchJob& job, ResultQueue& result_queue);
    void start_unweight_job(GeneratorBatchJob& job, ResultQueue& result_queue);
    std::size_t next_vegas_batch_size();
    void clear_events();
    void update_max_weight(Tensor weights);
    void write_events(const TensorVec& unweighted_events, double job_max_weight);
    void save(const std::string& file_name) const;

private:
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
    double _max_weight = 0.;
    std::size_t _unweighted_count = 0;
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
