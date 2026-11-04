#include "madspace/driver/generator_data.hpp"

#include <algorithm>
#include <cmath>

using namespace madspace;

std::size_t madspace::compute_generation_batch_event_count(
    std::size_t count_target,
    double count_unweighted,
    std::size_t count_requested,
    std::size_t abs_cross_section_count,
    double abs_cross_section_rel_error,
    const GeneratorConfig& config
) {
    double efficiency = count_requested > 0
        ? std::max(
              count_unweighted / static_cast<double>(count_requested),
              1. / count_requested
          )
        : 1.;

    double true_remaining =
        std::max(static_cast<double>(count_target) - count_unweighted, 0.);

    if (true_remaining <= config.finish_remaining_fraction * count_target) {
        return static_cast<std::size_t>(
            std::max(1., std::ceil(true_remaining / efficiency))
        );
    }

    double rel_error =
        abs_cross_section_count > 1 && std::isfinite(abs_cross_section_rel_error)
        ? abs_cross_section_rel_error
        : 1.;

    double target_uncertainty = static_cast<double>(count_target) * rel_error;
    double safe_target = static_cast<double>(count_target) -
        config.batch_overshoot_sigma * target_uncertainty;
    double safe_remaining = std::max(safe_target - count_unweighted, 0.);
    double capped = std::min(
        config.max_batch_fraction * static_cast<double>(count_target), safe_remaining
    );

    return static_cast<std::size_t>(std::max(1., std::ceil(capped / efficiency)));
}

std::size_t madspace::select_combine_channel_index(
    const std::vector<std::size_t>& cum_counts, std::size_t random_index
) {
    return select_combine_channel(
               cum_counts.begin(),
               cum_counts.end(),
               random_index,
               [](std::size_t cum_count) { return cum_count; }
           ) -
        cum_counts.begin();
}

void madspace::to_json(nlohmann::json& j, const GeneratorStatus& status) {
    j = nlohmann::json{
        {"subprocess", status.subprocess},
        {"name", status.name},
        {"mean", status.mean},
        {"error", status.error},
        {"mean_abs", status.mean_abs},
        {"error_abs", status.error_abs},
        {"rel_std_dev", status.rel_std_dev},
        {"count", status.count},
        {"count_opt", status.count_opt},
        {"count_after_cuts", status.count_after_cuts},
        {"count_after_cuts_opt", status.count_after_cuts_opt},
        {"count_unweighted", status.count_unweighted},
        {"count_target", status.count_target},
        {"iterations", status.iterations},
        {"iters_without_improvement", status.iters_without_improvement},
        {"optimized", status.optimized},
        {"done", status.done},
        {"count_invalid_step_values", status.count_invalid_step_values},
    };
}

void madspace::to_json(nlohmann::json& j, const Histogram& hist) {
    j = nlohmann::json{
        {"name", hist.name},
        {"min", hist.min},
        {"max", hist.max},
        {"bin_values", hist.bin_values},
        {"bin_errors", hist.bin_errors},
    };
}

StepChainResult madspace::evaluate_step_chain(
    double weight0,
    double max_weight0,
    const double* ratios,
    const double* max_weights,
    const std::vector<StepSettings>& steps
) {
    double abs_weight0 = std::abs(weight0);
    double acceptance = std::min(1., abs_weight0 / max_weight0);
    double overweight = std::max(1., abs_weight0 / max_weight0);
    double pending = std::copysign(1., weight0);
    for (std::size_t i = 0; auto& step : steps) {
        pending *= ratios[i];
        if (step.unweight) {
            double unweight_value = overweight * std::abs(pending);
            acceptance *= std::min(1., unweight_value / max_weights[i]);
            overweight = std::max(1., unweight_value / max_weights[i]);
            pending = std::copysign(1., pending);
        }
        ++i;
    }
    return {acceptance, overweight * pending};
}
