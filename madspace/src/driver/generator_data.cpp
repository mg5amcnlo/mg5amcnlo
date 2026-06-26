#include "madspace/driver/generator_data.hpp"

using namespace madspace;

void madspace::to_json(nlohmann::json& j, const GeneratorStatus& status) {
    j = nlohmann::json{
        {"subprocess", status.subprocess},
        {"name", status.name},
        {"mean", status.mean},
        {"error", status.error},
        {"rel_std_dev", status.rel_std_dev},
        {"count", status.count},
        {"count_opt", status.count_opt},
        {"count_after_cuts", status.count_after_cuts},
        {"count_after_cuts_opt", status.count_after_cuts_opt},
        {"count_unweighted", status.count_unweighted},
        {"count_target", status.count_target},
        {"iterations", status.iterations},
        {"optimized", status.optimized},
        {"done", status.done},
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
