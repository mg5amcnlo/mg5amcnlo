#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/systematics.hpp"
#include "madspace/phasespace/histograms.hpp"

namespace madspace {

/// One @ref EventHistograms histogram: an observable name and its binning.
struct EventHistogramSpec {
    /// Histogram name. Observables are matched to specs by position: the n-th
    /// spec without @ref from_weight uses the n-th observable of
    /// @ref SubprocessObservables::values.
    std::string name;
    /// Lower edge of the first bin.
    double min;
    /// Upper edge of the last bin.
    double max;
    /// Number of bins between @ref min and @ref max (excluding under/overflow).
    std::size_t bin_count;
    /// Histogram the event weight itself, divided by the reference weight
    /// given to @ref EventHistograms, instead of an observable of the
    /// momenta. Such a spec has no entry in @ref SubprocessObservables.
    bool from_weight = false;
};

/// The observables of one subprocess, evaluated on its external momenta.
struct SubprocessObservables {
    /// The @ref ObservableValues function generator for this subprocess.
    ObservableValues values;
    /// Number of external particles this subprocess's momenta carry.
    std::size_t particle_count;
};

/**
 * Histograms of the final (unweighted) event sample.
 *
 * Filled at combine time with the nominal weight and every @ref
 * SystematicsCalculator variation weight, so that each observable carries its
 * scale envelope and PDF uncertainty band alongside its central value.
 */
class EventHistograms {
public:
    /**
     * @param context      Context the per-subprocess observable runtimes are
     *                     built on.
     * @param specs        The histograms to fill.
     * @param observables  `observables[subprocess]` evaluates the
     *                     histogrammed observables, in the order of the specs
     *                     that are not @ref EventHistogramSpec::from_weight,
     *                     for that (unmerged) subprocess's events; a
     *                     `nullopt` entry skips the subprocess.
     * @param reference_weight  What a @ref EventHistogramSpec::from_weight
     *                     histogram divides the event weight by, so that its
     *                     axis does not depend on the size of the cross
     *                     section: pass the cross section (the mean event
     *                     weight) and a fully unweighted sample is a spike at
     *                     1. Zero or non-finite means the raw weight.
     */
    EventHistograms(
        ContextPtr context,
        const std::vector<EventHistogramSpec>& specs,
        const std::vector<std::optional<SubprocessObservables>>& observables,
        double reference_weight = 0.
    );

    /// The histogram definitions passed to the constructor.
    const std::vector<EventHistogramSpec>& specs() const { return _specs; }
    /// Number of systematic variation weights per event, in addition to the
    /// nominal weight. Zero until the first call of @ref fill.
    std::size_t weight_count() const { return _weight_count; }

    /**
     * Fill from a batch of combined events.
     *
     * `syst_weights` holds `weight_count` variation weights per event
     * (row-major), as computed by @ref SystematicsCalculator; the nominal
     * weight is taken from `buffer`. Thread-safe.
     */
    void fill(
        EventBuffer& buffer,
        const std::vector<double>& syst_weights,
        std::size_t weight_count
    );

    /// Bin contents normalized to cross sections (summing over bins gives the
    /// cross section of the corresponding weight); with `systematics`, the
    /// scale envelope and the PDF uncertainty bands are added per bin.
    nlohmann::json to_json(const SystematicsCalculator* systematics = nullptr) const;

private:
    struct RuntimeData {
        RuntimePtr runtime;
        std::size_t particle_count;
    };

    std::vector<EventHistogramSpec> _specs;
    // spec index -> its position in the observable runtime's outputs; unset
    // for a from_weight spec, which has no observable
    std::vector<std::optional<std::size_t>> _observable_index;
    double _reference_weight = 1.;
    std::vector<std::optional<RuntimeData>> _runtimes;
    std::size_t _weight_count = 0;
    // sums[spec][weight column][bin], bins: underflow, bin_count, overflow;
    // column 0 is the nominal weight
    std::vector<std::vector<std::vector<double>>> _sums;
    std::vector<std::vector<std::vector<double>>> _square_sums;
    std::size_t _event_count = 0;
    std::mutex _mutex;
};

} // namespace madspace
