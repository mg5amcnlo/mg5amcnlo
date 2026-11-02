#include "madspace/driver/event_histograms.hpp"

#include <cmath>
#include <map>

using namespace madspace;
using json = nlohmann::json;

namespace {

bool layout_has(const DataLayout& layout, const std::string& name) {
    for (auto& field : layout.event_layout()) {
        if (field.name == name) {
            return true;
        }
    }
    return false;
}

} // namespace

EventHistograms::EventHistograms(
    ContextPtr context,
    const std::vector<EventHistogramSpec>& specs,
    const std::vector<std::optional<SubprocessObservables>>& observables,
    double reference_weight
) :
    _specs(specs),
    _reference_weight(
        std::isfinite(reference_weight) && reference_weight != 0.
            ? reference_weight
            : 1.
    ) {
    if (!context) {
        throw std::invalid_argument("EventHistograms requires a context");
    }
    std::size_t observable_count = 0;
    for (auto& spec : _specs) {
        if (spec.bin_count == 0 || !(spec.max > spec.min)) {
            throw std::invalid_argument(
                std::format("invalid binning for histogram '{}'", spec.name)
            );
        }
        // a from_weight histogram is filled from the event record, so it does
        // not consume one of the subprocess's observables
        if (spec.from_weight) {
            _observable_index.push_back(std::nullopt);
        } else {
            _observable_index.push_back(observable_count);
            ++observable_count;
        }
    }
    for (auto& obs : observables) {
        if (!obs) {
            _runtimes.push_back(std::nullopt);
            continue;
        }
        if (obs->values.observables().size() != observable_count) {
            throw std::invalid_argument(
                "EventHistograms: one observable per histogram expected"
            );
        }
        _runtimes.push_back(
            RuntimeData{
                .runtime = build_runtime(obs->values.function(), context, false),
                .particle_count = obs->particle_count,
            }
        );
    }
}

void EventHistograms::fill(
    EventBuffer& buffer,
    const std::vector<double>& syst_weights,
    std::size_t weight_count
) {
    std::size_t count = buffer.event_count();
    if (count == 0) {
        return;
    }
    bool has_subproc = layout_has(buffer.layout(), "subprocess_index");

    // group the events by subprocess (the observables depend on the particle
    // content), evaluate the observables per group
    std::map<int, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < count; ++i) {
        int subproc = has_subproc ? buffer.event(i).subprocess_index().value() : 0;
        if (subproc >= 0 && static_cast<std::size_t>(subproc) < _runtimes.size() &&
            _runtimes.at(subproc)) {
            groups[subproc].push_back(i);
        }
    }
    // values[observable][event index in buffer], NaN when not evaluated
    std::vector<std::vector<double>> values(
        _specs.size(), std::vector<double>(count, std::nan(""))
    );
    std::vector<TensorVec> group_outputs;
    for (auto& [subproc, indices] : groups) {
        auto& runtime_data = _runtimes.at(subproc).value();
        std::size_t pc = runtime_data.particle_count;
        Tensor momenta(DataType::dt_float64, {indices.size(), pc, 4});
        auto view = momenta.view<double, 3>();
        for (std::size_t n = 0; n < indices.size(); ++n) {
            for (std::size_t j = 0; j < pc; ++j) {
                auto particle = buffer.particle(indices[n], j);
                view[n][j][0] = particle.energy();
                view[n][j][1] = particle.px();
                view[n][j][2] = particle.py();
                view[n][j][3] = particle.pz();
            }
        }
        TensorVec outputs;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            outputs = runtime_data.runtime->run({momenta});
        }
        for (std::size_t o = 0; o < _specs.size(); ++o) {
            if (!_observable_index.at(o)) {
                continue; // filled from the event weight below
            }
            Tensor out = outputs.at(_observable_index.at(o).value()).cpu().contiguous();
            if (out.shape().size() == 1) {
                auto out_view = out.view<double, 1>();
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    values[o][indices[n]] = out_view[n];
                }
            } else {
                // vector-valued observable: histogram its first component, as
                // op_histogram does
                auto out_view = out.view<double, 2>();
                for (std::size_t n = 0; n < indices.size(); ++n) {
                    values[o][indices[n]] = out_view[n][0];
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(_mutex);
    if (_sums.empty()) {
        _weight_count = weight_count;
        _sums.assign(
            _specs.size(),
            std::vector<std::vector<double>>(weight_count + 1, std::vector<double>())
        );
        _square_sums = _sums;
        for (std::size_t o = 0; o < _specs.size(); ++o) {
            for (auto& column : _sums[o]) {
                column.assign(_specs[o].bin_count + 2, 0.);
            }
            for (auto& column : _square_sums[o]) {
                column.assign(_specs[o].bin_count + 2, 0.);
            }
        }
    } else if (weight_count != _weight_count) {
        throw std::invalid_argument("EventHistograms: inconsistent weight count");
    }
    for (std::size_t i = 0; i < count; ++i) {
        double w0 = buffer.event(i).weight();
        for (std::size_t o = 0; o < _specs.size(); ++o) {
            auto& spec = _specs[o];
            double value = spec.from_weight ? w0 / _reference_weight : values[o][i];
            if (std::isnan(value)) {
                continue;
            }
            double frac = (value - spec.min) / (spec.max - spec.min);
            std::size_t bin;
            if (frac < 0.) {
                bin = 0;
            } else if (frac >= 1.) {
                bin = spec.bin_count + 1;
            } else {
                bin = static_cast<std::size_t>(std::floor(frac * spec.bin_count)) + 1;
            }
            _sums[o][0][bin] += w0;
            _square_sums[o][0][bin] += w0 * w0;
            for (std::size_t k = 0; k < weight_count; ++k) {
                double w = syst_weights.at(i * weight_count + k);
                _sums[o][k + 1][bin] += w;
                _square_sums[o][k + 1][bin] += w * w;
            }
        }
    }
    _event_count += count;
}

json EventHistograms::to_json(const SystematicsCalculator* systematics) const {
    json result = json::array();
    if (_event_count == 0) {
        for (auto& spec : _specs) {
            result.push_back({
                {"name", spec.name},
                {"min", spec.min},
                {"max", spec.max},
                {"bin_count", spec.bin_count},
            });
        }
        return result;
    }
    double n = static_cast<double>(_event_count);
    auto normalised =
        [&](const std::vector<double>& sums,
            const std::vector<double>& squares,
            std::vector<double>& values,
            std::vector<double>& errors) {
            values.resize(sums.size());
            errors.resize(sums.size());
            for (std::size_t b = 0; b < sums.size(); ++b) {
                // the event weights are "average" normalised: the cross section in a
                // bin is the mean of (weight x indicator), its error the standard
                // error of that mean
                values[b] = sums[b] / n;
                double var = (squares[b] - sums[b] * sums[b] / n) / n;
                errors[b] = var > 0. ? std::sqrt(var / n) : 0.;
            }
        };
    std::vector<int> ids;
    std::vector<std::size_t> scale_indices;
    std::vector<PdfGroupInfo> pdf_groups;
    if (systematics && systematics->weight_count() == _weight_count) {
        ids = systematics->weight_ids();
        scale_indices = systematics->scale_variation_indices();
        pdf_groups = systematics->pdf_groups();
    }
    for (std::size_t o = 0; o < _specs.size(); ++o) {
        auto& spec = _specs[o];
        std::vector<std::vector<double>> values(_weight_count + 1),
            errors(_weight_count + 1);
        for (std::size_t c = 0; c <= _weight_count; ++c) {
            normalised(_sums[o][c], _square_sums[o][c], values[c], errors[c]);
        }
        json weights = json::array();
        for (std::size_t k = 0; k < _weight_count; ++k) {
            json entry{{"bin_values", values[k + 1]}, {"bin_errors", errors[k + 1]}};
            if (!ids.empty()) {
                entry["id"] = ids.at(k);
            }
            weights.push_back(entry);
        }
        json hist{
            {"name", spec.name},
            {"min", spec.min},
            {"max", spec.max},
            {"bin_count", spec.bin_count},
            {"bin_values", values[0]},
            {"bin_errors", errors[0]},
            {"weights", weights},
        };
        std::size_t nbins = spec.bin_count + 2;
        if (!scale_indices.empty()) {
            std::vector<double> low = values[0], high = values[0];
            for (std::size_t k : scale_indices) {
                for (std::size_t b = 0; b < nbins; ++b) {
                    low[b] = std::min(low[b], values[k + 1][b]);
                    high[b] = std::max(high[b], values[k + 1][b]);
                }
            }
            hist["scale_envelope"] = {{"low", low}, {"high", high}};
        }
        if (!pdf_groups.empty()) {
            json pdf = json::array();
            for (auto& group : pdf_groups) {
                std::vector<double> central(nbins), up(nbins), down(nbins);
                bool any = false;
                for (std::size_t b = 0; b < nbins; ++b) {
                    std::vector<std::pair<int, double>> member_values;
                    for (auto [member, k] : group.members) {
                        member_values.push_back({member, values[k + 1][b]});
                    }
                    std::optional<double> nominal;
                    if (systematics &&
                        group.set_lhaid == systematics->config().nominal_lhaid) {
                        nominal = values[0][b];
                    }
                    auto [c, u, d] = SystematicsCalculator::pdf_uncertainty(
                        group.error_type, nominal, member_values
                    );
                    central[b] = std::isnan(c) ? 0. : c;
                    up[b] = std::isnan(u) ? 0. : u;
                    down[b] = std::isnan(d) ? 0. : d;
                    any |= !std::isnan(u);
                }
                if (any) {
                    pdf.push_back({
                        {"pdf_set", group.set_name},
                        {"pdf_lhaid", group.set_lhaid},
                        {"error_type", group.error_type},
                        {"central", central},
                        {"uncertainty_up", up},
                        {"uncertainty_down", down},
                    });
                }
            }
            hist["pdf_uncertainty"] = pdf;
        }
        result.push_back(hist);
    }
    return result;
}
