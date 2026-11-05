// SPDX-License-Identifier: 0BSD

#ifndef RUSTICOL_HPP
#define RUSTICOL_HPP

#include "rusticol.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rusticol {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string &message) : std::runtime_error(message) {}
};

inline std::string last_error() {
    std::size_t required = 0;
    rusticol_last_error_message(nullptr, 0, &required);
    if (required == 0) {
        return "unknown Rusticol error";
    }
    std::vector<char> buffer(required);
    rusticol_last_error_message(buffer.data(), buffer.size(), &required);
    return std::string(buffer.data());
}

inline void check(int status) {
    if (status != RUSTICOL_STATUS_OK) {
        throw Error(last_error());
    }
}

inline std::string supported_runtime_capabilities_json() {
    std::size_t required = 0;
    check(rusticol_supported_runtime_capabilities_json(nullptr, 0, &required));
    std::vector<char> buffer(required);
    check(rusticol_supported_runtime_capabilities_json(
        buffer.data(), buffer.size(), &required));
    return std::string(buffer.data());
}

struct ExternalParticle {
    std::size_t index{};
    std::int32_t pdg{};
};

struct HelicityConfiguration {
    std::string id;
    std::vector<std::int32_t> helicities;
};

struct ColorComponent {
    std::string id;
    std::string kind;
    std::vector<std::size_t> word;
};

struct ModelParameter {
    std::string name;
};

struct SpinCorrelationVector {
    std::size_t leg{};  // One-based public external leg.
    // One four-vector broadcasts; otherwise supply one per momentum point.
    std::vector<std::array<std::complex<double>, 4>> components;
};

struct CorrelatedRequest {
    std::string color_correlation{"born"};
    // nullopt inherits the setter; an empty vector selects physical helicities.
    std::optional<std::vector<SpinCorrelationVector>> spin_vectors;
};

struct CorrelatedEvaluation {
    std::vector<std::complex<double>> values;  // [request][point]
    std::size_t request_count{};
    std::size_t point_count{};

    std::complex<double> operator()(std::size_t request, std::size_t point) const {
        if (request >= request_count || point >= point_count) {
            throw std::out_of_range("correlated Rusticol result index is out of range");
        }
        return values.at(request * point_count + point);
    }
};

enum class WarmUpEventKind : std::uint32_t {
    start = RUSTICOL_WARM_UP_EVENT_START,
    update = RUSTICOL_WARM_UP_EVENT_UPDATE,
    end = RUSTICOL_WARM_UP_EVENT_END,
};

enum class WarmUpStage : std::uint32_t {
    process_preparation = RUSTICOL_WARM_UP_STAGE_PROCESS_PREPARATION,
    query_family = RUSTICOL_WARM_UP_STAGE_QUERY_FAMILY,
    family_finalization = RUSTICOL_WARM_UP_STAGE_FAMILY_FINALIZATION,
    first_evaluation = RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION,
};

struct WarmUpProgress {
    std::uint32_t schema_version{};
    WarmUpEventKind kind{};
    WarmUpStage stage{};
    std::uint64_t completed{};
    std::uint64_t total{};
    double elapsed_seconds{};
    std::uint64_t current_rss_bytes{};
    std::uint64_t peak_rss_bytes{};
    std::uint64_t workers{};
    bool current_rss_available{};
    bool peak_rss_available{};
};

struct WarmUpResult {
    std::uint32_t schema_version{};
    double elapsed_seconds{};
    std::uint64_t query_count{};
    std::uint64_t warmed_query_count{};
    std::uint64_t current_rss_bytes{};
    std::uint64_t peak_rss_bytes{};
    bool current_rss_available{};
    bool peak_rss_available{};
    bool already_warm{};
    bool first_evaluation_completed{};
};

namespace detail {

struct SpinCorrelationStorage {
    std::vector<std::vector<double>> components;
    std::vector<RusticolSpinCorrelationVector> vectors;

    explicit SpinCorrelationStorage(const std::vector<SpinCorrelationVector> &input) {
        components.reserve(input.size());
        vectors.reserve(input.size());
        for (const auto &item : input) {
            components.emplace_back();
            auto &flat = components.back();
            for (const auto &point : item.components) {
                for (const auto &value : point) {
                    flat.push_back(value.real());
                    flat.push_back(value.imag());
                }
            }
            vectors.push_back({item.leg, item.components.size(),
                               flat.empty() ? nullptr : flat.data(), flat.size()});
        }
    }
};

inline WarmUpProgress warm_up_progress(const RusticolWarmUpProgressEvent &event) {
    return {
        event.schema_version,
        static_cast<WarmUpEventKind>(event.kind),
        static_cast<WarmUpStage>(event.stage),
        event.completed,
        event.total,
        event.elapsed_seconds,
        event.current_rss_bytes,
        event.peak_rss_bytes,
        event.workers,
        event.current_rss_available != 0,
        event.peak_rss_available != 0,
    };
}

inline WarmUpResult warm_up_result(const RusticolWarmUpResult &result) {
    return {
        result.schema_version,
        result.elapsed_seconds,
        result.query_count,
        result.warmed_query_count,
        result.current_rss_bytes,
        result.peak_rss_bytes,
        result.current_rss_available != 0,
        result.peak_rss_available != 0,
        result.already_warm != 0,
        result.first_evaluation_completed != 0,
    };
}

struct WarmUpCallbackState {
    const std::function<bool(const WarmUpProgress &)> *callback{};
    std::exception_ptr exception;
};

inline int warm_up_progress_callback(
    const RusticolWarmUpProgressEvent *event,
    void *user_data) noexcept {
    if (event == nullptr || user_data == nullptr) {
        return 0;
    }
    auto &state = *static_cast<WarmUpCallbackState *>(user_data);
    if (state.callback == nullptr) {
        return 0;
    }
    try {
        return (*state.callback)(warm_up_progress(*event)) ? 1 : 0;
    } catch (...) {
        const bool terminal =
            event->kind == RUSTICOL_WARM_UP_EVENT_END &&
            event->stage == RUSTICOL_WARM_UP_STAGE_FIRST_EVALUATION;
        if (!terminal) {
            state.exception = std::current_exception();
            return 0;
        }
        // This event is emitted after the selected family and first evaluation
        // have committed. Match the core contract by treating it as a
        // non-cancellable notification.
        return 1;
    }
}

}  // namespace detail

class ResolvedEvaluation {
public:
    std::vector<double> values;
    std::size_t point_count{};
    std::vector<HelicityConfiguration> helicities;
    std::vector<ColorComponent> colors;

    double operator()(std::size_t point, std::size_t helicity, std::size_t color) const {
        if (point >= point_count || helicity >= helicities.size() || color >= colors.size()) {
            throw std::out_of_range("resolved Rusticol result index is out of range");
        }
        return values[(point * helicities.size() + helicity) * colors.size() + color];
    }

    std::vector<double> total() const {
        const std::size_t stride = helicities.size() * colors.size();
        std::vector<double> totals(point_count, 0.0);
        for (std::size_t point = 0; point < point_count; ++point) {
            for (std::size_t component = 0; component < stride; ++component) {
                totals[point] += values[point * stride + component];
            }
        }
        return totals;
    }
};

class Runtime {
public:
    Runtime(const std::string &process_dir,
            const std::string &process_key = {},
            const std::string &model_parameters = {}) {
        RusticolRuntimeHandle *loaded = nullptr;
        check(rusticol_runtime_load(
            process_dir.c_str(),
            process_key.empty() ? nullptr : process_key.c_str(),
            model_parameters.empty() ? nullptr : model_parameters.c_str(),
            &loaded));
        handle_ = loaded;
    }

    ~Runtime() {
        if (handle_ != nullptr) {
            rusticol_runtime_free(handle_);
        }
    }

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    Runtime(Runtime &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

    Runtime &operator=(Runtime &&other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) {
                rusticol_runtime_free(handle_);
            }
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    // Save the completed, retained OTF cache, without warming new selections.
    void save(const std::string &path) const {
        if (path.find('\0') != std::string::npos) {
            throw std::invalid_argument("cache path contains a NUL byte");
        }
        check(rusticol_runtime_save(handle_, path.c_str()));
    }

    // Restore an OTF cache after loading the matching process output normally.
    void load_cache(const std::string &path) {
        if (path.find('\0') != std::string::npos) {
            throw std::invalid_argument("cache path contains a NUL byte");
        }
        check(rusticol_runtime_load_cache(handle_, path.c_str()));
    }

    std::string process() const { return get_string(rusticol_runtime_process); }
    std::string process_key() const { return get_string(rusticol_runtime_process_key); }
    std::string representative_process_key() const {
        return get_string(rusticol_runtime_representative_process_key);
    }
    std::string color_accuracy() const { return get_string(rusticol_runtime_color_accuracy); }
    std::string execution_mode() const { return get_string(rusticol_runtime_execution_mode); }
    std::string metadata_json() const { return get_string(rusticol_runtime_metadata_json); }
    std::string physics_json() const { return get_string(rusticol_runtime_physics_json); }

    std::vector<ExternalParticle> external_particles() const {
        std::size_t count = 0;
        check(rusticol_runtime_external_count(handle_, &count));
        std::vector<ExternalParticle> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::int32_t pdg = 0;
            check(rusticol_runtime_external_pdg(handle_, index, &pdg));
            result.push_back({index, pdg});
        }
        return result;
    }

    std::vector<std::size_t> external_permutation() const {
        std::size_t required = 0;
        check(rusticol_runtime_external_permutation(
            handle_, nullptr, 0, &required));
        std::vector<std::size_t> result(required);
        if (required != 0) {
            check(rusticol_runtime_external_permutation(
                handle_, result.data(), result.size(), &required));
        }
        return result;
    }

    std::vector<double> load_kinematics_json(const std::string &path) const {
        std::size_t required = 0;
        check(rusticol_runtime_load_kinematics_json(
            handle_, path.c_str(), nullptr, 0, &required));
        std::vector<double> result(required);
        if (required != 0) {
            check(rusticol_runtime_load_kinematics_json(
                handle_, path.c_str(), result.data(), result.size(), &required));
        }
        return result;
    }

    std::vector<HelicityConfiguration> helicities() const {
        std::size_t count = 0;
        check(rusticol_runtime_helicity_count(handle_, &count));
        std::vector<HelicityConfiguration> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t vector_size = 0;
            check(rusticol_runtime_helicity_vector(handle_, index, nullptr, 0, &vector_size));
            std::vector<std::int32_t> vector(vector_size);
            check(rusticol_runtime_helicity_vector(
                handle_, index, vector.data(), vector.size(), &vector_size));
            result.push_back({get_indexed_string(rusticol_runtime_helicity_id, index), vector});
        }
        return result;
    }

    std::vector<ColorComponent> colors() const {
        std::size_t count = 0;
        check(rusticol_runtime_color_count(handle_, &count));
        std::vector<ColorComponent> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t word_size = 0;
            check(rusticol_runtime_color_word(handle_, index, nullptr, 0, &word_size));
            std::vector<std::size_t> word(word_size);
            check(rusticol_runtime_color_word(
                handle_, index, word.data(), word.size(), &word_size));
            result.push_back({
                get_indexed_string(rusticol_runtime_color_id, index),
                get_indexed_string(rusticol_runtime_color_kind, index),
                word,
            });
        }
        return result;
    }

    std::vector<ModelParameter> model_parameters() const {
        std::size_t count = 0;
        check(rusticol_runtime_model_parameter_count(handle_, &count));
        std::vector<ModelParameter> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            result.push_back({get_indexed_string(rusticol_runtime_model_parameter_name, index)});
        }
        return result;
    }

    std::vector<double> evaluate(const std::vector<double> &momenta,
                                 std::size_t point_count) {
        std::vector<double> values(point_count);
        check(rusticol_runtime_evaluate_f64(
            handle_, momenta.data(), momenta.size(), point_count, values.data(), values.size()));
        return values;
    }

    std::string color_correlation_catalogue_json() {
        return get_string_mut(rusticol_runtime_color_correlation_catalogue_json);
    }

    std::vector<std::string> color_correlation_ids() {
        std::size_t count = 0;
        check(rusticol_runtime_color_correlation_count(handle_, &count));
        std::vector<std::string> result;
        result.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            std::size_t required = 0;
            check(rusticol_runtime_color_correlation_id(handle_, index, nullptr, 0, &required));
            std::vector<char> buffer(required);
            check(rusticol_runtime_color_correlation_id(
                handle_, index, buffer.data(), buffer.size(), &required));
            result.emplace_back(buffer.data());
        }
        return result;
    }

    // Empty input clears defaults; this does not affect ordinary evaluate().
    void set_spin_correlation_vectors(const std::vector<SpinCorrelationVector> &vectors = {}) {
        const detail::SpinCorrelationStorage storage(vectors);
        check(rusticol_runtime_set_spin_correlation_vectors_f64(
            handle_, storage.vectors.empty() ? nullptr : storage.vectors.data(),
            storage.vectors.size()));
    }

    CorrelatedEvaluation evaluate_correlated_many(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::vector<CorrelatedRequest> &requests,
        const std::vector<std::string> &helicity_ids = {}) {
        if (point_count == 0 || requests.empty() ||
            point_count > std::numeric_limits<std::size_t>::max() / 2 / requests.size()) {
            throw std::invalid_argument("invalid correlated result dimensions");
        }
        std::vector<detail::SpinCorrelationStorage> storage;
        std::vector<RusticolCorrelatedRequest> raw_requests;
        const std::vector<SpinCorrelationVector> physical_helicities;
        storage.reserve(requests.size());
        raw_requests.reserve(requests.size());
        for (const auto &request : requests) {
            if (request.color_correlation.find('\0') != std::string::npos) {
                throw std::invalid_argument("colour correlation ID contains a NUL byte");
            }
            storage.emplace_back(request.spin_vectors ? *request.spin_vectors : physical_helicities);
            const auto &vectors = storage.back().vectors;
            raw_requests.push_back({request.color_correlation.c_str(),
                                    vectors.empty() ? nullptr : vectors.data(), vectors.size(),
                                    request.spin_vectors ? 0u : 1u});
        }
        for (const auto &id : helicity_ids) {
            if (id.find('\0') != std::string::npos) {
                throw std::invalid_argument("helicity ID contains a NUL byte");
            }
        }
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        std::vector<double> raw(requests.size() * point_count * 2);
        check(rusticol_runtime_evaluate_correlated_many_f64(
            handle_, momenta.data(), momenta.size(), point_count,
            raw_requests.data(), raw_requests.size(),
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(), helicity_ptrs.size(),
            raw.data(), raw.size()));
        CorrelatedEvaluation result{{}, requests.size(), point_count};
        result.values.reserve(raw.size() / 2);
        for (std::size_t index = 0; index < raw.size(); index += 2) {
            result.values.emplace_back(raw[index], raw[index + 1]);
        }
        return result;
    }

    std::vector<std::complex<double>> evaluate_correlated(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::string &color_correlation = "born",
        const std::optional<std::vector<SpinCorrelationVector>> &spin_vectors = std::nullopt,
        const std::vector<std::string> &helicity_ids = {}) {
        return evaluate_correlated_many(
            momenta, point_count, {{color_correlation, spin_vectors}}, helicity_ids).values;
    }

    WarmUpResult warm_up(
        const std::vector<double> &point,
        const std::vector<std::string> &helicity_ids = {},
        const std::vector<std::string> &color_ids = {},
        const std::function<bool(const WarmUpProgress &)> &progress = {},
        std::size_t n_cores = 0) {
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        const auto color_ptrs = c_string_pointers(color_ids);
        RusticolWarmUpResult result{};
        detail::WarmUpCallbackState callback_state{progress ? &progress : nullptr, {}};
        const int status = rusticol_runtime_warm_up_f64_with_cores(
            handle_,
            point.data(),
            point.size(),
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            n_cores,
            progress ? detail::warm_up_progress_callback : nullptr,
            progress ? static_cast<void *>(&callback_state) : nullptr,
            &result);
        if (callback_state.exception) {
            std::rethrow_exception(callback_state.exception);
        }
        check(status);
        const auto converted = detail::warm_up_result(result);
        if (converted.schema_version != 1 || !converted.first_evaluation_completed) {
            throw Error("Rusticol returned an invalid successful warm-up result");
        }
        return converted;
    }

    std::vector<double> evaluate_selected(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::vector<std::string> &helicity_ids = {},
        const std::vector<std::string> &color_ids = {},
        const std::vector<std::uint32_t> &helicity_by_point = {},
        const std::vector<std::uint32_t> &color_flow_by_point = {}) {
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        const auto color_ptrs = c_string_pointers(color_ids);
        std::vector<double> values(point_count);
        check(rusticol_runtime_evaluate_selected_f64(
            handle_,
            momenta.data(),
            momenta.size(),
            point_count,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            helicity_by_point.empty() ? nullptr : helicity_by_point.data(),
            helicity_by_point.size(),
            color_flow_by_point.empty() ? nullptr : color_flow_by_point.data(),
            color_flow_by_point.size(),
            values.data(),
            values.size()));
        return values;
    }

    ResolvedEvaluation evaluate_resolved(
        const std::vector<double> &momenta,
        std::size_t point_count,
        const std::vector<std::string> &helicity_ids = {},
        const std::vector<std::string> &color_ids = {}) {
        const auto helicity_ptrs = c_string_pointers(helicity_ids);
        const auto color_ptrs = c_string_pointers(color_ids);
        std::size_t helicity_count = 0;
        std::size_t color_count = 0;
        check(rusticol_runtime_resolved_shape(
            handle_,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            &helicity_count,
            &color_count));
        std::vector<double> values(point_count * helicity_count * color_count);
        check(rusticol_runtime_evaluate_resolved_f64(
            handle_,
            momenta.data(),
            momenta.size(),
            point_count,
            helicity_ptrs.empty() ? nullptr : helicity_ptrs.data(),
            helicity_ptrs.size(),
            color_ptrs.empty() ? nullptr : color_ptrs.data(),
            color_ptrs.size(),
            values.data(),
            values.size(),
            &helicity_count,
            &color_count));

        auto all_helicities = helicities();
        auto all_colors = colors();
        return {
            std::move(values),
            point_count,
            select_helicities(all_helicities, helicity_ids),
            select_colors(all_colors, color_ids),
        };
    }

    void set_model_parameter(const std::string &name, double real, double imaginary = 0.0) {
        check(rusticol_runtime_set_model_parameter(handle_, name.c_str(), real, imaginary));
    }

    void set_model_parameters(const std::vector<std::string> &names,
                              const std::vector<double> &real,
                              const std::vector<double> &imaginary) {
        if (names.size() != real.size() || names.size() != imaginary.size()) {
            throw std::invalid_argument("model parameter arrays have different lengths");
        }
        const auto pointers = c_string_pointers(names);
        check(rusticol_runtime_set_model_parameters(
            handle_, pointers.data(), real.data(), imaginary.data(), names.size()));
    }

    void set_model_parameters_json(const std::string &path) {
        check(rusticol_runtime_set_model_parameters_json(handle_, path.c_str()));
    }

    void mute_warnings() { check(rusticol_runtime_mute_warnings(handle_, 1)); }
    void unmute_warnings() { check(rusticol_runtime_mute_warnings(handle_, 0)); }
    std::string take_warnings_json() {
        return get_string_mut(rusticol_runtime_take_warnings_json);
    }

private:
    using StringFunction = int (*)(const RusticolRuntimeHandle *, char *, std::size_t, std::size_t *);
    using MutableStringFunction = int (*)(RusticolRuntimeHandle *, char *, std::size_t, std::size_t *);
    using IndexedStringFunction = int (*)(const RusticolRuntimeHandle *, std::size_t, char *, std::size_t, std::size_t *);

    std::string get_string(StringFunction function) const {
        std::size_t required = 0;
        check(function(handle_, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    std::string get_string_mut(MutableStringFunction function) {
        std::size_t required = 0;
        check(function(handle_, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    std::string get_indexed_string(IndexedStringFunction function, std::size_t index) const {
        std::size_t required = 0;
        check(function(handle_, index, nullptr, 0, &required));
        std::vector<char> buffer(required);
        check(function(handle_, index, buffer.data(), buffer.size(), &required));
        return std::string(buffer.data());
    }

    static std::vector<const char *> c_string_pointers(const std::vector<std::string> &values) {
        std::vector<const char *> pointers;
        pointers.reserve(values.size());
        for (const auto &value : values) {
            pointers.push_back(value.c_str());
        }
        return pointers;
    }

    static std::vector<HelicityConfiguration> select_helicities(
        const std::vector<HelicityConfiguration> &available,
        const std::vector<std::string> &selected) {
        if (selected.empty()) {
            return available;
        }
        std::vector<HelicityConfiguration> result;
        for (const auto &item : available) {
            for (const auto &id : selected) {
                if (item.id == id) {
                    result.push_back(item);
                }
            }
        }
        return result;
    }

    static std::vector<ColorComponent> select_colors(
        const std::vector<ColorComponent> &available,
        const std::vector<std::string> &selected) {
        if (selected.empty()) {
            return available;
        }
        std::vector<ColorComponent> result;
        for (const auto &item : available) {
            for (const auto &id : selected) {
                if (item.id == id) {
                    result.push_back(item);
                }
            }
        }
        return result;
    }

    RusticolRuntimeHandle *handle_{};
};

}  // namespace rusticol

#endif
