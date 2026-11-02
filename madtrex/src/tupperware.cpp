#include "madtrex/tupperware.hpp"

#include <algorithm>
#include <any>
#include <format>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace madtrex {

namespace detail {

std::mutex& api_call_mutex(const madspace::MatrixElementApi& api) {
    static std::mutex registry_mutex;
    static std::unordered_map<const madspace::MatrixElementApi*, std::unique_ptr<std::mutex>>
        registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto& slot = registry[&api];
    if (!slot) {
        slot = std::make_unique<std::mutex>();
    }
    return *slot;
}

void warn_once(const std::string& key, const std::string& message) {
    static std::mutex mutex;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(mutex);
    if (seen.insert(key).second) {
        std::cerr << "MadtRex warning: " << message << std::endl;
    }
}

namespace {

// Keyed by MatrixElementApi::file_name(), the library: the values set on it
// (see parameters_lock() in tupperware.hpp for why they are kept).
struct LibraryParameters {
    std::map<std::string, std::pair<double, double>> values;
};

std::mutex parameters_mutex;
std::unordered_map<std::string, LibraryParameters> parameters_registry;

LibraryParameters library_parameters(const std::string& library) {
    std::lock_guard<std::mutex> lock(parameters_mutex);
    auto found = parameters_registry.find(library);
    return found == parameters_registry.end() ? LibraryParameters{} : found->second;
}

void apply_parameters(const madspace::MatrixElementApi& target, const LibraryParameters& parameters) {
    for (auto& [location, value] : parameters.values) {
        target.set_parameter(location, value.first, value.second);
    }
}

} // namespace

std::shared_mutex& parameters_lock() {
    static std::shared_mutex mutex;
    return mutex;
}

void set_api_parameter(
    const madspace::MatrixElementApi& api, const std::string& location, double real, double imag
) {
    api.set_parameter(location, real, imag);
    std::lock_guard<std::mutex> lock(parameters_mutex);
    auto& entry = parameters_registry[api.file_name()];
    entry.values[location] = {real, imag};
}

void forget_api_parameters(const madspace::MatrixElementApi& api) {
    std::lock_guard<std::mutex> lock(parameters_mutex);
    parameters_registry.erase(api.file_name());
}

const madspace::MatrixElementApi&
reload_matrix_element(const ReloadSpec& reload, REX::process& proc) {
    std::string key = std::format(
        "madtrex::reload::{}::{}::{}", static_cast<const void*>(reload.context.get()),
        reload.me_path, reload.param_card
    );
    if (proc.has_device_data(key)) {
        return *std::any_cast<const madspace::MatrixElementApi*>(proc.get_device_data(key).tag);
    }
    const madspace::MatrixElementApi* api;
    {
        // no evaluation runs while the parameters are reset to the
        // param_card and set back (and a madspace::Context is not safe to
        // load into concurrently)
        std::unique_lock<std::shared_mutex> exclusive(parameters_lock());
        api = &reload.context->load_matrix_element(reload.me_path, reload.param_card);
        apply_parameters(*api, library_parameters(api->file_name()));
    }
    proc.set_device_data(key, nullptr, 0, std::any(api));
    return *api;
}

} // namespace detail

namespace {

using madspace::MatrixElementApi;

// umami_supported_inputs/umami_required_inputs/umami_supported_outputs are optional
// symbols: MatrixElementApi falls back to a stub that reports
// UMAMI_ERROR_NOT_IMPLEMENTED, which the query methods turn into a thrown
// runtime_error. Treat "not implemented" as "unknown" rather than propagating.
std::optional<std::vector<bool>> try_supported_inputs(const MatrixElementApi& api) {
    try {
        return api.supported_inputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<std::vector<bool>> try_required_inputs(const MatrixElementApi& api) {
    try {
        return api.required_inputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<std::vector<bool>> try_supported_outputs(const MatrixElementApi& api) {
    try {
        return api.supported_outputs();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// LHECompleter (madspace/src/driver/lhe_output.cpp, complete_event_data())
// can insert status==2 intermediate resonance lines into events whose
// propagator falls inside its Breit-Wigner window. Those lines matter for
// LHE consumers but are not part of what a compiled matrix element expects
// as external legs (api.particle_count() only counts incoming+outgoing
// partons), so the per-event view fed to UMAMI must exclude them. Builds a
// fresh process from copies of proc.events with per-event indices
// restricted to non-intermediate (status != 2) partons, leaving proc and
// its underlying (possibly shared) events untouched.
REX::process external_legs_process(REX::process& proc, const IncomingSwapped* incoming_swapped) {
    std::vector<REX::event> filtered_events;
    filtered_events.reserve(proc.events.size());
    for (auto& ev_ptr : proc.events) {
        REX::event copy = *ev_ptr;
        std::vector<std::size_t> external_indices;
        external_indices.reserve(copy.n_);
        for (std::size_t i = 0; i < copy.n_; ++i) {
            if (copy.status_[i] != 2) external_indices.push_back(i);
        }
        // the matrix element's leg order for a beam-swapped event (see
        // IncomingSwapped): the two incoming legs come first either way
        if (incoming_swapped && *incoming_swapped && external_indices.size() >= 2 &&
            (*incoming_swapped)(*ev_ptr)) {
            std::swap(external_indices[0], external_indices[1]);
        }
        copy.set_indices(external_indices);
        filtered_events.push_back(std::move(copy));
    }
    return REX::process(std::move(filtered_events), /*filter_partons=*/true, /*column_major=*/false);
}

void check_particle_counts(const MatrixElementApi& api, REX::process& proc) {
    std::size_t expected = api.particle_count();
    for (std::size_t i = 0; i < proc.size(); ++i) {
        std::size_t begin = (i == 0) ? 0 : proc.n_summed[i - 1];
        std::size_t end = proc.n_summed[i];
        if (end - begin != expected) {
            throw std::runtime_error(std::format(
                "make_weightor: event {} has {} particles, but matrix element {} "
                "expects {}",
                i,
                end - begin,
                api.file_name(),
                expected
            ));
        }
    }
}

// Check whether the process has been transposed to the UMAMI
// memory layout, and if not, transpose it
// Also checks that the number of particles in the process
// matches the number expected by the matrix element
bool gather_momenta(const MatrixElementApi& api, REX::process& proc) {
    std::size_t count = proc.size();
    std::size_t particles = api.particle_count();
    if (proc.umami_momenta().size() == 0) proc.to_umami();
    if (proc.umami_momenta().size() != count * particles * 4) {
        throw std::runtime_error(std::format(
            "make_weightor: process has {} events, but matrix element {} expects "
            "{} particles per event",
            count,
            api.file_name(),
            particles
        ));
    }
    return true;
}

// Owns one device-side scratch allocation used to stage a single input or
// output across to a CUDA/HIP matrix element library: unlike CPU UMAMI
// implementations, those expect device-resident pointers, not the host
// memory REX::process stores its data in. Freed through the same Device
// regardless of how evaluate() exits (including via check_umami_status's
// exceptions), so callers don't need their own cleanup path.
class DeviceScratch {
public:
    DeviceScratch() = default;
    DeviceScratch(madspace::DevicePtr device, std::size_t bytes) : _device(device) {
        if (bytes > 0) _ptr = _device->allocate(bytes, madspace::AllocHint::normal).first;
    }
    ~DeviceScratch() {
        if (_ptr) _device->free(_ptr);
    }
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
    DeviceScratch(DeviceScratch&& other) noexcept : _device(other._device), _ptr(other._ptr) {
        other._ptr = nullptr;
    }
    DeviceScratch& operator=(DeviceScratch&& other) noexcept {
        if (this != &other) {
            if (_ptr) _device->free(_ptr);
            _device = other._device;
            _ptr = other._ptr;
            other._ptr = nullptr;
        }
        return *this;
    }
    void* get() const { return _ptr; }

private:
    madspace::DevicePtr _device = nullptr;
    void* _ptr = nullptr;
};

using detail::warn_once;

// The input keys passed for proc, after checking that proc can be reweighted
// with them: with UMAMI_IN_HELICITY_INDEX (helicity reweighting) every event
// of proc must carry a helicity index (event::helicity_, set by the
// process's event checker from the LHE SPINUP column), ie a SPINUP of -1, 0
// or +1 for every particle. A sample without that helicity information (eg
// SPINUP 9, "unknown/summed") cannot be reweighted helicity by helicity:
// throws, rather than silently summing over helicities instead.
const std::vector<UmamiInputKey>& keys_for_process(
    const MatrixElementApi& api,
    const REX::process& proc,
    std::size_t count,
    const std::vector<UmamiInputKey>& input_keys
) {
    if (std::find(input_keys.begin(), input_keys.end(), UMAMI_IN_HELICITY_INDEX) ==
        input_keys.end()) {
        return input_keys;
    }
    const auto& helicity = proc.helicity();
    if (helicity.size() != count) {
        throw std::runtime_error(std::format(
            "helicity reweighting: the events matched to matrix element {} carry no "
            "helicity indices; reweight with helicity summation instead "
            "(set helicity_reweighting false in the reweight card, or "
            "--no-helicity-reweighting)",
            api.file_name()
        ));
    }
    auto missing = std::find_if(helicity.begin(), helicity.end(), [](int h) { return h < 0; });
    if (missing != helicity.end() && *missing == helicity_not_in_table) {
        throw std::runtime_error(std::format(
            "helicity reweighting: the helicity configuration (LHE SPINUP) of event {} "
            "(0-based) of the {} events matched to matrix element {} is not in the "
            "helicity table of its subprocess, so it has no helicity index in the "
            "matrix element; the sample does not match this process directory",
            missing - helicity.begin(),
            count,
            api.file_name()
        ));
    }
    if (missing != helicity.end()) {
        throw std::runtime_error(std::format(
            "helicity reweighting: event {} (0-based) of the {} events matched to "
            "matrix element {} has no helicity information (a SPINUP other than -1, "
            "0 or +1, eg 9, "
            "for at least one particle); reweight this sample with helicity "
            "summation instead (set helicity_reweighting false in the reweight card, "
            "or --no-helicity-reweighting)",
            missing - helicity.begin(),
            count,
            api.file_name()
        ));
    }
    return input_keys;
}

// Per-weightor-closure cache of everything evaluate() derives from a
// REX::process: momenta/alphaS/flavor/helicity (and, on cuda/hip, their
// device-resident copies) don't change between reweighting iterations --
// only UMAMI parameters do, via umami_set_parameter -- so gathering them
// (and, off-cpu, copying them to the device) only needs to happen once per
// process this weightor ends up bound to, not on every evaluate() call.
// Output storage (host buffers, and on-device their staging buffers) is
// also kept here so its *allocation* is reused across iterations, even
// though its *contents* are still overwritten by every call, since the
// matrix element's value does change iteration to iteration.
//
// Copying an EvalCache -- which happens whenever its owning weightor
// (and thus its owning procReweightor) is itself copied, eg
// TupperWare::build()'s rwgts.push_back(*weightor_ptr), or
// reweightor::extract_sorter()'s max_batch overflow cloning -- deliberately
// resets to a blank cache rather than duplicating the live buffers: the two
// copies always end up bound to different processes (or, for max_batch
// clones, may even run concurrently on different threads), so sharing
// state across them would be wrong or unsafe. Each repopulates
// independently on its own first call.
struct EvalCache {
    EvalCache() = default;
    EvalCache(const EvalCache&) : EvalCache() {}
    EvalCache& operator=(const EvalCache&) {
        *this = EvalCache();
        return *this;
    }
    EvalCache(EvalCache&&) noexcept = default;
    EvalCache& operator=(EvalCache&&) noexcept = default;

    const REX::process* proc_id = nullptr; // identifies which process this cache is valid for
    std::size_t count = 0;

    // The MatrixElementApi instance actually evaluated against for proc_id:
    // either the api the owning weightor was built with, or -- if built via
    // the ReloadSpec overload -- a freshly-loaded instance obtained the
    // first time this cache populated for proc_id. Resolved once per
    // process (see evaluate()) and reused for every later call against the
    // same process; resets to nullptr, like the rest of EvalCache, whenever
    // this weightor's closure is copied (see the copy ctor below), so each
    // copy resolves (and, with a ReloadSpec, reloads) independently on its
    // own first use.
    const MatrixElementApi* active_api = nullptr;

    // Backs input_ptrs[UMAMI_IN_MOMENTA] on cpu; must outlive the cache entry.
    REX::process external_proc;
    std::vector<UmamiInputKey> active_inputs;
    std::vector<void const*> input_ptrs;
    std::vector<DeviceScratch> input_scratch; // keeps device input buffers alive; empty on cpu

    std::unordered_map<UmamiOutputKey, std::vector<double>> output_buffers;
    std::vector<void*> output_ptrs; // positional, aligned to output_keys
    std::vector<DeviceScratch> output_scratch; // reused device output allocation; empty on cpu
};

std::shared_ptr<std::vector<double>> evaluate(
    const MatrixElementApi& api,
    REX::process& proc,
    const std::vector<UmamiInputKey>& input_keys,
    const std::vector<UmamiOutputKey>& output_keys,
    const std::vector<UmamiOutputKey>& result_keys,
    EvalCache& cache,
    const ReloadSpec* reload = nullptr,
    const IncomingSwapped* incoming_swapped = nullptr
) {
    std::size_t count = proc.size();
    if (count == 0) {
        return std::make_shared<std::vector<double>>();
    }

    // Which MatrixElementApi instance this call actually evaluates against:
    // whichever one was already resolved into the cache for this process
    // (reused verbatim, cache hit or not -- reloading again on every call
    // would defeat the purpose of a per-process reload), or, the first time
    // this cache sees a given process, either a fresh instance (reload set)
    // or api itself (no reload spec, the pre-existing behavior).
    const MatrixElementApi* resolved = cache.active_api;
    bool new_process = cache.proc_id != &proc || cache.count != count;
    if (!resolved || new_process) {
        resolved = reload ? &detail::reload_matrix_element(*reload, proc) : &api;
    }
    const MatrixElementApi& active_api = *resolved;

    // REX::process always stores its data on the host, but a cuda/hip matrix
    // element's umami_matrix_element expects device-resident input/output
    // pointers. When that's the case, inputs are staged through DeviceScratch
    // buffers copied from the host data below, and outputs are read back into
    // the host-side output_buffers after the call; on cpu, host pointers are
    // passed straight through as before.
    bool on_device = active_api.device_type() != madspace::DeviceType::cpu;
    madspace::DevicePtr device = on_device ? active_api.device() : nullptr;

    if (new_process) {
        cache = EvalCache{};
        cache.proc_id = &proc;
        cache.count = count;
        cache.active_api = resolved;

        // external_proc drops any status==2 intermediate resonance lines
        // LHECompleter may have inserted, so particle counts/momenta match
        // what the matrix element expects (its external legs); event-level
        // quantities (alphaS/flavor/helicity below) are unaffected by that
        // filtering and still come from proc directly.
        cache.external_proc = external_legs_process(proc, incoming_swapped);
        check_particle_counts(active_api, cache.external_proc);

        const auto& process_keys = keys_for_process(active_api, proc, count, input_keys);
        cache.active_inputs.reserve(process_keys.size());
        cache.input_ptrs.reserve(process_keys.size());

        auto stage_input = [&](UmamiInputKey key, const void* host_ptr, std::size_t bytes) {
            cache.active_inputs.push_back(key);
            if (on_device) {
                DeviceScratch scratch(device, bytes);
                device->memcpy(scratch.get(), const_cast<void*>(host_ptr), bytes);
                cache.input_ptrs.push_back(scratch.get());
                cache.input_scratch.push_back(std::move(scratch));
            } else {
                cache.input_ptrs.push_back(host_ptr);
            }
        };

        for (auto key : process_keys) {
            switch (key) {
            case UMAMI_IN_MOMENTA:
                if (!gather_momenta(active_api, cache.external_proc)) {
                    throw std::runtime_error(
                        "make_weightor: failed to gather momenta for UMAMI"
                    );
                }
                stage_input(
                    key,
                    cache.external_proc.umami_momenta().data(),
                    cache.external_proc.umami_momenta().size() * sizeof(double)
                );
                break;
            case UMAMI_IN_ALPHA_S:
                if (proc.alphaS().size() != count) {
                    throw std::runtime_error(
                        "make_weightor: process does not carry a per-event alpha_s "
                        "value for every event"
                    );
                }
                stage_input(key, proc.alphaS().data(), count * sizeof(double));
                break;
            case UMAMI_IN_FLAVOR_INDEX:
                if (proc.flavor().size() != count) {
                    throw std::runtime_error(
                        "make_weightor: process does not carry a per-event flavor "
                        "index for every event"
                    );
                }
                stage_input(key, proc.flavor().data(), count * sizeof(int));
                break;
            case UMAMI_IN_HELICITY_INDEX:
                if (proc.helicity().size() != count) {
                    throw std::runtime_error(
                        "make_weightor: process does not carry a per-event helicity "
                        "index for every event"
                    );
                }
                stage_input(key, proc.helicity().data(), count * sizeof(int));
                break;
            default:
                throw std::runtime_error(
                    "make_weightor: input key cannot be derived from a REX::process"
                );
            }
        }

        // On device, one DeviceScratch per output_keys entry (positionally
        // aligned with output_ptrs/output_keys) rather than per unique key,
        // so it's read back into the right host buffer below regardless of
        // call order. Allocated once here and reused (overwritten) by every
        // subsequent call for this process, not freed/reallocated per call.
        cache.output_ptrs.reserve(output_keys.size());
        for (auto key : output_keys) {
            std::size_t buf_size;
            switch (key) {
            case UMAMI_OUT_MATRIX_ELEMENT:
                buf_size = count;
                break;
            case UMAMI_OUT_DIAGRAM_AMP2:
                buf_size = active_api.diagram_count() * count;
                break;
            default:
                throw std::runtime_error(
                    "make_weightor: output key is not a reweighting-usable value"
                );
            }
            auto [it, inserted] =
                cache.output_buffers.emplace(key, std::vector<double>(buf_size, 0.0));
            if (on_device) {
                cache.output_scratch.emplace_back(device, buf_size * sizeof(double));
                cache.output_ptrs.push_back(cache.output_scratch.back().get());
            } else {
                cache.output_ptrs.push_back(it->second.data());
            }
        }
    }

    {
        std::shared_lock<std::shared_mutex> parameters(detail::parameters_lock());
        std::lock_guard<std::mutex> guard(detail::api_call_mutex(active_api));
        active_api.call(
            active_api.process_instance(),
            count,
            count,
            0,
            cache.active_inputs.size(),
            cache.active_inputs.data(),
            cache.input_ptrs.data(),
            output_keys.size(),
            output_keys.data(),
            cache.output_ptrs.data()
        );
        if (on_device) {
            // No caller here ever requests UMAMI_OUT_GPU_STREAM, so
            // umami_matrix_element is expected to have fully finished (not
            // just launched) by the time it returns, making it safe to read
            // the device outputs back immediately. Unlike the inputs above,
            // this copy-back runs on every call (cache hit or not): the
            // matrix element's value legitimately changes each iteration.
            for (std::size_t i = 0; i < output_keys.size(); ++i) {
                auto& host_buf = cache.output_buffers.at(output_keys[i]);
                device->memcpy(
                    host_buf.data(), cache.output_scratch[i].get(),
                    host_buf.size() * sizeof(double)
                );
            }
        }
    }

    auto result = std::make_shared<std::vector<double>>(count, 0.0);
    for (auto key : result_keys) {
        auto found = cache.output_buffers.find(key);
        if (found == cache.output_buffers.end()) {
            throw std::runtime_error(
                "make_weightor: result key was not among the requested output keys"
            );
        }
        if (key == UMAMI_OUT_MATRIX_ELEMENT) {
            for (std::size_t e = 0; e < count; ++e) {
                (*result)[e] += found->second[e];
            }
        } else { // UMAMI_OUT_DIAGRAM_AMP2
            std::size_t diagrams = active_api.diagram_count();
            for (std::size_t e = 0; e < count; ++e) {
                for (std::size_t d = 0; d < diagrams; ++d) {
                    (*result)[e] += found->second[d * count + e];
                }
            }
        }
    }
    return result;
}

// Per-(api, process) device-resident cache for the deferred/cuda-hip
// evaluation path, attached to the process itself via REX::process's
// generic, opaque device_data slot (Rex.h) rather than living inside the
// weightor closure -- unlike EvalCache above. This ties its lifetime
// directly to the process it was derived from (found/populated fresh
// whenever a process doesn't already have one under this api's key,
// destroyed automatically when proc is), and is exactly what
// process::device_data is documented for: "a standard place to stash and
// retrieve raw pointers/handles... for off-CPU reweighting functions."
// A pleasant consequence: unlike EvalCache, the closure that uses this
// needs no mutable captured state and no reset-on-copy trick to stay safe
// under procReweightor cloning (eg reweightor::extract_sorter()'s
// max_batch overflow clones) -- a cloned closure just looks up
// device_data on whatever process it ends up bound to, which naturally
// starts out unpopulated for a different process.
//
// Only ever used for cuda/hip apis (see make_deferred_weightor), so unlike
// EvalCache it needs no cpu fallback: every input is always staged to
// device, and every output always lives in an N-iteration-wide device
// buffer -- one contiguous allocation per output key, sized for exactly
// n_iterations calls, since deferring the copy-back means each iteration's
// result must survive, un-overwritten, until it's actually pulled at
// pull_deferred_backlog() time.
struct DeferredDeviceCache {
    std::size_t count = 0;
    std::size_t n_iterations = 0;
    std::size_t next_iter = 0;

    // As EvalCache::active_api: the MatrixElementApi instance actually
    // evaluated against for this process -- either the api the owning
    // deferred_weightor was built with, or, if built via the ReloadSpec
    // overload, a freshly-loaded instance obtained when this cache was
    // first populated. Since this cache already lives per-process (via
    // proc.device_data), no separate "is this a new process" bookkeeping is
    // needed here the way EvalCache needs it: a fresh process simply has no
    // device_data yet, so its cache -- and this field -- start blank.
    const MatrixElementApi* active_api = nullptr;

    std::vector<UmamiInputKey> active_inputs;
    std::vector<void const*> input_ptrs; // device pointers
    std::vector<DeviceScratch> input_scratch;

    std::unordered_map<UmamiOutputKey, DeviceScratch> output_scratch; // one N-iteration-wide buffer per key
    std::unordered_map<UmamiOutputKey, std::size_t> per_iter_elems;
};

// Unique per api instance (pointer identity, stable for the api's
// lifetime -- MatrixElementApis are owned indefinitely by their Context),
// so multiple deferred apis evaluating the same process don't collide on
// the same device_data slot.
std::string deferred_cache_key(const MatrixElementApi& api) {
    return std::format("madtrex::deferred_weightor::{}", static_cast<const void*>(&api));
}

REX::tea::weight_puller deferred_evaluate(
    const MatrixElementApi& api,
    REX::process& proc,
    const std::vector<UmamiInputKey>& input_keys,
    const std::vector<UmamiOutputKey>& output_keys,
    const std::vector<UmamiOutputKey>& result_keys,
    std::size_t n_iterations,
    const ReloadSpec* reload = nullptr,
    const IncomingSwapped* incoming_swapped = nullptr
) {
    std::size_t count = proc.size();
    if (count == 0) {
        return []() { return std::make_shared<std::vector<double>>(); };
    }

    std::string cache_key = deferred_cache_key(api);

    std::shared_ptr<DeferredDeviceCache> cache;
    if (proc.has_device_data(cache_key)) {
        cache = std::any_cast<std::shared_ptr<DeferredDeviceCache>>(proc.get_device_data(cache_key).tag);
    } else {
        cache = std::make_shared<DeferredDeviceCache>();
        cache->count = count;
        cache->n_iterations = n_iterations;
        cache->active_api = reload ? &detail::reload_matrix_element(*reload, proc) : &api;
        const MatrixElementApi& active_api = *cache->active_api;
        madspace::DevicePtr device = active_api.device();

        // Only needed transiently to compute the initial host-side momenta
        // for the one-time copy to device below; unlike EvalCache's
        // external_proc, nothing here keeps it around afterward, since the
        // deferred path never hands out host pointers.
        REX::process external_proc = external_legs_process(proc, incoming_swapped);
        check_particle_counts(active_api, external_proc);

        const auto& process_keys = keys_for_process(active_api, proc, count, input_keys);
        cache->active_inputs.reserve(process_keys.size());
        cache->input_ptrs.reserve(process_keys.size());

        auto stage_input = [&](UmamiInputKey key, const void* host_ptr, std::size_t bytes) {
            cache->active_inputs.push_back(key);
            DeviceScratch scratch(device, bytes);
            device->memcpy(scratch.get(), const_cast<void*>(host_ptr), bytes);
            cache->input_ptrs.push_back(scratch.get());
            cache->input_scratch.push_back(std::move(scratch));
        };

        for (auto key : process_keys) {
            switch (key) {
            case UMAMI_IN_MOMENTA:
                if (!gather_momenta(active_api, external_proc)) {
                    throw std::runtime_error(
                        "make_deferred_weightor: failed to gather momenta for UMAMI"
                    );
                }
                stage_input(
                    key,
                    external_proc.umami_momenta().data(),
                    external_proc.umami_momenta().size() * sizeof(double)
                );
                break;
            case UMAMI_IN_ALPHA_S:
                if (proc.alphaS().size() != count) {
                    throw std::runtime_error(
                        "make_deferred_weightor: process does not carry a per-event "
                        "alpha_s value for every event"
                    );
                }
                stage_input(key, proc.alphaS().data(), count * sizeof(double));
                break;
            case UMAMI_IN_FLAVOR_INDEX:
                if (proc.flavor().size() != count) {
                    throw std::runtime_error(
                        "make_deferred_weightor: process does not carry a per-event "
                        "flavor index for every event"
                    );
                }
                stage_input(key, proc.flavor().data(), count * sizeof(int));
                break;
            case UMAMI_IN_HELICITY_INDEX:
                if (proc.helicity().size() != count) {
                    throw std::runtime_error(
                        "make_deferred_weightor: process does not carry a per-event "
                        "helicity index for every event"
                    );
                }
                stage_input(key, proc.helicity().data(), count * sizeof(int));
                break;
            default:
                throw std::runtime_error(
                    "make_deferred_weightor: input key cannot be derived from a "
                    "REX::process"
                );
            }
        }

        for (auto key : output_keys) {
            std::size_t per_iter;
            switch (key) {
            case UMAMI_OUT_MATRIX_ELEMENT:
                per_iter = count;
                break;
            case UMAMI_OUT_DIAGRAM_AMP2:
                per_iter = active_api.diagram_count() * count;
                break;
            default:
                throw std::runtime_error(
                    "make_deferred_weightor: output key is not a reweighting-usable "
                    "value"
                );
            }
            cache->per_iter_elems[key] = per_iter;
            cache->output_scratch.emplace(
                key, DeviceScratch(device, per_iter * n_iterations * sizeof(double))
            );
        }

        proc.set_device_data(cache_key, nullptr, 0, std::any(cache));
    }

    // Whether this call populated the cache above or found it already
    // populated (ie a later iteration against the same process), active_api
    // is the instance that was resolved -- and, from here on, is the only
    // one ever evaluated against for this process.
    const MatrixElementApi& active_api = *cache->active_api;
    madspace::DevicePtr device = active_api.device();

    if (cache->count != count) {
        throw std::runtime_error(
            "make_deferred_weightor: process's event count changed between calls; a "
            "deferred weightor's device_data cache is only valid for a process whose "
            "events never change after its first evaluation"
        );
    }
    if (cache->next_iter >= cache->n_iterations) {
        throw std::runtime_error(
            "make_deferred_weightor: invoked more times than the reweighting run's "
            "declared iteration count; the n_iterations passed to "
            "TupperWare::build()/WareHouse must match how many times the reweightor "
            "actually iterates"
        );
    }
    std::size_t iter = cache->next_iter++;

    std::vector<void*> output_ptrs;
    output_ptrs.reserve(output_keys.size());
    for (auto key : output_keys) {
        auto& scratch = cache->output_scratch.at(key);
        std::size_t per_iter = cache->per_iter_elems.at(key);
        char* base = static_cast<char*>(scratch.get());
        output_ptrs.push_back(base + iter * per_iter * sizeof(double));
    }

    {
        std::shared_lock<std::shared_mutex> parameters(detail::parameters_lock());
        std::lock_guard<std::mutex> guard(detail::api_call_mutex(active_api));
        active_api.call(
            active_api.process_instance(),
            count,
            count,
            0,
            cache->active_inputs.size(),
            cache->active_inputs.data(),
            cache->input_ptrs.data(),
            output_keys.size(),
            output_keys.data(),
            output_ptrs.data()
        );
        // Deliberately no copy-back here: this iteration's outputs stay on
        // device, in their own dedicated slice, until the puller below is
        // invoked -- by pull_deferred_backlog(), once every iteration has
        // run, not after this one specifically.
    }

    std::size_t diagrams = active_api.diagram_count();
    std::vector<std::pair<UmamiOutputKey, void*>> this_iter_output_ptrs;
    this_iter_output_ptrs.reserve(output_keys.size());
    for (std::size_t i = 0; i < output_keys.size(); ++i) {
        this_iter_output_ptrs.emplace_back(output_keys[i], output_ptrs[i]);
    }

    // cache is captured too, purely as a keep-alive guard: its device
    // buffers are already guaranteed to outlive this puller via proc's own
    // device_data (proc stays alive at least through
    // reweightor::finalise_reweighting(), which is what invokes this), but
    // holding the shared_ptr here as well costs nothing and removes any
    // reliance on that reasoning staying true if this code is ever reused
    // in a context with different lifetime guarantees.
    return [device, count, diagrams, result_keys,
            this_iter_output_ptrs = std::move(this_iter_output_ptrs),
            cache]() -> std::shared_ptr<std::vector<double>> {
        std::unordered_map<UmamiOutputKey, std::vector<double>> host_buffers;
        for (auto& [key, dev_ptr] : this_iter_output_ptrs) {
            std::size_t elems = (key == UMAMI_OUT_DIAGRAM_AMP2) ? diagrams * count : count;
            std::vector<double> host_buf(elems, 0.0);
            device->memcpy(host_buf.data(), dev_ptr, elems * sizeof(double));
            host_buffers.emplace(key, std::move(host_buf));
        }
        auto result = std::make_shared<std::vector<double>>(count, 0.0);
        for (auto key : result_keys) {
            auto found = host_buffers.find(key);
            if (found == host_buffers.end()) {
                throw std::runtime_error(
                    "make_deferred_weightor: result key was not among the requested "
                    "output keys"
                );
            }
            if (key == UMAMI_OUT_MATRIX_ELEMENT) {
                for (std::size_t e = 0; e < count; ++e) {
                    (*result)[e] += found->second[e];
                }
            } else { // UMAMI_OUT_DIAGRAM_AMP2
                for (std::size_t e = 0; e < count; ++e) {
                    for (std::size_t d = 0; d < diagrams; ++d) {
                        (*result)[e] += found->second[d * count + e];
                    }
                }
            }
        }
        return result;
    };
}

// The normaliser of TupperWare::set_stored_normalisation(): each event's
// stored matrix element, process.extra[key], or evaluate's result for any
// process with an event that has none (another source, a file without them).
REX::tea::weightor stored_normaliser(std::string key, REX::tea::weightor evaluate) {
    return [key = std::move(key), evaluate = std::move(evaluate)](REX::process& proc) {
        auto it = proc.extra.find(key);
        if (it != proc.extra.end() && it->second.size() == proc.weight_.size()) {
            auto stored = std::make_shared<std::vector<double>>();
            stored->reserve(it->second.size());
            for (const auto& value : it->second) {
                const double* me = std::any_cast<double>(&value);
                if (me == nullptr) {
                    stored = nullptr;
                    break;
                }
                stored->push_back(*me);
            }
            if (stored) return stored;
        }
        return evaluate(proc);
    };
}

} // namespace

std::vector<UmamiInputKey> default_input_keys(const MatrixElementApi& api, bool helicity_reweighting) {
    auto supported = try_supported_inputs(api);
    if (!supported) {
        throw std::runtime_error(std::format(
            "make_weightor: matrix element {} does not report supported inputs; "
            "pass input keys explicitly",
            api.file_name()
        ));
    }
    auto required = try_required_inputs(api);

    static constexpr UmamiInputKey derivable[] = {
        UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_HELICITY_INDEX
    };

    std::vector<UmamiInputKey> keys;
    for (std::size_t i = 0; i < UMAMI_INPUT_KEY_COUNT; ++i) {
        auto key = static_cast<UmamiInputKey>(i);
        bool is_derivable = std::find(std::begin(derivable), std::end(derivable), key) !=
                             std::end(derivable);
        bool is_required = required && (*required)[i];
        if (is_required && !is_derivable) {
            throw std::runtime_error(std::format(
                "make_weightor: matrix element {} requires input key {}, which "
                "cannot be derived from a REX::process; pass input keys explicitly",
                api.file_name(),
                i
            ));
        }
        if (key == UMAMI_IN_HELICITY_INDEX && helicity_reweighting && !(*supported)[i]) {
            warn_once(
                "no-helicity-input:" + api.file_name(),
                std::format(
                    "helicity reweighting is on, but matrix element {} does not support "
                    "per-event helicities (UMAMI_IN_HELICITY_INDEX); using "
                    "helicity-summed matrix elements for it",
                    api.file_name()
                )
            );
        }
        if (!is_derivable || !(*supported)[i]) {
            continue;
        }
        // UMAMI_IN_HELICITY_INDEX only with helicity reweighting (see
        // default_input_keys()'s doc comment), or when the matrix element
        // requires it outright.
        if (key == UMAMI_IN_HELICITY_INDEX && !is_required && !helicity_reweighting) {
            continue;
        }
        keys.push_back(key);
    }
    return keys;
}

std::vector<UmamiOutputKey> default_output_keys(const MatrixElementApi& api) {
    auto supported = try_supported_outputs(api);
    if (!supported || !(*supported)[UMAMI_OUT_MATRIX_ELEMENT]) {
        throw std::runtime_error(std::format(
            "make_weightor: matrix element {} does not report supporting "
            "UMAMI_OUT_MATRIX_ELEMENT; pass output/result keys explicitly",
            api.file_name()
        ));
    }
    return {UMAMI_OUT_MATRIX_ELEMENT};
}

std::shared_ptr<REX::tea::weightor> make_weightor(const MatrixElementApi& api) {
    return make_weightor(
        api, default_input_keys(api), default_output_keys(api), {UMAMI_OUT_MATRIX_ELEMENT}
    );
}

std::shared_ptr<REX::tea::weightor> make_weightor(
    const MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    IncomingSwapped incoming_swapped
) {
    if (result_keys.empty()) {
        result_keys = {UMAMI_OUT_MATRIX_ELEMENT};
    }
    for (auto key : result_keys) {
        if (std::find(output_keys.begin(), output_keys.end(), key) == output_keys.end()) {
            throw std::runtime_error(
                "make_weightor: result key must be among the requested output keys"
            );
        }
    }

    const MatrixElementApi* api_ptr = &api;
    return std::make_shared<REX::tea::weightor>(
        [api_ptr,
         input_keys = std::move(input_keys),
         output_keys = std::move(output_keys),
         result_keys = std::move(result_keys),
         incoming_swapped = std::move(incoming_swapped),
         cache = EvalCache{}](REX::process& proc) mutable {
            return evaluate(
                *api_ptr, proc, input_keys, output_keys, result_keys, cache, nullptr,
                &incoming_swapped
            );
        }
    );
}

std::shared_ptr<REX::tea::weightor> make_weightor(
    const MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    ReloadSpec reload,
    IncomingSwapped incoming_swapped
) {
    if (result_keys.empty()) {
        result_keys = {UMAMI_OUT_MATRIX_ELEMENT};
    }
    for (auto key : result_keys) {
        if (std::find(output_keys.begin(), output_keys.end(), key) == output_keys.end()) {
            throw std::runtime_error(
                "make_weightor: result key must be among the requested output keys"
            );
        }
    }

    const MatrixElementApi* api_ptr = &api;
    return std::make_shared<REX::tea::weightor>(
        [api_ptr,
         input_keys = std::move(input_keys),
         output_keys = std::move(output_keys),
         result_keys = std::move(result_keys),
         reload = std::move(reload),
         incoming_swapped = std::move(incoming_swapped),
         cache = EvalCache{}](REX::process& proc) mutable {
            return evaluate(
                *api_ptr, proc, input_keys, output_keys, result_keys, cache, &reload,
                &incoming_swapped
            );
        }
    );
}

std::shared_ptr<REX::tea::deferred_weightor>
make_deferred_weightor(const MatrixElementApi& api, std::size_t n_iterations) {
    return make_deferred_weightor(
        api, default_input_keys(api), default_output_keys(api), {UMAMI_OUT_MATRIX_ELEMENT},
        n_iterations
    );
}

std::shared_ptr<REX::tea::deferred_weightor> make_deferred_weightor(
    const MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    std::size_t n_iterations,
    IncomingSwapped incoming_swapped
) {
    if (result_keys.empty()) {
        result_keys = {UMAMI_OUT_MATRIX_ELEMENT};
    }
    for (auto key : result_keys) {
        if (std::find(output_keys.begin(), output_keys.end(), key) == output_keys.end()) {
            throw std::runtime_error(
                "make_deferred_weightor: result key must be among the requested output "
                "keys"
            );
        }
    }

    const MatrixElementApi* api_ptr = &api;
    // Unlike make_weightor's closure, this one needs no mutable captured
    // state (and hence no `mutable`, no reset-on-copy handling): its cache
    // lives on the process passed in, not inside the closure -- see
    // DeferredDeviceCache's doc comment above.
    return std::make_shared<REX::tea::deferred_weightor>(
        [api_ptr,
         input_keys = std::move(input_keys),
         output_keys = std::move(output_keys),
         result_keys = std::move(result_keys),
         n_iterations,
         incoming_swapped = std::move(incoming_swapped)](REX::process& proc) -> REX::tea::weight_puller {
            return deferred_evaluate(
                *api_ptr, proc, input_keys, output_keys, result_keys, n_iterations, nullptr,
                &incoming_swapped
            );
        }
    );
}

std::shared_ptr<REX::tea::deferred_weightor> make_deferred_weightor(
    const MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    std::vector<UmamiOutputKey> result_keys,
    std::size_t n_iterations,
    ReloadSpec reload,
    IncomingSwapped incoming_swapped
) {
    if (result_keys.empty()) {
        result_keys = {UMAMI_OUT_MATRIX_ELEMENT};
    }
    for (auto key : result_keys) {
        if (std::find(output_keys.begin(), output_keys.end(), key) == output_keys.end()) {
            throw std::runtime_error(
                "make_deferred_weightor: result key must be among the requested output "
                "keys"
            );
        }
    }

    const MatrixElementApi* api_ptr = &api;
    return std::make_shared<REX::tea::deferred_weightor>(
        [api_ptr,
         input_keys = std::move(input_keys),
         output_keys = std::move(output_keys),
         result_keys = std::move(result_keys),
         n_iterations,
         reload = std::move(reload),
         incoming_swapped = std::move(incoming_swapped)](REX::process& proc) -> REX::tea::weight_puller {
            return deferred_evaluate(
                *api_ptr, proc, input_keys, output_keys, result_keys, n_iterations, &reload,
                &incoming_swapped
            );
        }
    );
}

TupperWare::TupperWare(const madspace::MatrixElementApi& api, REX::eventBelongs checker) {
    add_api(api);
    set_normaliser(api);
    set_event_checker(std::move(checker));
}

TupperWare::TupperWare(const madspace::MatrixElementApi& api, REX::event_bool_fn checker) {
    add_api(api);
    set_normaliser(api);
    set_event_checker(std::move(checker));
}

TupperWare& TupperWare::add_api(const madspace::MatrixElementApi& api) {
    return add_api(api, {}, {});
}

TupperWare& TupperWare::add_api(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys
) {
    _apis.push_back(&api);
    _input_keys_list.push_back(std::move(input_keys));
    _output_keys_list.push_back(std::move(output_keys));
    _reload_specs.push_back(std::nullopt);
    return *this;
}

TupperWare& TupperWare::add_api(const madspace::MatrixElementApi& api, ReloadSpec reload) {
    return add_api(api, {}, {}, std::move(reload));
}

TupperWare& TupperWare::add_api(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    ReloadSpec reload
) {
    add_api(api, std::move(input_keys), std::move(output_keys));
    _reload_specs.back() = std::move(reload);
    return *this;
}

TupperWare& TupperWare::add_api(std::vector<const madspace::MatrixElementApi*> apis) {
    for (const auto* api : apis) {
        add_api(*api);
    }
    return *this;
}

TupperWare& TupperWare::add_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<UmamiInputKey> input_keys_list,
    std::vector<UmamiOutputKey> output_keys_list
) {
    for (std::size_t i = 0; i < apis.size(); ++i) {
        add_api(*apis[i], input_keys_list, output_keys_list);
    }
    return *this;
}

TupperWare& TupperWare::add_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<std::vector<UmamiInputKey>> input_keys_list,
    std::vector<std::vector<UmamiOutputKey>> output_keys_list
) {
    for (std::size_t i = 0; i < apis.size(); ++i) {
        std::vector<UmamiInputKey> input_keys;
        if (i < input_keys_list.size()) {
            input_keys = std::move(input_keys_list[i]);
        }
        std::vector<UmamiOutputKey> output_keys;
        if (i < output_keys_list.size()) {
            output_keys = std::move(output_keys_list[i]);
        }
        add_api(*apis[i], input_keys, output_keys);
    }
    return *this;
}

TupperWare& TupperWare::set_api(std::vector<const madspace::MatrixElementApi*> apis) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    _reload_specs.clear();
    return add_api(std::move(apis));
}

TupperWare& TupperWare::set_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<UmamiInputKey> input_keys_list,
    std::vector<UmamiOutputKey> output_keys_list
) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    _reload_specs.clear();
    return add_api(std::move(apis), std::move(input_keys_list), std::move(output_keys_list));
}

TupperWare& TupperWare::set_api(
    std::vector<const madspace::MatrixElementApi*> apis,
    std::vector<std::vector<UmamiInputKey>> input_keys_list,
    std::vector<std::vector<UmamiOutputKey>> output_keys_list
) {
    _apis.clear();
    _input_keys_list.clear();
    _output_keys_list.clear();
    _reload_specs.clear();
    return add_api(std::move(apis), std::move(input_keys_list), std::move(output_keys_list));
}

TupperWare& TupperWare::set_stored_normalisation(std::string key) {
    _stored_normalisation = std::move(key);
    return *this;
}

const std::string& TupperWare::stored_normalisation() const { return _stored_normalisation; }

TupperWare& TupperWare::set_normaliser(const madspace::MatrixElementApi& api) {
    return set_normaliser(api, {}, {});
}

TupperWare& TupperWare::set_normaliser(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys
) {
    _normaliser = &api;
    _normaliser_input_keys = std::move(input_keys);
    _normaliser_output_keys = std::move(output_keys);
    _normaliser_reload.reset();
    return *this;
}

TupperWare& TupperWare::set_normaliser(const madspace::MatrixElementApi& api, ReloadSpec reload) {
    return set_normaliser(api, {}, {}, std::move(reload));
}

TupperWare& TupperWare::set_normaliser(
    const madspace::MatrixElementApi& api,
    std::vector<UmamiInputKey> input_keys,
    std::vector<UmamiOutputKey> output_keys,
    ReloadSpec reload
) {
    set_normaliser(api, std::move(input_keys), std::move(output_keys));
    _normaliser_reload = std::move(reload);
    return *this;
}

TupperWare& TupperWare::set_result_keys(std::vector<UmamiOutputKey> keys) {
    _result_keys = std::move(keys);
    return *this;
}

TupperWare& TupperWare::set_helicity_reweighting(bool helicity_reweighting) {
    _helicity_reweighting = helicity_reweighting;
    return *this;
}

TupperWare& TupperWare::set_use_helicity(bool use_helicity) {
    return set_helicity_reweighting(use_helicity);
}

TupperWare& TupperWare::set_incoming_swapped(IncomingSwapped incoming_swapped) {
    _incoming_swapped = std::move(incoming_swapped);
    return *this;
}

TupperWare& TupperWare::set_event_checker(REX::eventBelongs checker) {
    _event_checker = std::make_shared<REX::eventBelongs>(std::move(checker));
    _event_checker_fn = nullptr;
    return *this;
}

TupperWare& TupperWare::set_event_checker(std::shared_ptr<REX::eventBelongs> checker) {
    _event_checker = std::move(checker);
    _event_checker_fn = nullptr;
    return *this;
}

TupperWare& TupperWare::set_event_checker(REX::event_bool_fn checker) {
    _event_checker = nullptr;
    _event_checker_fn = std::move(checker);
    return *this;
}

std::shared_ptr<REX::tea::procReweightor> TupperWare::build(
    std::size_t n_iterations, bool reload_per_batch, std::size_t max_deferred_iterations
) {
    if (_apis.empty()) {
        throw std::runtime_error("TupperWare::build: no matrix elements registered");
    }
    if (!_event_checker && !_event_checker_fn) {
        throw std::runtime_error("TupperWare::build: no event checker set");
    }

    auto result_keys_for = [this]() -> std::vector<UmamiOutputKey> {
        return _result_keys.empty() ? std::vector<UmamiOutputKey>{UMAMI_OUT_MATRIX_ELEMENT}
                                     : _result_keys;
    };

    // Each api is routed to make_weightor() (cpu) or make_deferred_weightor()
    // (cuda/hip) by its own device_type() and registered onto proc_rwgt's
    // single, unified reweight_functions list via add_reweight_function()/
    // add_deferred_reweight_function(), in registration order -- so amp
    // slot i (see procReweightor::evaluate(amp)) ends up serviced by
    // whichever of the two an api resolves to. This is what lets a single
    // TupperWare mix cpu and off-cpu apis correctly; unlike the old
    // dual-array procReweightor, there's no longer a separate list per
    // case to build up and hand over as a batch, so each api is added to
    // proc_rwgt directly as it's resolved rather than staged into a
    // same-sized rwgts/deferred_rwgts pair first.
    _weightors.clear();
    _weightors.reserve(_apis.size());
    _deferred_weightors.clear();
    _deferred_weightors.reserve(_apis.size());
    auto proc_rwgt = std::make_shared<REX::tea::procReweightor>();
    // Helicity reweighting needs, besides a matrix element accepting
    // UMAMI_IN_HELICITY_INDEX, an event checker that can assign each event
    // its index in the matrix element's helicity table (helicity_index, as
    // Driver's event checkers do from subprocesses.json): without one there
    // is no index to pass, so the auto-derived keys sum over helicities.
    bool helicity_reweighting = _helicity_reweighting;
    bool auto_keys = (_normaliser && _normaliser_input_keys.empty()) ||
                     std::any_of(_input_keys_list.begin(), _input_keys_list.end(),
                                 [](auto& keys) { return keys.empty(); });
    if (helicity_reweighting && auto_keys && !(_event_checker && _event_checker->helicity_index)) {
        warn_once(
            "tupperware-no-helicity-index",
            "helicity reweighting is on, but a TupperWare's event checker assigns no "
            "helicity indices (REX::eventBelongs::helicity_index), so its events "
            "cannot be mapped onto the matrix element's helicities; using "
            "helicity-summed matrix elements for it"
        );
        helicity_reweighting = false;
    }
    for (std::size_t i = 0; i < _apis.size(); ++i) {
        const auto& api = *_apis[i];
        auto input_keys = _input_keys_list[i].empty() ? default_input_keys(api, helicity_reweighting)
                                                        : _input_keys_list[i];
        auto output_keys =
            _output_keys_list[i].empty() ? default_output_keys(api) : _output_keys_list[i];
        const ReloadSpec* reload =
            (reload_per_batch && _reload_specs[i]) ? &*_reload_specs[i] : nullptr;
        if (api.device_type() == madspace::DeviceType::cpu) {
            auto weightor_ptr = reload
                ? make_weightor(api, input_keys, output_keys, result_keys_for(), *reload, _incoming_swapped)
                : make_weightor(api, input_keys, output_keys, result_keys_for(), _incoming_swapped);
            _weightors.push_back(weightor_ptr);
            proc_rwgt->add_reweight_function(*weightor_ptr);
        } else {
            auto deferred_ptr = reload
                ? make_deferred_weightor(
                      api, input_keys, output_keys, result_keys_for(), n_iterations, *reload,
                      _incoming_swapped
                  )
                : make_deferred_weightor(
                      api, input_keys, output_keys, result_keys_for(), n_iterations,
                      _incoming_swapped
                  );
            _deferred_weightors.push_back(deferred_ptr);
            proc_rwgt->add_deferred_reweight_function(*deferred_ptr);
        }
    }

    if (_event_checker) {
        proc_rwgt->event_checker = _event_checker;
        proc_rwgt->event_checker_fn = _event_checker->get_event_bool();
    } else {
        proc_rwgt->set_event_checker(_event_checker_fn);
    }

    if (_normaliser) {
        auto input_keys = _normaliser_input_keys.empty()
            ? default_input_keys(*_normaliser, helicity_reweighting)
            : _normaliser_input_keys;
        auto output_keys = _normaliser_output_keys.empty() ? default_output_keys(*_normaliser)
                                                             : _normaliser_output_keys;
        const ReloadSpec* reload =
            (reload_per_batch && _normaliser_reload) ? &*_normaliser_reload : nullptr;
        auto norm_weightor = reload
            ? make_weightor(*_normaliser, input_keys, output_keys, result_keys_for(), *reload, _incoming_swapped)
            : make_weightor(*_normaliser, input_keys, output_keys, result_keys_for(), _incoming_swapped);
        bool helicity_summed = std::find(input_keys.begin(), input_keys.end(),
                                         UMAMI_IN_HELICITY_INDEX) == input_keys.end();
        if (!_stored_normalisation.empty() && helicity_summed) {
            proc_rwgt->set_normaliser(stored_normaliser(_stored_normalisation, *norm_weightor));
        } else {
            proc_rwgt->set_normaliser(*norm_weightor);
        }
    }

    proc_rwgt->max_deferred_iterations = max_deferred_iterations;

    return proc_rwgt;
}

TupperWare& WareHouse::add_process() {
    _processes.emplace_back();
    return _processes.back();
}

WareHouse& WareHouse::set_iterators(std::vector<REX::tea::iterator> iters) {
    _iterators = std::move(iters);
    return *this;
}

WareHouse& WareHouse::add_iterator(REX::tea::iterator iter) {
    _iterators.push_back(std::move(iter));
    return *this;
}

WareHouse& WareHouse::set_initialise(REX::tea::iterator init) {
    _initialise = std::move(init);
    return *this;
}

WareHouse& WareHouse::set_finalise(REX::tea::iterator fin) {
    _finalise = std::move(fin);
    return *this;
}

WareHouse& WareHouse::set_launch_names(std::vector<std::string> names) {
    _launch_names = std::move(names);
    return *this;
}

WareHouse& WareHouse::add_launch_name(const std::string& name) {
    _launch_names.push_back(name);
    return *this;
}

const std::vector<std::string>& WareHouse::launch_names() const { return _launch_names; }

WareHouse& WareHouse::set_max_batch(std::size_t max_batch) {
    _max_batch = max_batch;
    return *this;
}

std::size_t WareHouse::max_batch() const { return _max_batch; }

WareHouse& WareHouse::set_threads(std::size_t threads) {
    _threads = threads;
    return *this;
}

std::size_t WareHouse::threads() const { return _threads; }

WareHouse& WareHouse::set_min_batch(std::size_t min_batch) {
    _min_batch = min_batch;
    return *this;
}

std::size_t WareHouse::min_batch() const { return _min_batch; }

std::size_t WareHouse::batch_size_for(std::size_t n_events) const {
    if (_max_batch != REX::npos) {
        return _max_batch;
    }
    if (_min_batch == REX::npos || _min_batch == 0) {
        return REX::npos;
    }
    std::size_t threads = _threads ? _threads : std::thread::hardware_concurrency();
    if (threads <= 1 || n_events < 2 * _min_batch) {
        return REX::npos;
    }
    return std::max(_min_batch, (n_events + threads - 1) / threads);
}

WareHouse& WareHouse::set_max_deferred_iterations(std::size_t max_deferred_iterations) {
    _max_deferred_iterations = max_deferred_iterations;
    return *this;
}

std::size_t WareHouse::max_deferred_iterations() const { return _max_deferred_iterations; }

REX::verbosity& WareHouse::verbose() { return _verbose; }

const REX::verbosity& WareHouse::verbose() const { return _verbose; }

WareHouse& WareHouse::set_verbose(bool v) {
    _verbose.set_verbose(v);
    return *this;
}

bool WareHouse::is_verbose() const { return _verbose.is_verbose(); }

namespace {

std::vector<std::shared_ptr<REX::tea::procReweightor>> build_proc_reweightors(
    std::deque<TupperWare>& processes,
    std::size_t n_iterations,
    bool reload_per_batch,
    std::size_t max_deferred_iterations
) {
    if (processes.empty()) {
        throw std::runtime_error("WareHouse::build: no processes registered");
    }
    std::vector<std::shared_ptr<REX::tea::procReweightor>> result;
    result.reserve(processes.size());
    for (auto& process : processes) {
        result.push_back(process.build(n_iterations, reload_per_batch, max_deferred_iterations));
    }
    return result;
}

} // namespace

REX::tea::reweightor& WareHouse::build(REX::lhe&& mother) {
    // REX::tea::reweightor defaults to a single no-op iteration
    // ({true_function}) when set_iterators() is never called -- mirror
    // that here so a deferred/cuda-hip api's device output buffer is sized
    // for the same number of calls it will actually receive.
    std::size_t n_iterations = _iterators.empty() ? 1 : _iterators.size();
    // An explicit max_batch, or the split across threads of set_min_batch()
    // (a streaming lhe has no events yet: its batches are its own)
    mother.max_batch = batch_size_for(mother.events.size());
    // Once max_batch is actually set, every registered api with a
    // ReloadSpec reloads its own instance per batch instead of every
    // overflow clone sharing (and serializing calls through) the one
    // instance it was registered with; see TupperWare::build().
    bool reload_per_batch = mother.max_batch != REX::npos;
    auto proc_rwgts = build_proc_reweightors(
        _processes, n_iterations, reload_per_batch, _max_deferred_iterations
    );
    _reweightor =
        std::make_unique<REX::tea::reweightor>(std::move(mother), std::move(proc_rwgts));
    _reweightor->pool_threads = _threads;
    if (!_iterators.empty()) {
        _reweightor->set_iterators(_iterators);
    }
    if (_initialise) {
        _reweightor->set_initialise(_initialise);
    }
    if (_finalise) {
        _reweightor->set_finalise(_finalise);
    }
    if (!_launch_names.empty()) {
        _reweightor->set_launch_names(_launch_names);
    }
    // Muted by default (see _verbose's member initializer); linking here
    // shares the flag both ways, so verbose().set_verbose(true) unmutes the
    // reweightor (and, via reweightor::extract_sorter(), every
    // procReweightor/process batch it runs) after the fact too.
    _reweightor->verbose.link(_verbose);
    return *_reweightor;
}

REX::tea::reweightor& WareHouse::build(const REX::lhe& mother) {
    return build(REX::lhe(mother));
}

REX::tea::reweightor& WareHouse::get() {
    if (!_reweightor) {
        throw std::runtime_error("WareHouse::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

const REX::tea::reweightor& WareHouse::get() const {
    if (!_reweightor) {
        throw std::runtime_error("WareHouse::get: reweightor has not been built yet");
    }
    return *_reweightor;
}

bool WareHouse::built() const noexcept { return static_cast<bool>(_reweightor); }

TupperWare& WareHouse::process(std::size_t index) {
    if (index >= _processes.size()) {
        throw std::out_of_range("WareHouse::process: index out of range");
    }
    return _processes[index];
}

const TupperWare& WareHouse::process(std::size_t index) const {
    if (index >= _processes.size()) {
        throw std::out_of_range("WareHouse::process: index out of range");
    }
    return _processes[index];
}

std::deque<TupperWare>& WareHouse::processes() {
    return _processes;
}

const std::deque<TupperWare>& WareHouse::processes() const {
    return _processes;
}

WareHouse& WareHouse::set_processes(std::vector<TupperWare> processes) {
    _processes.assign(std::make_move_iterator(processes.begin()),
                      std::make_move_iterator(processes.end()));
    return *this;
}

WareHouse& WareHouse::add_process(TupperWare process) {
    _processes.push_back(std::move(process));
    return *this;
}

WareHouse& WareHouse::add_process(std::vector<TupperWare> processes) {
    _processes.insert(_processes.end(),
                      std::make_move_iterator(processes.begin()),
                      std::make_move_iterator(processes.end()));
    return *this;
}

} // namespace madtrex
