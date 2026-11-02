#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/tensor.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "madspace/umami.h"

namespace madspace {

/**
 * Table of UMAMI entry points, one per function of the UMAMI C interface.
 *
 * Used to build a @ref MatrixElementApi without a shared library, with @ref
 * Context::add_matrix_element. Each entry accepts a plain C function pointer
 * or a capturing callable. `get_meta`, `initialize`, `matrix_element` and
 * `free` are required. The others are optional: missing key queries report
 * `UMAMI_ERROR_NOT_IMPLEMENTED`, and a missing `set_parameter` only fails if
 * parameters have to be set.
 */
struct UmamiFunctions {
    std::function<std::remove_pointer_t<decltype(&umami_get_meta)>> get_meta;
    std::function<std::remove_pointer_t<decltype(&umami_supported_inputs)>>
        supported_inputs;
    std::function<std::remove_pointer_t<decltype(&umami_required_inputs)>>
        required_inputs;
    std::function<std::remove_pointer_t<decltype(&umami_supported_outputs)>>
        supported_outputs;
    std::function<std::remove_pointer_t<decltype(&umami_initialize)>> initialize;
    std::function<std::remove_pointer_t<decltype(&umami_set_parameter)>> set_parameter;
    std::function<std::remove_pointer_t<decltype(&umami_get_parameter)>> get_parameter;
    std::function<std::remove_pointer_t<decltype(&umami_matrix_element)>>
        matrix_element;
    std::function<std::remove_pointer_t<decltype(&umami_free)>> free;
};

/**
 * Loaded UMAMI shared library for one matrix element.
 *
 * UMAMI (Unified MAtrix eleMent Interface) is madspace's uniform way of
 * calling matrix-element code, currently the cudacpp plugin [1]; see @ref
 * MatrixElement for the compute-graph function built on top of it. A
 * `MatrixElementApi` dynamically loads the library, resolves its exported
 * UMAMI entry points, and exposes the metadata queries and the raw @ref call.
 * Obtained from @ref Context::load_matrix_element, or from @ref
 * Context::add_matrix_element for entry points given as @ref UmamiFunctions,
 * never constructed directly.
 *
 * **References**
 * - [1] S. Hageböck et al., "Data-parallel leading-order event generation in
 *   MadGraph5_aMC@NLO", https://arxiv.org/abs/2507.21039
 */
class MatrixElementApi {
public:
    MatrixElementApi(MatrixElementApi&&) noexcept = default;
    MatrixElementApi& operator=(MatrixElementApi&&) noexcept = default;
    MatrixElementApi(const MatrixElementApi&) = delete;
    MatrixElementApi& operator=(const MatrixElementApi&) = delete;
    /// Device the underlying library was built for.
    DeviceType device_type() const {
        UmamiDevice dev;
        check_umami_status(_functions.get_meta(UMAMI_META_DEVICE, &dev));
        switch (dev) {
        case UMAMI_DEVICE_CPU:
            return DeviceType::cpu;
        case UMAMI_DEVICE_CUDA:
            return DeviceType::cuda;
        case UMAMI_DEVICE_HIP:
            return DeviceType::hip;
        default:
            throw_error("matrix element device not known");
        }
    }
    /// Number of external particles.
    std::size_t particle_count() const {
        int count;
        check_umami_status(_functions.get_meta(UMAMI_META_PARTICLE_COUNT, &count));
        return count;
    }
    /// Number of Feynman diagrams.
    std::size_t diagram_count() const {
        int count;
        check_umami_status(_functions.get_meta(UMAMI_META_DIAGRAM_COUNT, &count));
        return count;
    }
    /// Number of helicity configurations.
    std::size_t helicity_count() const {
        int count;
        check_umami_status(_functions.get_meta(UMAMI_META_HELICITY_COUNT, &count));
        return count;
    }
    /// Index of this matrix element in its @ref Context, in load order.
    std::size_t index() const { return _index; }
    /// Path of the loaded UMAMI shared library, or the name given to @ref
    /// Context::add_matrix_element.
    const std::string& file_name() const { return _file_name; }
    /// UMAMI input keys this library can accept.
    std::vector<bool> supported_inputs() const {
        bool const* data;
        int count;
        check_umami_status(_functions.supported_inputs(&data, &count));
        std::vector<bool> result(UMAMI_INPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i) {
            result[i] = data[i];
        }
        return result;
    }
    /// UMAMI input keys this library needs; a subset of @ref supported_inputs.
    std::vector<bool> required_inputs() const {
        bool const* data;
        int count;
        check_umami_status(_functions.required_inputs(&data, &count));
        std::vector<bool> supported = supported_inputs();
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i) {
            if (data[i] && !supported[i]) {
                throw_error(
                    std::format(
                        "input key {} is reported as required but not as supported", i
                    )
                );
            }
        }
        std::vector<bool> result(UMAMI_INPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_INPUT_KEY_COUNT; ++i) {
            result[i] = data[i];
        }
        return result;
    }
    /// UMAMI output keys this library can produce.
    std::vector<bool> supported_outputs() const {
        bool const* data;
        int count;
        check_umami_status(_functions.supported_outputs(&data, &count));
        std::vector<bool> result(UMAMI_OUTPUT_KEY_COUNT, false);
        for (int i = 0; i < count && i < UMAMI_OUTPUT_KEY_COUNT; ++i) {
            result[i] = data[i];
        }
        return result;
    }

    /// Raw UMAMI call: evaluate the matrix element on a batch of `count`
    /// events for the requested `input_keys` / `output_keys`.
    void call(
        UmamiHandle handle,
        size_t count,
        size_t stride,
        size_t offset,
        size_t input_count,
        UmamiInputKey const* input_keys,
        void const* const* inputs,
        size_t output_count,
        UmamiOutputKey const* output_keys,
        void* const* outputs
    ) const {
        check_umami_status(_functions.matrix_element(
            handle,
            count,
            stride,
            offset,
            input_count,
            input_keys,
            inputs,
            output_count,
            output_keys,
            outputs
        ));
    }

    /// Opaque per-thread UMAMI process handle, initialized on first use.
    void* process_instance() const { return _instances.get().get(); }

    // The Device this api's library was loaded on (ie the owning Context's
    // device). Callers driving umami_matrix_element() need this to know
    // whether the inputs/outputs it exchanges have to be staged through
    // device-resident buffers (cuda/hip) or can be passed as plain host
    // pointers (cpu).
    DevicePtr device() const noexcept { return _device; }

    // Whether the loaded library actually implements umami_set_parameter/
    // umami_get_parameter (both optional symbols; a library built without
    // them still loads fine, it just can't be parameter-reweighted in place
    // and set_parameter()/get_parameter() will throw UMAMI_ERROR_NOT_IMPLEMENTED
    // if called anyway).
    bool supports_set_parameter() const noexcept { return bool(_functions.set_parameter); }
    bool supports_get_parameter() const noexcept { return bool(_functions.get_parameter); }

    // Directly sets/reads physics parameters according to whatever
    // convention the library has implemented.
    void set_parameter(const std::string& name, double real, double imag = 0.0) const;
    std::pair<double, double> get_parameter(const std::string& name) const;

private:
    MatrixElementApi(
        const std::string& file,
        const std::string& param_card,
        ThreadPool& thread_pool,
        DevicePtr device,
        std::size_t index = 0,
        const std::unordered_map<std::string, double>& parameters = {}
    );
    MatrixElementApi(
        UmamiFunctions functions,
        const std::string& name,
        const std::string& param_card,
        ThreadPool& thread_pool,
        DevicePtr device,
        std::size_t index = 0,
        const std::unordered_map<std::string, double>& parameters = {}
    );
    void init_functions(const std::unordered_map<std::string, double>& parameters);
    void init_instances(
        ThreadPool& thread_pool,
        DevicePtr device,
        const std::string& param_card,
        const std::unordered_map<std::string, double>& parameters
    );

    void check_umami_status(UmamiStatus status) const;
    [[noreturn]] void throw_error(const std::string& message) const;
    std::unique_ptr<void, std::function<void(void*)>> _shared_lib;
    UmamiFunctions _functions;
    using InstanceType = std::unique_ptr<void, std::function<void(void*)>>;
    ThreadResource<InstanceType> _instances;
    std::string _file_name;
    std::size_t _index;
    DevicePtr _device;

    friend class Context;
};

/**
 * Owns the trainable globals and loaded matrix elements for one device.
 *
 * The state a compiled @ref Function runs against: the named, persistent
 * @ref Tensor globals it reads and writes (network weights, VEGAS grids, ...)
 * and the @ref MatrixElementApi libraries loaded via @ref load_matrix_element.
 * All globals and matrix elements live on the same @ref device. Training and
 * generation typically use two contexts, one per role, and periodically sync
 * weights with @ref copy_globals_from.
 */
class Context {
public:
    /// A context on the CPU, with `thread_count` worker threads (`-1` for
    /// the hardware concurrency).
    Context(int thread_count = -1) :
        _device(cpu_device()),
        _thread_pool(std::make_unique<ThreadPool>(thread_count)),
        _tensor_cache(global_resource<TensorVec>(tensor_cache_resource_name, []() {
            return TensorVec{};
        })) {}
    /// A context on `device`, with `thread_count` worker threads (`-1` for
    /// the hardware concurrency).
    Context(DevicePtr device, int thread_count = -1) :
        _device(device),
        _thread_pool(std::make_unique<ThreadPool>(thread_count)),
        _tensor_cache(global_resource<TensorVec>(tensor_cache_resource_name, []() {
            return TensorVec{};
        })) {}
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    /// Load the UMAMI matrix element from `file`, initialized with
    /// `param_card`. Every entry of `parameters` is then passed to each
    /// process instance through `umami_set_parameter` (real value, e.g.
    /// `{"bwcutoff": 15.}`, the window of the `$`-excluded propagators);
    /// a library that rejects one of them is an error. Caches by `file`:
    /// loading the same file twice returns the same @ref MatrixElementApi.
    const MatrixElementApi& load_matrix_element(
        const std::string& file,
        const std::string& param_card,
        const std::unordered_map<std::string, double>& parameters = {}
    );
    /// Add a matrix element from the UMAMI entry points in `functions`,
    /// without a shared library. `name` identifies it in error messages, and
    /// `param_card` and `parameters` are passed on as in @ref
    /// load_matrix_element.
    const MatrixElementApi& add_matrix_element(
        UmamiFunctions functions,
        const std::string& name,
        const std::string& param_card = "",
        const std::unordered_map<std::string, double>& parameters = {}
    );
    /// Create and zero-initialize a new global named `name`.
    Tensor define_global(
        const std::string& name,
        DataType dtype,
        const SizeVec& shape,
        bool requires_grad = false
    );
    /// The global tensor named `name`.
    Tensor global(const std::string& name);
    /// Whether the global named `name` accumulates a gradient during training.
    bool global_requires_grad(const std::string& name);
    /// Set whether the global named `name` accumulates a gradient.
    void set_global_requires_grad(const std::string& name, bool value);
    /// Whether a global named `name` exists.
    bool global_exists(const std::string& name);
    /// Names of every global.
    std::vector<std::string> global_names() const;
    /// Remove the global named `name`.
    void delete_global(const std::string& name);
    /// Overwrite every global also present in `context` with its value there.
    void copy_globals_from(Context& context);
    /// Repack the globals in `names` into one contiguous allocation, for
    /// example before handing them to an external optimizer.
    Tensor reallocate_globals_contiguously(const std::vector<std::string>& names);
    /// The matrix element previously loaded at position `index`; see @ref
    /// load_matrix_element.
    const MatrixElementApi& matrix_element(std::size_t index) const;
    /// Write every global to the tar archive `file`, as one `.npy` file per name.
    void save_globals(const std::string& file) const;
    /// Define a global from every `.npy` file in the tar archive `file`, named
    /// after the file.
    void load_globals(const std::string& file);
    /// The device every global and matrix element of this context lives on.
    DevicePtr device() { return _device; }
    /// The thread pool backing this context's device-local resources.
    ThreadPool& thread_pool() { return *_thread_pool; }
    /// The next value of a per-context counter, used to derive independent
    /// RNG seeds; see @ref DerivedSeed.
    std::size_t unique_seed_index() { return _seed_index++; }
    /// A scratch tensor of at least `size` bytes, reused from a pool when
    /// possible.
    Tensor cached_tensor(std::size_t size);
    /// Drop every scratch tensor held by @ref cached_tensor.
    void reset_cache() {
        _tensor_cache = ThreadResource<TensorVec>(thread_pool(), []() {
            return TensorVec{};
        });
    }
    /// A per-thread resource named `name`, created with `constructor` on
    /// first access and torn down with `destructor`. Not thread-safe: callers
    /// must acquire the reference once during single-threaded initialization
    /// (e.g. a `Runtime` constructor) and reuse it.
    template <typename T>
    ThreadResource<T>& global_resource(
        const std::string& name,
        std::function<T()> constructor,
        std::optional<std::function<void(T&)>> destructor = std::nullopt
    ) {
        auto search = _resources.find(name);
        if (search == _resources.end()) {
            auto res = std::make_shared<ThreadResource<T>>(
                thread_pool(), constructor, destructor
            );
            _resources.emplace(
                name,
                std::pair<std::type_index, std::shared_ptr<void>>(
                    std::type_index(typeid(T)), res
                )
            );
            return *res;
        } else {
            auto& [tid, res] = search->second;
            if (std::type_index(typeid(T)) != tid) {
                throw std::runtime_error(
                    std::format("incompatible resource type for '{}'", name)
                );
            }
            return *std::static_pointer_cast<ThreadResource<T>>(res);
        }
    }

private:
    static constexpr const char* tensor_cache_resource_name = "__tensor_cache";
    DevicePtr _device;
    std::unique_ptr<ThreadPool> _thread_pool;
    std::unordered_map<std::string, std::pair<Tensor, bool>> _globals;
    std::vector<std::unique_ptr<MatrixElementApi>> _matrix_elements;
    std::vector<std::string> _param_card_paths;
    std::size_t _seed_index = 0;
    std::unordered_map<std::string, std::pair<std::type_index, std::shared_ptr<void>>>
        _resources;
    ThreadResource<TensorVec>& _tensor_cache;
};

/// Owning handle to a @ref Context.
using ContextPtr = std::shared_ptr<Context>;

/// Process-wide default CPU context, created on first call.
ContextPtr default_context();
/// Process-wide default CUDA context for device `index`, created on first
/// call.
ContextPtr default_cuda_context(std::size_t index = 0);
/// Process-wide default HIP context for device `index`, created on first
/// call.
ContextPtr default_hip_context(std::size_t index = 0);
/// Process-wide default context for `device`, created on first call.
ContextPtr default_device_context(DevicePtr device);

/// GPU stream that runtime calls on this thread run on without synchronizing;
/// 0 for the legacy default stream.
std::optional<std::uintptr_t> caller_stream();
/// Set the stream returned by @ref caller_stream for this thread.
void set_caller_stream(std::optional<std::uintptr_t> stream);

/// `name`, namespaced under `prefix` as `"prefix.name"`; `name` unchanged if
/// `prefix` is empty. Used to build unique global names for repeated
/// subnetworks; see for example @ref MLP.
inline std::string prefixed_name(const std::string& prefix, const std::string& name) {
    return prefix == "" ? name : std::format("{}.{}", prefix, name);
}

} // namespace madspace
