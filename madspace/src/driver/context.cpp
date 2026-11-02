#include "madspace/driver/context.hpp"

#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>

#include "madspace/driver/io.hpp"

using namespace madspace;
using json = nlohmann::json;

namespace {
UmamiStatus umami_key_query_not_implemented(bool const**, int*) {
    return UMAMI_ERROR_NOT_IMPLEMENTED;
}

// Fills `function` with the symbol `name` of `lib`, if it exists.
template <typename R, typename... Args>
void resolve_symbol(void* lib, const char* name, std::function<R(Args...)>& function) {
    void* address = dlsym(lib, name);
    if (address != nullptr) {
        function = reinterpret_cast<R (*)(Args...)>(address);
    }
}
thread_local std::optional<std::uintptr_t> current_caller_stream;
} // namespace

MatrixElementApi::MatrixElementApi(
    const std::string& file,
    const std::string& param_card,
    ThreadPool& thread_pool,
    DevicePtr device,
    std::size_t index,
    const std::unordered_map<std::string, double>& parameters
) :
    _file_name(file), _index(index), _device(device) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [file](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(
            std::format("Could not load shared object {}\n{}", file, dlerror())
        );
    }

    void* lib = _shared_lib.get();
    resolve_symbol(lib, "umami_get_meta", _functions.get_meta);
    resolve_symbol(lib, "umami_supported_inputs", _functions.supported_inputs);
    resolve_symbol(lib, "umami_required_inputs", _functions.required_inputs);
    resolve_symbol(lib, "umami_supported_outputs", _functions.supported_outputs);
    resolve_symbol(lib, "umami_initialize", _functions.initialize);
    resolve_symbol(lib, "umami_set_parameter", _functions.set_parameter);
    resolve_symbol(lib, "umami_get_parameter", _functions.get_parameter);
    resolve_symbol(lib, "umami_matrix_element", _functions.matrix_element);
    resolve_symbol(lib, "umami_free", _functions.free);
    init_functions(parameters);
    init_instances(thread_pool, device, param_card, parameters);
}

MatrixElementApi::MatrixElementApi(
    UmamiFunctions functions,
    const std::string& name,
    const std::string& param_card,
    ThreadPool& thread_pool,
    DevicePtr device,
    std::size_t index,
    const std::unordered_map<std::string, double>& parameters
) :
    _functions(std::move(functions)), _file_name(name), _index(index) {
    init_functions(parameters);
    init_instances(thread_pool, device, param_card, parameters);
}

void MatrixElementApi::init_functions(
    const std::unordered_map<std::string, double>& parameters
) {
    auto require = [&](bool present, const char* name) {
        if (!present) {
            throw_error(std::format("missing UMAMI function {}", name));
        }
    };
    require(bool(_functions.get_meta), "umami_get_meta");
    require(bool(_functions.initialize), "umami_initialize");
    require(bool(_functions.matrix_element), "umami_matrix_element");
    require(bool(_functions.free), "umami_free");
    // Implementations may not report which keys they support; queries for that
    // information then fail with UMAMI_ERROR_NOT_IMPLEMENTED.
    if (!_functions.supported_inputs) {
        _functions.supported_inputs = umami_key_query_not_implemented;
    }
    if (!_functions.required_inputs) {
        _functions.required_inputs = umami_key_query_not_implemented;
    }
    if (!_functions.supported_outputs) {
        _functions.supported_outputs = umami_key_query_not_implemented;
    }
    if (!parameters.empty()) {
        require(bool(_functions.set_parameter), "umami_set_parameter");
    }
}

void MatrixElementApi::init_instances(
    ThreadPool& thread_pool,
    DevicePtr device,
    const std::string& param_card,
    const std::unordered_map<std::string, double>& parameters
) {
    _instances = ThreadResource<InstanceType>(
        thread_pool,
        [this, device, param_card, parameters] {
            device->activate();
            void* instance;
            check_umami_status(_functions.initialize(&instance, param_card.c_str()));
            for (const auto& [name, value] : parameters) {
                UmamiStatus status =
                    _functions.set_parameter(instance, name.c_str(), value, 0.);
                if (status != UMAMI_SUCCESS) {
                    _functions.free(instance);
                    throw_error(std::format("could not set parameter {}", name));
                }
            }
            return InstanceType(instance, [this, device](void* proc) {
                device->activate();
                _functions.free(proc);
            });
        },
        std::nullopt,
        false
    );
}

void MatrixElementApi::check_umami_status(UmamiStatus status) const {
    std::string error;
    switch (status) {
    case UMAMI_SUCCESS:
        return;
    case UMAMI_ERROR:
        throw_error("unspecified error");
    case UMAMI_ERROR_NOT_IMPLEMENTED:
        throw_error("functionality not implemented");
    case UMAMI_ERROR_UNSUPPORTED_INPUT:
        throw_error("unsupported input key");
    case UMAMI_ERROR_UNSUPPORTED_OUTPUT:
        throw_error("unsupported output key");
    case UMAMI_ERROR_UNSUPPORTED_META:
        throw_error("unsupported metadata key");
    case UMAMI_ERROR_MISSING_INPUT:
        throw_error("missing input");
    case UMAMI_ERROR_UNKNOWN_PARAMETER:
        throw_error("unknown or non-settable parameter");
    default:
        throw_error("unknown error");
    }
}

void MatrixElementApi::throw_error(const std::string& message) const {
    throw std::runtime_error(
        std::format("Error in call to matrix element API {}: {}", _file_name, message)
    );
}

void MatrixElementApi::set_parameter(const std::string& name, double real, double imag) const {
    if (!_functions.set_parameter) {
        check_umami_status(UMAMI_ERROR_NOT_IMPLEMENTED);
    }
    check_umami_status(_functions.set_parameter(process_instance(), name.c_str(), real, imag));
}

std::pair<double, double> MatrixElementApi::get_parameter(const std::string& name) const {
    if (!_functions.get_parameter) {
        check_umami_status(UMAMI_ERROR_NOT_IMPLEMENTED);
    }
    double real = 0.0, imag = 0.0;
    check_umami_status(_functions.get_parameter(process_instance(), name.c_str(), &real, &imag));
    return {real, imag};
}

const MatrixElementApi& Context::load_matrix_element(
    const std::string& file,
    const std::string& param_card,
    const std::unordered_map<std::string, double>& parameters
) {
    _param_card_paths.push_back(param_card);
    _matrix_elements.push_back(
        std::unique_ptr<MatrixElementApi>(new MatrixElementApi(
            file,
            param_card,
            *_thread_pool,
            _device,
            _matrix_elements.size(),
            parameters
        ))
    );
    return *_matrix_elements.back().get();
}

const MatrixElementApi& Context::add_matrix_element(
    UmamiFunctions functions,
    const std::string& name,
    const std::string& param_card,
    const std::unordered_map<std::string, double>& parameters
) {
    _param_card_paths.push_back(param_card);
    _matrix_elements.push_back(
        std::unique_ptr<MatrixElementApi>(new MatrixElementApi(
            std::move(functions),
            name,
            param_card,
            *_thread_pool,
            _device,
            _matrix_elements.size(),
            parameters
        ))
    );
    return *_matrix_elements.back().get();
}

Tensor Context::define_global(
    const std::string& name, DataType dtype, const SizeVec& shape, bool requires_grad
) {
    SizeVec full_shape{1};
    full_shape.insert(full_shape.end(), shape.begin(), shape.end());
    if (_globals.contains(name)) {
        throw std::invalid_argument(
            std::format("Context already contains a global named {}", name)
        );
    }
    Tensor tensor(dtype, full_shape, _device);
    tensor.zero();
    _globals[name] = {tensor, requires_grad};
    return tensor;
}

Tensor Context::global(const std::string& name) {
    if (auto search = _globals.find(name); search != _globals.end()) {
        return search->second.first;
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

bool Context::global_exists(const std::string& name) {
    return _globals.find(name) != _globals.end();
}

bool Context::global_requires_grad(const std::string& name) {
    if (auto search = _globals.find(name); search != _globals.end()) {
        return search->second.second;
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

void Context::set_global_requires_grad(const std::string& name, bool value) {
    if (auto search = _globals.find(name); search != _globals.end()) {
        search->second.second = value;
    } else {
        throw std::invalid_argument(
            std::format("Context does not contain a global named {}", name)
        );
    }
}

std::vector<std::string> Context::global_names() const {
    std::vector<std::string> names;
    names.reserve(_globals.size());
    for (auto& [name, value] : _globals) {
        names.push_back(name);
    }
    return names;
}

void Context::delete_global(const std::string& name) { _globals.erase(name); }

void Context::copy_globals_from(Context& context) {
    for (auto& [name, value] : context._globals) {
        Tensor other_tensor = value.first;
        Tensor this_tensor;
        if (auto search = _globals.find(name); search != _globals.end()) {
            this_tensor = search->second.first;
            if (this_tensor.dtype() != other_tensor.dtype()) {
                throw std::runtime_error(
                    std::format("Global {}: incompatible data type", name)
                );
            }
            if (this_tensor.shape() != other_tensor.shape()) {
                throw std::runtime_error(
                    std::format("Global {}: incompatible shape", name)
                );
            }
        } else {
            this_tensor = define_global(
                name,
                other_tensor.dtype(),
                {other_tensor.shape().begin() + 1, other_tensor.shape().end()},
                value.second
            );
        }
        this_tensor.copy_from(other_tensor);
    }
}

Tensor Context::reallocate_globals_contiguously(const std::vector<std::string>& names) {
    std::vector<Sizes> shapes;
    shapes.reserve(names.size());
    std::size_t total_size = 0;
    DataType dtype;
    for (bool first = true; auto& name : names) {
        auto& glob = _globals.at(name).first;
        if (!glob.is_only_reference()) {
            throw std::runtime_error(
                std::format(
                    "Global {}: cannot reallocate as it is externally referenced", name
                )
            );
        }
        if (first) {
            dtype = glob.dtype();
            first = false;
        } else if (dtype != glob.dtype()) {
            throw std::runtime_error(
                std::format("Global {}: incompatible dtype", name)
            );
        }
        shapes.push_back(glob.shape());
        total_size += glob.shape().product();
    }
    Tensor parent(dtype, {total_size}, device());
    for (auto [name, tensor] : zip(names, parent.split_and_reshape(shapes))) {
        auto& global = _globals.at(name).first;
        tensor.copy_from(global);
        global = tensor;
    }
    return parent;
}

const MatrixElementApi& Context::matrix_element(std::size_t index) const {
    if (index >= _matrix_elements.size()) {
        throw std::runtime_error("Matrix element index out of bounds");
    }
    return *_matrix_elements.at(index).get();
}

void Context::save_globals(const std::string& file) const {
    // sorted, so that the archive does not depend on the hash map order
    std::vector<std::string> names;
    names.reserve(_globals.size());
    for (auto& [name, tensor_and_grad] : _globals) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());

    TarWriter tar(file);
    for (auto& name : names) {
        save_tensor(tar, name + ".npy", _globals.at(name).first);
    }
    tar.close();
}

void Context::load_globals(const std::string& file) {
    constexpr std::string_view extension = ".npy";
    TarReader tar(file);
    for (auto& file_name : tar.names()) {
        if (!file_name.ends_with(extension)) {
            continue;
        }
        std::string name = file_name.substr(0, file_name.size() - extension.size());
        Tensor tensor = load_tensor(tar, file_name);
        Tensor global_tensor = define_global(
            name,
            tensor.dtype(),
            {tensor.shape().begin() + 1, tensor.shape().end()},
            false
        );
        global_tensor.copy_from(tensor);
    }
}

Tensor Context::cached_tensor(std::size_t size) {
    auto& tensors = _tensor_cache.get();
    int largest_unused = -1, size_pos = 0;
    for (int i = 0; auto& tensor : tensors) {
        std::size_t byte_size = tensor.byte_size();
        if (size >= byte_size) {
            size_pos = i;
        }
        if (tensor.is_only_reference()) {
            if (byte_size >= size) {
                return tensor;
            }
            largest_unused = i;
        }
        ++i;
    }
    if (largest_unused != -1) {
        tensors.erase(tensors.begin() + largest_unused);
        if (size_pos > largest_unused) {
            --size_pos;
        }
    }
    std::size_t word_count = (size + 7) / 8;
    Tensor new_tensor(DataType::dt_float64, {word_count}, device());
    tensors.insert(tensors.begin() + size_pos, new_tensor);
    return new_tensor;
}

ContextPtr madspace::default_context() {
    static ContextPtr context = default_device_context(cpu_device());
    return context;
}

ContextPtr madspace::default_cuda_context(std::size_t index) {
    static ContextPtr context = default_device_context(cuda_device(index));
    return context;
}

ContextPtr madspace::default_hip_context(std::size_t index) {
    static ContextPtr context = default_device_context(hip_device(index));
    return context;
}

ContextPtr madspace::default_device_context(DevicePtr device) {
    static auto& default_contexts = *new std::unordered_map<DevicePtr, ContextPtr>;
    if (auto search = default_contexts.find(device); search != default_contexts.end()) {
        return search->second;
    } else {
        ContextPtr context = std::make_shared<Context>(device);
        default_contexts[device] = context;
        return context;
    }
}

std::optional<std::uintptr_t> madspace::caller_stream() {
    return current_caller_stream;
}

void madspace::set_caller_stream(std::optional<std::uintptr_t> stream) {
    if (stream && *stream == 2) {
        throw std::invalid_argument("the per-thread default stream is not supported");
    }
    current_caller_stream =
        stream && *stream == 1 ? std::optional<std::uintptr_t>(0) : stream;
}
