#include "madspace/driver/context.hpp"

#include <dlfcn.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "madspace/driver/io.hpp"

using namespace madspace;
using json = nlohmann::json;

MatrixElementApi::MatrixElementApi(
    const std::string& file,
    const std::string& param_card,
    ThreadPool& thread_pool,
    DevicePtr device,
    std::size_t index
) :
    _file_name(file), _index(index) {
    _shared_lib = std::unique_ptr<void, std::function<void(void*)>>(
        dlopen(file.c_str(), RTLD_NOW), [file](void* lib) { dlclose(lib); }
    );
    if (!_shared_lib) {
        throw std::runtime_error(
            std::format("Could not load shared object {}\n{}", file, dlerror())
        );
    }

    _get_meta = reinterpret_cast<decltype(&umami_get_meta)>(
        dlsym(_shared_lib.get(), "umami_get_meta")
    );
    if (_get_meta == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_get_meta in shared object {}", file)
        );
    }

    _initialize = reinterpret_cast<decltype(&umami_initialize)>(
        dlsym(_shared_lib.get(), "umami_initialize")
    );
    if (_initialize == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_initialize in shared object {}", file
            )
        );
    }

    _matrix_element = reinterpret_cast<decltype(&umami_matrix_element)>(
        dlsym(_shared_lib.get(), "umami_matrix_element")
    );
    if (_matrix_element == nullptr) {
        throw std::runtime_error(
            std::format(
                "Did not find symbol umami_matrix_element in shared object {}", file
            )
        );
    }

    _free =
        reinterpret_cast<decltype(&umami_free)>(dlsym(_shared_lib.get(), "umami_free"));
    if (_free == nullptr) {
        throw std::runtime_error(
            std::format("Did not find symbol umami_free in shared object {}", file)
        );
    }

    _instances = ThreadResource<InstanceType>(thread_pool, [&, device] {
        device->activate();
        void* instance;
        check_umami_status(_initialize(&instance, param_card.c_str()));
        return InstanceType(instance, [this, device](void* proc) {
            device->activate();
            _free(proc);
        });
    });
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
    default:
        throw_error("unknown error");
    }
}

void MatrixElementApi::throw_error(const std::string& message) const {
    throw std::runtime_error(
        std::format("Error in call to matrix element API {}: {}", _file_name, message)
    );
}

const MatrixElementApi&
Context::load_matrix_element(const std::string& file, const std::string& param_card) {
    _param_card_paths.push_back(param_card);
    _matrix_elements.push_back(
        std::unique_ptr<MatrixElementApi>(new MatrixElementApi(
            file, param_card, *_thread_pool, _device, _matrix_elements.size()
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

void Context::save_globals(const std::string& dir) const {
    namespace fs = std::filesystem;
    fs::path dir_path(dir);
    fs::create_directory(dir_path);

    for (auto& [name, tensor_and_grad] : _globals) {
        auto& [tensor, requires_grad] = tensor_and_grad;
        fs::path tensor_file = dir_path / name;
        tensor_file += ".npy";
        save_tensor(tensor_file, tensor);
    }
}

void Context::load_globals(const std::string& dir) {
    namespace fs = std::filesystem;
    for (auto& file : fs::directory_iterator(dir)) {
        if (file.path().extension() != ".npy") {
            continue;
        }
        std::string name = file.path().stem();
        Tensor tensor = load_tensor(file.path());
        Tensor global_tensor = define_global(
            name,
            tensor.dtype(),
            {tensor.shape().begin() + 1, tensor.shape().end()},
            false
        );
        global_tensor.copy_from(tensor);
    }
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
    static std::unordered_map<DevicePtr, ContextPtr> default_contexts;
    if (auto search = default_contexts.find(device); search != default_contexts.end()) {
        return search->second;
    } else {
        ContextPtr context = std::make_shared<Context>(device);
        default_contexts[device] = context;
        return context;
    }
}
