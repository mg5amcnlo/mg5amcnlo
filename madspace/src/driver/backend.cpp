#include "madspace/driver/backend.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <format>
#include <unordered_map>

using namespace madspace;

namespace {

struct LoadedBackend {
    inline static std::string lib_path = "";
    inline static int vector_size = -1;
    inline static std::unordered_map<DevicePtr, LoadedBackend*> device_backends;

    LoadedBackend(const std::string& file) {
#ifdef __APPLE__
        std::string so_ext = "dylib";
#else
        std::string so_ext = "so";
#endif
        shared_lib = std::shared_ptr<void>(
            dlopen(std::format("{}/{}.{}", lib_path, file, so_ext).c_str(), RTLD_NOW),
            [](void* lib) { dlclose(lib); }
        );
        if (!shared_lib) {
            throw std::runtime_error(
                std::format("Could not load shared object {}", file)
            );
        }
        device_count = reinterpret_cast<decltype(device_count)>(
            dlsym(shared_lib.get(), "device_count")
        );
        if (device_count == nullptr) {
            throw std::runtime_error(
                std::format(
                    "Did not find symbol device_count in shared object {}", file
                )
            );
        }
        get_device = reinterpret_cast<decltype(get_device)>(
            dlsym(shared_lib.get(), "get_device")
        );
        if (get_device == nullptr) {
            throw std::runtime_error(
                std::format("Did not find symbol get_device in shared object {}", file)
            );
        }
        build_runtime = reinterpret_cast<decltype(build_runtime)>(
            dlsym(shared_lib.get(), "build_runtime")
        );
        if (build_runtime == nullptr) {
            throw std::runtime_error(
                std::format(
                    "Did not find symbol build_runtime in shared object {}", file
                )
            );
        }

        for (int i = 0; i < device_count(); ++i) {
            device_backends[get_device(i)] = this;
        }
    }

    std::shared_ptr<void> shared_lib;
    int (*device_count)();
    DevicePtr (*get_device)(int);
    Runtime* (*build_runtime)(
        const Function& function, ContextPtr context, bool concurrent
    );
};

const LoadedBackend& cpu_backend() {
    static LoadedBackend backend = [&] {
        std::vector<int> supported_vector_sizes{1};
#ifdef SIMD_AVAILABLE
#ifdef __APPLE__
        supported_vector_sizes.push_back(2);
#else  // __APPLE__
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            supported_vector_sizes.push_back(4);
        }
        if (__builtin_cpu_supports("avx512f")) {
            supported_vector_sizes.push_back(8);
        }
#endif // __APPLE__
#endif // SIMD_AVAILABLE

        int vector_size = LoadedBackend::vector_size;
        if (vector_size == -1) {
            if (char* env_var = std::getenv("SIMD_VECTOR_SIZE")) {
                vector_size = std::atoi(env_var);
            } else {
                vector_size = 0;
            }
        }
        if (vector_size <= 0) {
#ifdef __APPLE__
            vector_size = 1;
#else
            // vector_size = supported_vector_sizes.back();
            vector_size = 1;
#endif
        } else if (std::find(
                       supported_vector_sizes.begin(),
                       supported_vector_sizes.end(),
                       vector_size
                   ) == supported_vector_sizes.end()) {
            throw std::runtime_error("unsupported SIMD vector size");
        }

        switch (vector_size) {
        case 2:
            return LoadedBackend("libmadspace_cpu_neon");
        case 4:
            return LoadedBackend("libmadspace_cpu_avx2");
        case 8:
            return LoadedBackend("libmadspace_cpu_avx512");
        default:
            return LoadedBackend("libmadspace_cpu");
        }
    }();
    return backend;
}

const LoadedBackend& cuda_backend() {
    static LoadedBackend backend("libmadspace_cuda");
    return backend;
}

const LoadedBackend& hip_backend() {
    static LoadedBackend backend("libmadspace_hip");
    return backend;
}

} // namespace

std::vector<std::string> madspace::available_backends() {
    std::vector<std::string> backends{"cpu_scalar"};
#if defined(__aarch64__) || defined(_M_ARM64)
    backends.push_back("cpu_vec128");
#elif defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("sse4.2")) {
        backends.push_back("cpu_vec128");
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        backends.push_back("cpu_vec256");
    }
    if (__builtin_cpu_supports("avx512f")) {
        backends.push_back("cpu_vec512");
    }
    if (__builtin_cpu_supports("avx512vl")) {
        backends.push_back("cpu_vec512y");
    }
#else
#error "Compiling for unsupported platform"
#endif
    return backends;
}

RuntimePtr
madspace::build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    auto& loaded_backend = LoadedBackend::device_backends.at(context->device());
    Runtime* runtime = loaded_backend->build_runtime(function, context, concurrent);
    runtime->shared_lib = loaded_backend->shared_lib;
    return RuntimePtr(runtime);
}

DevicePtr madspace::cpu_device() { return cpu_backend().get_device(0); }

DevicePtr madspace::cuda_device(std::size_t index) {
    return cuda_backend().get_device(index);
}

DevicePtr madspace::hip_device(std::size_t index) {
    return hip_backend().get_device(index);
}

void madspace::set_lib_path(const std::string& lib_path) {
    LoadedBackend::lib_path = lib_path;
}

void madspace::set_simd_vector_size(int vector_size) {
    LoadedBackend::vector_size = vector_size;
}
