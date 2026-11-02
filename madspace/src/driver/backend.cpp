#include "madspace/driver/backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <format>
#include <unordered_map>
#include <utility>

using namespace madspace;

namespace {

struct LoadedBackend {
    inline static std::string lib_path = "";
    // resolved SIMD mode requested with set_simd_mode, and the one actually loaded
    inline static std::string simd_mode = "";
    inline static std::string loaded_simd_mode = "";
    inline static std::unordered_map<DevicePtr, LoadedBackend*> device_backends;

    LoadedBackend(const std::string& file) {
#ifdef __APPLE__
        std::string so_ext = "dylib";
#else
        std::string so_ext = "so";
#endif
        shared_lib = std::shared_ptr<void>(
            dlopen(std::format("{}/{}.{}", lib_path, file, so_ext).c_str(), RTLD_NOW),
            // Unlike std::unique_ptr, std::shared_ptr's deleter is invoked
            // unconditionally on destruction, even when the stored pointer is
            // null -- if dlopen() above fails, the throw a few lines down
            // would otherwise unwind through a dlclose(nullptr), which
            // segfaults instead of letting the runtime_error propagate.
            [](void* lib) {
                if (lib) dlclose(lib);
            }
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

// SIMD modes supported by this build and host, and the one chosen by "auto"
std::pair<std::vector<std::string>, std::string> simd_mode_support() {
    std::vector<std::string> supported_modes{"scalar"};
    std::string auto_mode = "scalar";
#ifdef SIMD_AVAILABLE
#if defined(__aarch64__) || defined(_M_ARM64)
    supported_modes.push_back("simd_128");
    // auto stays scalar: the vectorized mode is currently slower on ARM
#elif defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        supported_modes.push_back("simd_256");
        auto_mode = "simd_256";
    }
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512dq")) {
        // never chosen by auto: slower than simd_512 in the x86 benchmarks
        supported_modes.push_back("avx512y");
    }
    if (__builtin_cpu_supports("avx512f")) {
        supported_modes.push_back("simd_512");
        auto_mode = "simd_512";
    }
#endif
#endif // SIMD_AVAILABLE
    return {supported_modes, auto_mode};
}

// SIMD mode that a requested mode ("auto" or an explicit one) resolves to
std::string resolve_simd_mode(const std::string& mode) {
    auto [supported_modes, auto_mode] = simd_mode_support();
    if (mode == "auto") {
        return auto_mode;
    }
    if (std::find(supported_modes.begin(), supported_modes.end(), mode) ==
        supported_modes.end()) {
        std::string supported;
        for (auto& supported_mode : supported_modes) {
            supported += std::format(" '{}'", supported_mode);
        }
        throw std::runtime_error(
            std::format(
                "unsupported madspace SIMD mode '{}', supported on this host: "
                "'auto'{}",
                mode,
                supported
            )
        );
    }
    return mode;
}

const LoadedBackend& cpu_backend() {
    static LoadedBackend backend = [&] {
        std::string mode = LoadedBackend::simd_mode;
        if (mode.empty()) {
            char* env_var = std::getenv("MADSPACE_SIMD_MODE");
            mode = resolve_simd_mode(env_var ? env_var : "auto");
        }
        LoadedBackend::loaded_simd_mode = mode;
        if (mode == "scalar") {
            return LoadedBackend("libmadspace_cpu");
        } else {
            return LoadedBackend(std::format("libmadspace_cpu_{}", mode));
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

std::vector<std::string> madspace::supported_simd_modes() {
    return simd_mode_support().first;
}

void madspace::set_simd_mode(const std::string& simd_mode) {
    std::string mode = resolve_simd_mode(simd_mode);
    const std::string& loaded_mode = LoadedBackend::loaded_simd_mode;
    if (!loaded_mode.empty() && mode != loaded_mode) {
        throw std::runtime_error(
            std::format(
                "can not switch the madspace SIMD mode to '{}': the CPU backend was "
                "already loaded with '{}'",
                mode,
                loaded_mode
            )
        );
    }
    LoadedBackend::simd_mode = mode;
}
