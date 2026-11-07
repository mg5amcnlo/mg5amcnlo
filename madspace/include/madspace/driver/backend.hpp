#pragma once

#include <cstdint>
#include <optional>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/random.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class Runtime {
public:
    virtual ~Runtime() = default;
    virtual TensorVec run(const TensorVec& inputs) = 0;
    virtual std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) = 0;
    virtual std::pair<TensorVec, TensorVec> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads = false
    ) = 0;
    virtual void set_seed(DerivedSeed seed) = 0;
    virtual void release_inputs() {}

    friend std::unique_ptr<Runtime>
    build_runtime(const Function& function, ContextPtr context, bool concurrent);

private:
    std::shared_ptr<void> shared_lib;
};

/// Owning handle to a `Runtime`.
using RuntimePtr = std::unique_ptr<Runtime>;
/// Compile `function` into a callable @ref Runtime for `context`'s device.
/// `concurrent` lets the CPU runtime run independent instructions concurrently on
/// the context's thread pool; the GPU runtime ignores it.
RuntimePtr
build_runtime(const Function& function, ContextPtr context, bool concurrent = true);
/// Names of the CPU backend variants usable on this host ("cpu_scalar",
/// "cpu_vec128", "cpu_vec256", ...); GPU backends are not listed.
std::vector<std::string> available_backends();
DevicePtr cpu_device();
DevicePtr cuda_device(std::size_t index = 0);
DevicePtr hip_device(std::size_t index = 0);
/// Directory to load the generated backend shared libraries from.
void set_lib_path(const std::string& lib_path);
/// SIMD mode of the CPU backend: "scalar", "simd_128" (ARM), "simd_256" (AVX2),
/// "avx512y" (AVX512 with 256-bit vectors), "simd_512" (AVX512) or "auto" (simd_512,
/// then simd_256 on x86 and scalar on ARM, where the vectorized mode is currently
/// slower). Defaults to $MADSPACE_SIMD_MODE or "auto". Must be set before the CPU
/// backend is first used; afterwards, requesting a different mode throws.
void set_simd_mode(const std::string& simd_mode);
/// SIMD modes supported by this build on this host (without "auto"), ordered by
/// increasing vector width; see @ref set_simd_mode.
std::vector<std::string> supported_simd_modes();

} // namespace madspace
