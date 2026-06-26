#pragma once

#include "gpu_abstraction.cuh"
#include "madspace/compgraphs/function.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/tensor.hpp"

#include <memory>

namespace madspace {
namespace gpu {

class GpuRuntime : public Runtime {
public:
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        std::vector<AllocHint> input_grad_alloc_hints;
        SizeVec output_indices;
        std::vector<AllocHint> output_alloc_hints;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        std::size_t batch_size_index;
        GpuRuntime& runtime;
        bool differentiable;
        std::size_t stream;
        SizeVec wait_events;
        int record_event;
        SizeVec backward_wait_events;
        int backward_record_event;
    };

    GpuRuntime(const Function& function, ContextPtr context);
    TensorVec run(const TensorVec& inputs) override;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) override;
    std::pair<TensorVec, TensorVec> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads
    ) override;
    Context& context() { return *_context; }
    gpublasHandle_t gpublas_handle() { return _gpublas_handle.get(); }
    gpurandGenerator_t gpurand_generator() { return _gpurand_generator.get(); }

private:
    std::vector<std::tuple<std::size_t, std::size_t, Tensor, bool>>
    load_pool_size_cache(bool backward);
    void update_pool_size_cache(
        const std::vector<std::pair<std::size_t, std::size_t>>& total_sizes,
        bool backward
    );
    void update_cached_tensors(
        const std::vector<std::pair<std::size_t, Tensor>>& tensors, bool backward
    );
    std::vector<Instruction> _instructions;
    SizeVec _output_indices;
    std::size_t _input_count;
    TensorVec _locals_init;
    std::vector<bool> _requires_grad_init;
    SizeVec _grad_global_indices;
    std::vector<Sizes> _grad_global_shapes;
    std::size_t _grad_global_total_size;
    ContextPtr _context;
    ThreadResource<std::vector<gpuStream_t>> _streams;
    ThreadResource<std::vector<gpuEvent_t>> _events;
    std::vector<std::size_t> _wait_events;
    std::vector<std::size_t> _backward_wait_events;
    ThreadResource<gpublasHandle_t> _gpublas_handle;
    ThreadResource<gpurandGenerator_t> _gpurand_generator;
    std::atomic<std::shared_ptr<std::unordered_map<std::size_t, std::size_t>>>
        _pool_size_cache;
    std::atomic<std::shared_ptr<std::unordered_map<std::size_t, std::size_t>>>
        _pool_size_cache_backward;
    ThreadResource<TensorVec> _prev_caches;
    ThreadResource<TensorVec> _prev_caches_backward;
};

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent);

} // namespace gpu
} // namespace madspace
