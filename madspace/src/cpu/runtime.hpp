#pragma once

#include <random>

#include "madspace/compgraphs/function.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {
namespace cpu {

class CpuRuntime : public Runtime {
public:
    template <bool is_grad>
    struct DummyAllocHints {
        AllocHint operator[](std::size_t index) const {
            return is_grad ? AllocHint::local_grad : AllocHint::local;
        }
    };
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        std::size_t batch_size_index;
        CpuRuntime& runtime;
        bool differentiable;
        SizeVec dependent_instructions;
        std::size_t dependency_count;
        SizeVec dependent_instructions_backward;
        std::size_t dependency_count_backward;
        DummyAllocHints<false> output_alloc_hints;
        DummyAllocHints<true> input_grad_alloc_hints;
    };

    CpuRuntime(const Function& function, ContextPtr context, bool concurrent);

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
    std::mt19937& rand_gen() { return _rand_gens.get(); }

private:
    TensorVec run_single(const TensorVec& inputs) const;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad_single(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_concurrent(
        const TensorVec& inputs,
        const std::vector<bool>& input_requires_grad,
        bool with_grad
    ) const;
    std::pair<TensorVec, TensorVec> run_backward_single(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads
    ) const;
    std::pair<TensorVec, TensorVec> run_backward_concurrent(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads
    ) const;

    std::vector<Instruction> _instructions;
    SizeVec _output_indices;
    std::size_t _input_count;
    TensorVec _locals_init;
    std::vector<bool> _requires_grad_init;
    SizeVec _grad_global_indices;
    std::vector<Sizes> _grad_global_shapes;
    std::size_t _grad_global_total_size;
    ContextPtr _context;
    ThreadResource<std::mt19937> _rand_gens;
    bool _concurrent;
    SizeVec _ready_instructions_init;
    SizeVec _ready_instructions_backward_init;
};

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent);

} // namespace cpu
} // namespace madspace
