#include "madspace/compgraphs/optimizer.hpp"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_map>

#include "madspace/util.hpp"

using namespace madspace;

InstructionDependencies::InstructionDependencies(const Function& function) :
    _size(function.instructions().size()), _matrix(_size * _size) {
    std::vector<int> local_source(function.locals().size(), -1);
    int index = 0;
    for (auto& instr : function.instructions()) {
        int rank = 0;
        for (auto& input : instr.inputs) {
            auto source_index = local_source.at(input.local_index);
            if (source_index == -1) {
                continue;
            }
            _matrix.at(index * _size + source_index) = true;
            for (int i = 0; i < _size; ++i) {
                _matrix.at(index * _size + i) = _matrix.at(index * _size + i) |
                    _matrix.at(source_index * _size + i);
            }
            int source_rank = _ranks.at(source_index);
            if (rank < source_rank) {
                rank = source_rank;
            }
        }
        for (auto& output : instr.outputs) {
            local_source.at(output.local_index) = index;
        }
        _ranks.push_back(rank + 1);
        ++index;
    }
}

LastUseOfLocals::LastUseOfLocals(const Function& function) :
    _last_used(function.instructions().size()) {
    std::vector<bool> seen_locals;
    for (auto& local : function.locals()) {
        seen_locals.push_back(
            !std::holds_alternative<std::monostate>(local.literal_value)
        );
    }
    for (auto& output : function.outputs()) {
        seen_locals.at(output.local_index) = true;
    }
    auto instr = function.instructions().rbegin();
    auto indices = _last_used.begin();
    for (; instr != function.instructions().rend(); ++instr, ++indices) {
        for (auto& input : instr->inputs) {
            auto index = input.local_index;
            if (!seen_locals.at(index)) {
                indices->push_back(index);
                seen_locals.at(index) = true;
            }
        }
    }
    std::reverse(_last_used.begin(), _last_used.end());
}

Function madspace::sort_breadth_first(const Function& function) {
    Function func_out = function;
    InstructionDependencies dependencies(function);
    auto order = dependencies.ranks();
    std::vector<std::size_t> instruction_perm(function.instructions().size());
    std::iota(instruction_perm.begin(), instruction_perm.end(), 0);
    std::stable_sort(
        instruction_perm.begin(),
        instruction_perm.end(),
        [&](std::size_t i, std::size_t j) { return order.at(i) < order.at(j); }
    );
    func_out._instructions.clear();
    for (std::size_t index : instruction_perm) {
        func_out._instructions.push_back(function._instructions.at(index));
    }
    return func_out;
}
