#pragma once

#include "madspace/compgraphs/function.hpp"

#include <vector>

namespace madspace {

class InstructionDependencies {
public:
    InstructionDependencies(const Function& function);
    bool depends(std::size_t test_index, std::size_t dependency_index) {
        return _matrix[test_index * _size + dependency_index];
    }
    const std::vector<int>& ranks() const { return _ranks; }

private:
    std::size_t _size;
    std::vector<bool> _matrix;
    std::vector<int> _ranks;
};

class LastUseOfLocals {
public:
    LastUseOfLocals(const Function& function);
    std::vector<std::size_t>& local_indices(std::size_t index) {
        return _last_used.at(index);
    }

private:
    std::vector<std::vector<std::size_t>> _last_used;
};

Function sort_breadth_first(const Function& function);

} // namespace madspace
