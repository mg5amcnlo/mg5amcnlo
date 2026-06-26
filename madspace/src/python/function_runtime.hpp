#pragma once

#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"

namespace py = pybind11;
using namespace madspace;

namespace madspace_py {

std::tuple<int, int> dlpack_device(Tensor tensor);
py::object tensor_to_dlpack(
    Tensor tensor,
    std::optional<int> stream = std::nullopt,
    std::optional<std::tuple<int, int>> max_version = std::nullopt,
    std::optional<int> dl_device = std::nullopt,
    std::optional<bool> copy = std::nullopt
);
Tensor dlpack_to_tensor(
    py::object tensor,
    std::optional<Type> expected_type = std::nullopt,
    std::size_t arg_index = 0,
    DevicePtr expected_device = nullptr,
    bool* dlpack_version_cache = nullptr
);

struct FunctionRuntime {
    FunctionRuntime(Function function) : _function(function), _context(nullptr) {}
    FunctionRuntime(Function function, ContextPtr context) :
        _function(function), _context(context) {}
    std::vector<Tensor> call(std::vector<py::object> args);
    std::tuple<
        std::vector<Tensor>,
        std::vector<std::optional<Tensor>>,
        std::vector<bool>>
    call_with_grad(
        const std::vector<py::object>& args,
        const std::vector<bool>& input_requires_grad
    );
    std::tuple<
        std::vector<std::optional<Tensor>>,
        std::vector<std::tuple<std::string, std::optional<Tensor>>>>
    call_backward(
        const std::vector<py::object>& output_grads,
        const std::vector<py::object>& stored_locals,
        const std::vector<bool>& eval_grad
    );

    Function _function;
    ContextPtr _context;
    std::unordered_map<DevicePtr, RuntimePtr> _runtimes;
    bool _dlpack_version_cache = false;
};

} // namespace madspace_py
