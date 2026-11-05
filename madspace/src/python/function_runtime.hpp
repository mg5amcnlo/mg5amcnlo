#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"

namespace py = pybind11;
using namespace madspace;

namespace madspace_py {

/// Holds the GIL while alive, for code that can run on madspace worker threads.
/// Does nothing once the interpreter is finalized.
class PythonGuard {
public:
    PythonGuard() : _active(Py_IsInitialized()) {
        if (_active) {
            _state = PyGILState_Ensure();
        }
    }
    ~PythonGuard() {
        if (_active) {
            PyGILState_Release(_state);
        }
    }
    PythonGuard(const PythonGuard&) = delete;
    PythonGuard& operator=(const PythonGuard&) = delete;

private:
    bool _active;
    PyGILState_STATE _state;
};

std::tuple<int, int> dlpack_device(Tensor tensor);
py::object tensor_to_dlpack(
    Tensor tensor,
    std::optional<std::int64_t> stream = std::nullopt,
    std::optional<std::tuple<int, int>> max_version = std::nullopt,
    std::optional<std::tuple<int, int>> dl_device = std::nullopt,
    std::optional<bool> copy = std::nullopt
);
Tensor dlpack_to_tensor(
    py::object tensor,
    std::optional<Type> expected_type = std::nullopt,
    std::size_t arg_index = 0,
    DevicePtr expected_device = nullptr,
    bool* dlpack_version_cache = nullptr,
    std::vector<PyTypeObject*>* ordered_producers = nullptr
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
    void release_inputs();

    Function _function;
    ContextPtr _context;
    std::unordered_map<DevicePtr, RuntimePtr> _runtimes;
    bool _dlpack_version_cache = false;
};

} // namespace madspace_py
