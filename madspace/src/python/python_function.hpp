#pragma once

#include <cstdint>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "madspace/driver/context.hpp"

namespace py = pybind11;

namespace madspace_py {

/// Non-owning view of `shape` and element `stride` onto the memory at
/// `address`, starting `offset` elements in.
madspace::Tensor tensor_view(
    std::uintptr_t address,
    madspace::DataType dtype,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& stride,
    std::size_t offset,
    const std::string& device_name,
    std::size_t device_index = 0
);

/// Add a UMAMI matrix element to `context` that calls `function._call` with
/// the raw UMAMI buffers, see madspace.PythonFunction.
const madspace::MatrixElementApi& add_python_function(
    madspace::Context& context,
    py::object function,
    const std::vector<int>& input_keys,
    const std::vector<int>& required_input_keys,
    const std::vector<int>& output_keys,
    const std::string& framework,
    const std::string& name,
    const std::vector<std::pair<std::string, double>>& parameter_defaults = {},
    const std::unordered_map<std::string, double>& parameters = {}
);

} // namespace madspace_py
