#include "python_function.hpp"

#include <array>
#include <format>
#include <stdexcept>

#include "function_runtime.hpp"

using namespace madspace;

namespace madspace_py {

namespace {

// Shared by the UMAMI entry points of one Python function. Python calls the
// `_call` method of `function` (a madspace.PythonFunction).
struct PythonFunctionState {
    py::object function;
    std::string name;
    std::string device_name;
    std::size_t device_index = 0;
    UmamiDevice umami_device;
    std::array<bool, UMAMI_INPUT_KEY_COUNT> supported_inputs{};
    std::array<bool, UMAMI_INPUT_KEY_COUNT> required_inputs{};
    std::array<bool, UMAMI_OUTPUT_KEY_COUNT> supported_outputs{};
    std::vector<std::pair<std::string, double>> parameter_defaults;

    ~PythonFunctionState() {
        if (Py_IsInitialized()) {
            PythonGuard guard;
            function = py::object();
        } else {
            function.release();
        }
    }
};

// One UMAMI instance, with its own parameter values. Kept free of Python
// objects, since instances are created on the thread pool without the GIL.
struct PythonFunctionInstance {
    std::vector<double> parameters;
};

} // namespace

Tensor tensor_view(
    std::uintptr_t address,
    DataType dtype,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& stride,
    std::size_t offset,
    const std::string& device_name,
    std::size_t device_index
) {
    DevicePtr device;
    if (device_name == "cpu") {
        device = cpu_device();
    } else if (device_name == "cuda") {
        device = cuda_device(device_index);
    } else if (device_name == "hip") {
        device = hip_device(device_index);
    } else {
        throw std::invalid_argument(std::format("unknown device {}", device_name));
    }
    std::size_t item_size =
        dtype == DataType::dt_float64 ? sizeof(double) : sizeof(me_int_t);
    void* data = reinterpret_cast<void*>(address + offset * item_size);
    return Tensor(dtype, Sizes(shape), Sizes(stride), device, data, [] {});
}

const MatrixElementApi& add_python_function(
    Context& context,
    py::object function,
    const std::vector<int>& input_keys,
    const std::vector<int>& required_input_keys,
    const std::vector<int>& output_keys,
    const std::string& framework,
    const std::string& name,
    const std::vector<std::pair<std::string, double>>& parameter_defaults,
    const std::unordered_map<std::string, double>& parameters
) {
    auto state = std::make_shared<PythonFunctionState>();
    state->parameter_defaults = parameter_defaults;
    state->function = function;
    state->name = name;
    state->device_index = context.device()->device_index();
    switch (context.device()->device_type()) {
    case DeviceType::cpu:
        state->umami_device = UMAMI_DEVICE_CPU;
        state->device_name = "cpu";
        break;
    case DeviceType::cuda:
        state->umami_device = UMAMI_DEVICE_CUDA;
        state->device_name = "cuda";
        break;
    case DeviceType::hip:
        state->umami_device = UMAMI_DEVICE_HIP;
        state->device_name = "hip";
        break;
    }
    if (framework == "numpy" && state->umami_device != UMAMI_DEVICE_CPU) {
        throw std::invalid_argument(
            std::format(
                "Python function {}: numpy functions run on the CPU only, use torch "
                "for GPU runs",
                name
            )
        );
    }
    for (int key : input_keys) {
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT) {
            throw std::invalid_argument("invalid UMAMI input key");
        }
        state->supported_inputs[key] = true;
    }
    for (int key : required_input_keys) {
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !state->supported_inputs[key]) {
            throw std::invalid_argument("required input key is not supported");
        }
        state->required_inputs[key] = true;
    }
    for (int key : output_keys) {
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || key == UMAMI_OUT_GPU_STREAM) {
            throw std::invalid_argument("invalid UMAMI output key");
        }
        state->supported_outputs[key] = true;
    }

    UmamiFunctions functions;
    functions.get_meta = [state](UmamiMetaKey meta_key, void* result) {
        if (meta_key == UMAMI_META_DEVICE) {
            *static_cast<UmamiDevice*>(result) = state->umami_device;
            return UMAMI_SUCCESS;
        }
        return UMAMI_ERROR_UNSUPPORTED_META;
    };
    functions.supported_inputs = [state](bool const** supported, int* count) {
        *supported = state->supported_inputs.data();
        *count = UMAMI_INPUT_KEY_COUNT;
        return UMAMI_SUCCESS;
    };
    functions.required_inputs = [state](bool const** required, int* count) {
        *required = state->required_inputs.data();
        *count = UMAMI_INPUT_KEY_COUNT;
        return UMAMI_SUCCESS;
    };
    functions.supported_outputs = [state](bool const** supported, int* count) {
        *supported = state->supported_outputs.data();
        *count = UMAMI_OUTPUT_KEY_COUNT;
        return UMAMI_SUCCESS;
    };
    functions.initialize = [state](UmamiHandle* handle, char const*) {
        auto instance = new PythonFunctionInstance;
        for (auto& [parameter_name, value] : state->parameter_defaults) {
            instance->parameters.push_back(value);
        }
        *handle = instance;
        return UMAMI_SUCCESS;
    };
    functions.set_parameter =
        [state](UmamiHandle handle, char const* parameter_name, double real, double) {
            auto instance = static_cast<PythonFunctionInstance*>(handle);
            for (std::size_t i = 0;
                 auto& [known_name, value] : state->parameter_defaults) {
                if (known_name == parameter_name) {
                    instance->parameters.at(i) = real;
                    return UMAMI_SUCCESS;
                }
                ++i;
            }
            return UMAMI_ERROR;
        };
    functions.free = [](UmamiHandle handle) {
        delete static_cast<PythonFunctionInstance*>(handle);
        return UMAMI_SUCCESS;
    };
    functions.matrix_element =
        [state](
            UmamiHandle handle,
            std::size_t count,
            std::size_t stride,
            std::size_t offset,
            std::size_t input_count,
            UmamiInputKey const* input_keys,
            void const* const* inputs,
            std::size_t output_count,
            UmamiOutputKey const* output_keys,
            void* const* outputs
        ) {
            std::array<bool, UMAMI_INPUT_KEY_COUNT> has_input{};
            for (std::size_t i = 0; i < input_count; ++i) {
                if (input_keys[i] < 0 || input_keys[i] >= UMAMI_INPUT_KEY_COUNT ||
                    !state->supported_inputs[input_keys[i]]) {
                    return UMAMI_ERROR_UNSUPPORTED_INPUT;
                }
                has_input[input_keys[i]] = true;
            }
            for (int key = 0; key < UMAMI_INPUT_KEY_COUNT; ++key) {
                if (state->required_inputs[key] && !has_input[key]) {
                    return UMAMI_ERROR_MISSING_INPUT;
                }
            }
            std::optional<std::uintptr_t> stream;
            for (std::size_t i = 0; i < output_count; ++i) {
                if (output_keys[i] == UMAMI_OUT_GPU_STREAM) {
                    // the GPU runtime passes the stream handle itself
                    stream = reinterpret_cast<std::uintptr_t>(outputs[i]);
                } else if (output_keys[i] < 0 ||
                           output_keys[i] >= UMAMI_OUTPUT_KEY_COUNT ||
                           !state->supported_outputs[output_keys[i]]) {
                    return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
                }
            }
            if (count == 0) {
                return UMAMI_SUCCESS;
            }

            std::string error;
            {
                PythonGuard guard;
                try {
                    py::dict input_dict, output_dict;
                    for (std::size_t i = 0; i < input_count; ++i) {
                        input_dict[py::int_(static_cast<int>(input_keys[i]))] =
                            py::int_(reinterpret_cast<std::uintptr_t>(inputs[i]));
                    }
                    for (std::size_t i = 0; i < output_count; ++i) {
                        if (output_keys[i] != UMAMI_OUT_GPU_STREAM) {
                            output_dict[py::int_(static_cast<int>(output_keys[i]))] =
                                py::int_(reinterpret_cast<std::uintptr_t>(outputs[i]));
                        }
                    }
                    py::object parameter_values = py::none();
                    if (!state->parameter_defaults.empty()) {
                        py::dict values;
                        auto instance = static_cast<PythonFunctionInstance*>(handle);
                        for (std::size_t i = 0; i < instance->parameters.size(); ++i) {
                            values[py::str(state->parameter_defaults.at(i).first)] =
                                instance->parameters.at(i);
                        }
                        parameter_values = values;
                    }
                    state->function.attr("_call")(
                        state->device_name,
                        state->device_index,
                        count,
                        stride,
                        offset,
                        input_dict,
                        output_dict,
                        stream ? py::object(py::int_(*stream)) : py::object(py::none()),
                        parameter_values
                    );
                } catch (py::error_already_set& e) {
                    error = e.what();
                }
            }
            if (!error.empty()) {
                throw std::runtime_error(
                    std::format("Error in Python function {}: {}", state->name, error)
                );
            }
            return UMAMI_SUCCESS;
        };
    return context.add_matrix_element(std::move(functions), name, "", parameters);
}

} // namespace madspace_py
