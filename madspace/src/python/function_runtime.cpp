#include "function_runtime.hpp"

#include <format>
#include <ranges>
#include <stdexcept>

#include "dlpack.h"

using namespace madspace_py;
using namespace pybind11::literals;

namespace {

struct ManagerContext {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    std::vector<int64_t> batch_sizes;
    Tensor tensor;
};

void deleter(struct DLManagedTensor* self) {
    delete static_cast<ManagerContext*>(self->manager_ctx);
    delete self;
};

Runtime* get_runtime(FunctionRuntime& func_runtime, DevicePtr expected_device) {
    Runtime* runtime;
    if (!expected_device) {
        expected_device =
            func_runtime._context ? func_runtime._context->device() : cpu_device();
    }
    if (auto search = func_runtime._runtimes.find(expected_device);
        search != func_runtime._runtimes.end()) {
        runtime = search->second.get();
    } else {
        RuntimePtr rt;
        if (func_runtime._context) {
            if (func_runtime._context->device() != expected_device) {
                throw std::invalid_argument(
                    "Given context does not have compatible device"
                );
            }
            rt = build_runtime(func_runtime._function, func_runtime._context);
        } else {
            rt = build_runtime(
                func_runtime._function,
                madspace::default_device_context(expected_device)
            );
        }
        runtime = rt.get();
        func_runtime._runtimes[expected_device] = std::move(rt);
    }
    return runtime;
}

std::tuple<std::vector<Tensor>, Runtime*> check_and_convert_args(
    const std::vector<py::object>& args,
    FunctionRuntime& func_runtime,
    bool* dlpack_version_cache
) {
    // TODO: check batch sizes
    auto n_args = func_runtime._function.inputs().size();
    if (args.size() != n_args) {
        throw std::invalid_argument(
            std::format(
                "Wrong number of arguments. Expected {}, got {}", n_args, args.size()
            )
        );
    }
    std::vector<Tensor> inputs;
    DevicePtr expected_device = nullptr;
    for (int i = 0; i < n_args; ++i) {
        auto& arg = args.at(i);
        auto& input_type = func_runtime._function.inputs().at(i).type;
        auto tensor =
            dlpack_to_tensor(arg, input_type, i, expected_device, dlpack_version_cache);
        if (expected_device == nullptr && tensor &&
            tensor.dtype() != DataType::batch_sizes) {
            expected_device = tensor.device();
        }
        inputs.push_back(tensor);
    }
    return {inputs, get_runtime(func_runtime, expected_device)};
}

} // namespace

std::tuple<int, int> madspace_py::dlpack_device(Tensor tensor) {
    switch (tensor.device()->device_type()) {
    case DeviceType::cpu:
        return {kDLCPU, 0};
    case DeviceType::cuda:
        return {kDLCUDA, 0};
    case DeviceType::hip:
        return {kDLROCM, 0};
    default:
        throw std::logic_error("unreachable");
    }
}

py::object madspace_py::tensor_to_dlpack(
    Tensor tensor,
    std::optional<int> stream,
    std::optional<std::tuple<int, int>> max_version,
    std::optional<int> dl_device,
    std::optional<bool> copy
) {
    // TODO: do something with the arguments
    if (!tensor) {
        return py::none();
    }

    DLManagedTensor* dl_tensor;
    if (tensor.dtype() == DataType::batch_sizes) {
        ManagerContext* context = new ManagerContext{
            {static_cast<int64_t>(tensor.batch_sizes().size())},
            {1},
            {tensor.batch_sizes().begin(), tensor.batch_sizes().end()},
            {}
        };
        dl_tensor = new DLManagedTensor{
            {static_cast<void*>(context->batch_sizes.data()),
             {kDLCPU, 0},
             static_cast<int32_t>(context->shape.size()),
             {kDLInt, 64, 1},
             context->shape.data(),
             context->stride.data(),
             0},
            static_cast<void*>(context),
            &deleter
        };
    } else {
        DLDataType dtype;
        switch (tensor.dtype()) {
        case DataType::dt_float:
            dtype = {kDLFloat, 64, 1};
            break;
        case DataType::dt_int:
            dtype = {kDLInt, 32, 1};
            break;
        default:
            break;
        }
        ManagerContext* context = new ManagerContext{
            {tensor.shape().begin(), tensor.shape().end()},
            {tensor.stride().begin(), tensor.stride().end()},
            {},
            tensor
        };
        auto [device_type, device_id] = dlpack_device(tensor);
        dl_tensor = new DLManagedTensor{
            {context->tensor.data(),
             DLDevice{static_cast<DLDeviceType>(device_type), device_id},
             static_cast<int32_t>(context->shape.size()),
             dtype,
             context->shape.data(),
             context->stride.data(),
             0},
            static_cast<void*>(context),
            &deleter
        };
    }

    return py::capsule(dl_tensor, "dltensor", [](PyObject* self) {
        // Implement capsule deleter following the example in
        // https://dmlc.github.io/dlpack/latest/python_spec.html
        if (PyCapsule_IsValid(self, "used_dltensor")) {
            return;
        }
        DLManagedTensor* managed =
            static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, "dltensor"));
        if (managed == NULL) {
            PyErr_WriteUnraisable(self);
            return;
        }
        if (managed->deleter) {
            managed->deleter(managed);
        }
    });
}

Tensor madspace_py::dlpack_to_tensor(
    py::object tensor,
    std::optional<Type> expected_type,
    std::size_t arg_index,
    DevicePtr expected_device,
    bool* dlpack_version_cache
) {
    if (tensor.is_none()) {
        return {};
    }

    py::object dlpack_func = tensor.attr("__dlpack__");
    py::object capsule_obj;

    // catching exceptions is extremely expensive so we cache whether to use the new or
    // old version of the dlpack protocol
    if (dlpack_version_cache == nullptr || !*dlpack_version_cache) {
        try {
            capsule_obj = dlpack_func(
                "max_version"_a =
                    std::make_tuple(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)
            );
        } catch (py::error_already_set& e) {
            if (e.matches(PyExc_TypeError)) {
                capsule_obj = dlpack_func();
                if (dlpack_version_cache != nullptr) {
                    *dlpack_version_cache = true;
                }
            } else {
                throw;
            }
        }
    } else {
        try {
            capsule_obj = dlpack_func();
        } catch (py::error_already_set& e) {
            capsule_obj = dlpack_func(
                "max_version"_a =
                    std::make_tuple(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)
            );
            *dlpack_version_cache = false;
        }
    }
    PyObject* capsule = capsule_obj.ptr();
    if (!capsule) {
        throw std::runtime_error("value must support the dlpack protocol");
    }

    void* managed_ptr;
    DLTensor* dl_tensor;
    bool versioned = PyCapsule_IsValid(capsule, "dltensor_versioned");
    if (versioned) {
        managed_ptr = PyCapsule_GetPointer(capsule, "dltensor_versioned");
        if (!managed_ptr) {
            throw std::runtime_error("value must support the dlpack protocol");
        }
        auto managed = static_cast<DLManagedTensorVersioned*>(managed_ptr);
        if (managed->version.major > 1) {
            throw std::runtime_error("unsupported dlpack version");
        }
        dl_tensor = &managed->dl_tensor;
    } else {
        managed_ptr = PyCapsule_GetPointer(capsule, "dltensor");
        if (!managed_ptr) {
            throw std::runtime_error("value must support the dlpack protocol");
        }
        auto managed = static_cast<DLManagedTensor*>(managed_ptr);
        dl_tensor = &managed->dl_tensor;
    }

    bool is_batch_sizes =
        expected_type ? expected_type->dtype == DataType::batch_sizes : false;
    DataType dtype;
    if (dl_tensor->dtype.code == kDLFloat && dl_tensor->dtype.bits == 64 &&
        dl_tensor->dtype.lanes == 1) {
        dtype = DataType::dt_float;
        if (expected_type && expected_type->dtype != DataType::dt_float) {
            throw std::invalid_argument(
                std::format("Argument {}: got unexpected dtype", arg_index + 1)
            );
        }
    } else if (dl_tensor->dtype.code == kDLInt && dl_tensor->dtype.bits == 32 &&
               dl_tensor->dtype.lanes == 1) {
        dtype = DataType::dt_int;
        if (expected_type && expected_type->dtype != DataType::dt_int &&
            !is_batch_sizes) {
            throw std::invalid_argument(
                std::format("Argument {}: got unexpected dtype", arg_index + 1)
            );
        }
    } else if (dl_tensor->dtype.code == kDLInt && dl_tensor->dtype.bits == 64 &&
               dl_tensor->dtype.lanes == 1) {
        dtype = DataType::batch_sizes;
        is_batch_sizes = true;
        if (!is_batch_sizes && expected_type) {
            throw std::invalid_argument(
                std::format("Argument {}: got unexpected dtype", arg_index + 1)
            );
        }
    } else {
        throw std::invalid_argument(
            std::format(
                "Argument {}: input dtype must be 64-bit float or 32-bit int",
                arg_index + 1
            )
        );
    }

    DevicePtr device;
    if (dl_tensor->device.device_type == kDLCUDA && dl_tensor->device.device_id == 0) {
        device = cuda_device();
    } else if (dl_tensor->device.device_type == kDLROCM &&
               dl_tensor->device.device_id == 0) {
        device = hip_device();
    } else if (dl_tensor->device.device_type == kDLCPU &&
               dl_tensor->device.device_id == 0) {
        device = cpu_device();
    } else {
        throw std::invalid_argument(
            std::format("Argument {}: device not supported", arg_index + 1)
        );
    }
    if (expected_device && !is_batch_sizes && device != expected_device) {
        throw std::invalid_argument(
            std::format("Argument {}: wrong device", arg_index + 1)
        );
    }

    Tensor ret_tensor;
    if (is_batch_sizes) {
        if (dl_tensor->ndim != 1) {
            throw std::invalid_argument(
                std::format(
                    "Argument {}: wrong input dimension. Expected 1, got {}",
                    arg_index + 1,
                    dl_tensor->ndim
                )
            );
        }
        std::size_t count = dl_tensor->shape[0];
        if (expected_type && count != expected_type->batch_size_list.size()) {
            throw std::invalid_argument(
                std::format(
                    "Argument {}, dimension 0: shape mismatch. Expected {}, got {}",
                    arg_index + 1,
                    expected_type->batch_size_list.size(),
                    dl_tensor->shape[0]
                )
            );
        }
        if (dl_tensor->device.device_type != kDLCPU ||
            dl_tensor->device.device_id != 0) {
            throw std::invalid_argument(
                std::format(
                    "Argument {}: batch size list must have device CPU", arg_index + 1
                )
            );
        }
        std::vector<std::size_t> batch_sizes(count);
        std::size_t bs_stride = dl_tensor->strides ? dl_tensor->strides[0] : 1;
        if (dl_tensor->dtype.bits == 64) {
            int64_t* data_ptr = reinterpret_cast<int64_t*>(
                static_cast<uint8_t*>(dl_tensor->data) + dl_tensor->byte_offset
            );
            for (std::size_t i = 0; i < count; ++i) {
                batch_sizes[i] = data_ptr[bs_stride * i];
            }
        } else {
            int* data_ptr = reinterpret_cast<int*>(
                static_cast<uint8_t*>(dl_tensor->data) + dl_tensor->byte_offset
            );
            for (std::size_t i = 0; i < count; ++i) {
                batch_sizes[i] = data_ptr[bs_stride * i];
            }
        }
        if (versioned) {
            auto ptr = static_cast<DLManagedTensorVersioned*>(managed_ptr);
            if (ptr->deleter) {
                ptr->deleter(ptr);
            }
        } else {
            auto ptr = static_cast<DLManagedTensor*>(managed_ptr);
            if (ptr->deleter) {
                ptr->deleter(ptr);
            }
        }
        ret_tensor = {batch_sizes};
    } else {
        bool is_batch = !expected_type || expected_type->batch_size != BatchSize::one;
        if (expected_type &&
            dl_tensor->ndim != expected_type->shape.size() + is_batch) {
            throw std::invalid_argument(
                std::format(
                    "Argument {}: wrong input dimension. Expected {}, got {}",
                    arg_index + 1,
                    expected_type->shape.size() + 1,
                    dl_tensor->ndim
                )
            );
        }
        std::vector<size_t> shape, stride;
        if (!is_batch) {
            shape.push_back(1);
            stride.push_back(1);
        }
        shape.insert(shape.end(), dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim);
        if (dl_tensor->strides) {
            stride.insert(
                stride.end(), dl_tensor->strides, dl_tensor->strides + dl_tensor->ndim
            );
        } else {
            stride.resize(shape.size());
            std::size_t stride_prod = 1;
            for (auto [size_i, stride_i] :
                 zip(std::views::reverse(shape), std::views::reverse(stride))) {
                stride_i = stride_prod;
                stride_prod *= size_i;
            }
        }
        if (expected_type) {
            for (int j = is_batch; j < dl_tensor->ndim; ++j) {
                if (shape.at(j) != expected_type->shape.at(j - is_batch)) {
                    throw std::invalid_argument(
                        std::format(
                            "Argument {}, dimension {}: shape mismatch. Expected {}, "
                            "got {}",
                            arg_index + 1,
                            j,
                            expected_type->shape.at(j - is_batch),
                            shape.at(j)
                        )
                    );
                }
            }
        }
        void* data_ptr = static_cast<void*>(
            static_cast<uint8_t*>(dl_tensor->data) + dl_tensor->byte_offset
        );
        std::function<void()> deleter;
        if (versioned) {
            deleter = [managed_ptr]() {
                auto ptr = static_cast<DLManagedTensorVersioned*>(managed_ptr);
                if (ptr->deleter) {
                    ptr->deleter(ptr);
                }
            };
        } else {
            deleter = [managed_ptr]() {
                auto ptr = static_cast<DLManagedTensor*>(managed_ptr);
                if (ptr->deleter) {
                    ptr->deleter(ptr);
                }
            };
        }
        ret_tensor = {dtype, shape, stride, device, data_ptr, deleter};
    }

    if (PyCapsule_SetName(
            capsule, versioned ? "used_dltensor_versioned" : "used_dltensor"
        ) < 0) {
        throw std::runtime_error("could not rename capsule");
    }
    return ret_tensor;
}

std::vector<Tensor> FunctionRuntime::call(std::vector<py::object> args) {
    auto [inputs, runtime] =
        check_and_convert_args(args, *this, &_dlpack_version_cache);
    return runtime->run(inputs);
}

std::tuple<std::vector<Tensor>, std::vector<std::optional<Tensor>>, std::vector<bool>>
FunctionRuntime::call_with_grad(
    const std::vector<py::object>& args, const std::vector<bool>& input_requires_grad
) {
    auto [inputs, runtime] =
        check_and_convert_args(args, *this, &_dlpack_version_cache);
    auto [outputs, loc_grad, eval_grad] =
        runtime->run_with_grad(inputs, input_requires_grad);
    std::vector<std::optional<Tensor>> local_grads;
    for (auto& grad : loc_grad) {
        if (grad) {
            local_grads.push_back(grad);
        } else {
            local_grads.push_back({});
        }
    }
    return {outputs, local_grads, eval_grad};
}

std::tuple<
    std::vector<std::optional<Tensor>>,
    std::vector<std::tuple<std::string, std::optional<Tensor>>>>
FunctionRuntime::call_backward(
    const std::vector<py::object>& output_grads,
    const std::vector<py::object>& stored_locals,
    const std::vector<bool>& eval_grad
) {
    std::vector<Tensor> arg_out;
    DevicePtr expected_device = nullptr;
    std::size_t arg_index = 0;
    for (auto& grad : output_grads) {
        auto tensor = dlpack_to_tensor(
            grad, std::nullopt, arg_index, expected_device, &_dlpack_version_cache
        );
        if (expected_device == nullptr && tensor &&
            tensor.dtype() != DataType::batch_sizes) {
            expected_device = tensor.device();
        }
        arg_out.push_back(tensor);
        ++arg_index;
    }
    std::vector<Tensor> arg_locals;
    for (auto& local : stored_locals) {
        auto tensor = dlpack_to_tensor(
            local, std::nullopt, arg_index, expected_device, &_dlpack_version_cache
        );
        if (expected_device == nullptr && tensor &&
            tensor.dtype() != DataType::batch_sizes) {
            expected_device = tensor.device();
        }
        arg_locals.push_back(tensor);
        ++arg_index;
    }
    // TODO: checks here
    Runtime* runtime = get_runtime(*this, expected_device);
    auto [ret_in_grads, ret_glob_grads] =
        runtime->run_backward(arg_out, arg_locals, eval_grad);
    std::vector<std::optional<Tensor>> input_grads;
    for (auto& grad : ret_in_grads) {
        if (grad) {
            input_grads.push_back(grad);
        } else {
            input_grads.push_back({});
        }
    }
    std::vector<std::tuple<std::string, std::optional<Tensor>>> global_grads;
    for (std::size_t ret_index = 0; auto& glob : _function.globals()) {
        if (!_context->global_requires_grad(glob.first)) {
            continue;
        }
        auto& grad = ret_glob_grads.at(ret_index);
        if (grad) {
            global_grads.push_back({glob.first, grad});
        } else {
            global_grads.push_back({glob.first, std::nullopt});
        }
        ++ret_index;
    }
    return {input_grads, global_grads};
}
