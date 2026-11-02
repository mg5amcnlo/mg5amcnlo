#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "docstrings.hpp"
#include "function_runtime.hpp"
#include "instruction_set.hpp"
#include "madspace/compgraphs.hpp"
#include "madspace/compgraphs/optimizer.hpp"
#include "madspace/driver.hpp"
#include "madspace/phasespace.hpp"
#include "madspace/util.hpp"
#include "python_function.hpp"

namespace py = pybind11;
using namespace madspace;
using namespace madspace_py;

namespace {

template <typename T>
auto to_string(const T& object) {
    std::ostringstream str;
    str << object;
    return str.str();
}

struct InstrCopy {
    std::string name;
    int opcode;
    InstrCopy(InstructionPtr instr) : name(instr->name()), opcode(instr->opcode()) {}
};

class PyMapping : public Mapping, py::trampoline_self_life_support {
public:
    using Mapping::Mapping;

    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_forward_impl, &fb, inputs, conditions
        );
    }

    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            Result, Mapping, build_inverse_impl, &fb, inputs, conditions
        );
    }
};

class PyFunctionGenerator : public FunctionGenerator, py::trampoline_self_life_support {
public:
    using FunctionGenerator::FunctionGenerator;

    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            NamedVector<Value>, FunctionGenerator, build_function_impl, &fb, &args
        );
    }
};

template <typename EnumType, typename ParentType>
void add_enum(
    ParentType& parent,
    const char* enum_name,
    std::initializer_list<std::pair<const std::string, EnumType>> values,
    const std::string& prefix = "",
    const char* doc = ""
) {
    std::unordered_map<std::string, EnumType> str_to_enum_map(values);
    py::enum_<EnumType> enumeration(parent, enum_name, doc);
    for (auto& [key, value] : values) {
        enumeration.value((prefix + key).c_str(), value);
    }
    enumeration.def(
        "__init__",
        [str_to_enum_map, enum_name](EnumType& self, const std::string& name) {
            if (auto search = str_to_enum_map.find(name);
                search != str_to_enum_map.end()) {
                self = search->second;
            } else {
                throw std::invalid_argument(
                    std::format("Value {} does not exist in enum {}", name, enum_name)
                );
            }
        },
        py::arg("name")
    );
    enumeration.export_values();
    py::implicitly_convertible<std::string, EnumType>();
}

template <typename T>
void named_vector_instance(py::module_& m, const char* name) {
    py::classh<NamedVector<T>>(m, name, pydoc::doc("NamedVector"))
        .def(py::init<>(), pydoc::doc("NamedVector::NamedVector"))
        .def(
            py::init<const std::vector<std::string>&, const std::vector<T>&>(),
            py::arg("keys"),
            py::arg("values"),
            pydoc::doc("NamedVector::NamedVector#2")
        )
        .def(
            py::init<const std::vector<std::pair<std::string, T>>&>(),
            py::arg("items"),
            pydoc::doc("NamedVector::NamedVector#4")
        )
        .def("__len__", &NamedVector<T>::size)
        .def(
            "__getitem__",
            py::overload_cast<std::size_t>(&NamedVector<T>::at, py::const_),
            py::arg("key")
        )
        .def(
            "__getitem__",
            py::overload_cast<const std::string&>(&NamedVector<T>::at, py::const_),
            py::arg("key")
        )
        .def("values", &NamedVector<T>::values, pydoc::doc("NamedVector::values"))
        .def(
            "index_map",
            &NamedVector<T>::index_map,
            pydoc::doc("NamedVector::index_map")
        )
        .def("keys", &NamedVector<T>::keys, pydoc::doc("NamedVector::keys"))
        .def(
            "push_back",
            &NamedVector<T>::push_back,
            py::arg("name"),
            py::arg("item"),
            pydoc::doc("NamedVector::push_back")
        );
}

} // namespace

PYBIND11_MODULE(_madspace_py, m) {
    add_enum<DataType>(
        m,
        "DataType",
        {
            {"int32", DataType::dt_int32},
            {"float32", DataType::dt_float32},
            {"float64", DataType::dt_float64},
            {"batch_sizes", DataType::batch_sizes},
        },
        "",
        pydoc::doc("DataType")
    );

    py::classh<BatchSize>(m, "BatchSize", pydoc::doc("BatchSize"))
        .def(py::init<>())
        .def(
            py::init<std::string>(), py::arg("name"), pydoc::doc("BatchSize::BatchSize")
        )
        .def_readonly_static("one", &BatchSize::one, pydoc::doc("BatchSize::one"))
        .def("__str__", &to_string<BatchSize>)
        .def("__repr__", &to_string<BatchSize>);
    m.attr("batch_size") = py::cast(batch_size);

    py::classh<Type>(m, "Type", pydoc::doc("Type"))
        .def(
            py::init<DataType, BatchSize, std::vector<int>>(),
            py::arg("dtype"),
            py::arg("batch_size"),
            py::arg("shape"),
            pydoc::doc("Type::Type")
        )
        .def(
            py::init<std::vector<BatchSize>>(),
            py::arg("batch_size_list"),
            pydoc::doc("Type::Type#2")
        )
        .def_readonly("dtype", &Type::dtype, pydoc::doc("Type::dtype"))
        .def_readonly("batch_size", &Type::batch_size, pydoc::doc("Type::batch_size"))
        .def_readonly("shape", &Type::shape, pydoc::doc("Type::shape"))
        .def("__str__", &to_string<Type>)
        .def("__repr__", &to_string<Type>);
    m.attr("single_float") = py::cast(single_float);
    m.attr("single_int") = py::cast(single_int);
    m.def(
        "multichannel_batch_size",
        &multichannel_batch_size,
        py::arg("count"),
        pydoc::doc("multichannel_batch_size")
    );
    m.attr("batch_float") = py::cast(batch_float);
    m.attr("batch_int") = py::cast(batch_int);
    m.attr("batch_four_vec") = py::cast(batch_four_vec);
    m.def(
        "batch_float_array",
        &batch_float_array,
        py::arg("count"),
        pydoc::doc("batch_float_array")
    );
    m.def(
        "batch_four_vec_array",
        &batch_four_vec_array,
        py::arg("count"),
        pydoc::doc("batch_four_vec_array")
    );

    py::classh<InstrCopy>(m, "Instruction", pydoc::doc("Instruction"))
        .def("__str__", [](const InstrCopy& instr) { return instr.name; })
        .def_readonly("name", &InstrCopy::name, pydoc::doc("Instruction::name"))
        .def_readonly("opcode", &InstrCopy::opcode, pydoc::doc("Instruction::opcode"));

    py::classh<Value>(m, "Value", pydoc::doc("Value"))
        .def(py::init<me_int_t>(), py::arg("value"), pydoc::doc("Value::Value#2"))
        .def(py::init<double>(), py::arg("value"), pydoc::doc("Value::Value#3"))
        // constant (non-batched) tensors, e.g. the frame_mask of
        // boost_to_frame or the permutations of permute_momenta
        .def(
            py::init<const std::vector<me_int_t>&, const std::vector<int>&>(),
            py::arg("values"),
            py::arg("shape") = std::vector<int>{}
        )
        .def(
            py::init<const std::vector<double>&, const std::vector<int>&>(),
            py::arg("values"),
            py::arg("shape") = std::vector<int>{}
        )
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type, pydoc::doc("Value::type"))
        .def_readonly(
            "literal_value", &Value::literal_value, pydoc::doc("Value::literal_value")
        )
        .def_readonly(
            "local_index", &Value::local_index, pydoc::doc("Value::local_index")
        );
    py::implicitly_convertible<me_int_t, Value>();
    py::implicitly_convertible<double, Value>();

    named_vector_instance<Value>(m, "NamedValues");
    named_vector_instance<Type>(m, "NamedTypes");

    py::classh<InstructionCall>(m, "InstructionCall", pydoc::doc("InstructionCall"))
        .def("__str__", &to_string<InstructionCall>)
        .def("__repr__", &to_string<InstructionCall>)
        .def_property_readonly(
            "instruction",
            [](const InstructionCall& call) -> InstrCopy { return call.instruction; },
            pydoc::doc("InstructionCall::instruction")
        )
        .def_readonly(
            "inputs", &InstructionCall::inputs, pydoc::doc("InstructionCall::inputs")
        )
        .def_readonly(
            "outputs", &InstructionCall::outputs, pydoc::doc("InstructionCall::outputs")
        );

    py::classh<Function>(m, "Function", pydoc::doc("Function"), py::dynamic_attr())
        .def("__str__", &to_string<Function>)
        .def("__repr__", &to_string<Function>)
        .def("save", &Function::save, py::arg("file"), pydoc::doc("Function::save"))
        .def_static(
            "load", &Function::load, py::arg("file"), pydoc::doc("Function::load")
        )
        .def_property_readonly(
            "inputs", &Function::inputs, pydoc::doc("Function::inputs")
        )
        .def_property_readonly(
            "outputs", &Function::outputs, pydoc::doc("Function::outputs")
        )
        .def_property_readonly(
            "locals", &Function::locals, pydoc::doc("Function::locals")
        )
        .def_property_readonly(
            "globals", &Function::globals, pydoc::doc("Function::globals")
        )
        .def_property_readonly(
            "instructions",
            &Function::instructions,
            pydoc::doc("Function::instructions")
        );

    m.def(
        "fuse_instructions",
        &fuse_instructions,
        py::arg("function"),
        "Fuses single-precision conversions and network activations into the "
        "instructions producing their inputs (the lowering the GPU backend applies "
        "when it builds a runtime)."
    );

    py::classh<Device> device(m, "Device", pydoc::doc("Device"));
    m.def(
        "cpu_device",
        &cpu_device,
        py::return_value_policy::reference,
        pydoc::doc("cpu_device")
    );
    m.def(
        "cuda_device",
        &cuda_device,
        py::arg("index") = 0,
        py::return_value_policy::reference,
        pydoc::doc("cuda_device")
    );
    m.def(
        "hip_device",
        &hip_device,
        py::arg("index") = 0,
        py::return_value_policy::reference,
        pydoc::doc("hip_device")
    );
    m.def("available_backends", &available_backends, pydoc::doc("available_backends"));

    py::classh<MatrixElementApi>(m, "MatrixElementApi", pydoc::doc("MatrixElementApi"))
        .def(
            "device_type",
            [](const MatrixElementApi& api) {
                switch (api.device_type()) {
                case DeviceType::cuda:
                    return "cuda";
                case DeviceType::hip:
                    return "hip";
                default:
                    return "cpu";
                }
            },
            "Device the library was built for: ``cpu``, ``cuda`` or ``hip``."
        )
        .def(
            "particle_count",
            &MatrixElementApi::particle_count,
            pydoc::doc("MatrixElementApi::particle_count")
        )
        .def(
            "diagram_count",
            &MatrixElementApi::diagram_count,
            pydoc::doc("MatrixElementApi::diagram_count")
        )
        .def(
            "helicity_count",
            &MatrixElementApi::helicity_count,
            pydoc::doc("MatrixElementApi::helicity_count")
        )
        .def("index", &MatrixElementApi::index, pydoc::doc("MatrixElementApi::index"))
        .def("file_name", &MatrixElementApi::file_name)
        .def("supported_inputs", &MatrixElementApi::supported_inputs)
        .def("required_inputs", &MatrixElementApi::required_inputs)
        .def("supported_outputs", &MatrixElementApi::supported_outputs)
        .def("supports_set_parameter", &MatrixElementApi::supports_set_parameter)
        .def("supports_get_parameter", &MatrixElementApi::supports_get_parameter)
        .def(
            "set_parameter",
            &MatrixElementApi::set_parameter,
            py::arg("name"),
            py::arg("real"),
            py::arg("imag") = 0.0
        )
        .def("get_parameter", &MatrixElementApi::get_parameter, py::arg("name"));

    py::classh<Tensor>(m, "Tensor", py::dynamic_attr(), pydoc::doc("Tensor"))
        .def(
            "__dlpack__",
            &tensor_to_dlpack,
            py::arg("stream") = std::nullopt,
            py::arg("max_version") = std::nullopt,
            py::arg("dl_device") = std::nullopt,
            py::arg("copy") = std::nullopt
        )
        .def("__dlpack_device__", &dlpack_device);

    py::classh<Context>(m, "Context", pydoc::doc("Context"))
        .def(
            py::init<int>(),
            py::arg("thread_count") = -1,
            pydoc::doc("Context::Context")
        )
        .def(
            py::init<DevicePtr, int>(),
            py::arg("device"),
            py::arg("thread_count") = -1,
            pydoc::doc("Context::Context#2")
        )
        .def(
            "load_matrix_element",
            &Context::load_matrix_element,
            py::arg("file"),
            py::arg("param_card"),
            py::arg("parameters") = std::unordered_map<std::string, double>{},
            py::return_value_policy::reference_internal,
            pydoc::doc("Context::load_matrix_element")
        )
        .def(
            "_add_python_function",
            &add_python_function,
            py::arg("function"),
            py::arg("input_keys"),
            py::arg("required_input_keys"),
            py::arg("output_keys"),
            py::arg("framework"),
            py::arg("name"),
            py::arg("parameter_defaults") =
                std::vector<std::pair<std::string, double>>{},
            py::arg("parameters") = std::unordered_map<std::string, double>{},
            py::return_value_policy::reference_internal
        )
        .def(
            "define_global",
            &Context::define_global,
            py::arg("name"),
            py::arg("dtype"),
            py::arg("shape"),
            py::arg("requires_grad") = false,
            pydoc::doc("Context::define_global")
        )
        .def(
            "get_global",
            &Context::global,
            py::arg("name"),
            pydoc::doc("Context::global")
        )
        .def(
            "global_requires_grad",
            &Context::global_requires_grad,
            py::arg("name"),
            pydoc::doc("Context::global_requires_grad")
        )
        .def(
            "global_exists",
            &Context::global_exists,
            py::arg("name"),
            pydoc::doc("Context::global_exists")
        )
        .def(
            "global_names", &Context::global_names, pydoc::doc("Context::global_names")
        )
        .def(
            "delete_global",
            &Context::delete_global,
            py::arg("name"),
            pydoc::doc("Context::delete_global")
        )
        .def(
            "copy_globals_from",
            &Context::copy_globals_from,
            py::arg("context"),
            pydoc::doc("Context::copy_globals_from")
        )
        .def(
            "matrix_element",
            &Context::matrix_element,
            py::arg("index"),
            py::return_value_policy::reference_internal,
            pydoc::doc("Context::matrix_element")
        )
        .def(
            "save_globals",
            &Context::save_globals,
            py::arg("dir"),
            pydoc::doc("Context::save_globals")
        )
        .def(
            "load_globals",
            &Context::load_globals,
            py::arg("dir"),
            pydoc::doc("Context::load_globals")
        )
        .def(
            "device",
            &Context::device,
            py::return_value_policy::reference,
            pydoc::doc("Context::device")
        );
    m.def("default_context", &default_context, pydoc::doc("default_context"));
    m.def(
        "default_cuda_context",
        &default_cuda_context,
        py::arg("index") = 0,
        pydoc::doc("default_cuda_context")
    );
    m.def(
        "default_hip_context",
        &default_hip_context,
        py::arg("index") = 0,
        pydoc::doc("default_hip_context")
    );
    m.def("get_stream", &caller_stream, pydoc::doc("caller_stream"));
    m.def(
        "set_stream",
        &set_caller_stream,
        py::arg("stream"),
        pydoc::doc("set_caller_stream")
    );

    m.def(
        "_tensor_view",
        &tensor_view,
        py::arg("address"),
        py::arg("dtype"),
        py::arg("shape"),
        py::arg("stride"),
        py::arg("offset"),
        py::arg("device"),
        py::arg("device_index") = 0
    );
    py::dict umami_input_keys, umami_output_keys;
    for (auto [name, key] : std::initializer_list<std::pair<const char*, int>>{
             {"momenta", UMAMI_IN_MOMENTA},
             {"alpha_s", UMAMI_IN_ALPHA_S},
             {"flavor_index", UMAMI_IN_FLAVOR_INDEX},
             {"random_color", UMAMI_IN_RANDOM_COLOR},
             {"random_helicity", UMAMI_IN_RANDOM_HELICITY},
             {"random_diagram", UMAMI_IN_RANDOM_DIAGRAM},
             {"helicity_index", UMAMI_IN_HELICITY_INDEX},
             {"diagram_index", UMAMI_IN_DIAGRAM_INDEX},
             {"channel_index", UMAMI_IN_CHANNEL_INDEX},
             {"particle_count", UMAMI_IN_PARTICLE_COUNT},
             {"pdg_ids", UMAMI_IN_PDG_IDS},
             {"momentum_fraction", UMAMI_IN_MOMENTUM_FRACTION},
             {"scale", UMAMI_IN_SCALE},
             {"pdg_id", UMAMI_IN_PDG_ID},
             {"beam_index", UMAMI_IN_BEAM_INDEX},
         }) {
        umami_input_keys[name] = key;
    }
    for (auto [name, key] : std::initializer_list<std::pair<const char*, int>>{
             {"matrix_element", UMAMI_OUT_MATRIX_ELEMENT},
             {"color_index", UMAMI_OUT_COLOR_INDEX},
             {"helicity_index", UMAMI_OUT_HELICITY_INDEX},
             {"diagram_index", UMAMI_OUT_DIAGRAM_INDEX},
             {"cut", UMAMI_OUT_CUT},
             {"ren_scale", UMAMI_OUT_REN_SCALE},
             {"fact_scale1", UMAMI_OUT_FACT_SCALE_1},
             {"fact_scale2", UMAMI_OUT_FACT_SCALE_2},
             {"pdf", UMAMI_OUT_PDF},
             {"bias", UMAMI_OUT_BIAS},
         }) {
        umami_output_keys[name] = key;
    }
    m.attr("_UMAMI_INPUT_KEYS") = umami_input_keys;
    m.attr("_UMAMI_OUTPUT_KEYS") = umami_output_keys;

    py::classh<FunctionRuntime>(m, "FunctionRuntime", py::dynamic_attr())
        .def(py::init<Function>(), py::arg("function"))
        .def(py::init<Function, ContextPtr>(), py::arg("function"), py::arg("context"))
        .def("call", &FunctionRuntime::call)
        .def("call_with_grad", &FunctionRuntime::call_with_grad)
        .def("call_backward", &FunctionRuntime::call_backward)
        .def("release_inputs", &FunctionRuntime::release_inputs);

    auto& fb =
        py::classh<FunctionBuilder>(m, "FunctionBuilder", pydoc::doc("FunctionBuilder"))
            .def(
                py::init<const NamedVector<Type>&, const NamedVector<Type>&>(),
                py::arg("input_types"),
                py::arg("output_types"),
                pydoc::doc("FunctionBuilder::FunctionBuilder")
            )
            .def(
                "input",
                &FunctionBuilder::input,
                py::arg("index"),
                pydoc::doc("FunctionBuilder::input")
            )
            .def(
                "input_range",
                &FunctionBuilder::input_range,
                py::arg("start_index"),
                py::arg("end_index"),
                pydoc::doc("FunctionBuilder::input_range")
            )
            .def(
                "output",
                &FunctionBuilder::output,
                py::arg("index"),
                py::arg("value"),
                pydoc::doc("FunctionBuilder::output")
            )
            .def(
                "output_range",
                &FunctionBuilder::output_range,
                py::arg("start_index"),
                py::arg("values"),
                pydoc::doc("FunctionBuilder::output_range")
            )
            .def(
                "get_global",
                &FunctionBuilder::global,
                py::arg("name"),
                py::arg("dtype"),
                py::arg("shape"),
                pydoc::doc("FunctionBuilder::global")
            )
            //.def("instruction", &FunctionBuilder::instruction, py::arg("name"),
            // py::arg("args"))
            .def(
                "product",
                &FunctionBuilder::product,
                py::arg("values"),
                pydoc::doc("FunctionBuilder::product")
            )
            .def(
                "current_stream",
                &FunctionBuilder::current_stream,
                pydoc::doc("FunctionBuilder::current_stream")
            )
            .def(
                "set_current_stream",
                &FunctionBuilder::set_current_stream,
                pydoc::doc("FunctionBuilder::set_current_stream")
            )
            .def(
                "function",
                &FunctionBuilder::function,
                pydoc::doc("FunctionBuilder::function")
            );
    add_instructions(fb);

    py::classh<Mapping, PyMapping>(
        m, "Mapping", pydoc::doc("Mapping"), py::dynamic_attr()
    )
        .def(
            py::init<
                const std::string&,
                const NamedVector<Type>&,
                const NamedVector<Type>&,
                const NamedVector<Type>&>(),
            py::arg("name"),
            py::arg("input_types"),
            py::arg("output_types"),
            py::arg("condition_types"),
            pydoc::doc("Mapping::Mapping")
        )
        .def(
            "forward_function",
            &Mapping::forward_function,
            pydoc::doc("Mapping::forward_function")
        )
        .def(
            "inverse_function",
            &Mapping::inverse_function,
            pydoc::doc("Mapping::inverse_function")
        )
        .def(
            "build_forward",
            py::overload_cast<FunctionBuilder&, const ValueVec&, const ValueVec&>(
                &Mapping::build_forward, py::const_
            ),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions"),
            pydoc::doc("Mapping::build_forward")
        )
        .def(
            "build_forward",
            py::overload_cast<
                FunctionBuilder&,
                const NamedVector<Value>&,
                const NamedVector<Value>&>(&Mapping::build_forward, py::const_),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions"),
            pydoc::doc("Mapping::build_forward")
        )
        .def(
            "build_inverse",
            py::overload_cast<FunctionBuilder&, const ValueVec&, const ValueVec&>(
                &Mapping::build_inverse, py::const_
            ),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions"),
            pydoc::doc("Mapping::build_inverse")
        )
        .def(
            "build_inverse",
            py::overload_cast<
                FunctionBuilder&,
                const NamedVector<Value>&,
                const NamedVector<Value>&>(&Mapping::build_inverse, py::const_),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions"),
            pydoc::doc("Mapping::build_inverse")
        );

    py::classh<FunctionGenerator, PyFunctionGenerator>(
        m, "FunctionGenerator", pydoc::doc("FunctionGenerator"), py::dynamic_attr()
    )
        .def(
            py::init<
                const std::string&,
                const NamedVector<Type>&,
                const NamedVector<Type>&>(),
            py::arg("name"),
            py::arg("arg_types"),
            py::arg("return_types"),
            pydoc::doc("FunctionGenerator::FunctionGenerator")
        )
        .def(
            "function",
            &FunctionGenerator::function,
            pydoc::doc("FunctionGenerator::function")
        )
        .def(
            "build_function",
            py::overload_cast<FunctionBuilder&, const ValueVec&>(
                &FunctionGenerator::build_function, py::const_
            ),
            py::arg("builder"),
            py::arg("args"),
            pydoc::doc("FunctionGenerator::build_function")
        )
        .def(
            "build_function",
            py::overload_cast<FunctionBuilder&, const NamedVector<Value>&>(
                &FunctionGenerator::build_function, py::const_
            ),
            py::arg("builder"),
            py::arg("args"),
            pydoc::doc("FunctionGenerator::build_function")
        );

    py::classh<Invariant, Mapping>(m, "Invariant", pydoc::doc("Invariant"))
        .def(
            py::init<double, double, double, double>(),
            py::arg("power") = 0.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.,
            py::arg("flat_window") = 0.,
            pydoc::doc("Invariant::Invariant")
        );

    py::classh<Luminosity, Mapping>(m, "Luminosity", pydoc::doc("Luminosity"))
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("s_lab"),
            py::arg("s_hat_min"),
            py::arg("s_hat_max") = 0.,
            py::arg("invariant_power") = 1.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.,
            pydoc::doc("Luminosity::Luminosity")
        );

    py::classh<TwoBodyDecay, Mapping>(m, "TwoBodyDecay", pydoc::doc("TwoBodyDecay"))
        .def(
            py::init<bool, double, double, double>(),
            py::arg("com"),
            py::arg("pt_min") = 0.,
            py::arg("y_max1") = -1.,
            py::arg("y_max2") = -1.,
            pydoc::doc("TwoBodyDecay::TwoBodyDecay")
        )
        .def(
            "random_dim",
            &TwoBodyDecay::random_dim,
            pydoc::doc("TwoBodyDecay::random_dim")
        );

    py::classh<TwoToTwoParticleScattering, Mapping>(
        m, "TwoToTwoParticleScattering", pydoc::doc("TwoToTwoParticleScattering")
    )
        .def(
            py::init<bool, double, double, double, bool, double, double, double>(),
            py::arg("com"),
            py::arg("invariant_power") = 0.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.,
            py::arg("has_cut") = false,
            py::arg("y_max1") = -1.,
            py::arg("y_max2") = -1.,
            py::arg("beam_sign") = 1.,
            pydoc::doc("TwoToTwoParticleScattering::TwoToTwoParticleScattering")
        );

    py::classh<DoubleT, Mapping>(m, "DoubleT", pydoc::doc("DoubleT"))
        .def(
            py::init<
                double,
                double,
                double,
                double,
                double,
                double,
                bool,
                double,
                double,
                double>(),
            py::arg("t1_invariant_power") = 0.,
            py::arg("t1_mass") = 0.,
            py::arg("t1_width") = 0.,
            py::arg("t2_invariant_power") = 0.,
            py::arg("t2_mass") = 0.,
            py::arg("t2_width") = 0.,
            py::arg("has_cut") = false,
            py::arg("y_max1") = -1.,
            py::arg("y_max2") = -1.,
            py::arg("beam_sign") = 1.,
            pydoc::doc("DoubleT::DoubleT")
        );

    py::classh<ThreeBodyDecay, Mapping>(
        m, "ThreeBodyDecay", pydoc::doc("ThreeBodyDecay")
    )
        .def(
            py::init<bool>(),
            py::arg("com"),
            pydoc::doc("ThreeBodyDecay::ThreeBodyDecay")
        )
        .def(
            "random_dim",
            &ThreeBodyDecay::random_dim,
            pydoc::doc("ThreeBodyDecay::random_dim")
        );

    py::classh<TwoToThreeParticleScattering, Mapping>(
        m, "TwoToThreeParticleScattering", pydoc::doc("TwoToThreeParticleScattering")
    )
        .def(
            py::init<
                double,
                double,
                double,
                double,
                double,
                double,
                bool,
                bool,
                bool,
                bool,
                bool>(),
            py::arg("t_invariant_power") = 0.,
            py::arg("t_mass") = 0.,
            py::arg("t_width") = 0.,
            py::arg("s_invariant_power") = 0.,
            py::arg("s_mass") = 0.,
            py::arg("s_width") = 0.,
            py::arg("has_cut") = false,
            py::arg("arcsine_s23") = true,
            py::arg("p12_condition") = false,
            py::arg("incoming_recoil") = false,
            py::arg("cut_aware_branch") = false,
            pydoc::doc("TwoToThreeParticleScattering::TwoToThreeParticleScattering")
        )
        .def(
            "discrete_dim",
            &TwoToThreeParticleScattering::discrete_dim,
            pydoc::doc("TwoToThreeParticleScattering::discrete_dim")
        );

    py::classh<Propagator>(m, "Propagator", pydoc::doc("Propagator"))
        .def(
            py::init<double, double, int, double, double, int, double>(),
            py::arg("mass") = 0.,
            py::arg("width") = 0.,
            py::arg("integration_order") = 0,
            py::arg("e_min") = 0.,
            py::arg("e_max") = 0.,
            py::arg("pdg_id") = 0,
            py::arg("flat_window") = 0.
        )
        .def_readonly("mass", &Propagator::mass, pydoc::doc("Propagator::mass"))
        .def_readonly("width", &Propagator::width, pydoc::doc("Propagator::width"))
        .def_readonly(
            "integration_order",
            &Propagator::integration_order,
            pydoc::doc("Propagator::integration_order")
        )
        .def_readonly("e_min", &Propagator::e_min, pydoc::doc("Propagator::e_min"))
        .def_readonly("e_max", &Propagator::e_max, pydoc::doc("Propagator::e_max"))
        .def_readonly("pdg_id", &Propagator::pdg_id, pydoc::doc("Propagator::pdg_id"))
        .def_readonly(
            "flat_window",
            &Propagator::flat_window,
            pydoc::doc("Propagator::flat_window")
        );

    py::classh<TPropagatorMapping, Mapping>(
        m, "TPropagatorMapping", pydoc::doc("TPropagatorMapping")
    )
        .def(
            py::init<
                std::vector<std::size_t>,
                double,
                std::vector<double>,
                std::vector<double>>(),
            py::arg("integration_order"),
            py::arg("invariant_power") = 0.8,
            py::arg("pt_min") = std::vector<double>{},
            py::arg("y_max") = std::vector<double>{},
            pydoc::doc("TPropagatorMapping::TPropagatorMapping")
        )
        .def(
            "random_dim",
            &TPropagatorMapping::random_dim,
            pydoc::doc("TPropagatorMapping::random_dim")
        );

    auto color_ordered_mapping = py::classh<ColorOrderedMapping, Mapping>(
        m, "ColorOrderedMapping", pydoc::doc("ColorOrderedMapping")
    );
    py::class_<ColorOrderedMapping::Options>(
        color_ordered_mapping, "Options", pydoc::doc("ColorOrderedMapping::Options")
    )
        .def(
            py::init([](const std::vector<std::size_t>& color_order,
                        std::optional<double> chain_invariant_power,
                        bool cut_aware_branch) {
                return ColorOrderedMapping::Options{
                    color_order, chain_invariant_power, cut_aware_branch
                };
            }),
            py::arg("color_order"),
            py::arg("chain_invariant_power") = std::nullopt,
            py::arg("cut_aware_branch") = true
        )
        .def_readwrite(
            "color_order",
            &ColorOrderedMapping::Options::color_order,
            pydoc::doc("ColorOrderedMapping::Options::color_order")
        )
        .def_readwrite(
            "chain_invariant_power",
            &ColorOrderedMapping::Options::chain_invariant_power,
            pydoc::doc("ColorOrderedMapping::Options::chain_invariant_power")
        )
        .def_readwrite(
            "cut_aware_branch",
            &ColorOrderedMapping::Options::cut_aware_branch,
            pydoc::doc("ColorOrderedMapping::Options::cut_aware_branch")
        );
    color_ordered_mapping
        .def(
            py::init<
                const ColorOrderedMapping::Options&,
                std::vector<double>,
                std::vector<std::vector<double>>,
                std::vector<std::vector<double>>,
                std::vector<double>>(),
            py::arg("options"),
            py::arg("pt_min") = std::vector<double>{},
            py::arg("m_inv_min") = std::vector<std::vector<double>>{},
            py::arg("dr_min") = std::vector<std::vector<double>>{},
            py::arg("y_max") = std::vector<double>{},
            pydoc::doc("ColorOrderedMapping::ColorOrderedMapping")
        )
        .def(
            "random_dim",
            &ColorOrderedMapping::random_dim,
            pydoc::doc("ColorOrderedMapping::random_dim")
        )
        .def(
            "discrete_dim",
            &ColorOrderedMapping::discrete_dim,
            pydoc::doc("ColorOrderedMapping::discrete_dim")
        );

    py::classh<ChiliMapping, Mapping>(m, "ChiliMapping", pydoc::doc("ChiliMapping"))
        .def(
            py::init<
                std::size_t,
                const std::vector<double>&,
                const std::vector<double>&>(),
            py::arg("n_particles"),
            py::arg("y_max"),
            py::arg("pt_min"),
            pydoc::doc("ChiliMapping::ChiliMapping")
        )
        .def(
            "random_dim",
            &ChiliMapping::random_dim,
            pydoc::doc("ChiliMapping::random_dim")
        );

    py::classh<VegasHistogram, FunctionGenerator>(
        m, "VegasHistogram", pydoc::doc("VegasHistogram")
    )
        .def(
            py::init<std::size_t, std::size_t>(),
            py::arg("dimension"),
            py::arg("bin_count"),
            pydoc::doc("VegasHistogram::VegasHistogram")
        );

    py::classh<VegasMapping, Mapping>(m, "VegasMapping", pydoc::doc("VegasMapping"))
        .def(
            py::init<std::size_t, std::size_t, const std::string&>(),
            py::arg("dimension"),
            py::arg("bin_count"),
            py::arg("prefix") = "",
            pydoc::doc("VegasMapping::VegasMapping")
        )
        .def(
            "grid_name", &VegasMapping::grid_name, pydoc::doc("VegasMapping::grid_name")
        )
        .def(
            "initialize_globals",
            &VegasMapping::initialize_globals,
            py::arg("context"),
            pydoc::doc("VegasMapping::initialize_globals")
        )
        .def(
            "dimension", &VegasMapping::dimension, pydoc::doc("VegasMapping::dimension")
        )
        .def(
            "bin_count", &VegasMapping::bin_count, pydoc::doc("VegasMapping::bin_count")
        );

    py::classh<FastRamboMapping, Mapping>(
        m, "FastRamboMapping", pydoc::doc("FastRamboMapping")
    )
        .def(
            py::init<std::size_t, bool, bool>(),
            py::arg("n_particles"),
            py::arg("massless"),
            py::arg("com") = true,
            pydoc::doc("FastRamboMapping::FastRamboMapping")
        )
        .def(
            "random_dim",
            &FastRamboMapping::random_dim,
            pydoc::doc("FastRamboMapping::random_dim")
        );

    py::classh<MultiChannelMapping, Mapping>(
        m, "MultiChannelMapping", pydoc::doc("MultiChannelMapping")
    )
        .def(
            py::init<std::vector<std::shared_ptr<Mapping>>&>(),
            py::arg("mappings"),
            pydoc::doc("MultiChannelMapping::MultiChannelMapping")
        );

    auto obs = py::classh<Observable, FunctionGenerator>(
        m, "Observable", pydoc::doc("Observable")
    );
    add_enum<Observable::ObservableOption>(
        obs,
        "ObservableOption",
        {
            {"e", Observable::obs_e},
            {"px", Observable::obs_px},
            {"py", Observable::obs_py},
            {"pz", Observable::obs_pz},
            {"mass", Observable::obs_mass},
            {"pt", Observable::obs_pt},
            {"p_mag", Observable::obs_p_mag},
            {"phi", Observable::obs_phi},
            {"theta", Observable::obs_theta},
            {"y", Observable::obs_y},
            {"y_abs", Observable::obs_y_abs},
            {"eta", Observable::obs_eta},
            {"eta_abs", Observable::obs_eta_abs},
            {"delta_eta", Observable::obs_delta_eta},
            {"delta_phi", Observable::obs_delta_phi},
            {"delta_r", Observable::obs_delta_r},
            {"pair_mass", Observable::obs_pair_mass},
            {"sqrt_s", Observable::obs_sqrt_s},
            {"sfos_pair_mass", Observable::obs_sfos_pair_mass},
        },
        "obs_"
    );
    obs.def(
           py::init<
               const std::vector<int>&,
               Observable::ObservableOption,
               const nested_vector2<int>&,
               bool,
               bool,
               const std::optional<Observable::ObservableOption>&,
               const std::vector<int>&,
               bool,
               const std::string&>(),
           py::arg("pids"),
           py::arg("observable"),
           py::arg("select_pids"),
           py::arg("sum_momenta") = false,
           py::arg("sum_observable") = false,
           py::arg("order_observable") = std::nullopt,
           py::arg("order_indices") = std::vector<int>{},
           py::arg("ignore_incoming") = true,
           py::arg("name") = "",
           pydoc::doc("Observable::Observable")
    )
        .def_readonly_static("jet_pids", &Observable::jet_pids)
        .def_readonly_static("bottom_pids", &Observable::bottom_pids)
        .def_readonly_static("lepton_pids", &Observable::lepton_pids)
        .def_readonly_static("missing_pids", &Observable::missing_pids)
        .def_readonly_static("photon_pids", &Observable::photon_pids);

    py::classh<CustomFunction, FunctionGenerator> custom_function(
        m, "CustomFunction", pydoc::doc("CustomFunction")
    );
    add_enum<CustomFunction::CustomFunctionType>(
        custom_function,
        "CustomFunctionType",
        {
            {"cut", CustomFunction::cut},
            {"scale", CustomFunction::scale},
            {"bias", CustomFunction::bias},
        }
    );
    custom_function
        .def(
            py::init<
                std::size_t,
                CustomFunction::CustomFunctionType,
                std::size_t,
                const nested_vector2<me_int_t>&>(),
            py::arg("function_index"),
            py::arg("type"),
            py::arg("particle_count"),
            py::arg("pdg_id_options") = nested_vector2<me_int_t>{},
            pydoc::doc("CustomFunction::CustomFunction")
        )
        .def(
            py::init<
                const MatrixElementApi&,
                CustomFunction::CustomFunctionType,
                std::size_t,
                const nested_vector2<me_int_t>&>(),
            py::arg("function_api"),
            py::arg("type"),
            py::arg("particle_count"),
            py::arg("pdg_id_options") = nested_vector2<me_int_t>{},
            pydoc::doc("CustomFunction::CustomFunction#2")
        )
        .def(
            "function_index",
            &CustomFunction::function_index,
            pydoc::doc("CustomFunction::function_index")
        )
        .def("type", &CustomFunction::type, pydoc::doc("CustomFunction::type"))
        .def(
            "dynamic_pdg_ids",
            &CustomFunction::dynamic_pdg_ids,
            pydoc::doc("CustomFunction::dynamic_pdg_ids")
        );
    py::classh<CustomPdf>(m, "CustomPdf", pydoc::doc("CustomPdf"))
        .def(
            py::init<std::size_t>(),
            py::arg("function_index"),
            pydoc::doc("CustomPdf::CustomPdf")
        )
        .def(
            py::init<const MatrixElementApi&>(),
            py::arg("function_api"),
            pydoc::doc("CustomPdf::CustomPdf#2")
        )
        .def_readonly(
            "function_index",
            &CustomPdf::function_index,
            pydoc::doc("CustomPdf::function_index")
        );

    auto cuts = py::classh<Cuts, FunctionGenerator>(m, "Cuts", pydoc::doc("Cuts"));
    add_enum<Cuts::CutMode>(
        cuts,
        "CutMode",
        {
            {"any", Cuts::any},
            {"all", Cuts::all},
        }
    );
    py::classh<Cuts::CutItem>(m, "CutItem", pydoc::doc("Cuts::CutItem"))
        .def(
            py::init<Observable, double, double, Cuts::CutMode>(),
            py::arg("observable"),
            py::arg("min") = -std::numeric_limits<double>::infinity(),
            py::arg("max") = std::numeric_limits<double>::infinity(),
            py::arg("mode") = Cuts::CutMode::all
        )
        .def_readonly(
            "observable",
            &Cuts::CutItem::observable,
            pydoc::doc("Cuts::CutItem::observable")
        )
        .def_readonly("min", &Cuts::CutItem::min, pydoc::doc("Cuts::CutItem::min"))
        .def_readonly("max", &Cuts::CutItem::max, pydoc::doc("Cuts::CutItem::max"))
        .def_readonly("mode", &Cuts::CutItem::mode, pydoc::doc("Cuts::CutItem::mode"));
    // Registered before the methods that return it: pybind11 writes a method's
    // signature when it is defined, and a type it does not know yet ends up as
    // its raw C++ name, which pybind11-stubgen rejects.
    py::classh<Cuts::PairMassAny>(cuts, "PairMassAny", pydoc::doc("Cuts::PairMassAny"))
        .def_readonly(
            "pairs", &Cuts::PairMassAny::pairs, pydoc::doc("Cuts::PairMassAny::pairs")
        )
        .def_readonly(
            "min", &Cuts::PairMassAny::min, pydoc::doc("Cuts::PairMassAny::min")
        );
    cuts.def(
            py::init<
                const std::vector<Cuts::CutItem>&,
                const std::optional<CustomFunction>&>(),
            py::arg("cut_data"),
            py::arg("custom_cut") = std::nullopt,
            pydoc::doc("Cuts::Cuts")
    )
        .def(
            py::init<std::size_t, const std::optional<CustomFunction>&>(),
            py::arg("particle_count"),
            py::arg("custom_cut") = std::nullopt,
            pydoc::doc("Cuts::Cuts#2")
        )
        .def("sqrt_s_min", &Cuts::sqrt_s_min, pydoc::doc("Cuts::sqrt_s_min"))
        .def("eta_max", &Cuts::eta_max, pydoc::doc("Cuts::eta_max"))
        .def("pt_min", &Cuts::pt_min, pydoc::doc("Cuts::pt_min"))
        .def("m_inv_min", &Cuts::m_inv_min, pydoc::doc("Cuts::m_inv_min"))
        .def("dr_min", &Cuts::dr_min, pydoc::doc("Cuts::dr_min"))
        .def(
            "pair_mass_any_min",
            &Cuts::pair_mass_any_min,
            pydoc::doc("Cuts::pair_mass_any_min")
        );

    py::classh<ObservableHistograms::HistItem>(
        m, "HistItem", pydoc::doc("ObservableHistograms::HistItem")
    )
        .def(
            py::init<Observable, double, double, std::size_t>(),
            py::arg("observable"),
            py::arg("min"),
            py::arg("max"),
            py::arg("bin_count")
        )
        .def_readonly(
            "observable",
            &ObservableHistograms::HistItem::observable,
            pydoc::doc("ObservableHistograms::HistItem::observable")
        )
        .def_readonly(
            "min",
            &ObservableHistograms::HistItem::min,
            pydoc::doc("ObservableHistograms::HistItem::min")
        )
        .def_readonly(
            "max",
            &ObservableHistograms::HistItem::max,
            pydoc::doc("ObservableHistograms::HistItem::max")
        )
        .def_readonly(
            "bin_count",
            &ObservableHistograms::HistItem::bin_count,
            pydoc::doc("ObservableHistograms::HistItem::bin_count")
        );
    py::classh<ObservableValues, FunctionGenerator>(
        m, "ObservableValues", pydoc::doc("ObservableValues")
    )
        .def(
            py::init<const std::vector<Observable>&>(),
            py::arg("observables"),
            pydoc::doc("ObservableValues::ObservableValues")
        )
        .def_property_readonly(
            "observables",
            &ObservableValues::observables,
            pydoc::doc("ObservableValues::observables")
        );
    py::classh<ObservableHistograms, FunctionGenerator>(
        m, "ObservableHistograms", pydoc::doc("ObservableHistograms")
    )
        .def(
            py::init<const std::vector<ObservableHistograms::HistItem>&>(),
            py::arg("observables"),
            pydoc::doc("ObservableHistograms::ObservableHistograms")
        )
        .def(
            "observables",
            &ObservableHistograms::observables,
            pydoc::doc("ObservableHistograms::observables")
        );

    auto line_ref =
        py::classh<Diagram::LineRef>(m, "LineRef", pydoc::doc("Diagram::LineRef"));
    add_enum<Diagram::LineType>(
        line_ref,
        "LineType",
        {
            {"incoming", Diagram::incoming},
            {"outgoing", Diagram::outgoing},
            {"propagator", Diagram::propagator},
        }
    );
    line_ref
        .def(
            py::init<Diagram::LineType, std::size_t>(),
            py::arg("type"),
            py::arg("index"),
            pydoc::doc("Diagram::LineRef::LineRef")
        )
        .def(
            py::init<std::string>(),
            py::arg("str"),
            pydoc::doc("Diagram::LineRef::LineRef#2")
        )
        .def("type", &Diagram::LineRef::type, pydoc::doc("Diagram::LineRef::type"))
        .def("index", &Diagram::LineRef::index, pydoc::doc("Diagram::LineRef::index"))
        .def("__repr__", &to_string<Diagram::LineRef>);
    py::implicitly_convertible<std::string, Diagram::LineRef>();
    py::classh<Diagram>(m, "Diagram", pydoc::doc("Diagram"))
        .def(
            py::init<
                std::vector<double>&,
                std::vector<double>&,
                std::vector<Propagator>&,
                std::vector<Diagram::Vertex>&>(),
            py::arg("incoming_masses"),
            py::arg("outgoing_masses"),
            py::arg("propagators"),
            py::arg("vertices"),
            pydoc::doc("Diagram::Diagram")
        )
        .def_property_readonly(
            "incoming_masses",
            &Diagram::incoming_masses,
            pydoc::doc("Diagram::incoming_masses")
        )
        .def_property_readonly(
            "outgoing_masses",
            &Diagram::outgoing_masses,
            pydoc::doc("Diagram::outgoing_masses")
        )
        .def_property_readonly(
            "propagators", &Diagram::propagators, pydoc::doc("Diagram::propagators")
        )
        .def_property_readonly(
            "vertices", &Diagram::vertices, pydoc::doc("Diagram::vertices")
        )
        .def_property_readonly(
            "incoming_vertices",
            &Diagram::incoming_vertices,
            pydoc::doc("Diagram::incoming_vertices")
        )
        .def_property_readonly(
            "outgoing_vertices",
            &Diagram::outgoing_vertices,
            pydoc::doc("Diagram::outgoing_vertices")
        )
        .def_property_readonly(
            "propagator_vertices",
            &Diagram::propagator_vertices,
            pydoc::doc("Diagram::propagator_vertices")
        );
    py::classh<Topology::Decay>(m, "Decay", pydoc::doc("Topology::Decay"))
        .def_readonly(
            "index", &Topology::Decay::index, pydoc::doc("Topology::Decay::index")
        )
        .def_readonly(
            "parent_index",
            &Topology::Decay::parent_index,
            pydoc::doc("Topology::Decay::parent_index")
        )
        .def_readonly(
            "child_indices",
            &Topology::Decay::child_indices,
            pydoc::doc("Topology::Decay::child_indices")
        )
        .def_readonly(
            "mass", &Topology::Decay::mass, pydoc::doc("Topology::Decay::mass")
        )
        .def_readonly(
            "width", &Topology::Decay::width, pydoc::doc("Topology::Decay::width")
        )
        .def_readonly(
            "e_min", &Topology::Decay::e_min, pydoc::doc("Topology::Decay::e_min")
        )
        .def_readonly(
            "e_max", &Topology::Decay::e_max, pydoc::doc("Topology::Decay::e_max")
        )
        .def_readonly(
            "pdg_id", &Topology::Decay::pdg_id, pydoc::doc("Topology::Decay::pdg_id")
        )
        .def_readonly(
            "on_shell",
            &Topology::Decay::on_shell,
            pydoc::doc("Topology::Decay::on_shell")
        )
        .def_readonly(
            "on_shell_boundary",
            &Topology::Decay::on_shell_boundary,
            pydoc::doc("Topology::Decay::on_shell_boundary")
        );
    auto& topology =
        py::classh<Topology>(m, "Topology", pydoc::doc("Topology"))
            .def(
                py::init<const Diagram&>(),
                py::arg("diagram"),
                pydoc::doc("Topology::Topology")
            )
            .def_static(
                "topologies",
                &Topology::topologies,
                py::arg("diagram"),
                pydoc::doc("Topology::topologies")
            )
            .def_property_readonly(
                "t_propagator_count",
                &Topology::t_propagator_count,
                pydoc::doc("Topology::t_propagator_count")
            )
            .def_property_readonly(
                "t_integration_order",
                &Topology::t_integration_order,
                pydoc::doc("Topology::t_integration_order")
            )
            .def_property_readonly(
                "t_propagator_masses",
                &Topology::t_propagator_masses,
                pydoc::doc("Topology::t_propagator_masses")
            )
            .def_property_readonly(
                "t_propagator_widths",
                &Topology::t_propagator_widths,
                pydoc::doc("Topology::t_propagator_widths")
            )
            .def_property_readonly(
                "decays", &Topology::decays, pydoc::doc("Topology::decays")
            )
            .def_property_readonly(
                "decay_integration_order",
                &Topology::decay_integration_order,
                pydoc::doc("Topology::decay_integration_order")
            )
            .def_property_readonly(
                "outgoing_indices",
                &Topology::outgoing_indices,
                pydoc::doc("Topology::outgoing_indices")
            )
            .def_property_readonly(
                "incoming_masses",
                &Topology::incoming_masses,
                pydoc::doc("Topology::incoming_masses")
            )
            .def_property_readonly(
                "outgoing_masses",
                &Topology::outgoing_masses,
                pydoc::doc("Topology::outgoing_masses")
            )
            .def(
                "propagator_momentum_terms",
                &Topology::propagator_momentum_terms,
                py::arg("only_decays") = false,
                pydoc::doc("Topology::propagator_momentum_terms")
            )
            .def("__str__", &Topology::to_string);
    py::classh<PhaseSpaceMapping, Mapping> psmap(
        m, "PhaseSpaceMapping", pydoc::doc("PhaseSpaceMapping")
    );
    add_enum<PhaseSpaceMapping::TChannelMode>(
        psmap,
        "TChannelMode",
        {
            {"propagator", PhaseSpaceMapping::propagator},
            {"rambo", PhaseSpaceMapping::rambo},
            {"chili", PhaseSpaceMapping::chili},
            {"color_ordered", PhaseSpaceMapping::color_ordered},
        }
    );
    psmap
        .def(
            py::init<
                const Topology&,
                double,
                bool,
                double,
                PhaseSpaceMapping::TChannelMode,
                const std::optional<Cuts>&,
                const nested_vector2<std::size_t>&,
                const std::optional<ColorOrderedMapping::Options>&,
                double,
                bool,
                std::optional<double>>(),
            py::arg("topology"),
            py::arg("cm_energy"),
            py::arg("leptonic") = false,
            py::arg("invariant_power") = 0.8,
            py::arg("t_channel_mode") = PhaseSpaceMapping::propagator,
            py::arg("cuts") = std::nullopt,
            py::arg("permutations") = nested_vector2<std::size_t>{},
            py::arg("color_ordered") = std::nullopt,
            py::arg("beam_rapidity") = 0.,
            py::arg("mirror_beams") = false,
            py::arg("s_hat_power") = std::nullopt,
            pydoc::doc("PhaseSpaceMapping::PhaseSpaceMapping")
        )
        .def(
            py::init<
                const std::vector<double>&,
                double,
                bool,
                double,
                PhaseSpaceMapping::TChannelMode,
                std::optional<Cuts>,
                const std::optional<ColorOrderedMapping::Options>&,
                double,
                bool,
                std::optional<double>>(),
            py::arg("external_masses"),
            py::arg("cm_energy"),
            py::arg("leptonic") = false,
            py::arg("invariant_power") = 0.8,
            py::arg("mode") = PhaseSpaceMapping::rambo,
            py::arg("cuts") = std::nullopt,
            py::arg("color_ordered") = std::nullopt,
            py::arg("beam_rapidity") = 0.,
            py::arg("mirror_beams") = false,
            py::arg("s_hat_power") = std::nullopt,
            pydoc::doc("PhaseSpaceMapping::PhaseSpaceMapping#2")
        )
        .def(
            "random_dim",
            &PhaseSpaceMapping::random_dim,
            pydoc::doc("PhaseSpaceMapping::random_dim")
        )
        .def(
            "discrete_dim",
            &PhaseSpaceMapping::discrete_dim,
            pydoc::doc("PhaseSpaceMapping::discrete_dim")
        )
        .def(
            "particle_count",
            &PhaseSpaceMapping::particle_count,
            pydoc::doc("PhaseSpaceMapping::particle_count")
        )
        .def(
            "channel_count",
            &PhaseSpaceMapping::channel_count,
            pydoc::doc("PhaseSpaceMapping::channel_count")
        )
        .def(
            "beam_rapidity",
            &PhaseSpaceMapping::beam_rapidity,
            pydoc::doc("PhaseSpaceMapping::beam_rapidity")
        )
        .def(
            "mirror_beams",
            &PhaseSpaceMapping::mirror_beams,
            pydoc::doc("PhaseSpaceMapping::mirror_beams")
        )
        .def("cuts", &PhaseSpaceMapping::cuts, pydoc::doc("PhaseSpaceMapping::cuts"))
        .def(
            "empty", &PhaseSpaceMapping::empty, pydoc::doc("PhaseSpaceMapping::empty")
        );

    py::classh<MultiChannelFunction, FunctionGenerator>(
        m, "MultiChannelFunction", pydoc::doc("MultiChannelFunction")
    )
        .def(
            py::init<std::vector<std::shared_ptr<FunctionGenerator>>&, bool>(),
            py::arg("functions"),
            py::arg("return_batch_sizes") = false,
            pydoc::doc("MultiChannelFunction::MultiChannelFunction")
        );

    py::classh<MatrixElement, FunctionGenerator> matrix_element(
        m, "MatrixElement", pydoc::doc("MatrixElement")
    );
    add_enum<MatrixElement::MatrixElementInput>(
        matrix_element,
        "MatrixElementInput",
        {
            {"momenta_in", MatrixElement::momenta_in},
            {"alpha_s_in", MatrixElement::alpha_s_in},
            {"flavor_in", MatrixElement::flavor_in},
            {"random_color_in", MatrixElement::random_color_in},
            {"random_helicity_in", MatrixElement::random_helicity_in},
            {"random_diagram_in", MatrixElement::random_diagram_in},
            {"helicity_in", MatrixElement::helicity_in},
            {"diagram_in", MatrixElement::diagram_in},
            {"channel_in", MatrixElement::channel_in},
            {"particle_count_in", MatrixElement::particle_count_in},
            {"pdg_ids_in", MatrixElement::pdg_ids_in},
            {"momentum_fraction_in", MatrixElement::momentum_fraction_in},
            {"scale_in", MatrixElement::scale_in},
            {"pdg_id_in", MatrixElement::pdg_id_in},
            {"beam_index_in", MatrixElement::beam_index_in},
        }
    );
    add_enum<MatrixElement::MatrixElementOutput>(
        matrix_element,
        "MatrixElementOutput",
        {
            {"matrix_element_out", MatrixElement::matrix_element_out},
            {"diagram_amp2_out", MatrixElement::diagram_amp2_out},
            {"color_index_out", MatrixElement::color_index_out},
            {"helicity_index_out", MatrixElement::helicity_index_out},
            {"diagram_index_out", MatrixElement::diagram_index_out},
            {"cut_out", MatrixElement::cut_out},
            {"ren_scale_out", MatrixElement::ren_scale_out},
            {"fact_scale1_out", MatrixElement::fact_scale1_out},
            {"fact_scale2_out", MatrixElement::fact_scale2_out},
            {"pdf_out", MatrixElement::pdf_out},
            {"bias_out", MatrixElement::bias_out},
        }
    );
    matrix_element
        .def(
            py::init<
                std::size_t,
                std::size_t,
                const std::vector<MatrixElement::MatrixElementInput>&,
                const std::vector<MatrixElement::MatrixElementOutput>&,
                std::size_t,
                bool,
                const std::vector<me_int_t>&,
                std::size_t>(),
            py::arg("matrix_element_index"),
            py::arg("particle_count"),
            // C++ defaults to {momenta_in} / {matrix_element_out}; kept required
            // in Python because pybind11-stubgen cannot render an enum-list
            // default (see the header @param docs).
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("diagram_count") = 1,
            py::arg("sample_random_inputs") = false,
            py::arg("me_frame") = std::vector<me_int_t>{},
            py::arg("incoming_count") = 2,
            pydoc::doc("MatrixElement::MatrixElement")
        )
        .def(
            py::init<
                const MatrixElementApi&,
                const std::vector<MatrixElement::MatrixElementInput>&,
                const std::vector<MatrixElement::MatrixElementOutput>&,
                bool,
                const std::vector<me_int_t>&,
                std::size_t>(),
            py::arg("matrix_element_api"),
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("sample_random_inputs") = false,
            py::arg("me_frame") = std::vector<me_int_t>{},
            py::arg("incoming_count") = 2,
            pydoc::doc("MatrixElement::MatrixElement#2")
        )
        .def(
            "matrix_element_index",
            &MatrixElement::matrix_element_index,
            pydoc::doc("MatrixElement::matrix_element_index")
        )
        .def(
            "diagram_count",
            &MatrixElement::diagram_count,
            pydoc::doc("MatrixElement::diagram_count")
        )
        .def(
            "particle_count",
            &MatrixElement::particle_count,
            pydoc::doc("MatrixElement::particle_count")
        )
        .def("inputs", &MatrixElement::inputs, pydoc::doc("MatrixElement::inputs"))
        .def("outputs", &MatrixElement::outputs, pydoc::doc("MatrixElement::outputs"))
        .def(
            "external_inputs",
            &MatrixElement::external_inputs,
            pydoc::doc("MatrixElement::external_inputs")
        )
        .def(
            "frame_mask",
            &MatrixElement::frame_mask,
            pydoc::doc("MatrixElement::frame_mask")
        )
        .def(
            "reference_mask",
            &MatrixElement::reference_mask,
            pydoc::doc("MatrixElement::reference_mask")
        );

    py::classh<MLP, FunctionGenerator> mlp(m, "MLP", pydoc::doc("MLP"));
    add_enum<MLP::Activation>(
        mlp,
        "Activation",
        {
            {"relu", MLP::relu},
            {"leaky_relu", MLP::leaky_relu},
            {"elu", MLP::elu},
            {"gelu", MLP::gelu},
            {"sigmoid", MLP::sigmoid},
            {"softplus", MLP::softplus},
            {"linear", MLP::linear},
        }
    );
    mlp.def(
           py::init<
               std::size_t,
               std::size_t,
               std::size_t,
               std::size_t,
               MLP::Activation,
               const std::string&,
               DataType>(),
           py::arg("input_dim"),
           py::arg("output_dim"),
           py::arg("hidden_dim") = 32,
           py::arg("layers") = 3,
           py::arg("activation") = MLP::leaky_relu,
           py::arg("prefix") = "",
           py::arg("precision") = DataType::dt_float64,
           pydoc::doc("MLP::MLP")
    )
        .def("input_dim", &MLP::input_dim, pydoc::doc("MLP::input_dim"))
        .def("output_dim", &MLP::output_dim, pydoc::doc("MLP::output_dim"))
        .def("precision", &MLP::precision, pydoc::doc("MLP::precision"))
        .def(
            "initialize_globals",
            &MLP::initialize_globals,
            py::arg("context"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("MLP::initialize_globals")
        )
        .def(
            "last_layer_bias_name",
            &MLP::last_layer_bias_name,
            pydoc::doc("MLP::last_layer_bias_name")
        )
        .def("global_names", &MLP::global_names, pydoc::doc("MLP::global_names"));

    py::classh<Flow, Mapping>(m, "Flow", pydoc::doc("Flow"))
        .def(
            py::init<
                std::size_t,
                std::size_t,
                const std::string&,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation,
                bool,
                DataType>(),
            py::arg("input_dim"),
            py::arg("condition_dim") = 0,
            py::arg("prefix") = "",
            py::arg("bin_count") = 10,
            py::arg("subnet_hidden_dim") = 32,
            py::arg("subnet_layers") = 3,
            py::arg("subnet_activation") = MLP::leaky_relu,
            py::arg("invert_spline") = true,
            py::arg("subnet_precision") = DataType::dt_float64,
            pydoc::doc("Flow::Flow")
        )
        .def("input_dim", &Flow::input_dim, pydoc::doc("Flow::input_dim"))
        .def("condition_dim", &Flow::condition_dim, pydoc::doc("Flow::condition_dim"))
        .def(
            "initialize_globals",
            &Flow::initialize_globals,
            py::arg("context"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("Flow::initialize_globals")
        )
        .def(
            "initialize_from_vegas",
            &Flow::initialize_from_vegas,
            py::arg("context"),
            py::arg("grid_name"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("Flow::initialize_from_vegas")
        );

    py::classh<PropagatorChannelWeights, FunctionGenerator>(
        m, "PropagatorChannelWeights", pydoc::doc("PropagatorChannelWeights")
    )
        .def(
            py::init<
                const std::vector<Topology>&,
                const nested_vector3<std::size_t>&,
                const nested_vector2<std::size_t>&>(),
            py::arg("topologies"),
            py::arg("permutations"),
            py::arg("channel_indices"),
            pydoc::doc("PropagatorChannelWeights::PropagatorChannelWeights")
        );

    py::classh<SubchannelWeights, FunctionGenerator>(
        m, "SubchannelWeights", pydoc::doc("SubchannelWeights")
    )
        .def(
            py::init<
                const nested_vector2<Topology>&,
                const nested_vector3<std::size_t>&,
                const nested_vector2<std::size_t>>(),
            py::arg("topologies"),
            py::arg("permutations"),
            py::arg("channel_indices"),
            pydoc::doc("SubchannelWeights::SubchannelWeights")
        )
        .def(
            "channel_count",
            &SubchannelWeights::channel_count,
            pydoc::doc("SubchannelWeights::channel_count")
        );

    py::classh<MomentumPreprocessing, FunctionGenerator>(
        m, "MomentumPreprocessing", pydoc::doc("MomentumPreprocessing")
    )
        .def(
            py::init<std::size_t>(),
            py::arg("particle_count"),
            pydoc::doc("MomentumPreprocessing::MomentumPreprocessing")
        )
        .def(
            "output_dim",
            &MomentumPreprocessing::output_dim,
            pydoc::doc("MomentumPreprocessing::output_dim")
        );

    py::classh<ChannelWeightNetwork, FunctionGenerator>(
        m, "ChannelWeightNetwork", pydoc::doc("ChannelWeightNetwork")
    )
        .def(
            py::init<
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation,
                const std::string&,
                bool,
                DataType>(),
            py::arg("channel_count"),
            py::arg("particle_count"),
            py::arg("hidden_dim") = 32,
            py::arg("layers") = 3,
            py::arg("activation") = MLP::leaky_relu,
            py::arg("prefix") = "",
            py::arg("include_preprocessing") = true,
            py::arg("precision") = DataType::dt_float64,
            pydoc::doc("ChannelWeightNetwork::ChannelWeightNetwork")
        )
        .def("mlp", &ChannelWeightNetwork::mlp, pydoc::doc("ChannelWeightNetwork::mlp"))
        .def(
            "preprocessing",
            &ChannelWeightNetwork::preprocessing,
            pydoc::doc("ChannelWeightNetwork::preprocessing")
        )
        .def(
            "mask_name",
            &ChannelWeightNetwork::mask_name,
            pydoc::doc("ChannelWeightNetwork::mask_name")
        )
        .def(
            "initialize_globals",
            &ChannelWeightNetwork::initialize_globals,
            py::arg("context"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("ChannelWeightNetwork::initialize_globals")
        );

    py::classh<DiscreteHistogram, FunctionGenerator>(
        m, "DiscreteHistogram", pydoc::doc("DiscreteHistogram")
    )
        .def(
            py::init<std::vector<std::size_t>>(),
            py::arg("option_counts"),
            pydoc::doc("DiscreteHistogram::DiscreteHistogram")
        );

    py::classh<DiscreteSampler, Mapping>(
        m, "DiscreteSampler", pydoc::doc("DiscreteSampler")
    )
        .def(
            py::init<
                const std::vector<std::size_t>&,
                const std::string&,
                const std::vector<std::size_t>&>(),
            py::arg("option_counts"),
            py::arg("prefix") = "",
            py::arg("dims_with_prior") = std::vector<std::size_t>{},
            pydoc::doc("DiscreteSampler::DiscreteSampler")
        )
        .def(
            "option_counts",
            &DiscreteSampler::option_counts,
            pydoc::doc("DiscreteSampler::option_counts")
        )
        .def(
            "prob_names",
            &DiscreteSampler::prob_names,
            pydoc::doc("DiscreteSampler::prob_names")
        )
        .def(
            "initialize_globals",
            &DiscreteSampler::initialize_globals,
            py::arg("context"),
            pydoc::doc("DiscreteSampler::initialize_globals")
        );

    py::classh<DiscreteFlow, Mapping>(m, "DiscreteFlow", pydoc::doc("DiscreteFlow"))
        .def(
            py::init<
                const std::vector<std::size_t>&,
                const std::string&,
                const std::vector<std::size_t>&,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation,
                DataType>(),
            py::arg("option_counts"),
            py::arg("prefix") = "",
            py::arg("dims_with_prior") = std::vector<std::size_t>{},
            py::arg("condition_dim") = 0,
            py::arg("subnet_hidden_dim") = 32,
            py::arg("subnet_layers") = 3,
            py::arg("subnet_activation") = MLP::leaky_relu,
            py::arg("subnet_precision") = DataType::dt_float64,
            pydoc::doc("DiscreteFlow::DiscreteFlow")
        )
        .def(
            "option_counts",
            &DiscreteFlow::option_counts,
            pydoc::doc("DiscreteFlow::option_counts")
        )
        .def(
            "condition_dim",
            &DiscreteFlow::condition_dim,
            pydoc::doc("DiscreteFlow::condition_dim")
        )
        .def(
            "initialize_globals",
            &DiscreteFlow::initialize_globals,
            py::arg("context"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("DiscreteFlow::initialize_globals")
        );

    py::classh<MLMClustering, FunctionGenerator> mlm(m, "MLMClustering");
    add_enum<ScaleScheme>(
        mlm,
        "ScaleScheme",
        {
            {"clustering_mean", ScaleScheme::clustering_mean},
            {"madevent", ScaleScheme::madevent},
        }
    );
    add_enum<JetScaleScheme>(
        mlm,
        "JetScaleScheme",
        {
            {"emission", JetScaleScheme::emission},
            {"production", JetScaleScheme::production},
        }
    );
    add_enum<AlphasScheme>(
        mlm,
        "AlphasScheme",
        {
            {"none", AlphasScheme::none},
            {"per_vertex", AlphasScheme::per_vertex},
            {"geometric_mean", AlphasScheme::geometric_mean},
        }
    );
    add_enum<ClusteringMeasure>(
        mlm,
        "ClusteringMeasure",
        {
            {"fxfx", ClusteringMeasure::fxfx},
            {"madevent", ClusteringMeasure::madevent},
        }
    );
    add_enum<ClusteringHistory>(
        mlm,
        "ClusteringHistory",
        {
            {"all_diagrams", ClusteringHistory::all_diagrams},
            {"diagram", ClusteringHistory::diagram},
            {"madevent", ClusteringHistory::madevent},
        }
    );
    add_enum<PartonLineScheme>(
        mlm,
        "PartonLineScheme",
        {
            {"flavor", PartonLineScheme::flavor},
            {"goodjet", PartonLineScheme::goodjet},
        }
    );
    mlm.def(
           py::init<
               std::vector<Topology>,
               nested_vector3<std::size_t>,
               nested_vector2<std::size_t>,
               double,
               JetScaleScheme,
               ScaleScheme,
               std::unordered_map<int, int>,
               double,
               double,
               double,
               bool,
               std::vector<int>,
               int,
               PartonLineScheme,
               AlphasScheme,
               bool,
               ClusteringMeasure,
               ClusteringHistory,
               bool>(),
           py::arg("topologies"),
           py::arg("permutations"),
           py::arg("diagram_indices"),
           py::arg("cm_energy"),
           py::arg("jet_scale_scheme") = JetScaleScheme::production,
           py::arg("scale_scheme") = ScaleScheme::clustering_mean,
           py::arg("pdg_color_types") = std::unordered_map<int, int>{},
           py::arg("xqcut") = 0.,
           py::arg("bw_cutoff") = 15.,
           py::arg("jet_radius") = 1.0,
           py::arg("hadronic") = true,
           py::arg("external_pdg_ids") = std::vector<int>{},
           py::arg("max_jet_flavor") = 4,
           py::arg("parton_line_scheme") = PartonLineScheme::goodjet,
           py::arg("alphas_scheme") = AlphasScheme::per_vertex,
           py::arg("pdf_reweighting") = true,
           py::arg("clustering_measure") = ClusteringMeasure::fxfx,
           py::arg("clustering_history") = ClusteringHistory::all_diagrams,
           py::arg("clustering_mean_pdf_ladder") = true
    )
        .def_property_readonly("pdf_absolute_pdgs", &MLMClustering::pdf_absolute_pdgs)
        .def_property_readonly("clustering_measure", &MLMClustering::clustering_measure)
        .def_property_readonly("clustering_history", &MLMClustering::clustering_history)
        .def_property_readonly(
            "diagram_start_states", &MLMClustering::diagram_start_states
        )
        .def_property_readonly(
            "cluster_state_machine", &MLMClustering::cluster_state_machine
        )
        .def_property_readonly("external_masses", &MLMClustering::external_masses)
        .def_property_readonly("bw_masses", &MLMClustering::bw_masses)
        .def_property_readonly("bw_widths", &MLMClustering::bw_widths);

    py::classh<MLMClusteringAlongDiagram, FunctionGenerator>(
        m, "MLMClusteringAlongDiagram"
    )
        .def(py::init<const MLMClustering&>(), py::arg("clustering"));

    py::classh<MLMClusteringWithFlavors, FunctionGenerator>(
        m, "MLMClusteringWithFlavors"
    )
        .def(
            py::init<const MLMClustering&, std::vector<me_int_t>>(),
            py::arg("clustering"),
            py::arg("leg_flavors")
        );

    py::classh<VegasGridOptimizer>(
        m, "VegasGridOptimizer", pydoc::doc("VegasGridOptimizer")
    )
        .def(
            "add_data",
            [](VegasGridOptimizer& opt, py::object values, py::object counts) {
                opt.add_data(
                    dlpack_to_tensor(values, batch_float, 0),
                    dlpack_to_tensor(counts, batch_float_array(opt.input_dim()), 1)
                );
            },
            py::arg("values"),
            py::arg("counts"),
            pydoc::doc("VegasGridOptimizer::add_data")
        )
        .def(
            "optimize",
            &VegasGridOptimizer::optimize,
            pydoc::doc("VegasGridOptimizer::optimize")
        )
        .def(
            py::init<const std::vector<ContextPtr>&, const std::string&, double>(),
            py::arg("contexts"),
            py::arg("grid_name"),
            py::arg("damping"),
            pydoc::doc("VegasGridOptimizer::VegasGridOptimizer")
        );

    py::classh<DiscreteOptimizer>(
        m, "DiscreteOptimizer", pydoc::doc("DiscreteOptimizer")
    )
        .def(
            "add_data",
            [](DiscreteOptimizer& opt, std::vector<py::object> values_and_counts) {
                TensorVec input_tensors;
                for (std::size_t i = 1; auto& input : values_and_counts) {
                    input_tensors.push_back(
                        dlpack_to_tensor(input, i % 2 == 0 ? batch_int : batch_float, i)
                    );
                    ++i;
                }
                opt.add_data(input_tensors);
            },
            py::arg("values_and_counts"),
            pydoc::doc("DiscreteOptimizer::add_data")
        )
        .def(
            "optimize",
            &DiscreteOptimizer::optimize,
            pydoc::doc("DiscreteOptimizer::optimize")
        )
        .def(
            py::init<const std::vector<ContextPtr>&, const std::vector<std::string>&>(),
            py::arg("contexts"),
            py::arg("prob_names"),
            pydoc::doc("DiscreteOptimizer::DiscreteOptimizer")
        );

    py::classh<AdamOptimizer> adam(m, "AdamOptimizer", pydoc::doc("AdamOptimizer"));
    add_enum<AdamOptimizer::LRSchedule>(
        adam,
        "LRSchedule",
        {
            {"none", AdamOptimizer::none},
            {"cosine", AdamOptimizer::cosine},
        },
        "",
        pydoc::doc("AdamOptimizer::LRSchedule")
    );
    adam.def(
            py::init<
                const Function&,
                ContextPtr,
                double,
                AdamOptimizer::LRSchedule,
                std::size_t,
                double,
                double,
                double,
                double,
                double>(),
            py::arg("function"),
            py::arg("context"),
            py::arg("learning_rate"),
            py::arg("schedule") = AdamOptimizer::none,
            py::arg("step_count") = 0,
            py::arg("beta1") = 0.9,
            py::arg("beta2") = 0.999,
            py::arg("eps") = 1e-8,
            py::arg("grad_clip_threshold") = 0.0,
            py::arg("weight_decay") = 0.0,
            pydoc::doc("AdamOptimizer::AdamOptimizer")
    )
        .def(
            "step",
            [](AdamOptimizer& opt, std::vector<py::object> inputs) {
                DevicePtr device = opt.context()->device();
                TensorVec tensors;
                tensors.reserve(inputs.size());
                bool dlpack_version_cache = false;
                for (std::size_t i = 0;
                     auto [input, type] : zip(inputs, opt.input_types())) {
                    tensors.push_back(
                        dlpack_to_tensor(input, type, i, device, &dlpack_version_cache)
                    );
                    ++i;
                }
                py::gil_scoped_release release;
                return opt.step(tensors);
            },
            py::arg("inputs"),
            pydoc::doc("AdamOptimizer::step")
        )
        .def(
            "learning_rate",
            &AdamOptimizer::learning_rate,
            pydoc::doc("AdamOptimizer::learning_rate")
        )
        .def(
            "input_types",
            &AdamOptimizer::input_types,
            pydoc::doc("AdamOptimizer::input_types")
        )
        .def("context", &AdamOptimizer::context, pydoc::doc("AdamOptimizer::context"));

    py::classh<PdfGrid>(m, "PdfGrid", pydoc::doc("PdfGrid"))
        .def(
            py::init<const std::string&>(),
            py::arg("file"),
            pydoc::doc("PdfGrid::PdfGrid")
        )
        .def_readonly("x", &PdfGrid::x, pydoc::doc("PdfGrid::x"))
        .def_readonly("logx", &PdfGrid::logx, pydoc::doc("PdfGrid::logx"))
        .def_readonly("q", &PdfGrid::q, pydoc::doc("PdfGrid::q"))
        .def_readonly("logq2", &PdfGrid::logq2, pydoc::doc("PdfGrid::logq2"))
        .def_readonly("pids", &PdfGrid::pids, pydoc::doc("PdfGrid::pids"))
        .def_readonly("values", &PdfGrid::values, pydoc::doc("PdfGrid::values"))
        .def_readonly(
            "region_sizes", &PdfGrid::region_sizes, pydoc::doc("PdfGrid::region_sizes")
        )
        .def_property_readonly(
            "grid_point_count",
            &PdfGrid::grid_point_count,
            pydoc::doc("PdfGrid::grid_point_count")
        )
        .def_property_readonly(
            "q_count", &PdfGrid::q_count, pydoc::doc("PdfGrid::q_count")
        )
        .def(
            "coefficients_shape",
            &PdfGrid::coefficients_shape,
            py::arg("batch_dim") = false,
            pydoc::doc("PdfGrid::coefficients_shape")
        )
        .def(
            "logx_shape",
            &PdfGrid::logx_shape,
            py::arg("batch_dim") = false,
            pydoc::doc("PdfGrid::logx_shape")
        )
        .def(
            "logq2_shape",
            &PdfGrid::logq2_shape,
            py::arg("batch_dim") = false,
            pydoc::doc("PdfGrid::logq2_shape")
        )
        .def(
            "initialize_globals",
            &PdfGrid::initialize_globals,
            py::arg("context"),
            py::arg("prefix") = "",
            pydoc::doc("PdfGrid::initialize_globals")
        );

    py::classh<PartonDensity, FunctionGenerator>(
        m, "PartonDensity", pydoc::doc("PartonDensity")
    )
        .def(
            py::init<
                const PdfGrid&,
                const std::vector<int>&,
                bool,
                const std::string&>(),
            py::arg("grid"),
            py::arg("pids"),
            py::arg("dynamic_pid") = false,
            py::arg("prefix") = "",
            pydoc::doc("PartonDensity::PartonDensity")
        )
        .def(
            py::init<const CustomPdf&, const std::vector<int>&, int, bool>(),
            py::arg("custom_pdf"),
            py::arg("pids"),
            py::arg("beam_index"),
            py::arg("dynamic_pid") = false,
            pydoc::doc("PartonDensity::PartonDensity#2")
        );

    py::classh<AlphaSGrid>(m, "AlphaSGrid", pydoc::doc("AlphaSGrid"))
        .def(
            py::init<const std::string&>(),
            py::arg("file"),
            pydoc::doc("AlphaSGrid::AlphaSGrid")
        )
        .def_readonly("q", &AlphaSGrid::q, pydoc::doc("AlphaSGrid::q"))
        .def_readonly("logq2", &AlphaSGrid::logq2, pydoc::doc("AlphaSGrid::logq2"))
        .def_readonly("values", &AlphaSGrid::values, pydoc::doc("AlphaSGrid::values"))
        .def_readonly(
            "region_sizes",
            &AlphaSGrid::region_sizes,
            pydoc::doc("AlphaSGrid::region_sizes")
        )
        .def_property_readonly(
            "q_count", &AlphaSGrid::q_count, pydoc::doc("AlphaSGrid::q_count")
        )
        .def(
            "coefficients_shape",
            &AlphaSGrid::coefficients_shape,
            py::arg("batch_dim") = false,
            pydoc::doc("AlphaSGrid::coefficients_shape")
        )
        .def(
            "logq2_shape",
            &AlphaSGrid::logq2_shape,
            py::arg("batch_dim") = false,
            pydoc::doc("AlphaSGrid::logq2_shape")
        )
        .def(
            "initialize_globals",
            &AlphaSGrid::initialize_globals,
            py::arg("context"),
            py::arg("prefix") = "",
            pydoc::doc("AlphaSGrid::initialize_globals")
        );

    py::classh<RunningCoupling, FunctionGenerator>(
        m, "RunningCoupling", pydoc::doc("RunningCoupling")
    )
        .def(
            py::init<const AlphaSGrid&, const std::string&>(),
            py::arg("grid"),
            py::arg("prefix") = "",
            pydoc::doc("RunningCoupling::RunningCoupling")
        );

    py::classh<EnergyScale, FunctionGenerator> scale(
        m, "EnergyScale", pydoc::doc("EnergyScale")
    );
    add_enum<EnergyScale::DynamicalScaleType>(
        scale,
        "DynamicalScaleType",
        {
            {"transverse_energy", EnergyScale::transverse_energy},
            {"transverse_mass", EnergyScale::transverse_mass},
            {"half_transverse_mass", EnergyScale::half_transverse_mass},
            {"partonic_energy", EnergyScale::partonic_energy},
        }
    );
    scale
        .def(
            py::init<std::size_t>(),
            py::arg("particle_count"),
            pydoc::doc("EnergyScale::EnergyScale")
        )
        .def(
            py::init<std::size_t, EnergyScale::DynamicalScaleType>(),
            py::arg("particle_count"),
            py::arg("type"),
            pydoc::doc("EnergyScale::EnergyScale#2")
        )
        .def(
            py::init<std::size_t, double>(),
            py::arg("particle_count"),
            py::arg("fixed_scale"),
            pydoc::doc("EnergyScale::EnergyScale#3")
        )
        .def(
            py::init<
                std::size_t,
                EnergyScale::DynamicalScaleType,
                bool,
                bool,
                double,
                double,
                double,
                double,
                const std::optional<CustomFunction>&,
                double,
                double>(),
            py::arg("particle_count"),
            py::arg("dynamical_scale_type"),
            py::arg("ren_scale_fixed"),
            py::arg("fact_scale_fixed"),
            py::arg("ren_scale"),
            py::arg("fact_scale1"),
            py::arg("fact_scale2"),
            py::arg("scale_factor") = 1.,
            py::arg("custom_scale") = std::nullopt,
            py::arg("min_scale") = 0.,
            py::arg("max_scale") = 0.,
            pydoc::doc("EnergyScale::EnergyScale#4")
        )
        .def(
            py::init<const MLMClustering&, double, double>(),
            py::arg("clustering"),
            py::arg("min_scale") = 0.,
            py::arg("max_scale") = 0.,
            pydoc::doc("EnergyScale::EnergyScale#5")
        )
        .def("is_mlm", &EnergyScale::is_mlm, pydoc::doc("EnergyScale::is_mlm"));

    py::classh<DifferentialCrossSection::CachedPdf>(
        m, "CachedPdf", pydoc::doc("DifferentialCrossSection::CachedPdf")
    )
        .def(py::init<>());
    py::classh<DifferentialCrossSection::CachedScale>(
        m, "CachedScale", pydoc::doc("DifferentialCrossSection::CachedScale")
    )
        .def(py::init<>());
    py::classh<DifferentialCrossSection, FunctionGenerator>(
        m, "DifferentialCrossSection", pydoc::doc("DifferentialCrossSection")
    )
        .def(
            py::init<
                const MatrixElement&,
                double,
                const std::optional<RunningCoupling>&,
                const std::variant<
                    std::monostate,
                    EnergyScale,
                    DifferentialCrossSection::CachedScale>&,
                const nested_vector2<me_int_t>&,
                const std::variant<
                    std::monostate,
                    PdfGrid,
                    DifferentialCrossSection::CachedPdf>&,
                const std::variant<
                    std::monostate,
                    PdfGrid,
                    DifferentialCrossSection::CachedPdf>&,
                bool,
                bool,
                bool>(),
            py::arg("matrix_element"),
            py::arg("cm_energy"),
            py::arg("running_coupling"),
            py::arg("energy_scale"),
            py::arg("pid_options") = nested_vector2<me_int_t>{},
            py::arg("pdf1") = std::monostate{},
            py::arg("pdf2") = std::monostate{},
            py::arg("input_momentum_fraction") = true,
            py::arg("decay") = false,
            py::arg("matrix_element_output") = false,
            pydoc::doc("DifferentialCrossSection::DifferentialCrossSection")
        )
        .def(
            "pid_options",
            &DifferentialCrossSection::pid_options,
            pydoc::doc("DifferentialCrossSection::pid_options")
        )
        .def(
            "has_pdf",
            &DifferentialCrossSection::has_pdf,
            py::arg("pdf_index"),
            pydoc::doc("DifferentialCrossSection::has_pdf")
        )
        .def(
            "matrix_element",
            &DifferentialCrossSection::matrix_element,
            pydoc::doc("DifferentialCrossSection::matrix_element")
        )
        .def(
            "running_coupling",
            &DifferentialCrossSection::running_coupling,
            pydoc::doc("DifferentialCrossSection::running_coupling")
        )
        .def(
            "matrix_element_output",
            &DifferentialCrossSection::matrix_element_output,
            pydoc::doc("DifferentialCrossSection::matrix_element_output")
        );

    py::classh<Unweighter, FunctionGenerator>(m, "Unweighter", pydoc::doc("Unweighter"))
        .def(
            py::init<const NamedVector<Type>&>(),
            py::arg("types"),
            pydoc::doc("Unweighter::Unweighter")
        );
    py::classh<BufferUnweighter, FunctionGenerator>(
        m, "BufferUnweighter", pydoc::doc("BufferUnweighter")
    )
        .def(
            py::init<const NamedVector<Type>&, double>(),
            py::arg("types"),
            py::arg("quantile") = 0.0,
            pydoc::doc("BufferUnweighter::BufferUnweighter")
        );
    py::classh<Integrand, FunctionGenerator>(m, "Integrand", pydoc::doc("Integrand"))
        .def(
            py::init<
                const PhaseSpaceMapping&,
                const std::vector<DifferentialCrossSection>&,
                const Integrand::AdaptiveMapping&,
                const Integrand::AdaptiveDiscrete&,
                const Integrand::AdaptiveDiscrete&,
                const nested_vector2<me_int_t>&,
                const Integrand::PdfSource&,
                const std::optional<RunningCoupling>&,
                const std::optional<EnergyScale>&,
                const std::optional<PropagatorChannelWeights>&,
                const std::optional<SubchannelWeights>&,
                const std::optional<ChannelWeightNetwork>&,
                const nested_vector2<me_int_t>&,
                std::size_t,
                const std::vector<me_int_t>&,
                std::size_t,
                bool,
                bool,
                bool,
                const std::vector<std::size_t>&,
                const nested_vector2<std::size_t>&,
                const std::vector<std::size_t>&,
                const std::vector<double>&,
                const std::vector<bool>&,
                const std::vector<std::size_t>&,
                const std::vector<std::size_t>&,
                const std::vector<std::size_t>&,
                std::size_t,
                const std::optional<CustomFunction>&,
                const std::optional<PdfGrid>&,
                int>(),
            py::arg("mapping"),
            py::arg("diff_xs"),
            py::arg("adaptive_map") = std::monostate{},
            py::arg("discrete_sym") = std::monostate{},
            py::arg("discrete_flavor") = std::monostate{},
            py::arg("pid_options") = nested_vector2<me_int_t>{},
            py::arg("pdf_grid") = std::monostate{},
            py::arg("running_coupling") = std::nullopt,
            py::arg("energy_scale") = std::nullopt,
            py::arg("prop_chan_weights") = std::nullopt,
            py::arg("subchan_weights") = std::nullopt,
            py::arg("chan_weight_net") = std::nullopt,
            py::arg("first_chan_weight_remap") = nested_vector2<me_int_t>{},
            py::arg("first_remapped_chan_count") = 0,
            py::arg("second_chan_weight_remap") = std::vector<me_int_t>{},
            py::arg("second_remapped_chan_count") = 0,
            py::arg("madnis_training") = false,
            py::arg("drop_cuts_and_rescale") = false,
            py::arg("partial_weights") = false,
            py::arg("channel_indices") = std::vector<std::size_t>{},
            py::arg("active_flavors") = nested_vector2<std::size_t>{},
            py::arg("flavor_remap") = std::vector<std::size_t>{},
            py::arg("flavor_factors") = std::vector<double>{},
            py::arg("flavor_mirror") = std::vector<bool>{},
            py::arg("flavor_diff_xs_indices") = std::vector<std::size_t>{},
            py::arg("flavor_subproc_indices") = std::vector<std::size_t>{},
            py::arg("flavor_per_subproc_remap") = std::vector<std::size_t>{},
            py::arg("compressed_channel_weight_count") = 50,
            py::arg("custom_bias") = std::nullopt,
            py::arg("pdf_grid2") = std::nullopt,
            py::arg("mlm_history_qcd_power") = -1,
            pydoc::doc("Integrand::Integrand")
        )
        .def(
            "particle_count",
            &Integrand::particle_count,
            pydoc::doc("Integrand::particle_count")
        )
        .def(
            "madnis_training",
            &Integrand::madnis_training,
            pydoc::doc("Integrand::madnis_training")
        )
        .def(
            "vegas_grid_name",
            &Integrand::vegas_grid_name,
            pydoc::doc("Integrand::vegas_grid_name")
        )
        .def(
            "vegas_dimension",
            &Integrand::vegas_dimension,
            pydoc::doc("Integrand::vegas_dimension")
        )
        .def(
            "vegas_bin_count",
            &Integrand::vegas_bin_count,
            pydoc::doc("Integrand::vegas_bin_count")
        )
        .def("mapping", &Integrand::mapping, pydoc::doc("Integrand::mapping"))
        .def("diff_xs", &Integrand::diff_xs, pydoc::doc("Integrand::diff_xs"))
        .def(
            "adaptive_map",
            &Integrand::adaptive_map,
            pydoc::doc("Integrand::adaptive_map")
        )
        .def(
            "discrete_sym",
            &Integrand::discrete_sym,
            pydoc::doc("Integrand::discrete_sym")
        )
        .def(
            "discrete_flavor",
            &Integrand::discrete_flavor,
            pydoc::doc("Integrand::discrete_flavor")
        )
        .def(
            "energy_scale",
            &Integrand::energy_scale,
            pydoc::doc("Integrand::energy_scale")
        )
        .def(
            "prop_chan_weights",
            &Integrand::prop_chan_weights,
            pydoc::doc("Integrand::prop_chan_weights")
        )
        .def(
            "chan_weight_net",
            &Integrand::chan_weight_net,
            pydoc::doc("Integrand::chan_weight_net")
        )
        .def("random_dim", &Integrand::random_dim, pydoc::doc("Integrand::random_dim"))
        .def(
            "latent_dims", &Integrand::latent_dims, pydoc::doc("Integrand::latent_dims")
        )
        .def(
            "channel_indices",
            &Integrand::channel_indices,
            pydoc::doc("Integrand::channel_indices")
        )
        .def(
            "active_flavors",
            &Integrand::active_flavors,
            pydoc::doc("Integrand::active_flavors")
        )
        .def_readonly_static(
            "matrix_element_inputs",
            &Integrand::matrix_element_inputs,
            pydoc::doc("Integrand::matrix_element_inputs")
        )
        .def_readonly_static(
            "matrix_element_outputs",
            &Integrand::matrix_element_outputs,
            pydoc::doc("Integrand::matrix_element_outputs")
        );
    py::classh<MultiChannelIntegrand, FunctionGenerator>(
        m, "MultiChannelIntegrand", pydoc::doc("MultiChannelIntegrand")
    )
        .def(
            py::init<const std::vector<std::shared_ptr<Integrand>>&, bool>(),
            py::arg("integrands"),
            py::arg("return_sizes") = false,
            pydoc::doc("MultiChannelIntegrand::MultiChannelIntegrand")
        );
    py::classh<IntegrandProbability, FunctionGenerator>(
        m, "IntegrandProbability", pydoc::doc("IntegrandProbability")
    )
        .def(
            py::init<const Integrand&>(),
            py::arg("integrand"),
            pydoc::doc("IntegrandProbability::IntegrandProbability")
        );

    py::classh<MadnisLoss, FunctionGenerator>(m, "MadnisLoss", pydoc::doc("MadnisLoss"))
        .def(
            py::init<
                const std::vector<std::shared_ptr<FunctionGenerator>>&,
                const std::optional<ChannelWeightNetwork>&,
                double,
                std::size_t>(),
            py::arg("functions"),
            py::arg("cwnet"),
            py::arg("softclip_threshold") = 0.0,
            py::arg("compressed_channel_weight_count") = 50,
            pydoc::doc("MadnisLoss::MadnisLoss")
        );

    add_enum<Verbosity>(
        m,
        "Verbosity",
        {
            {"silent", Verbosity::silent},
            {"log", Verbosity::log},
            {"pretty", Verbosity::pretty},
        },
        "",
        pydoc::doc("Verbosity")
    );

    py::classh<MadnisTraining::Config>(
        m, "MadnisConfig", pydoc::doc("MadnisTraining::Config")
    )
        .def(py::init<>())
        .def_readwrite(
            "learning_rate",
            &MadnisTraining::Config::learning_rate,
            pydoc::doc("MadnisTraining::Config::learning_rate")
        )
        .def_readwrite(
            "batches",
            &MadnisTraining::Config::batches,
            pydoc::doc("MadnisTraining::Config::batches")
        )
        .def_readwrite(
            "log_interval",
            &MadnisTraining::Config::log_interval,
            pydoc::doc("MadnisTraining::Config::log_interval")
        )
        .def_readwrite(
            "integration_history_length",
            &MadnisTraining::Config::integration_history_length,
            pydoc::doc("MadnisTraining::Config::integration_history_length")
        )
        .def_readwrite(
            "channel_dropping_interval",
            &MadnisTraining::Config::channel_dropping_interval,
            pydoc::doc("MadnisTraining::Config::channel_dropping_interval")
        )
        .def_readwrite(
            "channel_dropping_threshold",
            &MadnisTraining::Config::channel_dropping_threshold,
            pydoc::doc("MadnisTraining::Config::channel_dropping_threshold")
        )
        .def_readwrite(
            "cpu_generator_batch_size",
            &MadnisTraining::Config::cpu_generator_batch_size,
            pydoc::doc("MadnisTraining::Config::cpu_generator_batch_size")
        )
        .def_readwrite(
            "gpu_generator_batch_size",
            &MadnisTraining::Config::gpu_generator_batch_size,
            pydoc::doc("MadnisTraining::Config::gpu_generator_batch_size")
        )
        .def_readwrite(
            "gpu_generator_batch_granularity",
            &MadnisTraining::Config::gpu_generator_batch_granularity,
            pydoc::doc("MadnisTraining::Config::gpu_generator_batch_granularity")
        )
        .def_readwrite(
            "generator_target_size_factor",
            &MadnisTraining::Config::generator_target_size_factor,
            pydoc::doc("MadnisTraining::Config::generator_target_size_factor")
        )
        .def_readwrite(
            "batch_size_offset",
            &MadnisTraining::Config::batch_size_offset,
            pydoc::doc("MadnisTraining::Config::batch_size_offset")
        )
        .def_readwrite(
            "batch_size_per_channel",
            &MadnisTraining::Config::batch_size_per_channel,
            pydoc::doc("MadnisTraining::Config::batch_size_per_channel")
        )
        .def_readwrite(
            "uniform_channel_ratio",
            &MadnisTraining::Config::uniform_channel_ratio,
            pydoc::doc("MadnisTraining::Config::uniform_channel_ratio")
        )
        .def_readwrite(
            "lr_schedule",
            &MadnisTraining::Config::lr_schedule,
            pydoc::doc("MadnisTraining::Config::lr_schedule")
        )
        .def_readwrite(
            "adam_beta1",
            &MadnisTraining::Config::adam_beta1,
            pydoc::doc("MadnisTraining::Config::adam_beta1")
        )
        .def_readwrite(
            "adam_beta2",
            &MadnisTraining::Config::adam_beta2,
            pydoc::doc("MadnisTraining::Config::adam_beta2")
        )
        .def_readwrite(
            "adam_eps",
            &MadnisTraining::Config::adam_eps,
            pydoc::doc("MadnisTraining::Config::adam_eps")
        )
        .def_readwrite(
            "adam_weight_decay",
            &MadnisTraining::Config::adam_weight_decay,
            pydoc::doc("MadnisTraining::Config::adam_weight_decay")
        )
        .def_readwrite(
            "grad_clip_threshold",
            &MadnisTraining::Config::grad_clip_threshold,
            pydoc::doc("MadnisTraining::Config::grad_clip_threshold")
        )
        .def_readwrite(
            "buffer_capacity",
            &MadnisTraining::Config::buffer_capacity,
            pydoc::doc("MadnisTraining::Config::buffer_capacity")
        )
        .def_readwrite(
            "minimum_buffer_size",
            &MadnisTraining::Config::minimum_buffer_size,
            pydoc::doc("MadnisTraining::Config::minimum_buffer_size")
        )
        .def_readwrite(
            "buffered_steps_fraction",
            &MadnisTraining::Config::buffered_steps_fraction,
            pydoc::doc("MadnisTraining::Config::buffered_steps_fraction")
        )
        .def_readwrite(
            "buffer_skip_batches",
            &MadnisTraining::Config::buffer_skip_batches,
            pydoc::doc("MadnisTraining::Config::buffer_skip_batches")
        )
        .def_readwrite(
            "buffer_unweighting_quantile",
            &MadnisTraining::Config::buffer_unweighting_quantile,
            pydoc::doc("MadnisTraining::Config::buffer_unweighting_quantile")
        )
        .def_readwrite(
            "fixed_cwnet_fraction",
            &MadnisTraining::Config::fixed_cwnet_fraction,
            pydoc::doc("MadnisTraining::Config::fixed_cwnet_fraction")
        )
        .def_readwrite(
            "softclip_threshold",
            &MadnisTraining::Config::softclip_threshold,
            pydoc::doc("MadnisTraining::Config::softclip_threshold")
        )
        .def_readwrite(
            "compressed_channel_weight_count",
            &MadnisTraining::Config::compressed_channel_weight_count,
            pydoc::doc("MadnisTraining::Config::compressed_channel_weight_count")
        );

    py::classh<MadnisTraining>(m, "MadnisTraining", pydoc::doc("MadnisTraining"))
        .def(
            py::init<
                ContextPtr,
                ContextPtr,
                const MadnisTraining::Config&,
                const std::vector<std::shared_ptr<Integrand>>&,
                const std::optional<ChannelWeightNetwork>&,
                std::optional<std::uint64_t>>(),
            py::arg("generator_context"),
            py::arg("optimizer_context"),
            py::arg("config"),
            py::arg("integrands"),
            py::arg("cwnet"),
            py::arg("seed") = std::nullopt,
            pydoc::doc("MadnisTraining::MadnisTraining")
        )
        .def(
            "train_step",
            &MadnisTraining::train_step,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("batch_index"),
            pydoc::doc("MadnisTraining::train_step")
        )
        .def(
            "active_channels",
            &MadnisTraining::active_channels,
            pydoc::doc("MadnisTraining::active_channels")
        )
        .def(
            "active_channel_count",
            &MadnisTraining::active_channel_count,
            pydoc::doc("MadnisTraining::active_channel_count")
        );

    py::classh<StatusFile>(m, "StatusFile", pydoc::doc("StatusFile"))
        .def(
            py::init<const std::string&, double>(),
            py::arg("file_name"),
            py::arg("min_interval_sec") = 10.0,
            pydoc::doc("StatusFile::StatusFile")
        );

    py::classh<MultiMadnisTraining::TrainingArgs>(
        m, "TrainingArgs", pydoc::doc("MultiMadnisTraining::TrainingArgs")
    )
        .def(
            py::init<
                const MadnisTraining::Config&,
                const std::vector<std::shared_ptr<Integrand>>&,
                const std::optional<ChannelWeightNetwork>&>(),
            py::arg("config"),
            py::arg("integrands"),
            py::arg("cwnet")
        );

    py::classh<MultiMadnisTraining>(
        m, "MultiMadnisTraining", pydoc::doc("MultiMadnisTraining")
    )
        .def(
            py::init<
                ContextPtr,
                ContextPtr,
                const std::vector<MultiMadnisTraining::TrainingArgs>&,
                Verbosity,
                std::shared_ptr<StatusFile>,
                std::optional<std::uint64_t>>(),
            py::arg("generator_context"),
            py::arg("optimizer_context"),
            py::arg("training_args"),
            py::arg("verbosity"),
            py::arg("status_file") = std::shared_ptr<StatusFile>(),
            py::arg("seed") = std::nullopt,
            pydoc::doc("MultiMadnisTraining::MultiMadnisTraining")
        )
        .def(
            "train",
            &MultiMadnisTraining::train,
            py::call_guard<py::gil_scoped_release>(),
            pydoc::doc("MultiMadnisTraining::train")
        )
        .def(
            "active_channels",
            &MultiMadnisTraining::active_channels,
            pydoc::doc("MultiMadnisTraining::active_channels")
        );

    py::classh<GeneratorConfig>(m, "GeneratorConfig", pydoc::doc("GeneratorConfig"))
        .def(py::init<>())
        .def_readwrite(
            "target_count",
            &GeneratorConfig::target_count,
            pydoc::doc("GeneratorConfig::target_count")
        )
        .def_readwrite(
            "vegas_damping",
            &GeneratorConfig::vegas_damping,
            pydoc::doc("GeneratorConfig::vegas_damping")
        )
        .def_readwrite(
            "max_overweight_truncation",
            &GeneratorConfig::max_overweight_truncation,
            pydoc::doc("GeneratorConfig::max_overweight_truncation")
        )
        .def_readwrite(
            "freeze_max_weight_after",
            &GeneratorConfig::freeze_max_weight_after,
            pydoc::doc("GeneratorConfig::freeze_max_weight_after")
        )
        .def_readwrite(
            "start_batch_size",
            &GeneratorConfig::start_batch_size,
            pydoc::doc("GeneratorConfig::start_batch_size")
        )
        .def_readwrite(
            "max_batch_size",
            &GeneratorConfig::max_batch_size,
            pydoc::doc("GeneratorConfig::max_batch_size")
        )
        .def_readwrite(
            "survey_min_iters",
            &GeneratorConfig::survey_min_iters,
            pydoc::doc("GeneratorConfig::survey_min_iters")
        )
        .def_readwrite(
            "survey_max_iters",
            &GeneratorConfig::survey_max_iters,
            pydoc::doc("GeneratorConfig::survey_max_iters")
        )
        .def_readwrite(
            "survey_target_precision",
            &GeneratorConfig::survey_target_precision,
            pydoc::doc("GeneratorConfig::survey_target_precision")
        )
        .def_readwrite(
            "optimization_patience",
            &GeneratorConfig::optimization_patience,
            pydoc::doc("GeneratorConfig::optimization_patience")
        )
        .def_readwrite(
            "optimization_threshold",
            &GeneratorConfig::optimization_threshold,
            pydoc::doc("GeneratorConfig::optimization_threshold")
        )
        .def_readwrite(
            "cpu_batch_size",
            &GeneratorConfig::cpu_batch_size,
            pydoc::doc("GeneratorConfig::cpu_batch_size")
        )
        .def_readwrite(
            "gpu_batch_size",
            &GeneratorConfig::gpu_batch_size,
            pydoc::doc("GeneratorConfig::gpu_batch_size")
        )
        .def_readwrite(
            "verbosity",
            &GeneratorConfig::verbosity,
            pydoc::doc("GeneratorConfig::verbosity")
        )
        .def_readwrite(
            "write_live_data",
            &GeneratorConfig::write_live_data,
            pydoc::doc("GeneratorConfig::write_live_data")
        )
        .def_readwrite(
            "combine_thread_count",
            &GeneratorConfig::combine_thread_count,
            pydoc::doc("GeneratorConfig::combine_thread_count")
        )
        .def_readwrite(
            "cut_efficiency_threshold",
            &GeneratorConfig::cut_efficiency_threshold,
            pydoc::doc("GeneratorConfig::cut_efficiency_threshold")
        )
        .def_readwrite(
            "max_cut_repetitions",
            &GeneratorConfig::max_cut_repetitions,
            pydoc::doc("GeneratorConfig::max_cut_repetitions")
        )
        .def_readwrite(
            "finish_remaining_fraction",
            &GeneratorConfig::finish_remaining_fraction,
            pydoc::doc("GeneratorConfig::finish_remaining_fraction")
        )
        .def_readwrite(
            "max_batch_fraction",
            &GeneratorConfig::max_batch_fraction,
            pydoc::doc("GeneratorConfig::max_batch_fraction")
        )
        .def_readwrite(
            "batch_overshoot_sigma",
            &GeneratorConfig::batch_overshoot_sigma,
            pydoc::doc("GeneratorConfig::batch_overshoot_sigma")
        );

    m.def(
        "compute_generation_batch_event_count",
        &compute_generation_batch_event_count,
        py::arg("count_target"),
        py::arg("count_unweighted"),
        py::arg("count_requested"),
        py::arg("abs_cross_section_count"),
        py::arg("abs_cross_section_rel_error"),
        py::arg("config"),
        pydoc::doc("compute_generation_batch_event_count")
    );

    m.def(
        "select_combine_channel_index",
        &select_combine_channel_index,
        py::arg("cum_counts"),
        py::arg("random_index"),
        pydoc::doc("select_combine_channel_index")
    );

    py::classh<GeneratorStatus>(m, "GeneratorStatus", pydoc::doc("GeneratorStatus"))
        .def(py::init<>())
        .def_readwrite(
            "subprocess",
            &GeneratorStatus::subprocess,
            pydoc::doc("GeneratorStatus::subprocess")
        )
        .def_readwrite(
            "name", &GeneratorStatus::name, pydoc::doc("GeneratorStatus::name")
        )
        .def_readwrite(
            "mean", &GeneratorStatus::mean, pydoc::doc("GeneratorStatus::mean")
        )
        .def_readwrite(
            "error", &GeneratorStatus::error, pydoc::doc("GeneratorStatus::error")
        )
        .def_readwrite(
            "mean_abs",
            &GeneratorStatus::mean_abs,
            pydoc::doc("GeneratorStatus::mean_abs")
        )
        .def_readwrite(
            "error_abs",
            &GeneratorStatus::error_abs,
            pydoc::doc("GeneratorStatus::error_abs")
        )
        .def_readwrite(
            "rel_std_dev",
            &GeneratorStatus::rel_std_dev,
            pydoc::doc("GeneratorStatus::rel_std_dev")
        )
        .def_readwrite(
            "count", &GeneratorStatus::count, pydoc::doc("GeneratorStatus::count")
        )
        .def_readwrite(
            "count_opt",
            &GeneratorStatus::count_opt,
            pydoc::doc("GeneratorStatus::count_opt")
        )
        .def_readwrite(
            "count_after_cuts",
            &GeneratorStatus::count_after_cuts,
            pydoc::doc("GeneratorStatus::count_after_cuts")
        )
        .def_readwrite(
            "count_after_cuts_opt",
            &GeneratorStatus::count_after_cuts_opt,
            pydoc::doc("GeneratorStatus::count_after_cuts_opt")
        )
        .def_readwrite(
            "count_unweighted",
            &GeneratorStatus::count_unweighted,
            pydoc::doc("GeneratorStatus::count_unweighted")
        )
        .def_readwrite(
            "count_target",
            &GeneratorStatus::count_target,
            pydoc::doc("GeneratorStatus::count_target")
        )
        .def_readwrite(
            "iterations",
            &GeneratorStatus::iterations,
            pydoc::doc("GeneratorStatus::iterations")
        )
        .def_readwrite(
            "iters_without_improvement",
            &GeneratorStatus::iters_without_improvement,
            pydoc::doc("GeneratorStatus::iters_without_improvement")
        )
        .def_readwrite(
            "optimized",
            &GeneratorStatus::optimized,
            pydoc::doc("GeneratorStatus::optimized")
        )
        .def_readwrite(
            "done", &GeneratorStatus::done, pydoc::doc("GeneratorStatus::done")
        );

    py::classh<Histogram>(m, "Histogram", pydoc::doc("Histogram"))
        .def_readonly("name", &Histogram::name, pydoc::doc("Histogram::name"))
        .def_readonly("min", &Histogram::min, pydoc::doc("Histogram::min"))
        .def_readonly("max", &Histogram::max, pydoc::doc("Histogram::max"))
        .def_readonly(
            "bin_values", &Histogram::bin_values, pydoc::doc("Histogram::bin_values")
        )
        .def_readonly(
            "bin_errors", &Histogram::bin_errors, pydoc::doc("Histogram::bin_errors")
        );

    py::classh<LHEHeader>(m, "LHEHeader", pydoc::doc("LHEHeader"))
        .def(
            py::init<std::string, std::string, bool>(),
            py::arg("name") = "",
            py::arg("content") = "",
            py::arg("escape_content") = false
        )
        .def_readwrite("name", &LHEHeader::name, pydoc::doc("LHEHeader::name"))
        .def_readwrite("content", &LHEHeader::content, pydoc::doc("LHEHeader::content"))
        .def_readwrite(
            "escape_content",
            &LHEHeader::escape_content,
            pydoc::doc("LHEHeader::escape_content")
        );
    py::classh<LHEProcess>(m, "LHEProcess", pydoc::doc("LHEProcess"))
        .def(
            py::init<double, double, double, int>(),
            py::arg("cross_section") = 0.,
            py::arg("cross_section_error") = 0.,
            py::arg("max_weight") = 0.,
            py::arg("process_id") = 0
        )
        .def_readwrite(
            "cross_section",
            &LHEProcess::cross_section,
            pydoc::doc("LHEProcess::cross_section")
        )
        .def_readwrite(
            "cross_section_error",
            &LHEProcess::cross_section_error,
            pydoc::doc("LHEProcess::cross_section_error")
        )
        .def_readwrite(
            "max_weight", &LHEProcess::max_weight, pydoc::doc("LHEProcess::max_weight")
        )
        .def_readwrite(
            "process_id", &LHEProcess::process_id, pydoc::doc("LHEProcess::process_id")
        );
    py::classh<LHEMeta>(m, "LHEMeta", pydoc::doc("LHEMeta"))
        .def(
            py::init<
                int,
                int,
                double,
                double,
                int,
                int,
                int,
                int,
                int,
                std::vector<LHEProcess>,
                std::vector<LHEHeader>>(),
            py::arg("beam1_pdg_id") = 0,
            py::arg("beam2_pdg_id") = 0,
            py::arg("beam1_energy") = 0.,
            py::arg("beam2_energy") = 0.,
            py::arg("beam1_pdf_authors") = 0,
            py::arg("beam2_pdf_authors") = 0,
            py::arg("beam1_pdf_id") = 0,
            py::arg("beam2_pdf_id") = 0,
            py::arg("weight_mode") = 0,
            py::arg("processes") = std::vector<LHEProcess>{},
            py::arg("headers") = std::vector<LHEHeader>{}
        )
        .def_readwrite(
            "beam1_pdg_id", &LHEMeta::beam1_pdg_id, pydoc::doc("LHEMeta::beam1_pdg_id")
        )
        .def_readwrite(
            "beam2_pdg_id", &LHEMeta::beam2_pdg_id, pydoc::doc("LHEMeta::beam2_pdg_id")
        )
        .def_readwrite(
            "beam1_energy", &LHEMeta::beam1_energy, pydoc::doc("LHEMeta::beam1_energy")
        )
        .def_readwrite(
            "beam2_energy", &LHEMeta::beam2_energy, pydoc::doc("LHEMeta::beam2_energy")
        )
        .def_readwrite(
            "beam1_pdf_authors",
            &LHEMeta::beam1_pdf_authors,
            pydoc::doc("LHEMeta::beam1_pdf_authors")
        )
        .def_readwrite(
            "beam2_pdf_authors",
            &LHEMeta::beam2_pdf_authors,
            pydoc::doc("LHEMeta::beam2_pdf_authors")
        )
        .def_readwrite(
            "beam1_pdf_id", &LHEMeta::beam1_pdf_id, pydoc::doc("LHEMeta::beam1_pdf_id")
        )
        .def_readwrite(
            "beam2_pdf_id", &LHEMeta::beam2_pdf_id, pydoc::doc("LHEMeta::beam2_pdf_id")
        )
        .def_readwrite(
            "weight_mode", &LHEMeta::weight_mode, pydoc::doc("LHEMeta::weight_mode")
        )
        .def_readwrite(
            "processes", &LHEMeta::processes, pydoc::doc("LHEMeta::processes")
        )
        .def_readwrite("headers", &LHEMeta::headers, pydoc::doc("LHEMeta::headers"));
    py::classh<LHEParticle>(m, "LHEParticle", pydoc::doc("LHEParticle"))
        .def(
            py::init<
                int,
                int,
                int,
                int,
                int,
                int,
                double,
                double,
                double,
                double,
                double,
                double,
                double>(),
            py::arg("pdg_id") = 0,
            py::arg("status_code") = 0,
            py::arg("mother1") = 0,
            py::arg("mother2") = 0,
            py::arg("color") = 0,
            py::arg("anti_color") = 0,
            py::arg("px") = 0.,
            py::arg("py") = 0.,
            py::arg("pz") = 0.,
            py::arg("energy") = 0.,
            py::arg("mass") = 0.,
            py::arg("lifetime") = 0.,
            py::arg("spin") = 0.
        )
        .def_readonly_static("status_incoming", &LHEParticle::status_incoming)
        .def_readonly_static("status_outgoing", &LHEParticle::status_outgoing)
        .def_readonly_static(
            "status_intermediate_resonance", &LHEParticle::status_intermediate_resonance
        )
        .def_readwrite(
            "pdg_id", &LHEParticle::pdg_id, pydoc::doc("LHEParticle::pdg_id")
        )
        .def_readwrite(
            "status_code",
            &LHEParticle::status_code,
            pydoc::doc("LHEParticle::status_code")
        )
        .def_readwrite(
            "mother1", &LHEParticle::mother1, pydoc::doc("LHEParticle::mother1")
        )
        .def_readwrite(
            "mother2", &LHEParticle::mother2, pydoc::doc("LHEParticle::mother2")
        )
        .def_readwrite("color", &LHEParticle::color, pydoc::doc("LHEParticle::color"))
        .def_readwrite(
            "anti_color",
            &LHEParticle::anti_color,
            pydoc::doc("LHEParticle::anti_color")
        )
        .def_readwrite("px", &LHEParticle::px, pydoc::doc("LHEParticle::px"))
        .def_readwrite("py", &LHEParticle::py, pydoc::doc("LHEParticle::py"))
        .def_readwrite("pz", &LHEParticle::pz, pydoc::doc("LHEParticle::pz"))
        .def_readwrite(
            "energy", &LHEParticle::energy, pydoc::doc("LHEParticle::energy")
        )
        .def_readwrite("mass", &LHEParticle::mass, pydoc::doc("LHEParticle::mass"))
        .def_readwrite(
            "lifetime", &LHEParticle::lifetime, pydoc::doc("LHEParticle::lifetime")
        )
        .def_readwrite("spin", &LHEParticle::spin, pydoc::doc("LHEParticle::spin"));
    py::classh<LHEEvent>(m, "LHEEvent", pydoc::doc("LHEEvent"))
        .def(
            py::init<int, double, double, double, double, std::vector<LHEParticle>>(),
            py::arg("process_id") = 0,
            py::arg("weight") = 0.,
            py::arg("scale") = 0.,
            py::arg("alpha_qed") = 0.,
            py::arg("alpha_qcd") = 0.,
            py::arg("particles") = std::vector<LHEParticle>{}
        )
        .def_readwrite(
            "process_id", &LHEEvent::process_id, pydoc::doc("LHEEvent::process_id")
        )
        .def_readwrite("weight", &LHEEvent::weight, pydoc::doc("LHEEvent::weight"))
        .def_readwrite("scale", &LHEEvent::scale, pydoc::doc("LHEEvent::scale"))
        .def_readwrite(
            "alpha_qed", &LHEEvent::alpha_qed, pydoc::doc("LHEEvent::alpha_qed")
        )
        .def_readwrite(
            "alpha_qcd", &LHEEvent::alpha_qcd, pydoc::doc("LHEEvent::alpha_qcd")
        )
        .def_readwrite(
            "particles", &LHEEvent::particles, pydoc::doc("LHEEvent::particles")
        )
        .def_readwrite(
            "rwgt_ids", &LHEEvent::rwgt_ids, pydoc::doc("LHEEvent::rwgt_ids")
        )
        .def_readwrite("rwgt", &LHEEvent::rwgt, pydoc::doc("LHEEvent::rwgt"))
        .def(
            "format",
            [](const LHEEvent& event) {
                std::string buffer;
                event.format_to(buffer);
                return buffer;
            },
            pydoc::doc("LHEEvent::format_to")
        );
    py::classh<LHECompleter::SubprocArgs>(
        m, "SubprocArgs", pydoc::doc("LHECompleter::SubprocArgs")
    )
        .def(
            py::init<
                int,
                std::vector<Topology>,
                nested_vector3<std::size_t>,
                nested_vector2<std::size_t>,
                nested_vector3<std::size_t>,
                nested_vector2<std::tuple<int, int>>,
                std::unordered_map<int, int>,
                nested_vector2<double>,
                nested_vector3<int>,
                nested_vector3<int>>(),
            py::arg("process_id") = 0,
            py::arg("topologies") = std::vector<Topology>{},
            py::arg("permutations") = nested_vector3<std::size_t>{},
            py::arg("diagram_indices") = nested_vector2<std::size_t>{},
            py::arg("diagram_color_indices") = nested_vector3<std::size_t>{},
            py::arg("color_flows") = nested_vector2<std::tuple<int, int>>{},
            py::arg("pdg_color_types") = std::unordered_map<int, int>{},
            py::arg("helicities") = nested_vector2<double>{},
            py::arg("pdg_ids") = nested_vector3<int>{},
            py::arg("diagram_propagator_pdgs") = nested_vector3<int>{}
        )
        .def_readwrite(
            "process_id",
            &LHECompleter::SubprocArgs::process_id,
            pydoc::doc("LHECompleter::SubprocArgs::process_id")
        )
        .def_readwrite(
            "topologies",
            &LHECompleter::SubprocArgs::topologies,
            pydoc::doc("LHECompleter::SubprocArgs::topologies")
        )
        .def_readwrite(
            "permutations",
            &LHECompleter::SubprocArgs::permutations,
            pydoc::doc("LHECompleter::SubprocArgs::permutations")
        )
        .def_readwrite(
            "diagram_indices",
            &LHECompleter::SubprocArgs::diagram_indices,
            pydoc::doc("LHECompleter::SubprocArgs::diagram_indices")
        )
        .def_readwrite(
            "diagram_color_indices",
            &LHECompleter::SubprocArgs::diagram_color_indices,
            pydoc::doc("LHECompleter::SubprocArgs::diagram_color_indices")
        )
        .def_readwrite(
            "color_flows",
            &LHECompleter::SubprocArgs::color_flows,
            pydoc::doc("LHECompleter::SubprocArgs::color_flows")
        )
        .def_readwrite(
            "pdg_color_types",
            &LHECompleter::SubprocArgs::pdg_color_types,
            pydoc::doc("LHECompleter::SubprocArgs::pdg_color_types")
        )
        .def_readwrite(
            "helicities",
            &LHECompleter::SubprocArgs::helicities,
            pydoc::doc("LHECompleter::SubprocArgs::helicities")
        )
        .def_readwrite(
            "pdg_ids",
            &LHECompleter::SubprocArgs::pdg_ids,
            pydoc::doc("LHECompleter::SubprocArgs::pdg_ids")
        )
        .def_readwrite(
            "diagram_propagator_pdgs",
            &LHECompleter::SubprocArgs::diagram_propagator_pdgs,
            pydoc::doc("LHECompleter::SubprocArgs::diagram_propagator_pdgs")
        );
    py::classh<MixMaxRandom>(m, "MixMaxRandom", pydoc::doc("MixMaxRandom"))
        .def(py::init<>(), pydoc::doc("MixMaxRandom::MixMaxRandom"))
        .def(
            py::init<std::uint64_t>(),
            py::arg("seed"),
            pydoc::doc("MixMaxRandom::MixMaxRandom#3")
        );
    py::classh<LHECompleter>(m, "LHECompleter", pydoc::doc("LHECompleter"))
        .def(
            py::init<const std::vector<LHECompleter::SubprocArgs>&, double>(),
            py::arg("subproc_args"),
            py::arg("bw_cutoff"),
            pydoc::doc("LHECompleter::LHECompleter")
        )
        .def(
            "complete_event_data",
            &LHECompleter::complete_event_data,
            py::arg("event"),
            py::arg("subprocess_index"),
            py::arg("diagram_index"),
            py::arg("color_index"),
            py::arg("flavor_index"),
            py::arg("helicity_index"),
            py::arg("rand_gen"),
            pydoc::doc("LHECompleter::complete_event_data")
        )
        .def(
            "save",
            &LHECompleter::save,
            py::arg("file"),
            pydoc::doc("LHECompleter::save")
        )
        .def_static(
            "load",
            &LHECompleter::load,
            py::arg("file"),
            pydoc::doc("LHECompleter::load")
        )
        .def_property_readonly(
            "max_particle_count",
            &LHECompleter::max_particle_count,
            pydoc::doc("LHECompleter::max_particle_count")
        );
    py::classh<LHEFileWriter>(m, "LHEFileWriter", pydoc::doc("LHEFileWriter"))
        .def(
            py::init<const std::string&, const LHEMeta&>(),
            py::arg("file_name"),
            py::arg("meta"),
            pydoc::doc("LHEFileWriter::LHEFileWriter")
        )
        .def(
            "write",
            &LHEFileWriter::write,
            py::arg("event"),
            pydoc::doc("LHEFileWriter::write")
        )
        .def(
            "write_string",
            &LHEFileWriter::write_string,
            py::arg("str"),
            pydoc::doc("LHEFileWriter::write_string")
        );

    m.def(
        "format_si_prefix",
        &format_si_prefix,
        py::arg("value"),
        pydoc::doc("format_si_prefix")
    );
    m.def(
        "format_with_error",
        &format_with_error,
        py::arg("value"),
        py::arg("error"),
        pydoc::doc("format_with_error")
    );
    m.def(
        "format_progress",
        &format_progress,
        py::arg("progress"),
        py::arg("width"),
        pydoc::doc("format_progress")
    );
    py::classh<PrettyBox>(m, "PrettyBox", pydoc::doc("PrettyBox"))
        .def(
            py::init<
                const std::string&,
                std::size_t,
                const std::vector<std::size_t>&,
                std::size_t,
                std::size_t>(),
            py::arg("title"),
            py::arg("rows"),
            py::arg("columns"),
            py::arg("offset") = 0,
            py::arg("box_width") = 91,
            pydoc::doc("PrettyBox::PrettyBox#2")
        )
        .def(
            "set_row",
            &PrettyBox::set_row,
            py::arg("row"),
            py::arg("values"),
            pydoc::doc("PrettyBox::set_row")
        )
        .def(
            "set_column",
            &PrettyBox::set_column,
            py::arg("column"),
            py::arg("values"),
            pydoc::doc("PrettyBox::set_column")
        )
        .def(
            "set_cell",
            &PrettyBox::set_cell,
            py::arg("row"),
            py::arg("column"),
            py::arg("value"),
            pydoc::doc("PrettyBox::set_cell")
        )
        .def(
            "print_first", &PrettyBox::print_first, pydoc::doc("PrettyBox::print_first")
        )
        .def(
            "print_update",
            &PrettyBox::print_update,
            pydoc::doc("PrettyBox::print_update")
        )
        .def_property_readonly(
            "line_count", &PrettyBox::line_count, pydoc::doc("PrettyBox::line_count")
        );

    py::classh<ChannelEventGenerator>(
        m, "ChannelEventGenerator", pydoc::doc("ChannelEventGenerator")
    )
        .def_static(
            "load",
            &ChannelEventGenerator::load,
            py::arg("channel_file"),
            py::arg("contexts"),
            py::arg("event_file"),
            py::arg("weight_file"),
            py::arg("config"),
            pydoc::doc("ChannelEventGenerator::load")
        )
        .def_static(
            "load_json",
            &ChannelEventGenerator::load_json,
            py::arg("channel_json"),
            py::arg("contexts"),
            py::arg("event_file"),
            py::arg("weight_file"),
            py::arg("config"),
            pydoc::doc("ChannelEventGenerator::load_json")
        )
        .def(
            py::init<
                const std::vector<ContextPtr>&,
                const Integrand&,
                const std::string&,
                const std::string&,
                const GeneratorConfig&,
                std::size_t,
                const std::string&,
                const std::optional<ObservableHistograms>&>(),
            py::arg("contexts"),
            py::arg("integrand"),
            py::arg("event_file"),
            py::arg("weight_file"),
            py::arg("config"),
            py::arg("subprocess_index"),
            py::arg("name"),
            py::arg("histograms"),
            pydoc::doc("ChannelEventGenerator::ChannelEventGenerator")
        )
        .def(
            "status",
            &ChannelEventGenerator::status,
            pydoc::doc("ChannelEventGenerator::status")
        )
        .def(
            "save",
            &ChannelEventGenerator::save,
            py::arg("save"),
            pydoc::doc("ChannelEventGenerator::save")
        )
        .def(
            "to_json",
            &ChannelEventGenerator::to_json,
            py::arg("include_estimates") = false,
            pydoc::doc("ChannelEventGenerator::to_json")
        )
        .def(
            "set_fixed_max_weight",
            &ChannelEventGenerator::set_fixed_max_weight,
            py::arg("max_weight"),
            pydoc::doc("ChannelEventGenerator::set_fixed_max_weight")
        );

    py::classh<PdfMemberSpec>(m, "PdfMemberSpec", pydoc::doc("PdfMemberSpec"))
        .def(
            py::init([](const std::string& set_name,
                        int set_lhaid,
                        int member,
                        const std::string& grid_file,
                        const std::string& info_file,
                        const std::string& error_type,
                        const std::string& description) {
                return PdfMemberSpec{
                    set_name,
                    set_lhaid,
                    member,
                    grid_file,
                    info_file,
                    error_type,
                    description
                };
            }),
            py::arg("set_name"),
            py::arg("set_lhaid"),
            py::arg("member"),
            py::arg("grid_file"),
            py::arg("info_file"),
            py::arg("error_type") = "",
            py::arg("description") = ""
        )
        .def_readwrite(
            "set_name", &PdfMemberSpec::set_name, pydoc::doc("PdfMemberSpec::set_name")
        )
        .def_readwrite(
            "set_lhaid",
            &PdfMemberSpec::set_lhaid,
            pydoc::doc("PdfMemberSpec::set_lhaid")
        )
        .def_readwrite(
            "member", &PdfMemberSpec::member, pydoc::doc("PdfMemberSpec::member")
        )
        .def_readwrite(
            "grid_file",
            &PdfMemberSpec::grid_file,
            pydoc::doc("PdfMemberSpec::grid_file")
        )
        .def_readwrite(
            "info_file",
            &PdfMemberSpec::info_file,
            pydoc::doc("PdfMemberSpec::info_file")
        )
        .def_readwrite(
            "error_type",
            &PdfMemberSpec::error_type,
            pydoc::doc("PdfMemberSpec::error_type")
        )
        .def_readwrite(
            "description",
            &PdfMemberSpec::description,
            pydoc::doc("PdfMemberSpec::description")
        );
    py::classh<SystematicsConfig>(
        m, "SystematicsConfig", pydoc::doc("SystematicsConfig")
    )
        .def(py::init<>())
        .def_readwrite(
            "mur", &SystematicsConfig::mur, pydoc::doc("SystematicsConfig::mur")
        )
        .def_readwrite(
            "muf", &SystematicsConfig::muf, pydoc::doc("SystematicsConfig::muf")
        )
        .def_readwrite(
            "alps", &SystematicsConfig::alps, pydoc::doc("SystematicsConfig::alps")
        )
        .def_readwrite(
            "pdf_ladder",
            &SystematicsConfig::pdf_ladder,
            pydoc::doc("SystematicsConfig::pdf_ladder")
        )
        .def_readwrite(
            "ladder_min_scale",
            &SystematicsConfig::ladder_min_scale,
            pydoc::doc("SystematicsConfig::ladder_min_scale")
        )
        .def_readwrite(
            "ladder_max_scale",
            &SystematicsConfig::ladder_max_scale,
            pydoc::doc("SystematicsConfig::ladder_max_scale")
        )
        .def_readwrite(
            "together",
            &SystematicsConfig::together,
            pydoc::doc("SystematicsConfig::together")
        )
        .def_readwrite(
            "dyn_scales",
            &SystematicsConfig::dyn_scales,
            pydoc::doc("SystematicsConfig::dyn_scales")
        )
        .def_readwrite(
            "scale_factor",
            &SystematicsConfig::scale_factor,
            pydoc::doc("SystematicsConfig::scale_factor")
        )
        .def_readwrite(
            "pdf_members",
            &SystematicsConfig::pdf_members,
            pydoc::doc("SystematicsConfig::pdf_members")
        )
        .def_readwrite(
            "nominal_set_name",
            &SystematicsConfig::nominal_set_name,
            pydoc::doc("SystematicsConfig::nominal_set_name")
        )
        .def_readwrite(
            "nominal_lhaid",
            &SystematicsConfig::nominal_lhaid,
            pydoc::doc("SystematicsConfig::nominal_lhaid")
        )
        .def_readwrite(
            "nominal_error_type",
            &SystematicsConfig::nominal_error_type,
            pydoc::doc("SystematicsConfig::nominal_error_type")
        )
        .def_readwrite(
            "nominal_description",
            &SystematicsConfig::nominal_description,
            pydoc::doc("SystematicsConfig::nominal_description")
        )
        .def_readwrite(
            "has_pdf",
            &SystematicsConfig::has_pdf,
            pydoc::doc("SystematicsConfig::has_pdf")
        )
        .def_readwrite(
            "write_inputs",
            &SystematicsConfig::write_inputs,
            pydoc::doc("SystematicsConfig::write_inputs")
        )
        .def_readwrite(
            "first_id",
            &SystematicsConfig::first_id,
            pydoc::doc("SystematicsConfig::first_id")
        )
        .def(
            "to_json",
            [](const SystematicsConfig& config) {
                return nlohmann::json(config).dump();
            }
        )
        .def_static("from_json", [](const std::string& text) {
            return nlohmann::json::parse(text).get<SystematicsConfig>();
        });
    py::classh<SubprocessSystArgs>(
        m, "SubprocessSystArgs", pydoc::doc("SubprocessSystArgs")
    )
        .def(
            py::init([](int qcd_power, const nested_vector2<int>& beam_pdgs) {
                return SubprocessSystArgs{qcd_power, beam_pdgs};
            }),
            py::arg("qcd_power"),
            py::arg("beam_pdgs")
        )
        .def_readwrite(
            "qcd_power",
            &SubprocessSystArgs::qcd_power,
            pydoc::doc("SubprocessSystArgs::qcd_power")
        )
        .def_readwrite(
            "beam_pdgs",
            &SubprocessSystArgs::beam_pdgs,
            pydoc::doc("SubprocessSystArgs::beam_pdgs")
        )
        .def(
            "to_json",
            [](const SubprocessSystArgs& args) { return nlohmann::json(args).dump(); }
        )
        .def_static("from_json", [](const std::string& text) {
            return nlohmann::json::parse(text).get<SubprocessSystArgs>();
        });
    py::classh<Variation>(m, "Variation", pydoc::doc("Variation"))
        .def_readonly("id", &Variation::id, pydoc::doc("Variation::id"))
        .def_readonly("mur", &Variation::mur, pydoc::doc("Variation::mur"))
        .def_readonly("muf", &Variation::muf, pydoc::doc("Variation::muf"))
        .def_readonly(
            "pdf_index", &Variation::pdf_index, pydoc::doc("Variation::pdf_index")
        )
        .def_readonly("dyn", &Variation::dyn, pydoc::doc("Variation::dyn"))
        .def_readonly("alps", &Variation::alps, pydoc::doc("Variation::alps"))
        .def_property_readonly(
            "is_scale", &Variation::is_scale, pydoc::doc("Variation::is_scale")
        )
        .def_property_readonly(
            "is_emission", &Variation::is_emission, pydoc::doc("Variation::is_emission")
        );
    py::classh<PdfGroupInfo>(m, "PdfGroupInfo", pydoc::doc("PdfGroupInfo"))
        .def_readonly(
            "set_name", &PdfGroupInfo::set_name, pydoc::doc("PdfGroupInfo::set_name")
        )
        .def_readonly(
            "set_lhaid", &PdfGroupInfo::set_lhaid, pydoc::doc("PdfGroupInfo::set_lhaid")
        )
        .def_readonly(
            "error_type",
            &PdfGroupInfo::error_type,
            pydoc::doc("PdfGroupInfo::error_type")
        )
        .def_readonly(
            "members", &PdfGroupInfo::members, pydoc::doc("PdfGroupInfo::members")
        );
    py::classh<SystematicsCalculator>(
        m, "SystematicsCalculator", pydoc::doc("SystematicsCalculator")
    )
        .def(
            py::init<
                const SystematicsConfig&,
                const std::vector<SubprocessSystArgs>&,
                const std::optional<PdfGrid>&,
                const std::optional<AlphaSGrid>&,
                ContextPtr,
                const std::vector<std::optional<MatrixElement>>&,
                const nested_vector2<me_int_t>&,
                const std::optional<PdfGrid>&>(),
            py::arg("config"),
            py::arg("subproc_args"),
            py::arg("nominal_pdf") = std::nullopt,
            py::arg("nominal_alpha_s") = std::nullopt,
            py::arg("context") = nullptr,
            py::arg("matrix_elements") = std::vector<std::optional<MatrixElement>>{},
            py::arg("me_flavor_remap") = nested_vector2<me_int_t>{},
            py::arg("nominal_pdf2") = std::nullopt,
            pydoc::doc("SystematicsCalculator::SystematicsCalculator")
        )
        .def_property_readonly(
            "config",
            &SystematicsCalculator::config,
            pydoc::doc("SystematicsCalculator::config")
        )
        .def_property_readonly(
            "scale_variation_indices",
            &SystematicsCalculator::scale_variation_indices,
            pydoc::doc("SystematicsCalculator::scale_variation_indices")
        )
        .def_property_readonly(
            "emission_variation_indices",
            &SystematicsCalculator::emission_variation_indices,
            pydoc::doc("SystematicsCalculator::emission_variation_indices")
        )
        .def_property_readonly(
            "pdf_groups",
            &SystematicsCalculator::pdf_groups,
            pydoc::doc("SystematicsCalculator::pdf_groups")
        )
        .def_static(
            "pdf_uncertainty",
            &SystematicsCalculator::pdf_uncertainty,
            py::arg("error_type"),
            py::arg("central"),
            py::arg("member_values"),
            pydoc::doc("SystematicsCalculator::pdf_uncertainty")
        )
        .def_static(
            "dynamical_scale",
            &SystematicsCalculator::dynamical_scale,
            py::arg("dyn"),
            py::arg("momenta"),
            pydoc::doc("SystematicsCalculator::dynamical_scale")
        )
        .def_property_readonly(
            "variations",
            &SystematicsCalculator::variations,
            pydoc::doc("SystematicsCalculator::variations")
        )
        .def_property_readonly(
            "weight_count",
            &SystematicsCalculator::weight_count,
            pydoc::doc("SystematicsCalculator::weight_count")
        )
        .def_property_readonly(
            "weight_ids",
            &SystematicsCalculator::weight_ids,
            pydoc::doc("SystematicsCalculator::weight_ids")
        )
        .def_property_readonly(
            "members",
            &SystematicsCalculator::members,
            pydoc::doc("SystematicsCalculator::members")
        )
        .def_property_readonly(
            "warnings",
            &SystematicsCalculator::warnings,
            pydoc::doc("SystematicsCalculator::warnings")
        )
        .def(
            "initrwgt",
            &SystematicsCalculator::initrwgt,
            pydoc::doc("SystematicsCalculator::initrwgt")
        )
        .def(
            "summary",
            [](const SystematicsCalculator& calc) { return calc.summary().dump(); },
            pydoc::doc("SystematicsCalculator::summary")
        )
        .def(
            "weights",
            [](const SystematicsCalculator& calc,
               const std::vector<double>& event_weight,
               const std::vector<int>& subprocess_index,
               const std::vector<int>& flavor_index,
               const std::vector<double>& ren_scale,
               const std::vector<double>& x1,
               const std::vector<double>& fact_scale1,
               const std::vector<double>& x2,
               const std::vector<double>& fact_scale2,
               const std::vector<double>& partial_weight_product,
               const nested_vector3<double>& momenta,
               const std::vector<double>& alpha_qcd,
               const nested_vector2<double>& alphas_scales,
               const nested_vector3<double>& ladder,
               const nested_vector2<double>& ladder_lead) {
                // build a combined-layout buffer from columns (testing / scripting)
                std::size_t count = event_weight.size();
                std::size_t particle_count = momenta.empty() ? 0 : momenta.at(0).size();
                // MLM vertex v rides on outgoing particle 2 + v
                for (auto& row : alphas_scales) {
                    particle_count = std::max(particle_count, row.size() + 3);
                }
                // the pdf ladder: step s ([x, pdg, q, info]) on particle s,
                // and [q_lead1, q_lead2] per event
                for (auto& steps : ladder) {
                    particle_count = std::max(particle_count, steps.size());
                }
                if (!ladder_lead.empty()) {
                    particle_count = std::max<std::size_t>(particle_count, 2);
                }
                int particle_flags =
                    (momenta.empty() ? ParticleRecord::f_none
                                     : ParticleRecord::f_particle_data) |
                    (alphas_scales.empty()
                         ? ParticleRecord::f_none
                         : ParticleRecord::f_alphas_scales) |
                    (ladder_lead.empty()
                         ? ParticleRecord::f_none
                         : ParticleRecord::f_pdf_ladder);
                DataLayout layout(
                    EventRecord::layout(
                        EventRecord::f_weight | EventRecord::f_subproc_index |
                        EventRecord::f_event_data | EventRecord::f_beam1 |
                        EventRecord::f_beam2 | EventRecord::f_partial_weights
                    ),
                    ParticleRecord::layout(particle_flags)
                );
                EventBuffer buffer(count, particle_count, layout);
                for (std::size_t i = 0; i < count; ++i) {
                    auto event = buffer.event(i);
                    event.weight() = event_weight.at(i);
                    event.subprocess_index() = subprocess_index.at(i);
                    event.flavor_index() = flavor_index.at(i);
                    event.diagram_index() = 0;
                    event.color_index() = 0;
                    event.helicity_index() = 0;
                    event.ren_scale() = ren_scale.at(i);
                    event.alpha_qcd() = alpha_qcd.empty() ? 0. : alpha_qcd.at(i);
                    event.x1() = x1.at(i);
                    event.fact_scale1() = fact_scale1.at(i);
                    event.x2() = x2.at(i);
                    event.fact_scale2() = fact_scale2.at(i);
                    event.partial_weight_product() = partial_weight_product.at(i);
                    for (std::size_t j = 0;
                         !momenta.empty() && j < momenta.at(i).size();
                         ++j) {
                        auto particle = buffer.particle(i, j);
                        auto& p = momenta.at(i).at(j);
                        particle.energy() = p.at(0);
                        particle.px() = p.at(1);
                        particle.py() = p.at(2);
                        particle.pz() = p.at(3);
                    }
                    for (std::size_t j = 0;
                         !alphas_scales.empty() && j < particle_count;
                         ++j) {
                        auto& row = alphas_scales.at(i);
                        buffer.particle(i, j).alphas_scale() =
                            j >= 2 && j - 2 < row.size() ? row.at(j - 2) : 0.;
                    }
                    for (std::size_t j = 0; !ladder_lead.empty() && j < particle_count;
                         ++j) {
                        auto particle = buffer.particle(i, j);
                        auto& lead = ladder_lead.at(i);
                        particle.ladder_lead_q() = j < 2 ? lead.at(j) : 0.;
                        bool has_step = !ladder.empty() && j < ladder.at(i).size();
                        auto step =
                            has_step ? ladder.at(i).at(j) : std::vector<double>(4, 0.);
                        particle.ladder_x() = step.at(0);
                        particle.ladder_pdg() = step.at(1);
                        particle.ladder_q() = step.at(2);
                        particle.ladder_info() = step.at(3);
                    }
                }
                std::vector<double> weights;
                calc.compute(buffer, weights);
                nested_vector2<double> result(count);
                std::size_t var_count = calc.weight_count();
                for (std::size_t i = 0; i < count; ++i) {
                    result[i].assign(
                        weights.begin() + i * var_count,
                        weights.begin() + (i + 1) * var_count
                    );
                }
                return result;
            },
            py::arg("event_weight"),
            py::arg("subprocess_index"),
            py::arg("flavor_index"),
            py::arg("ren_scale"),
            py::arg("x1"),
            py::arg("fact_scale1"),
            py::arg("x2"),
            py::arg("fact_scale2"),
            py::arg("partial_weight_product"),
            py::arg("momenta") = nested_vector3<double>{},
            py::arg("alpha_qcd") = std::vector<double>{},
            py::arg("alphas_scales") = nested_vector2<double>{},
            py::arg("ladder") = nested_vector3<double>{},
            py::arg("ladder_lead") = nested_vector2<double>{}
        );
    py::classh<EventHistogramSpec>(
        m, "EventHistogramSpec", pydoc::doc("EventHistogramSpec")
    )
        .def(
            py::init([](const std::string& name,
                        double min,
                        double max,
                        std::size_t bin_count,
                        bool from_weight) {
                return EventHistogramSpec{name, min, max, bin_count, from_weight};
            }),
            py::arg("name"),
            py::arg("min"),
            py::arg("max"),
            py::arg("bin_count"),
            py::arg("from_weight") = false
        )
        .def_readwrite(
            "name", &EventHistogramSpec::name, pydoc::doc("EventHistogramSpec::name")
        )
        .def_readwrite(
            "min", &EventHistogramSpec::min, pydoc::doc("EventHistogramSpec::min")
        )
        .def_readwrite(
            "max", &EventHistogramSpec::max, pydoc::doc("EventHistogramSpec::max")
        )
        .def_readwrite(
            "bin_count",
            &EventHistogramSpec::bin_count,
            pydoc::doc("EventHistogramSpec::bin_count")
        )
        .def_readwrite(
            "from_weight",
            &EventHistogramSpec::from_weight,
            pydoc::doc("EventHistogramSpec::from_weight")
        );
    py::classh<SubprocessObservables>(
        m, "SubprocessObservables", pydoc::doc("SubprocessObservables")
    )
        .def(
            py::init([](const ObservableValues& values, std::size_t particle_count) {
                return SubprocessObservables{values, particle_count};
            }),
            py::arg("values"),
            py::arg("particle_count")
        );
    py::classh<EventHistograms>(m, "EventHistograms", pydoc::doc("EventHistograms"))
        .def(
            py::init<
                ContextPtr,
                const std::vector<EventHistogramSpec>&,
                const std::vector<std::optional<SubprocessObservables>>&,
                double>(),
            py::arg("context"),
            py::arg("specs"),
            py::arg("observables"),
            py::arg("reference_weight") = 0.,
            pydoc::doc("EventHistograms::EventHistograms")
        )
        .def_property_readonly(
            "specs", &EventHistograms::specs, pydoc::doc("EventHistograms::specs")
        )
        .def_property_readonly(
            "weight_count",
            &EventHistograms::weight_count,
            pydoc::doc("EventHistograms::weight_count")
        )
        .def(
            "fill",
            [](EventHistograms& hists,
               const std::vector<double>& event_weight,
               const std::vector<int>& subprocess_index,
               const nested_vector3<double>& momenta,
               const nested_vector2<double>& syst_weights) {
                std::size_t count = event_weight.size();
                std::size_t particle_count = momenta.empty() ? 0 : momenta.at(0).size();
                DataLayout layout(
                    EventRecord::layout(
                        EventRecord::f_weight | EventRecord::f_subproc_index |
                        EventRecord::f_event_data
                    ),
                    ParticleRecord::layout(ParticleRecord::f_particle_data)
                );
                EventBuffer buffer(count, particle_count, layout);
                std::size_t weight_count =
                    syst_weights.empty() ? 0 : syst_weights.at(0).size();
                std::vector<double> flat;
                for (std::size_t i = 0; i < count; ++i) {
                    auto event = buffer.event(i);
                    event.weight() = event_weight.at(i);
                    event.subprocess_index() = subprocess_index.at(i);
                    event.diagram_index() = 0;
                    event.color_index() = 0;
                    event.flavor_index() = 0;
                    event.helicity_index() = 0;
                    event.ren_scale() = 0.;
                    event.alpha_qcd() = 0.;
                    for (std::size_t j = 0; j < particle_count; ++j) {
                        auto particle = buffer.particle(i, j);
                        auto& p = momenta.at(i).at(j);
                        particle.energy() = p.at(0);
                        particle.px() = p.at(1);
                        particle.py() = p.at(2);
                        particle.pz() = p.at(3);
                    }
                    if (weight_count > 0) {
                        auto& row = syst_weights.at(i);
                        flat.insert(flat.end(), row.begin(), row.end());
                    }
                }
                hists.fill(buffer, flat, weight_count);
            },
            py::arg("event_weight"),
            py::arg("subprocess_index"),
            py::arg("momenta"),
            py::arg("syst_weights") = nested_vector2<double>{}
        )
        .def(
            "to_json",
            [](const EventHistograms& hists, const SystematicsCalculator* systematics) {
                return hists.to_json(systematics).dump();
            },
            py::arg("systematics") = nullptr,
            pydoc::doc("EventHistograms::to_json")
        );

    py::classh<EventGenerator>(m, "EventGenerator", pydoc::doc("EventGenerator"))
        .def_readonly_static("default_config", &EventGenerator::default_config)
        .def(
            py::init<
                const std::vector<ContextPtr>&,
                const std::vector<std::shared_ptr<ChannelEventGenerator>>&,
                std::uint64_t,
                std::shared_ptr<StatusFile>,
                const GeneratorConfig&>(),
            py::arg("contexts"),
            py::arg("channels"),
            py::arg("seed"),
            py::arg("status_file") = std::shared_ptr<StatusFile>(),
            py::arg_v(
                "config",
                EventGenerator::default_config,
                "EventGenerator.default_config"
            ),
            pydoc::doc("EventGenerator::EventGenerator")
        )
        .def(
            "survey",
            &EventGenerator::survey,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("survey_pass") = 0,
            pydoc::doc("EventGenerator::survey")
        )
        .def(
            "generate",
            &EventGenerator::generate,
            py::call_guard<py::gil_scoped_release>(),
            pydoc::doc("EventGenerator::generate")
        )
        .def(
            "optimize",
            &EventGenerator::optimize,
            pydoc::doc("EventGenerator::optimize")
        )
        .def(
            "fix_max_weights",
            &EventGenerator::fix_max_weights,
            pydoc::doc("EventGenerator::fix_max_weights")
        )
        .def(
            "combine_to_compact_npy",
            &EventGenerator::combine_to_compact_npy,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("file_name"),
            py::arg("systematics") = nullptr,
            py::arg("histograms") = nullptr,
            pydoc::doc("EventGenerator::combine_to_compact_npy")
        )
        .def(
            "combine_to_lhe_npy",
            &EventGenerator::combine_to_lhe_npy,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("file_name"),
            py::arg("lhe_completer"),
            py::arg("systematics") = nullptr,
            py::arg("histograms") = nullptr,
            pydoc::doc("EventGenerator::combine_to_lhe_npy")
        )
        .def(
            "combine_to_lhe",
            &EventGenerator::combine_to_lhe,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("file_name"),
            py::arg("lhe_completer"),
            py::arg_v("meta", LHEMeta{}, "LHEMeta()"),
            py::arg("systematics") = nullptr,
            py::arg("histograms") = nullptr,
            pydoc::doc("EventGenerator::combine_to_lhe")
        )
        .def("status", &EventGenerator::status, pydoc::doc("EventGenerator::status"))
        .def(
            "channel_status",
            &EventGenerator::channel_status,
            pydoc::doc("EventGenerator::channel_status")
        )
        .def(
            "histograms",
            &EventGenerator::histograms,
            pydoc::doc("EventGenerator::histograms")
        )
        .def(
            "used_globals",
            &EventGenerator::used_globals,
            pydoc::doc("EventGenerator::used_globals")
        )
        .def(
            "channels",
            &EventGenerator::channels,
            pydoc::doc("EventGenerator::channels")
        );

    py::classh<Logger> logger(m, "Logger", pydoc::doc("Logger"));
    add_enum<Logger::LogLevel>(
        logger,
        "LogLevel",
        {
            {"level_debug", Logger::level_debug},
            {"level_info", Logger::level_info},
            {"level_warning", Logger::level_warning},
            {"level_error", Logger::level_error},
        },
        "",
        pydoc::doc("Logger::LogLevel")
    );
    logger
        .def_static(
            "log",
            &Logger::log,
            py::arg("level"),
            py::arg("message"),
            pydoc::doc("Logger::log")
        )
        .def_static(
            "debug", &Logger::debug, py::arg("message"), pydoc::doc("Logger::debug")
        )
        .def_static(
            "info", &Logger::info, py::arg("message"), pydoc::doc("Logger::info")
        )
        .def_static(
            "warning",
            &Logger::warning,
            py::arg("message"),
            pydoc::doc("Logger::warning")
        )
        .def_static(
            "error", &Logger::error, py::arg("message"), pydoc::doc("Logger::error")
        )
        .def_static(
            "set_log_handler",
            &Logger::set_log_handler,
            py::arg("func"),
            pydoc::doc("Logger::set_log_handler")
        )
        .def_static(
            "clear_log_handler",
            &Logger::clear_log_handler,
            pydoc::doc("Logger::clear_log_handler")
        );

    // prevent memory error due to static lifetime of log handler
    py::module_::import("atexit").attr("register")(py::cpp_function([]() {
        Logger::clear_log_handler();
    }));

    m.def(
        "initialize_vegas_grid",
        &initialize_vegas_grid,
        py::arg("context"),
        py::arg("grid_name"),
        pydoc::doc("initialize_vegas_grid")
    );
    m.def(
        "set_lib_path", &set_lib_path, py::arg("lib_path"), pydoc::doc("set_lib_path")
    );
    m.def(
        "set_simd_mode",
        &set_simd_mode,
        py::arg("simd_mode"),
        pydoc::doc("set_simd_mode")
    );
    m.def(
        "supported_simd_modes",
        &supported_simd_modes,
        pydoc::doc("supported_simd_modes")
    );

    auto abort_check_function = [] {
        // the long-running calls release the GIL
        py::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };
    EventGenerator::set_abort_check_function(abort_check_function);
    MadnisTraining::set_abort_check_function(abort_check_function);
}
