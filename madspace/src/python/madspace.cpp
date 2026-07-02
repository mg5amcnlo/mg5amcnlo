#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "function_runtime.hpp"
#include "instruction_set.hpp"
#include "madspace/compgraphs.hpp"
#include "madspace/driver.hpp"
#include "madspace/phasespace.hpp"
#include "madspace/util.hpp"

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
    const std::string& prefix = ""
) {
    std::unordered_map<std::string, EnumType> str_to_enum_map(values);
    py::enum_<EnumType> enumeration(parent, enum_name);
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
    py::classh<NamedVector<T>>(m, name)
        .def(py::init<>())
        .def(
            py::init<const std::vector<std::string>&, const std::vector<T>&>(),
            py::arg("keys"),
            py::arg("values")
        )
        .def(
            py::init<const std::vector<std::pair<std::string, T>>&>(), py::arg("items")
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
        .def("values", &NamedVector<T>::values)
        .def("index_map", &NamedVector<T>::index_map)
        .def("keys", &NamedVector<T>::keys)
        .def("push_back", &NamedVector<T>::push_back, py::arg("name"), py::arg("item"));
}

} // namespace

PYBIND11_MODULE(_madspace_py, m) {
    add_enum<DataType>(
        m,
        "DataType",
        {
            {"int", DataType::dt_int},
            {"float", DataType::dt_float},
            {"batch_sizes", DataType::batch_sizes},
        }
    );

    py::classh<BatchSize>(m, "BatchSize")
        .def(py::init<>())
        .def(py::init<std::string>(), py::arg("name"))
        .def_readonly_static("one", &BatchSize::one)
        .def("__str__", &to_string<BatchSize>)
        .def("__repr__", &to_string<BatchSize>);
    m.attr("batch_size") = py::cast(batch_size);

    py::classh<Type>(m, "Type")
        .def(
            py::init<DataType, BatchSize, std::vector<int>>(),
            py::arg("dtype"),
            py::arg("batch_size"),
            py::arg("shape")
        )
        .def(py::init<std::vector<BatchSize>>(), py::arg("batch_size_list"))
        .def_readonly("dtype", &Type::dtype)
        .def_readonly("batch_size", &Type::batch_size)
        .def_readonly("shape", &Type::shape)
        .def("__str__", &to_string<Type>)
        .def("__repr__", &to_string<Type>);
    m.attr("single_float") = py::cast(single_float);
    m.attr("single_int") = py::cast(single_int);
    m.def("multichannel_batch_size", &multichannel_batch_size, py::arg("count"));
    m.attr("batch_float") = py::cast(batch_float);
    m.attr("batch_int") = py::cast(batch_int);
    m.attr("batch_four_vec") = py::cast(batch_four_vec);
    m.def("batch_float_array", &batch_float_array, py::arg("count"));
    m.def("batch_four_vec_array", &batch_four_vec_array, py::arg("count"));

    py::classh<InstrCopy>(m, "Instruction")
        .def("__str__", [](const InstrCopy& instr) { return instr.name; })
        .def_readonly("name", &InstrCopy::name)
        .def_readonly("opcode", &InstrCopy::opcode);

    py::classh<Value>(m, "Value")
        .def(py::init<me_int_t>(), py::arg("value"))
        .def(py::init<double>(), py::arg("value"))
        .def("__str__", &to_string<Value>)
        .def("__repr__", &to_string<Value>)
        .def_readonly("type", &Value::type)
        .def_readonly("literal_value", &Value::literal_value)
        .def_readonly("local_index", &Value::local_index);
    py::implicitly_convertible<me_int_t, Value>();
    py::implicitly_convertible<double, Value>();

    named_vector_instance<Value>(m, "NamedValues");
    named_vector_instance<Type>(m, "NamedTypes");

    py::classh<InstructionCall>(m, "InstructionCall")
        .def("__str__", &to_string<InstructionCall>)
        .def("__repr__", &to_string<InstructionCall>)
        .def_property_readonly(
            "instruction",
            [](const InstructionCall& call) -> InstrCopy { return call.instruction; }
        )
        .def_readonly("inputs", &InstructionCall::inputs)
        .def_readonly("outputs", &InstructionCall::outputs);

    py::classh<Function>(m, "Function", py::dynamic_attr())
        .def("__str__", &to_string<Function>)
        .def("__repr__", &to_string<Function>)
        .def("save", &Function::save, py::arg("file"))
        .def_static("load", &Function::load, py::arg("file"))
        .def_property_readonly("inputs", &Function::inputs)
        .def_property_readonly("outputs", &Function::outputs)
        .def_property_readonly("locals", &Function::locals)
        .def_property_readonly("globals", &Function::globals)
        .def_property_readonly("instructions", &Function::instructions);

    py::classh<Device> device(m, "Device");
    m.def("cpu_device", &cpu_device, py::return_value_policy::reference);
    m.def(
        "cuda_device",
        &cuda_device,
        py::arg("index") = 0,
        py::return_value_policy::reference
    );
    m.def(
        "hip_device",
        &hip_device,
        py::arg("index") = 0,
        py::return_value_policy::reference
    );
    m.def("available_backends", &available_backends);

    py::classh<MatrixElementApi>(m, "MatrixElementApi")
        //.def("device", &MatrixElementApi::device)
        .def("particle_count", &MatrixElementApi::particle_count)
        .def("diagram_count", &MatrixElementApi::diagram_count)
        .def("helicity_count", &MatrixElementApi::helicity_count)
        .def("index", &MatrixElementApi::index);

    py::classh<Tensor>(m, "Tensor", py::dynamic_attr())
        .def(
            "__dlpack__",
            &tensor_to_dlpack,
            py::arg("stream") = std::nullopt,
            py::arg("max_version") = std::nullopt,
            py::arg("dl_device") = std::nullopt,
            py::arg("copy") = std::nullopt
        )
        .def("__dlpack_device__", &dlpack_device);

    py::classh<Context>(m, "Context")
        .def(py::init<int>(), py::arg("thread_count") = -1)
        .def(
            py::init<DevicePtr, int>(), py::arg("device"), py::arg("thread_count") = -1
        )
        .def(
            "load_matrix_element",
            &Context::load_matrix_element,
            py::arg("file"),
            py::arg("param_card"),
            py::return_value_policy::reference_internal
        )
        .def(
            "define_global",
            &Context::define_global,
            py::arg("name"),
            py::arg("dtype"),
            py::arg("shape"),
            py::arg("requires_grad") = false
        )
        .def("get_global", &Context::global, py::arg("name"))
        .def("global_requires_grad", &Context::global_requires_grad, py::arg("name"))
        .def("global_exists", &Context::global_exists, py::arg("name"))
        .def("global_names", &Context::global_names)
        .def("delete_global", &Context::delete_global, py::arg("name"))
        .def("copy_globals_from", &Context::copy_globals_from, py::arg("context"))
        .def(
            "matrix_element",
            &Context::matrix_element,
            py::arg("index"),
            py::return_value_policy::reference_internal
        )
        .def("save_globals", &Context::save_globals, py::arg("dir"))
        .def("load_globals", &Context::load_globals, py::arg("dir"))
        .def("device", &Context::device, py::return_value_policy::reference);
    m.def("default_context", &default_context);
    m.def("default_cuda_context", &default_cuda_context, py::arg("index") = 0);
    m.def("default_hip_context", &default_hip_context, py::arg("index") = 0);

    py::classh<FunctionRuntime>(m, "FunctionRuntime", py::dynamic_attr())
        .def(py::init<Function>(), py::arg("function"))
        .def(py::init<Function, ContextPtr>(), py::arg("function"), py::arg("context"))
        .def("call", &FunctionRuntime::call)
        .def("call_with_grad", &FunctionRuntime::call_with_grad)
        .def("call_backward", &FunctionRuntime::call_backward);

    auto& fb =
        py::classh<FunctionBuilder>(m, "FunctionBuilder")
            .def(
                py::init<const NamedVector<Type>&, const NamedVector<Type>&>(),
                py::arg("input_types"),
                py::arg("output_types")
            )
            .def("input", &FunctionBuilder::input, py::arg("index"))
            .def(
                "input_range",
                &FunctionBuilder::input_range,
                py::arg("start_index"),
                py::arg("end_index")
            )
            .def("output", &FunctionBuilder::output, py::arg("index"), py::arg("value"))
            .def(
                "output_range",
                &FunctionBuilder::output_range,
                py::arg("start_index"),
                py::arg("values")
            )
            .def(
                "get_global",
                &FunctionBuilder::global,
                py::arg("name"),
                py::arg("dtype"),
                py::arg("shape")
            )
            //.def("instruction", &FunctionBuilder::instruction, py::arg("name"),
            // py::arg("args"))
            .def("product", &FunctionBuilder::product, py::arg("values"))
            .def("current_stream", &FunctionBuilder::current_stream)
            .def("set_current_stream", &FunctionBuilder::set_current_stream)
            .def("function", &FunctionBuilder::function);
    add_instructions(fb);

    py::classh<Mapping, PyMapping>(m, "Mapping", py::dynamic_attr())
        .def(
            py::init<
                const std::string&,
                const NamedVector<Type>&,
                const NamedVector<Type>&,
                const NamedVector<Type>&>(),
            py::arg("name"),
            py::arg("input_types"),
            py::arg("output_types"),
            py::arg("condition_types")
        )
        .def("forward_function", &Mapping::forward_function)
        .def("inverse_function", &Mapping::inverse_function)
        .def(
            "build_forward",
            py::overload_cast<FunctionBuilder&, const ValueVec&, const ValueVec&>(
                &Mapping::build_forward, py::const_
            ),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions")
        )
        .def(
            "build_forward",
            py::overload_cast<
                FunctionBuilder&,
                const NamedVector<Value>&,
                const NamedVector<Value>&>(&Mapping::build_forward, py::const_),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions")
        )
        .def(
            "build_inverse",
            py::overload_cast<FunctionBuilder&, const ValueVec&, const ValueVec&>(
                &Mapping::build_inverse, py::const_
            ),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions")
        )
        .def(
            "build_inverse",
            py::overload_cast<
                FunctionBuilder&,
                const NamedVector<Value>&,
                const NamedVector<Value>&>(&Mapping::build_inverse, py::const_),
            py::arg("builder"),
            py::arg("inputs"),
            py::arg("conditions")
        );

    py::classh<FunctionGenerator, PyFunctionGenerator>(
        m, "FunctionGenerator", py::dynamic_attr()
    )
        .def(
            py::init<
                const std::string&,
                const NamedVector<Type>&,
                const NamedVector<Type>&>(),
            py::arg("name"),
            py::arg("arg_types"),
            py::arg("return_types")
        )
        .def("function", &FunctionGenerator::function)
        .def(
            "build_function",
            py::overload_cast<FunctionBuilder&, const ValueVec&>(
                &FunctionGenerator::build_function, py::const_
            ),
            py::arg("builder"),
            py::arg("args")
        )
        .def(
            "build_function",
            py::overload_cast<FunctionBuilder&, const NamedVector<Value>&>(
                &FunctionGenerator::build_function, py::const_
            ),
            py::arg("builder"),
            py::arg("args")
        );

    py::classh<Invariant, Mapping>(m, "Invariant")
        .def(
            py::init<double, double, double>(),
            py::arg("power") = 0.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.
        );

    py::classh<Luminosity, Mapping>(m, "Luminosity")
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("s_lab"),
            py::arg("s_hat_min"),
            py::arg("s_hat_max") = 0.,
            py::arg("invariant_power") = 0.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.
        );

    py::classh<TwoBodyDecay, Mapping>(m, "TwoBodyDecay")
        .def(py::init<bool>(), py::arg("com"));

    py::classh<TwoToTwoParticleScattering, Mapping>(m, "TwoToTwoParticleScattering")
        .def(
            py::init<bool, double, double, double>(),
            py::arg("com"),
            py::arg("invariant_power") = 0.,
            py::arg("mass") = 0.,
            py::arg("width") = 0.
        );

    py::classh<DoubleT, Mapping>(m, "DoubleT")
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("t1_invariant_power") = 0.,
            py::arg("t1_mass") = 0.,
            py::arg("t1_width") = 0.,
            py::arg("t2_invariant_power") = 0.,
            py::arg("t2_mass") = 0.,
            py::arg("t2_width") = 0.
        );

    py::classh<ThreeBodyDecay, Mapping>(m, "ThreeBodyDecay")
        .def(py::init<bool>(), py::arg("com"));

    py::classh<TwoToThreeParticleScattering, Mapping>(m, "TwoToThreeParticleScattering")
        .def(
            py::init<double, double, double, double, double, double>(),
            py::arg("t_invariant_power") = 0.,
            py::arg("t_mass") = 0.,
            py::arg("t_width") = 0.,
            py::arg("s_invariant_power") = 0.,
            py::arg("s_mass") = 0.,
            py::arg("s_width") = 0.
        );

    py::classh<Propagator>(m, "Propagator")
        .def(
            py::init<double, double, int, double, double, int>(),
            py::arg("mass") = 0.,
            py::arg("width") = 0.,
            py::arg("integration_order") = 0,
            py::arg("e_min") = 0.,
            py::arg("e_max") = 0.,
            py::arg("pdg_id") = 0
        )
        .def_readonly("mass", &Propagator::mass)
        .def_readonly("width", &Propagator::width)
        .def_readonly("integration_order", &Propagator::integration_order)
        .def_readonly("e_min", &Propagator::e_min)
        .def_readonly("e_max", &Propagator::e_max)
        .def_readonly("pdg_id", &Propagator::pdg_id);

    py::classh<TPropagatorMapping, Mapping>(m, "TPropagatorMapping")
        .def(
            py::init<std::vector<std::size_t>, double>(),
            py::arg("integration_order"),
            py::arg("invariant_power") = 0.
        )
        .def("random_dim", &TPropagatorMapping::random_dim);

    py::classh<ColorOrderedMapping, Mapping>(m, "ColorOrderedMapping")
        .def(
            py::init<std::vector<std::size_t>, double, double>(),
            py::arg("color_order"),
            py::arg("t_invariant_power") = 0.,
            py::arg("s_invariant_power") = 0.
        )
        .def("random_dim", &ColorOrderedMapping::random_dim);

    py::classh<VegasHistogram, FunctionGenerator>(m, "VegasHistogram")
        .def(
            py::init<std::size_t, std::size_t>(),
            py::arg("dimension"),
            py::arg("bin_count")
        );

    py::classh<VegasMapping, Mapping>(m, "VegasMapping")
        .def(
            py::init<std::size_t, std::size_t, const std::string&>(),
            py::arg("dimension"),
            py::arg("bin_count"),
            py::arg("prefix") = ""
        )
        .def("grid_name", &VegasMapping::grid_name)
        .def(
            "initialize_globals", &VegasMapping::initialize_globals, py::arg("context")
        );

    py::classh<FastRamboMapping, Mapping>(m, "FastRamboMapping")
        .def(
            py::init<std::size_t, bool>(), py::arg("n_particles"), py::arg("massless")
        );

    py::classh<MultiChannelMapping, Mapping>(m, "MultiChannelMapping")
        .def(py::init<std::vector<std::shared_ptr<Mapping>>&>(), py::arg("mappings"));

    auto obs = py::classh<Observable, FunctionGenerator>(m, "Observable");
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
            {"sqrt_s", Observable::obs_sqrt_s},
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
           py::arg("name") = ""
    )
        .def_readonly_static("jet_pids", &Observable::jet_pids)
        .def_readonly_static("bottom_pids", &Observable::bottom_pids)
        .def_readonly_static("lepton_pids", &Observable::lepton_pids)
        .def_readonly_static("missing_pids", &Observable::missing_pids)
        .def_readonly_static("photon_pids", &Observable::photon_pids);

    auto cuts = py::classh<Cuts, FunctionGenerator>(m, "Cuts");
    add_enum<Cuts::CutMode>(
        cuts,
        "CutMode",
        {
            {"any", Cuts::any},
            {"all", Cuts::all},
        }
    );
    py::classh<Cuts::CutItem>(m, "CutItem")
        .def(
            py::init<Observable, double, double, Cuts::CutMode>(),
            py::arg("observable"),
            py::arg("min") = -std::numeric_limits<double>::infinity(),
            py::arg("max") = std::numeric_limits<double>::infinity(),
            py::arg("mode") = Cuts::CutMode::all
        )
        .def_readonly("observable", &Cuts::CutItem::observable)
        .def_readonly("min", &Cuts::CutItem::min)
        .def_readonly("max", &Cuts::CutItem::max)
        .def_readonly("mode", &Cuts::CutItem::mode);
    cuts.def(py::init<const std::vector<Cuts::CutItem>&>(), py::arg("cut_data"))
        .def(py::init<std::size_t>(), py::arg("particle_count"))
        .def("sqrt_s_min", &Cuts::sqrt_s_min)
        .def("eta_max", &Cuts::eta_max)
        .def("pt_min", &Cuts::pt_min);

    py::classh<ObservableHistograms::HistItem>(m, "HistItem")
        .def(
            py::init<Observable, double, double, std::size_t>(),
            py::arg("observable"),
            py::arg("min"),
            py::arg("max"),
            py::arg("bin_count")
        )
        .def_readonly("observable", &ObservableHistograms::HistItem::observable)
        .def_readonly("min", &ObservableHistograms::HistItem::min)
        .def_readonly("max", &ObservableHistograms::HistItem::max)
        .def_readonly("bin_count", &ObservableHistograms::HistItem::bin_count);
    py::classh<ObservableHistograms, FunctionGenerator>(m, "ObservableHistograms")
        .def(
            py::init<const std::vector<ObservableHistograms::HistItem>&>(),
            py::arg("observables")
        );

    py::classh<Diagram::LineRef>(m, "LineRef")
        .def(py::init<std::string>(), py::arg("str"))
        .def("__repr__", &to_string<Diagram::LineRef>);
    py::implicitly_convertible<std::string, Diagram::LineRef>();
    py::classh<Diagram>(m, "Diagram")
        .def(
            py::init<
                std::vector<double>&,
                std::vector<double>&,
                std::vector<Propagator>&,
                std::vector<Diagram::Vertex>&>(),
            py::arg("incoming_masses"),
            py::arg("outgoing_masses"),
            py::arg("propagators"),
            py::arg("vertices")
        )
        .def_property_readonly("incoming_masses", &Diagram::incoming_masses)
        .def_property_readonly("outgoing_masses", &Diagram::outgoing_masses)
        .def_property_readonly("propagators", &Diagram::propagators)
        .def_property_readonly("vertices", &Diagram::vertices)
        .def_property_readonly("incoming_vertices", &Diagram::incoming_vertices)
        .def_property_readonly("outgoing_vertices", &Diagram::outgoing_vertices)
        .def_property_readonly("propagator_vertices", &Diagram::propagator_vertices);
    py::classh<Topology::Decay>(m, "Decay")
        .def_readonly("index", &Topology::Decay::index)
        .def_readonly("parent_index", &Topology::Decay::parent_index)
        .def_readonly("child_indices", &Topology::Decay::child_indices)
        .def_readonly("mass", &Topology::Decay::mass)
        .def_readonly("width", &Topology::Decay::width)
        .def_readonly("e_min", &Topology::Decay::e_min)
        .def_readonly("e_max", &Topology::Decay::e_max)
        .def_readonly("pdg_id", &Topology::Decay::pdg_id)
        .def_readonly("on_shell", &Topology::Decay::on_shell);
    auto& topology =
        py::classh<Topology>(m, "Topology")
            .def(py::init<const Diagram&>(), py::arg("diagram"))
            .def_static("topologies", &Topology::topologies, py::arg("diagram"))
            .def_property_readonly("t_propagator_count", &Topology::t_propagator_count)
            .def_property_readonly(
                "t_integration_order", &Topology::t_integration_order
            )
            .def_property_readonly(
                "t_propagator_masses", &Topology::t_propagator_masses
            )
            .def_property_readonly(
                "t_propagator_widths", &Topology::t_propagator_widths
            )
            .def_property_readonly("decays", &Topology::decays)
            .def_property_readonly(
                "decay_integration_order", &Topology::decay_integration_order
            )
            .def_property_readonly("outgoing_indices", &Topology::outgoing_indices)
            .def_property_readonly("incoming_masses", &Topology::incoming_masses)
            .def_property_readonly("outgoing_masses", &Topology::outgoing_masses)
            .def("propagator_momentum_terms", &Topology::propagator_momentum_terms);
    py::classh<PhaseSpaceMapping, Mapping> psmap(m, "PhaseSpaceMapping");
    add_enum<PhaseSpaceMapping::TChannelMode>(
        psmap,
        "TChannelMode",
        {
            {"propagator", PhaseSpaceMapping::propagator},
            {"rambo", PhaseSpaceMapping::rambo},
            {"chili", PhaseSpaceMapping::chili},
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
                const nested_vector2<std::size_t>&>(),
            py::arg("topology"),
            py::arg("cm_energy"),
            py::arg("leptonic") = false,
            py::arg("invariant_power") = 0.8,
            py::arg("t_channel_mode") = PhaseSpaceMapping::propagator,
            py::arg("cuts") = std::nullopt,
            py::arg("permutations") = std::vector<Topology>{}
        )
        .def(
            py::init<
                const std::vector<double>&,
                double,
                bool,
                double,
                PhaseSpaceMapping::TChannelMode,
                std::optional<Cuts>>(),
            py::arg("masses"),
            py::arg("cm_energy"),
            py::arg("leptonic") = false,
            py::arg("invariant_power") = 0.8,
            py::arg("mode") = PhaseSpaceMapping::rambo,
            py::arg("cuts") = std::nullopt
        )
        .def("random_dim", &PhaseSpaceMapping::random_dim)
        .def("particle_count", &PhaseSpaceMapping::particle_count)
        .def("channel_count", &PhaseSpaceMapping::channel_count);

    py::classh<MultiChannelFunction, FunctionGenerator>(m, "MultiChannelFunction")
        .def(
            py::init<std::vector<std::shared_ptr<FunctionGenerator>>&>(),
            py::arg("functions")
        );

    py::classh<MatrixElement, FunctionGenerator> matrix_element(m, "MatrixElement");
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
                bool>(),
            py::arg("matrix_element_index"),
            py::arg("particle_count"),
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("diagram_count") = 1,
            py::arg("sample_random_inputs") = false
        )
        .def(
            py::init<
                const MatrixElementApi&,
                const std::vector<MatrixElement::MatrixElementInput>&,
                const std::vector<MatrixElement::MatrixElementOutput>&,
                bool>(),
            py::arg("matrix_element_api"),
            py::arg("inputs"),
            py::arg("outputs"),
            py::arg("sample_random_inputs") = false
        )
        .def("matrix_element_index", &MatrixElement::diagram_count)
        .def("diagram_count", &MatrixElement::diagram_count)
        .def("particle_count", &MatrixElement::particle_count);

    py::classh<MLP, FunctionGenerator> mlp(m, "MLP");
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
               const std::string&>(),
           py::arg("input_dim"),
           py::arg("output_dim"),
           py::arg("hidden_dim") = 32,
           py::arg("layers") = 3,
           py::arg("activation") = MLP::leaky_relu,
           py::arg("prefix") = ""
    )
        .def("input_dim", &MLP::input_dim)
        .def("output_dim", &MLP::output_dim)
        .def("initialize_globals", &MLP::initialize_globals, py::arg("context"));

    py::classh<Flow, Mapping>(m, "Flow")
        .def(
            py::init<
                std::size_t,
                std::size_t,
                const std::string&,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation,
                bool>(),
            py::arg("input_dim"),
            py::arg("condition_dim") = 0,
            py::arg("prefix") = "",
            py::arg("bin_count") = 10,
            py::arg("subnet_hidden_dim") = 32,
            py::arg("subnet_layers") = 3,
            py::arg("subnet_activation") = MLP::leaky_relu,
            py::arg("invert_spline") = true
        )
        .def("input_dim", &Flow::input_dim)
        .def("condition_dim", &Flow::condition_dim)
        .def("initialize_globals", &Flow::initialize_globals, py::arg("context"))
        .def(
            "initialize_from_vegas",
            &Flow::initialize_from_vegas,
            py::arg("context"),
            py::arg("grid_name")
        );

    py::classh<PropagatorChannelWeights, FunctionGenerator>(
        m, "PropagatorChannelWeights"
    )
        .def(
            py::init<
                const std::vector<Topology>&,
                const nested_vector3<std::size_t>&,
                const nested_vector2<std::size_t>&>(),
            py::arg("topologies"),
            py::arg("permutations"),
            py::arg("channel_indices")
        );

    py::classh<SubchannelWeights, FunctionGenerator>(m, "SubchannelWeights")
        .def(
            py::init<
                const nested_vector2<Topology>&,
                const nested_vector3<std::size_t>&,
                const nested_vector2<std::size_t>>(),
            py::arg("topologies"),
            py::arg("permutations"),
            py::arg("channel_indices")
        )
        .def("channel_count", &SubchannelWeights::channel_count);

    py::classh<MomentumPreprocessing, FunctionGenerator>(m, "MomentumPreprocessing")
        .def(py::init<std::size_t>(), py::arg("particle_count"))
        .def("output_dim", &MomentumPreprocessing::output_dim);

    py::classh<ChannelWeightNetwork, FunctionGenerator>(m, "ChannelWeightNetwork")
        .def(
            py::init<
                std::size_t,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation,
                const std::string&,
                bool>(),
            py::arg("channel_count"),
            py::arg("particle_count"),
            py::arg("hidden_dim") = 32,
            py::arg("layers") = 3,
            py::arg("activation") = MLP::leaky_relu,
            py::arg("prefix") = "",
            py::arg("include_preprocessing") = true
        )
        .def("mlp", &ChannelWeightNetwork::mlp)
        .def("preprocessing", &ChannelWeightNetwork::preprocessing)
        .def("mask_name", &ChannelWeightNetwork::mask_name)
        .def(
            "initialize_globals",
            &ChannelWeightNetwork::initialize_globals,
            py::arg("context")
        );

    py::classh<DiscreteHistogram, FunctionGenerator>(m, "DiscreteHistogram")
        .def(py::init<std::vector<std::size_t>>(), py::arg("option_counts"));

    py::classh<DiscreteSampler, Mapping>(m, "DiscreteSampler")
        .def(
            py::init<
                const std::vector<std::size_t>&,
                const std::string&,
                const std::vector<std::size_t>&>(),
            py::arg("option_counts"),
            py::arg("prefix") = "",
            py::arg("dims_with_prior") = std::vector<std::size_t>{}
        )
        .def("option_counts", &DiscreteSampler::option_counts)
        .def("prob_names", &DiscreteSampler::prob_names)
        .def(
            "initialize_globals",
            &DiscreteSampler::initialize_globals,
            py::arg("context")
        );

    py::classh<DiscreteFlow, Mapping>(m, "DiscreteFlow")
        .def(
            py::init<
                const std::vector<std::size_t>&,
                const std::string&,
                const std::vector<std::size_t>&,
                std::size_t,
                std::size_t,
                std::size_t,
                MLP::Activation>(),
            py::arg("option_counts"),
            py::arg("prefix") = "",
            py::arg("dims_with_prior") = std::vector<std::size_t>{},
            py::arg("condition_dim") = 0,
            py::arg("subnet_hidden_dim") = 32,
            py::arg("subnet_layers") = 3,
            py::arg("subnet_activation") = MLP::leaky_relu
        )
        .def("option_counts", &DiscreteFlow::option_counts)
        .def("condition_dim", &DiscreteFlow::condition_dim)
        .def(
            "initialize_globals", &DiscreteFlow::initialize_globals, py::arg("context")
        );

    py::classh<VegasGridOptimizer>(m, "VegasGridOptimizer")
        .def(
            "add_data",
            [](VegasGridOptimizer& opt, py::object values, py::object counts) {
                opt.add_data(
                    dlpack_to_tensor(values, batch_float, 0),
                    dlpack_to_tensor(counts, batch_float_array(opt.input_dim()), 1)
                );
            },
            py::arg("values"),
            py::arg("counts")
        )
        .def("optimize", &VegasGridOptimizer::optimize)
        .def(
            py::init<const std::vector<ContextPtr>&, const std::string&, double>(),
            py::arg("contexts"),
            py::arg("grid_name"),
            py::arg("damping")
        );

    py::classh<DiscreteOptimizer>(m, "DiscreteOptimizer")
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
            py::arg("values_and_counts")
        )
        .def("optimize", &DiscreteOptimizer::optimize)
        .def(
            py::init<const std::vector<ContextPtr>&, const std::vector<std::string>&>(),
            py::arg("contexts"),
            py::arg("prob_names")
        );

    py::classh<AdamOptimizer> adam(m, "AdamOptimizer");
    add_enum<AdamOptimizer::LRSchedule>(
        adam,
        "LRSchedule",
        {
            {"none", AdamOptimizer::none},
            {"cosine", AdamOptimizer::cosine},
        }
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
                double>(),
            py::arg("function"),
            py::arg("context"),
            py::arg("learning_rate"),
            py::arg("schedule") = AdamOptimizer::none,
            py::arg("step_count") = 0,
            py::arg("beta1") = 0.9,
            py::arg("beta2") = 0.999,
            py::arg("eps") = 1e-8
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
                return opt.step(tensors);
            },
            py::arg("inputs")
        )
        .def("learning_rate", &AdamOptimizer::learning_rate)
        .def("input_types", &AdamOptimizer::input_types)
        .def("context", &AdamOptimizer::context);

    py::classh<PdfGrid>(m, "PdfGrid")
        .def(py::init<const std::string&>(), py::arg("file"))
        .def_readonly("x", &PdfGrid::x)
        .def_readonly("logx", &PdfGrid::logx)
        .def_readonly("q", &PdfGrid::q)
        .def_readonly("logq2", &PdfGrid::logq2)
        .def_readonly("pids", &PdfGrid::pids)
        .def_readonly("values", &PdfGrid::values)
        .def_readonly("region_sizes", &PdfGrid::region_sizes)
        .def_property_readonly("grid_point_count", &PdfGrid::grid_point_count)
        .def_property_readonly("q_count", &PdfGrid::q_count)
        .def(
            "coefficients_shape",
            &PdfGrid::coefficients_shape,
            py::arg("batch_dim") = false
        )
        .def("logx_shape", &PdfGrid::logx_shape, py::arg("batch_dim") = false)
        .def("logq2_shape", &PdfGrid::logq2_shape, py::arg("batch_dim") = false)
        .def(
            "initialize_globals",
            &PdfGrid::initialize_globals,
            py::arg("context"),
            py::arg("prefix") = ""
        );

    py::classh<PartonDensity, FunctionGenerator>(m, "PartonDensity")
        .def(
            py::init<
                const PdfGrid&,
                const std::vector<int>&,
                bool,
                const std::string&>(),
            py::arg("grid"),
            py::arg("pids"),
            py::arg("dynamic_pid") = false,
            py::arg("prefix") = ""
        );

    py::classh<AlphaSGrid>(m, "AlphaSGrid")
        .def(py::init<const std::string&>(), py::arg("file"))
        .def_readonly("q", &AlphaSGrid::q)
        .def_readonly("logq2", &AlphaSGrid::logq2)
        .def_readonly("values", &AlphaSGrid::values)
        .def_readonly("region_sizes", &AlphaSGrid::region_sizes)
        .def_property_readonly("q_count", &AlphaSGrid::q_count)
        .def(
            "coefficients_shape",
            &AlphaSGrid::coefficients_shape,
            py::arg("batch_dim") = false
        )
        .def("logq2_shape", &AlphaSGrid::logq2_shape, py::arg("batch_dim") = false)
        .def(
            "initialize_globals",
            &AlphaSGrid::initialize_globals,
            py::arg("context"),
            py::arg("prefix") = ""
        );

    py::classh<RunningCoupling, FunctionGenerator>(m, "RunningCoupling")
        .def(
            py::init<const AlphaSGrid&, const std::string&>(),
            py::arg("grid"),
            py::arg("prefix") = ""
        );

    py::classh<EnergyScale, FunctionGenerator> scale(m, "EnergyScale");
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
    scale.def(py::init<std::size_t>(), py::arg("particle_count"))
        .def(
            py::init<std::size_t, EnergyScale::DynamicalScaleType>(),
            py::arg("particle_count"),
            py::arg("type")
        )
        .def(
            py::init<std::size_t, double>(),
            py::arg("particle_count"),
            py::arg("fixed_scale")
        )
        .def(
            py::init<
                std::size_t,
                EnergyScale::DynamicalScaleType,
                bool,
                bool,
                double,
                double,
                double>(),
            py::arg("particle_count"),
            py::arg("dynamical_scale_type"),
            py::arg("ren_scale_fixed"),
            py::arg("fact_scale_fixed"),
            py::arg("ren_scale"),
            py::arg("fact_scale1"),
            py::arg("fact_scale2")
        );

    py::classh<DifferentialCrossSection::CachedPdf>(m, "CachedPdf").def(py::init<>());
    py::classh<DifferentialCrossSection::CachedScale>(m, "CachedScale")
        .def(py::init<>());
    py::classh<DifferentialCrossSection, FunctionGenerator>(
        m, "DifferentialCrossSection"
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
                bool>(),
            py::arg("matrix_element"),
            py::arg("cm_energy"),
            py::arg("running_coupling"),
            py::arg("energy_scale"),
            py::arg("pid_options") = nested_vector2<me_int_t>{},
            py::arg("pdf1") = std::monostate{},
            py::arg("pdf2") = std::monostate{},
            py::arg("input_momentum_fraction") = true
        )
        .def("pid_options", &DifferentialCrossSection::pid_options)
        .def("matrix_element", &DifferentialCrossSection::matrix_element);

    py::classh<Unweighter, FunctionGenerator>(m, "Unweighter")
        .def(py::init<const NamedVector<Type>&>(), py::arg("types"));
    py::classh<Integrand, FunctionGenerator>(m, "Integrand")
        .def(
            py::init<
                const PhaseSpaceMapping&,
                const DifferentialCrossSection&,
                const Integrand::AdaptiveMapping&,
                const Integrand::AdaptiveDiscrete&,
                const Integrand::AdaptiveDiscrete&,
                const std::optional<PdfGrid>&,
                const std::optional<RunningCoupling>&,
                const std::optional<EnergyScale>&,
                const std::optional<PropagatorChannelWeights>&,
                const std::optional<SubchannelWeights>&,
                const std::optional<ChannelWeightNetwork>&,
                const std::vector<me_int_t>&,
                std::size_t,
                bool,
                bool,
                bool,
                const std::vector<std::size_t>&,
                const std::vector<std::size_t>&,
                const std::vector<std::size_t>&,
                const std::vector<double>&,
                const std::vector<bool>&>(),
            py::arg("mapping"),
            py::arg("diff_xs"),
            py::arg("adaptive_map") = std::monostate{},
            py::arg("discrete_before") = std::monostate{},
            py::arg("discrete_after") = std::monostate{},
            py::arg("pdf_grid") = std::nullopt,
            py::arg("running_coupling") = std::nullopt,
            py::arg("energy_scale") = std::nullopt,
            py::arg("prop_chan_weights") = std::nullopt,
            py::arg("subchan_weights") = std::nullopt,
            py::arg("chan_weight_net") = std::nullopt,
            py::arg("chan_weight_remap") = std::vector<me_int_t>{},
            py::arg("remapped_chan_count") = 0,
            py::arg("madnis_training") = false,
            py::arg("drop_cuts_and_rescale") = false,
            py::arg("partial_weights") = false,
            py::arg("channel_indices") = std::vector<std::size_t>{},
            py::arg("active_flavors") = std::vector<std::size_t>{},
            py::arg("flavor_remap") = std::vector<std::size_t>{},
            py::arg("flavor_factors") = std::vector<double>{},
            py::arg("flavor_mirror") = std::vector<bool>{}
        )
        .def("particle_count", &Integrand::particle_count)
        .def("madnis_training", &Integrand::madnis_training)
        .def("vegas_grid_name", &Integrand::vegas_grid_name)
        .def("mapping", &Integrand::mapping)
        .def("diff_xs", &Integrand::diff_xs)
        .def("adaptive_map", &Integrand::adaptive_map)
        .def("discrete_before", &Integrand::discrete_before)
        .def("discrete_after", &Integrand::discrete_after)
        .def("energy_scale", &Integrand::energy_scale)
        .def("prop_chan_weights", &Integrand::prop_chan_weights)
        .def("chan_weight_net", &Integrand::chan_weight_net)
        .def("random_dim", &Integrand::random_dim)
        .def("latent_dims", &Integrand::latent_dims)
        .def_readonly_static("matrix_element_inputs", &Integrand::matrix_element_inputs)
        .def_readonly_static(
            "matrix_element_outputs", &Integrand::matrix_element_outputs
        );
    py::classh<MultiChannelIntegrand, FunctionGenerator>(m, "MultiChannelIntegrand")
        .def(
            py::init<const std::vector<std::shared_ptr<Integrand>>&, bool>(),
            py::arg("integrands"),
            py::arg("return_sizes") = false
        );
    py::classh<IntegrandProbability, FunctionGenerator>(m, "IntegrandProbability")
        .def(py::init<const Integrand&>(), py::arg("integrand"));

    py::classh<MadnisLoss, FunctionGenerator>(m, "MadnisLoss")
        .def(
            py::init<
                const std::vector<std::shared_ptr<FunctionGenerator>>&,
                const std::optional<ChannelWeightNetwork>&,
                double>(),
            py::arg("functions"),
            py::arg("cwnet"),
            py::arg("softclip_threshold") = 0.0
        );

    add_enum<Verbosity>(
        m,
        "Verbosity",
        {
            {"silent", Verbosity::silent},
            {"log", Verbosity::log},
            {"pretty", Verbosity::pretty},
        }
    );

    py::classh<MadnisTraining::Config>(m, "MadnisConfig")
        .def(py::init<>())
        .def_readwrite("verbosity", &MadnisTraining::Config::verbosity)
        .def_readwrite("learning_rate", &MadnisTraining::Config::learning_rate)
        .def_readwrite("batches", &MadnisTraining::Config::batches)
        .def_readwrite("log_interval", &MadnisTraining::Config::log_interval)
        .def_readwrite(
            "integration_history_length",
            &MadnisTraining::Config::integration_history_length
        )
        .def_readwrite(
            "channel_dropping_interval",
            &MadnisTraining::Config::channel_dropping_interval
        )
        .def_readwrite(
            "channel_dropping_threshold",
            &MadnisTraining::Config::channel_dropping_threshold
        )
        .def_readwrite(
            "cpu_generator_batch_size",
            &MadnisTraining::Config::cpu_generator_batch_size
        )
        .def_readwrite(
            "gpu_generator_batch_size",
            &MadnisTraining::Config::gpu_generator_batch_size
        )
        .def_readwrite(
            "gpu_generator_batch_granularity",
            &MadnisTraining::Config::gpu_generator_batch_granularity
        )
        .def_readwrite(
            "generator_target_size_factor",
            &MadnisTraining::Config::generator_target_size_factor
        )
        .def_readwrite("batch_size_offset", &MadnisTraining::Config::batch_size_offset)
        .def_readwrite(
            "batch_size_per_channel", &MadnisTraining::Config::batch_size_per_channel
        )
        .def_readwrite(
            "uniform_channel_ratio", &MadnisTraining::Config::uniform_channel_ratio
        )
        .def_readwrite("lr_schedule", &MadnisTraining::Config::lr_schedule)
        .def_readwrite("adam_beta1", &MadnisTraining::Config::adam_beta1)
        .def_readwrite("adam_beta2", &MadnisTraining::Config::adam_beta2)
        .def_readwrite("adam_eps", &MadnisTraining::Config::adam_eps)
        .def_readwrite("buffer_capacity", &MadnisTraining::Config::buffer_capacity)
        .def_readwrite(
            "minimum_buffer_size", &MadnisTraining::Config::minimum_buffer_size
        )
        .def_readwrite("buffered_steps", &MadnisTraining::Config::buffered_steps)
        .def_readwrite(
            "buffer_unweighting_quantile",
            &MadnisTraining::Config::buffer_unweighting_quantile
        )
        .def_readwrite(
            "fixed_cwnet_fraction", &MadnisTraining::Config::fixed_cwnet_fraction
        )
        .def_readwrite(
            "softclip_threshold", &MadnisTraining::Config::softclip_threshold
        );

    py::classh<MadnisTraining>(m, "MadnisTraining")
        .def(
            py::init<
                ContextPtr,
                ContextPtr,
                const MadnisTraining::Config&,
                const std::vector<std::shared_ptr<Integrand>>&,
                const std::optional<ChannelWeightNetwork>&>(),
            py::arg("generator_context"),
            py::arg("optimizer_context"),
            py::arg("config"),
            py::arg("integrands"),
            py::arg("cwnet")
        )
        .def("train_step", &MadnisTraining::train_step, py::arg("batch_index"))
        .def("active_channels", &MadnisTraining::active_channels)
        .def("active_channel_count", &MadnisTraining::active_channel_count);

    py::classh<MultiMadnisTraining>(m, "MultiMadnisTraining")
        .def(
            py::init<
                ContextPtr,
                ContextPtr,
                const MadnisTraining::Config&,
                const nested_vector2<std::shared_ptr<Integrand>>&,
                const std::vector<std::optional<ChannelWeightNetwork>>&>(),
            py::arg("generator_context"),
            py::arg("optimizer_context"),
            py::arg("config"),
            py::arg("integrands"),
            py::arg("cwnets")
        )
        .def("train", &MultiMadnisTraining::train)
        .def("active_channels", &MultiMadnisTraining::active_channels);

    py::classh<GeneratorConfig>(m, "GeneratorConfig")
        .def(py::init<>())
        .def_readwrite("target_count", &GeneratorConfig::target_count)
        .def_readwrite("vegas_damping", &GeneratorConfig::vegas_damping)
        .def_readwrite(
            "max_overweight_truncation", &GeneratorConfig::max_overweight_truncation
        )
        .def_readwrite(
            "freeze_max_weight_after", &GeneratorConfig::freeze_max_weight_after
        )
        .def_readwrite("start_batch_size", &GeneratorConfig::start_batch_size)
        .def_readwrite("max_batch_size", &GeneratorConfig::max_batch_size)
        .def_readwrite("survey_min_iters", &GeneratorConfig::survey_min_iters)
        .def_readwrite("survey_max_iters", &GeneratorConfig::survey_max_iters)
        .def_readwrite(
            "survey_target_precision", &GeneratorConfig::survey_target_precision
        )
        .def_readwrite("optimization_patience", &GeneratorConfig::optimization_patience)
        .def_readwrite(
            "optimization_threshold", &GeneratorConfig::optimization_threshold
        )
        .def_readwrite("cpu_batch_size", &GeneratorConfig::cpu_batch_size)
        .def_readwrite("gpu_batch_size", &GeneratorConfig::gpu_batch_size)
        .def_readwrite("verbosity", &GeneratorConfig::verbosity)
        .def_readwrite("write_live_data", &GeneratorConfig::write_live_data)
        .def_readwrite("combine_thread_count", &GeneratorConfig::combine_thread_count)
        .def_readwrite(
            "cut_efficiency_threshold", &GeneratorConfig::cut_efficiency_threshold
        )
        .def_readwrite("max_cut_repetitions", &GeneratorConfig::max_cut_repetitions);

    py::classh<GeneratorStatus>(m, "GeneratorStatus")
        .def(py::init<>())
        .def_readwrite("subprocess", &GeneratorStatus::subprocess)
        .def_readwrite("name", &GeneratorStatus::name)
        .def_readwrite("mean", &GeneratorStatus::mean)
        .def_readwrite("error", &GeneratorStatus::error)
        .def_readwrite("rel_std_dev", &GeneratorStatus::rel_std_dev)
        .def_readwrite("count", &GeneratorStatus::count)
        .def_readwrite("count_opt", &GeneratorStatus::count_opt)
        .def_readwrite("count_after_cuts", &GeneratorStatus::count_after_cuts)
        .def_readwrite("count_after_cuts_opt", &GeneratorStatus::count_after_cuts_opt)
        .def_readwrite("count_unweighted", &GeneratorStatus::count_unweighted)
        .def_readwrite("count_target", &GeneratorStatus::count_target)
        .def_readwrite("iterations", &GeneratorStatus::iterations)
        .def_readwrite("optimized", &GeneratorStatus::optimized)
        .def_readwrite("done", &GeneratorStatus::done);

    py::classh<Histogram>(m, "Histogram")
        .def_readonly("name", &Histogram::name)
        .def_readonly("min", &Histogram::min)
        .def_readonly("max", &Histogram::max)
        .def_readonly("bin_values", &Histogram::bin_values)
        .def_readonly("bin_errors", &Histogram::bin_errors);

    py::classh<LHEHeader>(m, "LHEHeader")
        .def(
            py::init<std::string, std::string, bool>(),
            py::arg("name") = "",
            py::arg("content") = "",
            py::arg("escape_content") = false
        )
        .def_readwrite("name", &LHEHeader::name)
        .def_readwrite("content", &LHEHeader::content)
        .def_readwrite("escape_content", &LHEHeader::escape_content);
    py::classh<LHEProcess>(m, "LHEProcess")
        .def(
            py::init<double, double, double, int>(),
            py::arg("cross_section") = 0.,
            py::arg("cross_section_error") = 0.,
            py::arg("max-weight") = 0.,
            py::arg("process_id") = 0
        )
        .def_readwrite("cross_section", &LHEProcess::cross_section)
        .def_readwrite("cross_section_error", &LHEProcess::cross_section_error)
        .def_readwrite("max_weight", &LHEProcess::max_weight)
        .def_readwrite("process_id", &LHEProcess::process_id);
    py::classh<LHEMeta>(m, "LHEMeta")
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
        .def_readwrite("beam1_pdg_id", &LHEMeta::beam1_pdg_id)
        .def_readwrite("beam2_pdg_id", &LHEMeta::beam2_pdg_id)
        .def_readwrite("beam1_energy", &LHEMeta::beam1_energy)
        .def_readwrite("beam2_energy", &LHEMeta::beam2_energy)
        .def_readwrite("beam1_pdf_authors", &LHEMeta::beam1_pdf_authors)
        .def_readwrite("beam2_pdf_authors", &LHEMeta::beam2_pdf_authors)
        .def_readwrite("beam1_pdf_id", &LHEMeta::beam1_pdf_id)
        .def_readwrite("beam2_pdf_id", &LHEMeta::beam2_pdf_id)
        .def_readwrite("weight_mode", &LHEMeta::weight_mode)
        .def_readwrite("processes", &LHEMeta::processes)
        .def_readwrite("headers", &LHEMeta::headers);
    py::classh<LHEParticle>(m, "LHEParticle")
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
            py::arg("p_x") = 0.,
            py::arg("p_y") = 0.,
            py::arg("p_z") = 0.,
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
        .def_readwrite("pdg_id", &LHEParticle::pdg_id)
        .def_readwrite("status_code", &LHEParticle::status_code)
        .def_readwrite("mother1", &LHEParticle::mother1)
        .def_readwrite("mother2", &LHEParticle::mother2)
        .def_readwrite("color", &LHEParticle::color)
        .def_readwrite("anti_color", &LHEParticle::anti_color)
        .def_readwrite("px", &LHEParticle::px)
        .def_readwrite("py", &LHEParticle::py)
        .def_readwrite("pz", &LHEParticle::pz)
        .def_readwrite("energy", &LHEParticle::energy)
        .def_readwrite("mass", &LHEParticle::mass)
        .def_readwrite("lifetime", &LHEParticle::lifetime)
        .def_readwrite("spin", &LHEParticle::spin);
    py::classh<LHEEvent>(m, "LHEEvent")
        .def(
            py::init<int, double, double, double, double, std::vector<LHEParticle>>(),
            py::arg("process_id") = 0,
            py::arg("weight") = 0.,
            py::arg("scale") = 0.,
            py::arg("alpha_qed") = 0.,
            py::arg("alpha_qcd") = 0.,
            py::arg("particles") = std::vector<LHEParticle>{}
        )
        .def_readwrite("process_id", &LHEEvent::process_id)
        .def_readwrite("weight", &LHEEvent::weight)
        .def_readwrite("scale", &LHEEvent::scale)
        .def_readwrite("alpha_qed", &LHEEvent::alpha_qed)
        .def_readwrite("alpha_qcd", &LHEEvent::process_id)
        .def_readwrite("particles", &LHEEvent::particles);
    py::classh<LHECompleter::SubprocArgs>(m, "SubprocArgs")
        .def(
            py::init<
                int,
                std::vector<Topology>,
                nested_vector3<std::size_t>,
                nested_vector2<std::size_t>,
                nested_vector3<std::size_t>,
                nested_vector3<std::tuple<int, int>>,
                std::unordered_map<int, int>,
                nested_vector2<double>,
                nested_vector3<int>,
                std::vector<std::size_t>>(),
            py::arg("process_id") = 0,
            py::arg("topologies") = std::vector<Topology>{},
            py::arg("permutations") = nested_vector3<std::size_t>{},
            py::arg("diagram_indices") = nested_vector2<std::size_t>{},
            py::arg("diagram_color_indices") = nested_vector3<std::size_t>{},
            py::arg("color_flows") = nested_vector3<std::tuple<int, int>>{},
            py::arg("pdg_color_types") = std::unordered_map<int, int>{},
            py::arg("helicities") = nested_vector2<double>{},
            py::arg("pdg_ids") = nested_vector3<int>{},
            py::arg("matrix_flavor_indices") = std::vector<std::size_t>{}
        )
        .def_readwrite("process_id", &LHECompleter::SubprocArgs::process_id)
        .def_readwrite("topologies", &LHECompleter::SubprocArgs::topologies)
        .def_readwrite("permutations", &LHECompleter::SubprocArgs::permutations)
        .def_readwrite("diagram_indices", &LHECompleter::SubprocArgs::diagram_indices)
        .def_readwrite(
            "diagram_color_indices", &LHECompleter::SubprocArgs::diagram_color_indices
        )
        .def_readwrite("color_flows", &LHECompleter::SubprocArgs::color_flows)
        .def_readwrite("pdg_color_types", &LHECompleter::SubprocArgs::pdg_color_types)
        .def_readwrite("helicities", &LHECompleter::SubprocArgs::helicities)
        .def_readwrite("pdg_ids", &LHECompleter::SubprocArgs::pdg_ids)
        .def_readwrite(
            "matrix_flavor_indices", &LHECompleter::SubprocArgs::matrix_flavor_indices
        );
    py::classh<LHECompleter>(m, "LHECompleter")
        .def(
            py::init<const std::vector<LHECompleter::SubprocArgs>&, double>(),
            py::arg("subproc_args"),
            py::arg("bw_cutoff")
        )
        /*.def(
            "complete_event_data",
            &LHECompleter::complete_event_data,
            py::arg("event"),
            py::arg("subprocess_index"),
            py::arg("diagram_index"),
            py::arg("color_index"),
            py::arg("flavor_index"),
            py::arg("helicity_index")
        )*/
        .def("save", &LHECompleter::save, py::arg("file"))
        .def_static("load", &LHECompleter::load, py::arg("file"))
        .def_property_readonly("max_particle_count", &LHECompleter::max_particle_count);
    py::classh<LHEFileWriter>(m, "LHEFileWriter")
        .def(
            py::init<const std::string&, const LHEMeta&>(),
            py::arg("file_name"),
            py::arg("meta")
        )
        .def("write", &LHEFileWriter::write, py::arg("event"))
        .def("write_string", &LHEFileWriter::write_string, py::arg("str"));

    m.def("format_si_prefix", &format_si_prefix, py::arg("value"));
    m.def("format_with_error", &format_with_error, py::arg("value"), py::arg("error"));
    m.def("format_progress", &format_progress, py::arg("progress"), py::arg("width"));
    py::classh<PrettyBox>(m, "PrettyBox")
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
            py::arg("box_width") = 91
        )
        .def("set_row", &PrettyBox::set_row, py::arg("row"), py::arg("values"))
        .def("set_column", &PrettyBox::set_column, py::arg("column"), py::arg("values"))
        .def(
            "set_cell",
            &PrettyBox::set_cell,
            py::arg("row"),
            py::arg("column"),
            py::arg("value")
        )
        .def("print_first", &PrettyBox::print_first)
        .def("print_update", &PrettyBox::print_update)
        .def_property_readonly("line_count", &PrettyBox::line_count);

    py::classh<ChannelEventGenerator>(m, "ChannelEventGenerator")
        .def_static(
            "load",
            &ChannelEventGenerator::load,
            py::arg("channel_file"),
            py::arg("contexts"),
            py::arg("event_file"),
            py::arg("weight_file"),
            py::arg("config")
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
            py::arg("histograms")
        )
        .def("status", &ChannelEventGenerator::status)
        .def("save", &ChannelEventGenerator::save, py::arg("save"));

    py::classh<EventGenerator>(m, "EventGenerator")
        .def_readonly_static("default_config", &EventGenerator::default_config)
        .def(
            py::init<
                const std::vector<ContextPtr>&,
                const std::vector<std::shared_ptr<ChannelEventGenerator>>&,
                const std::string&,
                const GeneratorConfig&>(),
            py::arg("contexts"),
            py::arg("channels"),
            py::arg("status_file") = "",
            py::arg_v(
                "config",
                EventGenerator::default_config,
                "EventGenerator.default_config"
            )
        )
        .def("survey", &EventGenerator::survey)
        .def("generate", &EventGenerator::generate)
        .def(
            "combine_to_compact_npy",
            &EventGenerator::combine_to_compact_npy,
            py::arg("file_name")
        )
        .def(
            "combine_to_lhe_npy",
            &EventGenerator::combine_to_lhe_npy,
            py::arg("file_name"),
            py::arg("lhe_completer")
        )
        .def(
            "combine_to_lhe",
            &EventGenerator::combine_to_lhe,
            py::arg("file_name"),
            py::arg("lhe_completer")
        )
        .def("status", &EventGenerator::status)
        .def("channel_status", &EventGenerator::channel_status)
        .def("histograms", &EventGenerator::histograms)
        .def("used_globals", &EventGenerator::used_globals)
        .def("channels", &EventGenerator::channels);

    py::classh<Logger> logger(m, "Logger");
    add_enum<Logger::LogLevel>(
        logger,
        "LogLevel",
        {
            {"level_debug", Logger::level_debug},
            {"level_info", Logger::level_info},
            {"level_warning", Logger::level_warning},
            {"level_error", Logger::level_error},
        }
    );
    logger.def_static("log", &Logger::log, py::arg("level"), py::arg("message"))
        .def_static("debug", &Logger::debug, py::arg("message"))
        .def_static("info", &Logger::info, py::arg("message"))
        .def_static("warning", &Logger::warning, py::arg("message"))
        .def_static("error", &Logger::error, py::arg("message"))
        .def_static("set_log_handler", &Logger::set_log_handler, py::arg("func"));

    m.def(
        "initialize_vegas_grid",
        &initialize_vegas_grid,
        py::arg("context"),
        py::arg("grid_name")
    );
    m.def("set_lib_path", &set_lib_path, py::arg("lib_path"));
    m.def("set_simd_vector_size", &set_simd_vector_size, py::arg("vector_size"));

    auto abort_check_function = [] {
        if (PyErr_CheckSignals() != 0) {
            throw py::error_already_set();
        }
    };
    EventGenerator::set_abort_check_function(abort_check_function);
    MadnisTraining::set_abort_check_function(abort_check_function);
}
