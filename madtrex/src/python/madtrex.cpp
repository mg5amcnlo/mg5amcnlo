#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "madtrex/madtrex.hpp"
#include "madtrex/pdfware.hpp"

namespace py = pybind11;
using namespace madtrex;
using madspace::LHEMeta;
using madspace::MatrixElementApi;

PYBIND11_MODULE(_madtrex_py, m) {
    m.doc() = "Python bindings for madtrex::TupperWare/WareHouse, the generic "
               "MatrixElementApi <-> REX::tea reweighting bridge";

    // madtrex is a separate extension module layered on top of madspace: the
    // MatrixElementApi/Context/LHEMeta types its bindings take and return are
    // registered by madspace's own _madspace_py, so import it first (pybind11
    // shares registered types across modules built against the same pybind11
    // version, see CMakeLists.txt) rather than binding them again here.
    py::module_::import("madspace");

    // TupperWare/WareHouse build REX::tea::procReweightor/reweightor objects and
    // consume REX::eventBelongs/event_bool_fn event checkers -- all types already
    // bound by the separately-built `pyrex` package (pyrex / pyrex.tea; named
    // "pyrex" rather than "rex" because "rex" is already an established,
    // unrelated package on the index). Import them first so those types are
    // registered in the process before any madtrex binding below references
    // them, and do NOT re-register py::classh<...> for any of them here:
    // pybind11 raises "generic_type: type ... is already registered!" the
    // moment both `pyrex` and `madtrex` are imported into the same process
    // and both sides independently declare a class binding for the same C++
    // type. Cross-module reuse (verified empirically: an independently
    // compiled procReweightor, returned from a second, unrelated extension
    // module, casts cleanly to the already-registered pyrex.tea.ProcReweightor
    // Python type) is what makes objects built here actually usable with
    // pyrex.tea.Reweightor.run() etc.
    py::module_::import("pyrex");
    py::module_::import("pyrex.tea");

    // ---- UmamiInputKey / UmamiOutputKey ----
    // Plain C enums from madspace/umami.h; not bound anywhere else in
    // madspace itself. Every value is exposed (not just the subset TupperWare can
    // auto-derive from a REX::process) since the enum also indexes
    // MatrixElementApi.supported_inputs()/required_inputs()/supported_outputs().
    py::enum_<UmamiInputKey>(m, "UmamiInputKey")
        .value("MOMENTA", UMAMI_IN_MOMENTA)
        .value("ALPHA_S", UMAMI_IN_ALPHA_S)
        .value("FLAVOR_INDEX", UMAMI_IN_FLAVOR_INDEX)
        .value("RANDOM_COLOR", UMAMI_IN_RANDOM_COLOR)
        .value("RANDOM_HELICITY", UMAMI_IN_RANDOM_HELICITY)
        .value("RANDOM_DIAGRAM", UMAMI_IN_RANDOM_DIAGRAM)
        .value("HELICITY_INDEX", UMAMI_IN_HELICITY_INDEX)
        .value("DIAGRAM_INDEX", UMAMI_IN_DIAGRAM_INDEX)
        .value("CHANNEL_INDEX", UMAMI_IN_CHANNEL_INDEX);

    py::enum_<UmamiOutputKey>(m, "UmamiOutputKey")
        .value("MATRIX_ELEMENT", UMAMI_OUT_MATRIX_ELEMENT)
        .value("DIAGRAM_AMP2", UMAMI_OUT_DIAGRAM_AMP2)
        .value("COLOR_INDEX", UMAMI_OUT_COLOR_INDEX)
        .value("HELICITY_INDEX", UMAMI_OUT_HELICITY_INDEX)
        .value("DIAGRAM_INDEX", UMAMI_OUT_DIAGRAM_INDEX)
        .value("GPU_STREAM", UMAMI_OUT_GPU_STREAM);

    // default_input_keys/default_output_keys are exposed as the documented
    // starting point for a manual add_api/set_normaliser override.
    // make_weightor() itself is intentionally not bound: it returns a raw
    // shared_ptr<REX::tea::weightor>, which -- like the plain weightor type in
    // tearex.cpp -- isn't something any bound API can consume back from Python
    // (procReweightor's Python constructors only take Python-callable
    // std::functions, not a pre-built weightor handle). TupperWare/WareHouse's
    // add_api()/build() cover the actual construction path end to end.
    m.def(
        "default_input_keys",
        &default_input_keys,
        py::arg("api"),
        py::arg("helicity_reweighting") = false
    );
    m.def("default_output_keys", &default_output_keys, py::arg("api"));

    // ---- ReloadSpec ----
    // Plain data: how TupperWare::add_api()/set_normaliser() can reload an
    // independent MatrixElementApi instance per WareHouse.set_max_batch()
    // overflow batch instead of every batch sharing the one instance it was
    // registered with -- see WareHouse.set_max_batch() below. Driver builds
    // and attaches one of these automatically for every api it loads, so
    // Python code driving reweighting through Driver never needs to touch
    // this directly; it's exposed for TupperWare/WareHouse used standalone.
    py::classh<ReloadSpec>(m, "ReloadSpec")
        .def(
            py::init<madspace::ContextPtr, std::string, std::string>(),
            py::arg("context"),
            py::arg("me_path"),
            py::arg("param_card")
        )
        .def_readwrite("context", &ReloadSpec::context)
        .def_readwrite("me_path", &ReloadSpec::me_path)
        .def_readwrite("param_card", &ReloadSpec::param_card);

    // ---- TupperWare ----
    // set_event_checker is bound only in its shared_ptr<eventBelongs> form, not
    // the sibling by-value eventBelongs overload: a Python EventBelongs object
    // converts to shared_ptr<...> transparently via classh either way, and
    // pybind11 can't otherwise distinguish which of the two overloads a given
    // EventBelongs argument was meant for (same rationale as
    // procReweightor's selector argument in Rex's own tearex.cpp). The
    // convenience constructor taking a plain eventBelongs has no shared_ptr
    // sibling to conflict with, so it's bound as-is.
    //
    // Lifetimes: a TupperWare only keeps raw pointers to its MatrixElementApi
    // objects, so every method taking an api (or a list of them) keeps that
    // argument alive as long as the TupperWare (keep_alive<1, 2>; a Python
    // MatrixElementApi in turn keeps its madspace Context, which owns the
    // loaded library, alive). The builder methods return *this
    // (reference_internal), so tw.add_api(a).set_normaliser(b) configures tw
    // itself rather than a detached copy.
    py::classh<TupperWare>(m, "TupperWare")
        .def(py::init<>())
        .def(
            py::init<const MatrixElementApi&, REX::eventBelongs>(),
            py::arg("api"),
            py::arg("checker"),
            py::keep_alive<1, 2>()
        )
        .def(
            py::init<const MatrixElementApi&, REX::event_bool_fn>(),
            py::arg("api"),
            py::arg("checker"),
            py::keep_alive<1, 2>()
        )
        .def(
            "add_api",
            py::overload_cast<const MatrixElementApi&>(&TupperWare::add_api),
            py::arg("api"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::add_api),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<const MatrixElementApi&, ReloadSpec>(&TupperWare::add_api),
            py::arg("api"),
            py::arg("reload"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>,
                ReloadSpec>(&TupperWare::add_api),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::arg("reload"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<std::vector<const MatrixElementApi*>>(&TupperWare::add_api),
            py::arg("apis"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::add_api),
            py::arg("apis"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<std::vector<UmamiInputKey>>,
                std::vector<std::vector<UmamiOutputKey>>>(&TupperWare::add_api),
            py::arg("apis"),
            py::arg("input_keys_list"),
            py::arg("output_keys_list"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_api",
            py::overload_cast<std::vector<const MatrixElementApi*>>(&TupperWare::set_api),
            py::arg("apis"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::set_api),
            py::arg("apis"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_api",
            py::overload_cast<
                std::vector<const MatrixElementApi*>,
                std::vector<std::vector<UmamiInputKey>>,
                std::vector<std::vector<UmamiOutputKey>>>(&TupperWare::set_api),
            py::arg("apis"),
            py::arg("input_keys_list"),
            py::arg("output_keys_list"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_normaliser",
            py::overload_cast<const MatrixElementApi&>(&TupperWare::set_normaliser),
            py::arg("api"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_normaliser",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>>(&TupperWare::set_normaliser),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_normaliser",
            py::overload_cast<const MatrixElementApi&, ReloadSpec>(&TupperWare::set_normaliser),
            py::arg("api"),
            py::arg("reload"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_normaliser",
            py::overload_cast<
                const MatrixElementApi&,
                std::vector<UmamiInputKey>,
                std::vector<UmamiOutputKey>,
                ReloadSpec>(&TupperWare::set_normaliser),
            py::arg("api"),
            py::arg("input_keys"),
            py::arg("output_keys"),
            py::arg("reload"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def("set_result_keys", &TupperWare::set_result_keys, py::arg("keys"), py::return_value_policy::reference_internal)
        .def(
            "set_helicity_reweighting",
            &TupperWare::set_helicity_reweighting,
            py::arg("helicity_reweighting"),
            py::return_value_policy::reference_internal
        )
        .def("helicity_reweighting", &TupperWare::helicity_reweighting)
        .def(
            "set_stored_normalisation",
            &TupperWare::set_stored_normalisation,
            py::arg("key"),
            py::return_value_policy::reference_internal
        )
        .def("stored_normalisation", &TupperWare::stored_normalisation)
        .def("set_use_helicity", &TupperWare::set_use_helicity, py::arg("use_helicity"), py::return_value_policy::reference_internal)
        .def(
            "set_event_checker",
            py::overload_cast<std::shared_ptr<REX::eventBelongs>>(
                &TupperWare::set_event_checker
            ),
            py::arg("checker"),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_event_checker",
            py::overload_cast<REX::event_bool_fn>(&TupperWare::set_event_checker),
            py::arg("checker"),
            py::return_value_policy::reference_internal
        )
        .def(
            "build",
            &TupperWare::build,
            py::arg("n_iterations") = 1,
            py::arg("reload_per_batch") = false,
            py::arg("max_deferred_iterations") = REX::npos
        );

    // ---- WareHouse ----
    // Move-only (copy ctor/assignment deleted in C++, mirroring
    // REX::tea::reweightor's own single-owner semantics); py::classh's
    // smart_holder handles that without needing a copy constructor. The
    // lhe&&-taking build() overload is skipped, same rationale as Reweightor's
    // own constructor in tearex.cpp: Python has no rvalue distinction, and the
    // const lhe& overload covers every practical call (it just wraps its
    // argument in a local REX::lhe copy before delegating).
    py::classh<WareHouse>(m, "WareHouse")
        .def(py::init<>())
        .def(
            "add_process",
            py::overload_cast<TupperWare>(&WareHouse::add_process),
            py::arg("process"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_process",
            py::overload_cast<std::vector<TupperWare>>(&WareHouse::add_process),
            py::arg("processes"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_process",
            py::overload_cast<>(&WareHouse::add_process),
            py::return_value_policy::reference_internal
        )
        .def("set_processes", &WareHouse::set_processes, py::arg("processes"), py::keep_alive<1, 2>(), py::return_value_policy::reference_internal)
        .def(
            "process",
            py::overload_cast<std::size_t>(&WareHouse::process),
            py::arg("index"),
            py::return_value_policy::reference_internal
        )
        .def("processes", py::overload_cast<>(&WareHouse::processes))
        .def("set_iterators", &WareHouse::set_iterators, py::arg("iterators"), py::return_value_policy::reference_internal)
        .def("add_iterator", &WareHouse::add_iterator, py::arg("iterator"), py::return_value_policy::reference_internal)
        .def("set_initialise", &WareHouse::set_initialise, py::arg("init"), py::return_value_policy::reference_internal)
        .def("set_finalise", &WareHouse::set_finalise, py::arg("fin"), py::return_value_policy::reference_internal)
        .def("set_launch_names", &WareHouse::set_launch_names, py::arg("names"), py::return_value_policy::reference_internal)
        .def("add_launch_name", &WareHouse::add_launch_name, py::arg("name"), py::return_value_policy::reference_internal)
        .def("launch_names", &WareHouse::launch_names)
        .def("set_max_batch", &WareHouse::set_max_batch, py::arg("max_batch"), py::return_value_policy::reference_internal)
        .def("max_batch", &WareHouse::max_batch)
        .def("set_threads", &WareHouse::set_threads, py::arg("threads"), py::return_value_policy::reference_internal)
        .def("threads", &WareHouse::threads)
        .def("set_min_batch", &WareHouse::set_min_batch, py::arg("min_batch"), py::return_value_policy::reference_internal)
        .def("min_batch", &WareHouse::min_batch)
        .def("batch_size_for", &WareHouse::batch_size_for, py::arg("n_events"))
        .def(
            "set_max_deferred_iterations",
            &WareHouse::set_max_deferred_iterations,
            py::arg("max_deferred_iterations"),
            py::return_value_policy::reference_internal
        )
        .def("max_deferred_iterations", &WareHouse::max_deferred_iterations)
        .def("set_verbose", &WareHouse::set_verbose, py::arg("v"), py::return_value_policy::reference_internal)
        .def("is_verbose", &WareHouse::is_verbose)
        .def(
            "build",
            py::overload_cast<const REX::lhe&>(&WareHouse::build),
            py::arg("mother"),
            py::return_value_policy::reference_internal
        )
        .def(
            "get",
            py::overload_cast<>(&WareHouse::get),
            py::return_value_policy::reference_internal
        )
        .def("built", &WareHouse::built);

    // ---- LHE loading/writing, LheFormat ----
    // detect_lhe_format()/load_lhe() sniff a path's leading bytes (NumPy magic
    // vs LHEF XML) rather than trusting the extension; see madtrex.hpp/
    // binary_io.hpp. The std::istream-taking load_lhe_xml() and
    // madspace::EventFile-taking load_lhe_binary() overloads aren't bound --
    // Python callers have a path string, not an already-open C++ stream/file
    // handle.
    py::enum_<LheFormat>(m, "LheFormat")
        .value("XML", LheFormat::xml)
        .value("BINARY", LheFormat::binary);

    m.def("detect_lhe_format", &detect_lhe_format, py::arg("path"));
    m.def(
        "load_lhe_xml",
        py::overload_cast<const std::string&>(&load_lhe_xml),
        py::arg("path")
    );
    m.def("load_lhe", &load_lhe, py::arg("path"));
    m.def(
        "load_lhe_binary",
        py::overload_cast<const std::string&, const LHEMeta&>(&load_lhe_binary),
        py::arg("path"),
        py::arg("meta") = LHEMeta{}
    );
    m.def(
        "save_weights_binary", &save_weights_binary, py::arg("lhe"), py::arg("path_prefix")
    );

    // ---- Splitting a job across independent shards, and recombining them ----
    // The generic REX::split_source() itself is intentionally not bound
    // (would need REX::eventSource/eventSink exposed to Python too, which
    // pyrex doesn't currently do); split_events()/combine_shards() are the
    // two madtrex-level entry points that actually matter for driving this
    // workflow from Python -- see madtrex.hpp for the full picture.
    py::classh<SplitResult>(m, "SplitResult")
        .def_readonly("paths", &SplitResult::paths)
        .def_readonly("counts", &SplitResult::counts);

    m.def(
        "split_events",
        &split_events,
        py::arg("path"),
        py::arg("output_prefix"),
        py::arg("n_shards"),
        py::arg("particle_count") = 0,
        py::arg("batch_size") = 10000,
        py::arg("meta") = LHEMeta{}
    );
    m.def(
        "combine_shards",
        &combine_shards,
        py::arg("shard_event_paths"),
        py::arg("shard_weight_prefixes"),
        py::arg("launch_names"),
        py::arg("output_path"),
        py::arg("binary_output") = false,
        py::arg("meta") = LHEMeta{}
    );

    // ---- SubProcessSpec / FlavorChannel ----
    // Read-only introspection views: both are produced by
    // Driver.load_process()/load_process_directory() parsing a
    // subprocesses.json (see madtrex.hpp), not meant to be built from
    // scratch in Python.
    py::classh<FlavorChannel>(m, "FlavorChannel")
        .def_readonly("index", &FlavorChannel::index)
        .def_readonly("pdgs", &FlavorChannel::pdgs)
        .def_readonly("mirrored", &FlavorChannel::mirrored);

    py::classh<SubProcessSpec>(m, "SubProcessSpec")
        .def_readonly("incoming", &SubProcessSpec::incoming)
        .def_readonly("outgoing", &SubProcessSpec::outgoing)
        .def_readonly("me_path_template", &SubProcessSpec::me_path_template)
        .def_readonly("source_path", &SubProcessSpec::source_path)
        .def_readonly("diagram_count", &SubProcessSpec::diagram_count)
        .def_readonly("flavors", &SubProcessSpec::flavors)
        .def_readonly("helicities", &SubProcessSpec::helicities);

    // ---- ParamHandler ----
    // Builder-style methods are bound with reference_internal (as
    // TupperWare's and WareHouse's above): ParamHandler is meant to be
    // configured incrementally and its identity matters once its iterators()
    // are handed to a WareHouse, so returning *this by reference rather than
    // a detached copy keeps chaining behave like the C++ side.
    py::classh<ParamHandler>(m, "ParamHandler")
        .def(py::init<>())
        .def(
            "add_api",
            &ParamHandler::add_api,
            py::arg("api"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_apis",
            py::overload_cast<const std::vector<const MatrixElementApi*>&>(
                &ParamHandler::add_apis
            ),
            py::arg("apis"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_apis",
            py::overload_cast<const WareHouse&>(&ParamHandler::add_apis),
            py::arg("house"),
            py::keep_alive<1, 2>(),
            py::return_value_policy::reference_internal
        )
        .def("apis", &ParamHandler::apis)
        .def(
            "read_rwgt_card",
            py::overload_cast<const std::string&>(&ParamHandler::read_rwgt_card),
            py::arg("rwgt_path"),
            py::return_value_policy::reference_internal
        )
        .def("launch_count", &ParamHandler::launch_count)
        .def("apply_launch", &ParamHandler::apply_launch, py::arg("idx"))
        .def("iterators", &ParamHandler::iterators)
        .def("launch_names", &ParamHandler::launch_names)
        .def("weight_context", &ParamHandler::weight_context);

    // ---- Driver ----
    // Driver is not copy-constructible (it owns a WareHouse, which isn't
    // copyable either), so every method below that returns Driver& must use
    // reference_internal explicitly -- the default policy would otherwise
    // try to copy-construct the return value and fail to compile.
    py::classh<Driver>(m, "Driver")
        .def(py::init<>())
        .def(py::init<madspace::ContextPtr>(), py::arg("context"))
        .def(
            "set_device_priority",
            &Driver::set_device_priority,
            py::arg("priority"),
            py::return_value_policy::reference_internal
        )
        .def("device_priority", &Driver::device_priority)
        .def(
            "set_fptype_priority",
            &Driver::set_fptype_priority,
            py::arg("priority"),
            py::return_value_policy::reference_internal
        )
        .def("fptype_priority", &Driver::fptype_priority)
        .def(
            "set_normaliser_fptype_priority",
            &Driver::set_normaliser_fptype_priority,
            py::arg("priority"),
            py::return_value_policy::reference_internal
        )
        .def("normaliser_fptype_priority", &Driver::normaliser_fptype_priority)
        .def(
            "set_helicity_reweighting",
            &Driver::set_helicity_reweighting,
            py::arg("helicity_reweighting"),
            py::return_value_policy::reference_internal
        )
        .def("helicity_reweighting", &Driver::helicity_reweighting)
        .def(
            "set_normaliser_process_directory",
            &Driver::set_normaliser_process_directory,
            py::arg("process_directory"),
            py::arg("param_card") = "",
            py::return_value_policy::reference_internal
        )
        .def("normaliser_process_directory", &Driver::normaliser_process_directory)
        .def(
            "set_normaliser_param_card",
            &Driver::set_normaliser_param_card,
            py::arg("param_card"),
            py::return_value_policy::reference_internal
        )
        .def("normaliser_param_card", &Driver::normaliser_param_card)
        .def(
            "set_use_stored_normalisation",
            &Driver::set_use_stored_normalisation,
            py::arg("use_stored_normalisation"),
            py::return_value_policy::reference_internal
        )
        .def("use_stored_normalisation", &Driver::use_stored_normalisation)
        .def(
            "set_use_helicity",
            &Driver::set_use_helicity,
            py::arg("use_helicity"),
            py::return_value_policy::reference_internal
        )
        .def("use_helicity", &Driver::use_helicity)
        .def(
            "set_verbose",
            &Driver::set_verbose,
            py::arg("v"),
            py::return_value_policy::reference_internal
        )
        .def("is_verbose", &Driver::is_verbose)
        .def(
            "set_max_batch",
            &Driver::set_max_batch,
            py::arg("max_batch"),
            py::return_value_policy::reference_internal
        )
        .def("max_batch", &Driver::max_batch)
        .def(
            "set_threads",
            &Driver::set_threads,
            py::arg("threads"),
            py::return_value_policy::reference_internal
        )
        .def("threads", &Driver::threads)
        .def(
            "set_min_batch",
            &Driver::set_min_batch,
            py::arg("min_batch"),
            py::return_value_policy::reference_internal
        )
        .def("min_batch", &Driver::min_batch)
        .def(
            "set_max_deferred_iterations",
            &Driver::set_max_deferred_iterations,
            py::arg("max_deferred_iterations"),
            py::return_value_policy::reference_internal
        )
        .def("max_deferred_iterations", &Driver::max_deferred_iterations)
        .def(
            "set_streaming",
            &Driver::set_streaming,
            py::arg("enable"),
            py::arg("batch_size") = 10000,
            py::return_value_policy::reference_internal
        )
        .def("streaming", &Driver::streaming)
        .def("stream_batch_size", &Driver::stream_batch_size)
        .def(
            "load_process",
            &Driver::load_process,
            py::arg("subprocesses_json_path"),
            py::arg("param_card") = "",
            py::return_value_policy::reference_internal
        )
        .def(
            "load_process_directory",
            &Driver::load_process_directory,
            py::arg("process_directory"),
            py::arg("param_card") = "",
            py::return_value_policy::reference_internal
        )
        .def(
            "load_param_reweighting",
            &Driver::load_param_reweighting,
            py::arg("rwgt_path"),
            py::return_value_policy::reference_internal
        )
        .def(
            "param_handler",
            py::overload_cast<>(&Driver::param_handler),
            py::return_value_policy::reference_internal
        )
        .def(
            "load_events",
            &Driver::load_events,
            py::arg("path"),
            py::arg("meta") = LHEMeta{},
            py::return_value_policy::reference_internal
        )
        .def("binary_output", &Driver::binary_output)
        .def(
            "set_binary_output",
            &Driver::set_binary_output,
            py::arg("binary"),
            py::return_value_policy::reference_internal
        )
        .def(
            "write_weights",
            &Driver::write_weights,
            py::arg("path"),
            py::return_value_policy::reference_internal
        )
        .def(
            "run_streaming",
            &Driver::run_streaming,
            py::arg("output_path"),
            py::arg("include_ids") = false,
            py::return_value_policy::reference_internal
        )
        .def(
            "reweight",
            &Driver::reweight,
            py::arg("process_directory"),
            py::arg("events_path"),
            py::arg("output_path"),
            py::arg("rwgt_path") = "",
            py::arg("param_card") = "",
            py::arg("meta") = LHEMeta{},
            py::return_value_policy::reference_internal
        )
        .def("context", &Driver::context)
        .def(
            "warehouse",
            py::overload_cast<>(&Driver::warehouse),
            py::return_value_policy::reference_internal
        )
        .def("specs", &Driver::specs);

    // ---- ScaleChoice ----
    // Either an absolute scale (GeV) or a multiplicative factor of an
    // event's own recorded muF/muR -- see its own doc comment in
    // pdfware.hpp. A plain Python float passed where a ScaleChoice is
    // expected (set_baseline()/add_hypothesis()/add_scale_variation_envelope()
    // below) is implicitly treated as ScaleChoice.factor(...), matching the
    // C++ side's own non-explicit single-double constructor -- the
    // pre-existing, default behaviour.
    py::classh<ScaleChoice> scale_choice(m, "ScaleChoice");
    scale_choice
        .def(py::init<>())
        .def(py::init<double>(), py::arg("value"))
        .def(py::init<double, bool>(), py::arg("value"), py::arg("absolute"))
        .def_readwrite("value", &ScaleChoice::value)
        .def_readwrite("absolute", &ScaleChoice::absolute)
        .def_static("factor", &ScaleChoice::factor, py::arg("value"))
        .def_static("fixed", &ScaleChoice::fixed, py::arg("value"))
        .def("scaled_by", &ScaleChoice::scaled_by, py::arg("extra_factor"))
        .def("resolve", &ScaleChoice::resolve, py::arg("nominal"));
    py::implicitly_convertible<double, ScaleChoice>();
    py::implicitly_convertible<int, ScaleChoice>();

    // ---- ScaleVariationOutput ----
    py::enum_<ScaleVariationOutput>(m, "ScaleVariationOutput")
        .value("NONE", ScaleVariationOutput::none)
        .value("ENVELOPE", ScaleVariationOutput::envelope)
        .value("ALL", ScaleVariationOutput::all);

    // ---- PdfDriver ----
    // PdfDriver builds/runs a single REX::tea::reweightor doing PDF-only (not
    // matrix-element) reweighting -- see pdfware.hpp. PdfSet/PdfHypothesis/
    // make_pdf_weightor are internal building blocks (same "not bound" reasoning
    // as make_weightor above -- load_pdf_set()/set_baseline()/add_hypothesis()
    // cover construction end to end from Python) and aren't exposed here.
    // PdfDriver isn't copy-constructible either (it owns a
    // unique_ptr<REX::tea::reweightor>), so every PdfDriver&-returning method
    // uses reference_internal, same as Driver above.
    py::classh<PdfDriver>(m, "PdfDriver")
        .def(py::init<>())
        .def(py::init<madspace::ContextPtr>(), py::arg("context"))
        .def(
            "load_pdf_set",
            &PdfDriver::load_pdf_set,
            py::arg("name"),
            py::arg("grid_file"),
            py::arg("alpha_s_grid_file") = "",
            py::arg("pids") = PdfDriver::default_pids(),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_pdf_set",
            &PdfDriver::add_pdf_set,
            py::arg("name"),
            py::arg("grid"),
            py::arg("alpha_s_grid") = nullptr,
            py::arg("pids") = PdfDriver::default_pids(),
            py::arg("existing_globals_prefix") = "",
            py::return_value_policy::reference_internal
        )
        .def(
            "set_alpha_s_power",
            &PdfDriver::set_alpha_s_power,
            py::arg("power"),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_baseline",
            py::overload_cast<const std::string&, ScaleChoice, ScaleChoice>(
                &PdfDriver::set_baseline),
            py::arg("pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_baseline",
            py::overload_cast<const std::string&, const std::string&, ScaleChoice, ScaleChoice>(
                &PdfDriver::set_baseline),
            py::arg("beam1_pdf_set_name"),
            py::arg("beam2_pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_hypothesis",
            py::overload_cast<const std::string&, const std::string&, ScaleChoice, ScaleChoice>(
                &PdfDriver::add_hypothesis),
            py::arg("launch_name"),
            py::arg("pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_hypothesis",
            py::overload_cast<
                const std::string&, const std::string&, const std::string&, ScaleChoice,
                ScaleChoice>(&PdfDriver::add_hypothesis),
            py::arg("launch_name"),
            py::arg("beam1_pdf_set_name"),
            py::arg("beam2_pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_scale_variation_envelope",
            py::overload_cast<const std::string&, const std::string&, ScaleChoice, ScaleChoice>(
                &PdfDriver::add_scale_variation_envelope),
            py::arg("group_name"),
            py::arg("pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "add_scale_variation_envelope",
            py::overload_cast<
                const std::string&, const std::string&, const std::string&, ScaleChoice,
                ScaleChoice>(&PdfDriver::add_scale_variation_envelope),
            py::arg("group_name"),
            py::arg("beam1_pdf_set_name"),
            py::arg("beam2_pdf_set_name"),
            py::arg("muF_scale") = ScaleChoice::factor(1.0),
            py::arg("muR_scale") = ScaleChoice::factor(1.0),
            py::return_value_policy::reference_internal
        )
        .def(
            "set_scale_variation_output",
            &PdfDriver::set_scale_variation_output,
            py::arg("mode"),
            py::return_value_policy::reference_internal
        )
        .def("scale_variation_output", &PdfDriver::scale_variation_output)
        .def(
            "set_verbose",
            &PdfDriver::set_verbose,
            py::arg("v"),
            py::return_value_policy::reference_internal
        )
        .def("is_verbose", &PdfDriver::is_verbose)
        .def(
            "set_beam_energies",
            &PdfDriver::set_beam_energies,
            py::arg("beam1_energy"),
            py::arg("beam2_energy"),
            py::return_value_policy::reference_internal
        )
        .def(
            "load_events",
            &PdfDriver::load_events,
            py::arg("path"),
            py::arg("meta") = LHEMeta{},
            py::return_value_policy::reference_internal
        )
        .def("binary_output", &PdfDriver::binary_output)
        .def(
            "set_binary_output",
            &PdfDriver::set_binary_output,
            py::arg("binary"),
            py::return_value_policy::reference_internal
        )
        .def(
            "write_weights",
            &PdfDriver::write_weights,
            py::arg("path"),
            py::return_value_policy::reference_internal
        )
        .def(
            "keep_weights",
            &PdfDriver::keep_weights,
            py::arg("keep_names"),
            py::return_value_policy::reference_internal
        )
        .def(
            "reweight",
            &PdfDriver::reweight,
            py::arg("events_path"),
            py::arg("output_path"),
            py::arg("meta") = LHEMeta{},
            py::return_value_policy::reference_internal
        )
        .def("context", &PdfDriver::context)
        .def(
            "get",
            py::overload_cast<>(&PdfDriver::get),
            py::return_value_policy::reference_internal
        )
        .def("built", &PdfDriver::built)
        .def_static("default_pids", &PdfDriver::default_pids);
}
