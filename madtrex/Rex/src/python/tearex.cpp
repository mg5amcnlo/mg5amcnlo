#include "tearex.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "teaRex.h"

namespace py = pybind11;
using namespace REX;
using namespace REX::tea;

namespace {

// weightor is std::function<shared_ptr<vector<double>>(process&)>. pybind11's
// functional.h caster can't handle that return type directly: its
// shared_ptr<T> caster assumes T is a pybind11-registered class (for which it
// has a holder-based caster), not an STL container like vector<double> (which
// is instead handled by stl.h's *non*-holder vector caster) -- so
// shared_ptr<vector<double>> falls through both and fails to compile. Route
// Python callables through this plain-vector-returning signature instead,
// which pybind11 can cast natively, and adapt to the real weightor here in
// hand-written glue rather than via automatic function casting
using py_weightor = std::function<std::vector<double>(process &)>;

weightor to_weightor(py_weightor f) {
    return [f = std::move(f)](process &p) -> std::shared_ptr<std::vector<double>> {
        return std::make_shared<std::vector<double>>(f(p));
    };
}

std::vector<weightor> to_weightors(std::vector<py_weightor> fs) {
    std::vector<weightor> out;
    out.reserve(fs.size());
    for (auto &f : fs) out.push_back(to_weightor(std::move(f)));
    return out;
}

// Same rationale as py_weightor above, one level further removed: a
// deferred_weightor is std::function<weight_puller(process&)>, and
// weight_puller is itself std::function<shared_ptr<vector<double>>()> --
// two holder-vs-non-holder shared_ptr<vector<double>> casters that can't be
// bound directly. Python callables instead produce a plain vector<double>
// puller (py_weight_puller), and the outer Python callable returns *that*
// rather than a real weight_puller; both layers get adapted to the real
// deferred_weightor/weight_puller here
using py_weight_puller = std::function<std::vector<double>()>;
using py_deferred_weightor = std::function<py_weight_puller(process &)>;

deferred_weightor to_deferred_weightor(py_deferred_weightor f) {
    return [f = std::move(f)](process &p) -> weight_puller {
        py_weight_puller puller = f(p);
        return [puller = std::move(puller)]() -> std::shared_ptr<std::vector<double>> {
            return std::make_shared<std::vector<double>>(puller());
        };
    };
}

std::vector<deferred_weightor> to_deferred_weightors(std::vector<py_deferred_weightor> fs) {
    std::vector<deferred_weightor> out;
    out.reserve(fs.size());
    for (auto &f : fs) out.push_back(to_deferred_weightor(std::move(f)));
    return out;
}

} // namespace

void bind_tearex(py::module_ m) {
    m.doc() = "REX::tea (teaRex) generic event reweighting bindings";

    m.def("true_function", &true_function);
    m.def("helicity_selector", &helicity_selector, py::arg("helicity"));
    m.def("flavor_selector", &flavor_selector, py::arg("flavor"));
    m.def(
        "helicity_flavor_selector",
        &helicity_flavor_selector,
        py::arg("helicity"),
        py::arg("flavor")
    );

    // ---- procReweightor ----
    // Constructors taking an event selector are bound only in their
    // shared_ptr<eventBelongs> form, not the sibling eventBelongs-by-value
    // overloads: a Python EventBelongs object converts to shared_ptr<...>
    // transparently via classh either way, and keeping only one form avoids
    // pybind11 having two simultaneously-viable overloads for the same call
    //
    // The deferred_weightor-taking constructors are NOT bound as further
    // py::init(...) overloads alongside these: pybind11's std::function
    // caster accepts any Python callable regardless of what it returns, so
    // a single-callable or list-of-callables argument is equally "valid"
    // for both a py_weightor- and a py_deferred_weightor-shaped overload --
    // pybind11 can't tell them apart the way it can eg a callable from a
    // list, and the first-registered (eager) overload would silently
    // shadow the deferred one for every call. They're bound as separate
    // ProcReweightor.from_deferred(...) static factories below instead,
    // which sidesteps the collision entirely since they're never in the
    // same overload-resolution pool as these
    py::classh<procReweightor>(m, "ProcReweightor")
        .def(py::init<>())
        .def(
            py::init([](py_weightor f) { return procReweightor(to_weightor(std::move(f))); }),
            py::arg("reweight_function")
        )
        .def(
            py::init([](py_weightor f, std::shared_ptr<eventBelongs> sel) {
                return procReweightor(to_weightor(std::move(f)), sel);
            }),
            py::arg("reweight_function"),
            py::arg("selector")
        )
        .def(
            py::init([](std::vector<py_weightor> fs) {
                return procReweightor(to_weightors(std::move(fs)));
            }),
            py::arg("reweight_functions")
        )
        .def(
            py::init([](std::vector<py_weightor> fs, std::shared_ptr<eventBelongs> sel) {
                return procReweightor(to_weightors(std::move(fs)), sel);
            }),
            py::arg("reweight_functions"),
            py::arg("selector")
        )
        .def(
            py::init([](std::vector<py_weightor> fs,
                        std::shared_ptr<eventBelongs> sel,
                        py_weightor norm) {
                return procReweightor(
                    to_weightors(std::move(fs)), sel, to_weightor(std::move(norm))
                );
            }),
            py::arg("reweight_functions"),
            py::arg("selector"),
            py::arg("normaliser")
        )
        // ---- deferred_weightor construction (see comment above) ----
        .def_static(
            "from_deferred",
            [](py_deferred_weightor f) {
                return procReweightor(to_deferred_weightor(std::move(f)));
            },
            py::arg("reweight_function")
        )
        .def_static(
            "from_deferred",
            [](py_deferred_weightor f, std::shared_ptr<eventBelongs> sel) {
                return procReweightor(to_deferred_weightor(std::move(f)), sel);
            },
            py::arg("reweight_function"),
            py::arg("selector")
        )
        .def_static(
            "from_deferred",
            [](std::vector<py_deferred_weightor> fs) {
                return procReweightor(to_deferred_weightors(std::move(fs)));
            },
            py::arg("reweight_functions")
        )
        .def_static(
            "from_deferred",
            [](std::vector<py_deferred_weightor> fs, std::shared_ptr<eventBelongs> sel) {
                return procReweightor(to_deferred_weightors(std::move(fs)), sel);
            },
            py::arg("reweight_functions"),
            py::arg("selector")
        )
        .def_static(
            "from_deferred",
            [](std::vector<py_deferred_weightor> fs,
               std::shared_ptr<eventBelongs> sel,
               py_weightor norm) {
                return procReweightor(
                    to_deferred_weightors(std::move(fs)), sel, to_weightor(std::move(norm))
                );
            },
            py::arg("reweight_functions"),
            py::arg("selector"),
            py::arg("normaliser")
        )
        .def_readonly("normalisation", &procReweightor::normalisation)
        .def_readonly("proc", &procReweightor::proc)
        // procReweightor::verbose is a REX::verbosity, not a plain bool --
        // exposed as one anyway since that's all Python callers need; a
        // procReweightor owned by a Reweightor has its verbosity linked to
        // its owner's (see Reweightor.verbose below), so muting the latter
        // mutes this one too, but it can still be read/overridden directly
        // here (eg for a procReweightor used standalone)
        .def_property(
            "verbose",
            [](const procReweightor &self) { return bool(self.verbose); },
            [](procReweightor &self, bool v) { self.verbose = v; }
        )
        // procReweightor::backlog is now std::vector<backlog_entry>, a
        // struct holding a raw weight_puller (a C++ closure, not itself
        // pybind11-registered) plus the immediate flag -- no longer the
        // plain std::vector<std::vector<double>> stl.h could convert
        // automatically. Exposing its length is enough to let callers
        // check whether anything is pending without reaching into C++
        // closures from Python
        .def_property_readonly(
            "backlog_size", [](const procReweightor &self) { return self.backlog.size(); }
        )
        .def(
            "set_event_checker",
            py::overload_cast<eventBelongs>(&procReweightor::set_event_checker)
        )
        .def(
            "set_event_checker",
            py::overload_cast<REX::event_bool_fn>(&procReweightor::set_event_checker)
        )
        .def(
            "set_normaliser",
            [](procReweightor &self, py_weightor f) -> procReweightor & {
                return self.set_normaliser(to_weightor(std::move(f)));
            },
            py::arg("normaliser")
        )
        // ---- normalisation access besides evaluating a normaliser ----
        // calc_normalisation derives the normalisation array from a vector
        // of target weights (relative to proc's original weight_);
        // set_initial_weights does the same but also marks the
        // normalisation as explicitly set, so initialise() won't overwrite
        // it later; set_normalisation instead takes the normalisation array
        // itself, verbatim
        .def(
            "calc_normalisation", &procReweightor::calc_normalisation, py::arg("wgts")
        )
        .def(
            "set_initial_weights", &procReweightor::set_initial_weights, py::arg("weights")
        )
        .def(
            "set_normalisation", &procReweightor::set_normalisation, py::arg("normalisation")
        )
        .def(
            "set_reweight_functions",
            [](procReweightor &self, py_weightor f) -> procReweightor & {
                return self.set_reweight_functions(to_weightor(std::move(f)));
            }
        )
        .def(
            "set_reweight_functions",
            [](procReweightor &self, std::vector<py_weightor> fs) -> procReweightor & {
                return self.set_reweight_functions(to_weightors(std::move(fs)));
            }
        )
        .def(
            "add_reweight_function",
            [](procReweightor &self, py_weightor f) -> procReweightor & {
                return self.add_reweight_function(to_weightor(std::move(f)));
            },
            py::arg("rwgt")
        )
        .def(
            "set_deferred_reweight_functions",
            [](procReweightor &self, py_deferred_weightor f) -> procReweightor & {
                return self.set_deferred_reweight_functions(to_deferred_weightor(std::move(f)));
            }
        )
        .def(
            "set_deferred_reweight_functions",
            [](procReweightor &self, std::vector<py_deferred_weightor> fs) -> procReweightor & {
                return self.set_deferred_reweight_functions(to_deferred_weightors(std::move(fs))
                );
            }
        )
        .def(
            "add_deferred_reweight_function",
            [](procReweightor &self, py_deferred_weightor f) -> procReweightor & {
                return self.add_deferred_reweight_function(to_deferred_weightor(std::move(f)));
            },
            py::arg("rwgt")
        )
        .def("set_process", &procReweightor::set_process, py::arg("process"))
        .def("initialise", py::overload_cast<>(&procReweightor::initialise))
        .def(
            "initialise",
            py::overload_cast<std::shared_ptr<process>>(&procReweightor::initialise),
            py::arg("process")
        )
        .def("evaluate", py::overload_cast<>(&procReweightor::evaluate))
        .def(
            "evaluate", py::overload_cast<std::size_t>(&procReweightor::evaluate), py::arg("amp")
        )
        .def("append_zero_weights", &procReweightor::append_zero_weights)
        .def("append_backlog", &procReweightor::append_backlog)
        .def("pull_deferred_backlog", &procReweightor::pull_deferred_backlog);

    // ---- xSecAccumulator ----
    // Running-reduction accumulator standing in for
    // calc_xSec_from_weights()/calc_norm()/calc_xSecs()/calc_xErrs() when
    // driving Reweightor.run_streaming()'s per-batch loop by hand instead of
    // calling run_streaming() itself -- absorb() folds one finished batch's
    // events into a handful of running scalars, finalise_into() reproduces
    // the same xSec/xSecErr/rwgt_xSec/rwgt_xErr a full run_streaming() call
    // would set on the given Reweightor. See run_streaming's own reference
    // sequence in teaRex.cc: pull_batch() -> extract_sorter_stream() ->
    // initialise_reweightors() -> run_all_iterations() ->
    // pull_deferred_backlog() per procReweightor -> absorb(events) -> write
    // each event -> repeat per batch -> finalise_into(reweightor) once, only
    // after every batch has been absorbed
    py::classh<xSecAccumulator>(m, "XSecAccumulator")
        .def(py::init<>())
        .def_readonly("n_events", &xSecAccumulator::n_events)
        .def_readonly("weight_sum", &xSecAccumulator::weight_sum)
        .def_readonly("weight_sumSq", &xSecAccumulator::weight_sumSq)
        .def_readonly("wgt_sum", &xSecAccumulator::wgt_sum)
        .def_readonly("ratio_sum", &xSecAccumulator::ratio_sum)
        .def_readonly("ratio_sumSq", &xSecAccumulator::ratio_sumSq)
        .def("absorb", &xSecAccumulator::absorb, py::arg("events"))
        .def("finalise_into", &xSecAccumulator::finalise_into, py::arg("reweightor"));

    // ---- reweightor ----
    // Bound as a real subclass of Lhe (registered in rex.cpp, which runs
    // before this function is called). Constructor overloads taking
    // vector<procReweightor> by value are skipped in favour of the
    // shared_ptr<procReweightor> forms, same rationale as above; the
    // lhe&&-taking (move) overloads are skipped since Python has no rvalue
    // distinction -- the const lhe& forms cover every practical call
    py::classh<reweightor, lhe>(m, "Reweightor")
        .def(py::init<>())
        .def(py::init<const lhe &>(), py::arg("mother"))
        .def(
            py::init<
                const lhe &,
                std::vector<std::shared_ptr<procReweightor>>,
                std::vector<iterator>>(),
            py::arg("mother"),
            py::arg("reweightors"),
            // Must match reweightor::iterators's own default member
            // initializer ({true_function}), not an empty vector: passing an
            // explicit empty vector here would override that default and
            // leave run_all_iterations()'s `curr_iter < iterators.size()`
            // loop with nothing to iterate over
            py::arg("iterators") = std::vector<iterator>{true_function}
        )
        .def_readwrite("reweightors", &reweightor::reweightors)
        .def_readwrite("launch_names", &reweightor::launch_names)
        .def_readwrite("pool_threads", &reweightor::pool_threads)
        // Controls whether non-forced diagnostic output is emitted (eg the
        // per-iteration "." progress marker); forced warnings still get
        // through regardless
        .def_property(
            "verbose",
            [](const reweightor &self) { return bool(self.verbose); },
            [](reweightor &self, bool v) { self.verbose = v; }
        )
        .def_readonly("norm_factor", &reweightor::norm_factor)
        .def_readonly("rwgt_xSec", &reweightor::rwgt_xSec)
        .def_readonly("rwgt_xErr", &reweightor::rwgt_xErr)
        // curr_iter/n_amps are read-write: setup()/run_streaming() maintain
        // them automatically for run()/run_streaming(), but a caller driving
        // run_iteration()/run_all_iterations() by hand per streamed batch
        // (see the manual-loop half of streaming_demo() in showcase.py)
        // needs to reset curr_iter to 0 and resync n_amps (via
        // sync_n_amps() below) itself once per batch
        .def_readwrite("curr_iter", &reweightor::curr_iter)
        .def_readwrite("n_amps", &reweightor::n_amps)
        .def(
            "set_reweightors",
            py::overload_cast<std::vector<std::shared_ptr<procReweightor>>>(
                &reweightor::set_reweightors
            )
        )
        .def(
            "add_reweightor",
            py::overload_cast<std::shared_ptr<procReweightor>>(&reweightor::add_reweightor)
        )
        .def("set_initialise", &reweightor::set_initialise, py::arg("init"))
        .def("set_finalise", &reweightor::set_finalise, py::arg("fin"))
        .def("set_iterators", &reweightor::set_iterators, py::arg("iterators"))
        .def(
            "add_iterator",
            [](reweightor &self, iterator it) -> reweightor & {
                return self.add_iterator(it);
            },
            py::arg("iterator")
        )
        .def("set_launch_names", &reweightor::set_launch_names, py::arg("names"))
        .def("add_launch_name", &reweightor::add_launch_name, py::arg("name"))
        .def("extract_sorter", &reweightor::extract_sorter)
        // Per-batch counterpart to extract_sorter(), used by run_streaming()
        // (and available here for callers driving the streaming loop by
        // hand): `base` is the procReweightor list as registered before
        // streaming began (eg self.reweightors, saved off beforehand) --
        // unlike extract_sorter(), this never rebinds into `base` itself, so
        // the same list stays a valid template call after call, once per
        // batch
        .def(
            "extract_sorter_stream",
            &reweightor::extract_sorter_stream,
            py::arg("base")
        )
        .def("initialise_reweightors", &reweightor::initialise_reweightors)
        .def("finalise_reweighting", &reweightor::finalise_reweighting)
        .def("sync_n_amps", &reweightor::sync_n_amps)
        .def("setup", &reweightor::setup)
        // run_iteration()/run_all_iterations()/run() dispatch procReweightor
        // evaluation onto REX::tea::threadPool's worker threads and then
        // block the calling thread in pool->wait_batch() until they finish.
        // Since Python-defined weightor/event_bool_fn callables need the GIL
        // to run (pybind11's functional.h caster acquires it automatically
        // before calling in), leaving the GIL held on the blocked calling
        // thread would deadlock every worker against it. Release it for the
        // duration of the call so the workers can actually acquire it
        .def("run_iteration", &reweightor::run_iteration, py::call_guard<py::gil_scoped_release>())
        .def(
            "run_all_iterations",
            &reweightor::run_all_iterations,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("run", &reweightor::run, py::call_guard<py::gil_scoped_release>())
        // Streaming counterpart to run(): repeatedly pulls a batch from
        // self's source (see Lhe.open_streaming()), reweights it, writes it
        // to sink, and frees it before pulling the next -- see run_streaming's
        // C++ doc comment in teaRex.h for the full per-batch sequence. Same
        // GIL-release rationale as run_iteration()/run_all_iterations()/run()
        .def(
            "run_streaming",
            &reweightor::run_streaming,
            py::arg("sink"),
            py::arg("include_ids") = false,
            py::call_guard<py::gil_scoped_release>()
        )
        .def("calc_norm", &reweightor::calc_norm)
        .def("set_norm", &reweightor::set_norm, py::arg("norm"))
        .def("calc_xSecs", &reweightor::calc_xSecs)
        .def("calc_xErrs", &reweightor::calc_xErrs);

    // ---- param_rwgt ----
    // rwgt_slha/rwgt_card/rwgt_block stay unbound for now (mirrors slha's own
    // scope trim) -- read_slha_rwgt(slha_file, rwgt_file) is the practical
    // entry point for the SLHA-reweight-card workflow
    py::classh<param_rwgt, reweightor>(m, "ParamRwgt")
        .def(py::init<>())
        .def(py::init<const lhe &>(), py::arg("mother"))
        .def(
            py::init<const lhe &, std::vector<std::shared_ptr<procReweightor>>>(),
            py::arg("mother"),
            py::arg("reweightors")
        )
        .def(
            py::init<
                const lhe &,
                std::vector<std::shared_ptr<procReweightor>>,
                const std::string &,
                const std::string &>(),
            py::arg("mother"),
            py::arg("reweightors"),
            py::arg("slha_path"),
            py::arg("rwgt_path")
        )
        .def(
            "read_slha_rwgt",
            py::overload_cast<const std::string &, const std::string &>(
                &param_rwgt::read_slha_rwgt
            ),
            py::arg("slha_file"),
            py::arg("rwgt_file")
        );
}
