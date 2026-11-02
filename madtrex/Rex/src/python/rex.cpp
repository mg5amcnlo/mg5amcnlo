#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Rex.h"
#include "tearex.hpp"

#include <sstream>

namespace py = pybind11;
using namespace REX;

namespace {

// ---- arrN<T,N> / vecArrN<T,N> <-> plain Python lists ----
//
// Rex's fixed-size arrN<T,N> (a single momentum/mother/icol tuple) and
// vecArrN<T,N> (a flat, strided vector of those) aren't STL types, so
// pybind11/stl.h doesn't know how to convert them. Rather than writing a
// generic type_caster, bind them as ordinary list-valued properties: read
// copies out to a list, write rebuilds the arrN/vecArrN from a list

template <typename T, std::size_t N>
std::vector<T> arrN_to_vec(const arrN<T, N> &a) {
    std::vector<T> v(N);
    for (std::size_t i = 0; i < N; ++i) v[i] = a[i];
    return v;
}

template <typename T, std::size_t N>
arrN<T, N> vec_to_arrN(const std::vector<T> &v) {
    if (v.size() != N) {
        throw std::invalid_argument(
            "expected a sequence of length " + std::to_string(N) + ", got "
            + std::to_string(v.size())
        );
    }
    arrN<T, N> a;
    for (std::size_t i = 0; i < N; ++i) a[i] = v[i];
    return a;
}

template <typename T, std::size_t N>
std::vector<std::vector<T>> vecArrN_to_nested(const vecArrN<T, N> &v) {
    std::vector<std::vector<T>> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i].resize(N);
        for (std::size_t j = 0; j < N; ++j) out[i][j] = v[i][j];
    }
    return out;
}

template <typename T, std::size_t N>
vecArrN<T, N> nested_to_vecArrN(const std::vector<std::vector<T>> &in) {
    vecArrN<T, N> out(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i].size() != N) {
            throw std::invalid_argument(
                "expected inner sequences of length " + std::to_string(N)
            );
        }
        for (std::size_t j = 0; j < N; ++j) out[i][j] = in[i][j];
    }
    return out;
}

template <typename T, std::size_t N, typename C>
void bind_arrN_property(py::classh<C> &cls, const char *name, arrN<T, N> C::*field) {
    cls.def_property(
        name,
        [field](const C &self) { return arrN_to_vec(self.*field); },
        [field](C &self, const std::vector<T> &v) { self.*field = vec_to_arrN<T, N>(v); }
    );
}

template <typename T, std::size_t N, typename C>
void bind_vecArrN_property(py::classh<C> &cls, const char *name, vecArrN<T, N> C::*field) {
    cls.def_property(
        name,
        [field](const C &self) { return vecArrN_to_nested(self.*field); },
        [field](C &self, const std::vector<std::vector<T>> &v) {
            self.*field = nested_to_vecArrN<T, N>(v);
        }
    );
}

// ---- arrNRef<T,N> <-> plain Python lists ----
//
// event::particle/const_particle expose their momentum/mother/icol fields
// as arrNRef<T,N> -- a raw-pointer proxy into the *owning event's* own
// storage vectors (not an owning array like arrN<T,N> above), so writing
// through it writes straight back into that event. Converts the same way:
// read copies out to a list, write copies a list's values back through the
// proxy (never rebinding what it points at)
template <typename T, std::size_t N>
std::vector<std::remove_const_t<T>> arrNRef_to_vec(const arrNRef<T, N> &a) {
    std::vector<std::remove_const_t<T>> v(N);
    for (std::size_t i = 0; i < N; ++i) v[i] = a[i];
    return v;
}

template <typename T, std::size_t N>
void vec_to_arrNRef(arrNRef<T, N> &a, const std::vector<std::remove_const_t<T>> &v) {
    if (v.size() != N) {
        throw std::invalid_argument(
            "expected a sequence of length " + std::to_string(N) + ", got "
            + std::to_string(v.size())
        );
    }
    for (std::size_t i = 0; i < N; ++i) a[i] = v[i];
}

// ---- index-based iterator helpers for Event/EventView ----
//
// event::particle_iterator/event::event_view::iterator only implement
// operator!=, not operator==, but pybind11::make_iterator (this version)
// needs operator== to detect the end. Wrapping plain indices here (rather
// than the library's own iterator types) sidesteps that without changing
// Rex.h's public iterator API
struct event_particle_iter {
    event *evt;
    std::size_t i;
    event::particle operator*() const { return evt->get_particle(i); }
    event_particle_iter &operator++() {
        ++i;
        return *this;
    }
    bool operator==(const event_particle_iter &o) const { return i == o.i; }
    bool operator!=(const event_particle_iter &o) const { return i != o.i; }
};

struct event_view_particle_iter {
    event::event_view *view;
    std::size_t i;
    event::particle operator*() const { return (*view)[i]; }
    event_view_particle_iter &operator++() {
        ++i;
        return *this;
    }
    bool operator==(const event_view_particle_iter &o) const { return i == o.i; }
    bool operator!=(const event_view_particle_iter &o) const { return i != o.i; }
};

} // namespace

PYBIND11_MODULE(_rex_py, m) {
    m.doc() = "Python bindings for the Rex/teaRex LHE parsing and reweighting libraries";
    // Generic "not found"/"unset" sentinel used eg by Lhe.max_batch (unset)
    // EventSorter.position()/sort() (no match), and Lhe.process_group (catch-all)
    m.attr("npos") = npos;

    // ---- parton ----
    auto parton_cls =
        py::classh<parton>(m, "Parton")
            .def(py::init<>())
            .def_readwrite("mass", &parton::mass_)
            .def_readwrite("vtim", &parton::vtim_)
            .def_readwrite("spin", &parton::spin_)
            .def_readwrite("pdg", &parton::pdg_)
            .def_readwrite("status", &parton::status_)
            .def_property_readonly("pT", &parton::pT)
            .def_property_readonly("pT2", &parton::pT2)
            .def_property_readonly("pL", &parton::pL)
            .def_property_readonly("pL2", &parton::pL2)
            .def_property_readonly("eT", &parton::eT)
            .def_property_readonly("eT2", &parton::eT2)
            .def_property_readonly("phi", &parton::phi)
            .def_property_readonly("theta", &parton::theta)
            .def_property_readonly("eta", &parton::eta)
            .def_property_readonly("rap", &parton::rap)
            .def_property_readonly("mT", &parton::mT)
            .def_property_readonly("mT2", &parton::mT2)
            .def_property_readonly("m2", &parton::m2);
    bind_arrN_property(parton_cls, "momentum", &parton::momenta_);
    bind_arrN_property(parton_cls, "mother", &parton::mother_);
    bind_arrN_property(parton_cls, "icol", &parton::icol_);

    // ---- event::particle / event::const_particle ----
    // Live views into the *owning event's* own storage vectors (arrNRef
    // members are raw-pointer proxies, not copies) -- so writing to a
    // Particle's fields writes straight back into the Event it came from.
    // Bound before Event itself since Event's get_particle/__getitem__/
    // __iter__/view() return these. keep_alive<0, 1>() on every factory
    // that returns one (see below) ties the returned view's lifetime to
    // its parent Event, so the proxied vectors can't be freed out from
    // under it
    py::classh<event::particle>(m, "Particle")
        .def_property(
            "momentum",
            [](const event::particle &p) { return arrNRef_to_vec(p.momentum_); },
            [](event::particle &p, const std::vector<double> &v) {
                vec_to_arrNRef(p.momentum_, v);
            }
        )
        .def_property(
            "mass",
            [](const event::particle &p) { return p.mass_; },
            [](event::particle &p, double v) { p.mass_ = v; }
        )
        .def_property(
            "vtim",
            [](const event::particle &p) { return p.vtim_; },
            [](event::particle &p, double v) { p.vtim_ = v; }
        )
        .def_property(
            "spin",
            [](const event::particle &p) { return p.spin_; },
            [](event::particle &p, double v) { p.spin_ = v; }
        )
        .def_property(
            "pdg",
            [](const event::particle &p) { return p.pdg_; },
            [](event::particle &p, long int v) { p.pdg_ = v; }
        )
        .def_property(
            "status",
            [](const event::particle &p) { return p.status_; },
            [](event::particle &p, short int v) { p.status_ = v; }
        )
        .def_property(
            "mother",
            [](const event::particle &p) { return arrNRef_to_vec(p.mother_); },
            [](event::particle &p, const std::vector<short int> &v) {
                vec_to_arrNRef(p.mother_, v);
            }
        )
        .def_property(
            "icol",
            [](const event::particle &p) { return arrNRef_to_vec(p.icol_); },
            [](event::particle &p, const std::vector<short int> &v) {
                vec_to_arrNRef(p.icol_, v);
            }
        )
        .def_property_readonly("pT", &event::particle::pT)
        .def_property_readonly("pT2", &event::particle::pT2)
        .def_property_readonly("pL", &event::particle::pL)
        .def_property_readonly("pL2", &event::particle::pL2)
        .def_property_readonly("eT", &event::particle::eT)
        .def_property_readonly("eT2", &event::particle::eT2)
        .def_property_readonly("phi", &event::particle::phi)
        .def_property_readonly("theta", &event::particle::theta)
        .def_property_readonly("eta", &event::particle::eta)
        .def_property_readonly("rap", &event::particle::rap)
        .def_property_readonly("mT", &event::particle::mT)
        .def_property_readonly("mT2", &event::particle::mT2)
        .def_property_readonly("m2", &event::particle::m2)
        .def("__str__", [](const event::particle &p) {
            std::ostringstream ss;
            p.print(ss);
            return ss.str();
        });

    py::classh<event::const_particle>(m, "ConstParticle")
        .def_property_readonly(
            "momentum", [](const event::const_particle &p) { return arrNRef_to_vec(p.momentum_); }
        )
        .def_property_readonly("mass", [](const event::const_particle &p) { return p.mass_; })
        .def_property_readonly("vtim", [](const event::const_particle &p) { return p.vtim_; })
        .def_property_readonly("spin", [](const event::const_particle &p) { return p.spin_; })
        .def_property_readonly("pdg", [](const event::const_particle &p) { return p.pdg_; })
        .def_property_readonly("status", [](const event::const_particle &p) { return p.status_; })
        .def_property_readonly(
            "mother", [](const event::const_particle &p) { return arrNRef_to_vec(p.mother_); }
        )
        .def_property_readonly(
            "icol", [](const event::const_particle &p) { return arrNRef_to_vec(p.icol_); }
        )
        .def_property_readonly("pT", &event::const_particle::pT)
        .def_property_readonly("pT2", &event::const_particle::pT2)
        .def_property_readonly("pL", &event::const_particle::pL)
        .def_property_readonly("pL2", &event::const_particle::pL2)
        .def_property_readonly("eT", &event::const_particle::eT)
        .def_property_readonly("eT2", &event::const_particle::eT2)
        .def_property_readonly("phi", &event::const_particle::phi)
        .def_property_readonly("theta", &event::const_particle::theta)
        .def_property_readonly("eta", &event::const_particle::eta)
        .def_property_readonly("rap", &event::const_particle::rap)
        .def_property_readonly("mT", &event::const_particle::mT)
        .def_property_readonly("mT2", &event::const_particle::mT2)
        .def_property_readonly("m2", &event::const_particle::m2)
        .def("__str__", [](const event::const_particle &p) {
            std::ostringstream ss;
            p.print(ss);
            return ss.str();
        });

    // ---- event::event_view ----
    // A reordered, read/write view over an event's particles according to
    // its `indices` (see event.set_indices()/view() below), without
    // touching the underlying storage vectors
    py::classh<event::event_view>(m, "EventView")
        .def("__len__", &event::event_view::size)
        .def(
            "__getitem__",
            [](event::event_view &self, std::size_t i) { return self[i]; },
            py::keep_alive<0, 1>()
        )
        .def(
            "__iter__",
            [](event::event_view &self) {
                return py::make_iterator(
                    event_view_particle_iter{&self, 0},
                    event_view_particle_iter{&self, self.size()}
                );
            },
            py::keep_alive<0, 1>()
        );

    // ---- event ----
    auto event_cls =
        py::classh<event>(m, "Event")
            .def(py::init<>())
            .def(py::init<std::size_t>(), py::arg("n_particles"))
            .def_readwrite("n", &event::n_)
            .def_readwrite("proc_id", &event::proc_id_)
            .def_readwrite("weight", &event::weight_)
            .def_readwrite("scale", &event::scale_)
            .def_readwrite("muF", &event::muF_)
            .def_readwrite("muR", &event::muR_)
            .def_readwrite("muPS", &event::muPS_)
            .def_readwrite("alphaEW", &event::alphaEW_)
            .def_readwrite("alphaS", &event::alphaS_)
            .def_readwrite("mass", &event::mass_)
            .def_readwrite("vtim", &event::vtim_)
            .def_readwrite("spin", &event::spin_)
            .def_readwrite("pdg", &event::pdg_)
            .def_readwrite("status", &event::status_)
            .def_readwrite("wgts", &event::wgts_)
            .def_readwrite("helicity", &event::helicity_)
            .def_readwrite("flavor", &event::flavor_)
            .def_readwrite("indices", &event::indices)
            .def(
                "add_particle",
                py::overload_cast<const parton &>(&event::add_particle)
            )
            .def("size", &event::size)
            .def("__len__", &event::size)
            .def(
                "__getitem__",
                [](event &self, std::size_t i) { return self.get_particle(i); },
                py::arg("index"),
                py::keep_alive<0, 1>()
            )
            .def(
                "__iter__",
                [](event &self) {
                    return py::make_iterator(
                        event_particle_iter{&self, 0}, event_particle_iter{&self, self.size()}
                    );
                },
                py::keep_alive<0, 1>()
            )
            .def(
                "get_particle",
                [](event &self, std::size_t i) { return self.get_particle(i); },
                py::arg("index"),
                py::keep_alive<0, 1>()
            )
            .def(
                "view", [](event &self) { return self.view(); }, py::keep_alive<0, 1>()
            )
            .def("set_indices", py::overload_cast<>(&event::set_indices))
            .def(
                "set_indices",
                py::overload_cast<const event &, bool>(&event::set_indices),
                py::arg("other"),
                py::arg("fail_on_mismatch") = false
            )
            .def(
                "set_indices",
                py::overload_cast<const std::vector<std::size_t> &>(&event::set_indices),
                py::arg("indices")
            )
            .def("get_muF", &event::get_muF)
            .def("get_muR", &event::get_muR)
            .def("get_muPS", &event::get_muPS)
            .def("__eq__", [](const event &a, const event &b) { return a == b; })
            .def("__ne__", [](const event &a, const event &b) { return a != b; })
            .def("validate", &event::validate);
    bind_vecArrN_property(event_cls, "momenta", &event::momenta_);
    bind_vecArrN_property(event_cls, "mother", &event::mother_);
    bind_vecArrN_property(event_cls, "icol", &event::icol_);
    m.def("default_event_equal", &default_event_equal, py::arg("lhs"), py::arg("rhs"));
    // set_event_comparator's only overload takes cevent_equal_fn -- unlike
    // eg eventBelongs/eventSorter's constructor pairs, there's no sibling
    // event_equal_fn overload of the same name to collide with, so this is
    // bindable directly despite the const-vs-mutable comparator ambiguity 
    m.def("set_event_comparator", &set_event_comparator, py::arg("comparator"));
    m.def("reset_event_comparator", &reset_event_comparator);

    // ---- eventComparatorConfig ----
    py::classh<eventComparatorConfig>(m, "EventComparatorConfig")
        .def(py::init<>())
        .def_readwrite("status_filter", &eventComparatorConfig::status_filter)
        .def_readwrite("compare_momentum", &eventComparatorConfig::compare_momentum)
        .def_readwrite("compare_momentum_x", &eventComparatorConfig::compare_momentum_x)
        .def_readwrite("compare_momentum_y", &eventComparatorConfig::compare_momentum_y)
        .def_readwrite("compare_momentum_z", &eventComparatorConfig::compare_momentum_z)
        .def_readwrite("compare_momentum_E", &eventComparatorConfig::compare_momentum_E)
        .def_readwrite("compare_mass", &eventComparatorConfig::compare_mass)
        .def_readwrite("compare_vtim", &eventComparatorConfig::compare_vtim)
        .def_readwrite("compare_spin", &eventComparatorConfig::compare_spin)
        .def_readwrite("compare_pdg", &eventComparatorConfig::compare_pdg)
        .def_readwrite("compare_status", &eventComparatorConfig::compare_status)
        .def_readwrite("compare_mother", &eventComparatorConfig::compare_mother)
        .def_readwrite("compare_icol", &eventComparatorConfig::compare_icol)
        .def_readwrite("compare_n", &eventComparatorConfig::compare_n)
        .def_readwrite("compare_proc_id", &eventComparatorConfig::compare_proc_id)
        .def_readwrite("compare_weight", &eventComparatorConfig::compare_weight)
        .def_readwrite("compare_scale", &eventComparatorConfig::compare_scale)
        .def_readwrite("compare_alphaEW", &eventComparatorConfig::compare_alphaEW)
        .def_readwrite("compare_alphaS", &eventComparatorConfig::compare_alphaS)
        .def_readwrite("mass_tol", &eventComparatorConfig::mass_tol)
        .def_readwrite("vtim_tol", &eventComparatorConfig::vtim_tol)
        .def_readwrite("spin_tol", &eventComparatorConfig::spin_tol)
        .def_readwrite("momentum_tol", &eventComparatorConfig::momentum_tol)
        .def_readwrite("weight_tol", &eventComparatorConfig::weight_tol)
        .def_readwrite("scale_tol", &eventComparatorConfig::scale_tol)
        .def_readwrite("alphaEW_tol", &eventComparatorConfig::alphaEW_tol)
        .def_readwrite("alphaS_tol", &eventComparatorConfig::alphaS_tol);
    m.def("compare_legs_only", &compare_legs_only);
    m.def("compare_final_state_only", &compare_final_state_only);
    m.def("compare_physics_fields", &compare_physics_fields);

    // ---- eventBelongs ----
    // Only the mutable-event (event_equal_fn) comparator overload is bound,
    // not its cevent_equal_fn twin: pybind11 can't distinguish a Python
    // callable meant for one std::function signature from another of the
    // same arity, so binding both would leave the second unreachable. The
    // mutable path is also what belongs(event&)/__call__ actually use
    py::classh<eventBelongs>(m, "EventBelongs")
        .def(py::init<>())
        .def(py::init<const event &>(), py::arg("event"))
        .def(py::init<std::vector<event>>(), py::arg("events"))
        .def_readwrite("events", &eventBelongs::events)
        .def(
            "add_event", py::overload_cast<const event &>(&eventBelongs::add_event)
        )
        .def(
            "add_event",
            py::overload_cast<const std::vector<event> &>(&eventBelongs::add_event)
        )
        .def(
            "set_events", py::overload_cast<const event &>(&eventBelongs::set_events)
        )
        .def(
            "set_events",
            py::overload_cast<const std::vector<event> &>(&eventBelongs::set_events)
        )
        .def(
            "set_comparator",
            py::overload_cast<event_equal_fn>(&eventBelongs::set_comparator)
        )
        .def(
            "set_comparator",
            py::overload_cast<const eventComparatorConfig &>(&eventBelongs::set_comparator)
        )
        .def("belongs", py::overload_cast<event &>(&eventBelongs::belongs))
        .def("__call__", py::overload_cast<event &>(&eventBelongs::operator()));
    m.def("all_events_belong", &all_events_belong);
    m.def("external_legs_comparator", &external_legs_comparator);
    m.def("always_true", &always_true);

    // ---- eventSorter ----
    // Same mutable-vs-const ambiguity as eventBelongs applies to its
    // event_bool_fn/cevent_bool_fn constructor twins; only event_bool_fn is
    // bound. Unlike that documented omission, the single-vs-vector forms
    // below are ALL bound in pairs (constructor, add_bool, set_event_sets,
    // set_bools), matching add_event_set's existing single+vector pair
    py::classh<eventSorter>(m, "EventSorter")
        .def(py::init<>())
        .def(py::init<const eventBelongs &>(), py::arg("event_set"))
        .def(py::init<std::vector<eventBelongs>>(), py::arg("event_sets"))
        .def(py::init<event_bool_fn>(), py::arg("comparator"))
        .def(py::init<std::vector<event_bool_fn>>(), py::arg("comparators"))
        .def(
            "add_event_set",
            py::overload_cast<const eventBelongs &>(&eventSorter::add_event_set)
        )
        .def(
            "add_event_set",
            py::overload_cast<const std::vector<eventBelongs> &>(&eventSorter::add_event_set)
        )
        .def("add_bool", py::overload_cast<event_bool_fn>(&eventSorter::add_bool))
        .def(
            "add_bool",
            py::overload_cast<std::vector<event_bool_fn>>(&eventSorter::add_bool)
        )
        .def(
            "set_event_sets",
            py::overload_cast<const eventBelongs &>(&eventSorter::set_event_sets)
        )
        .def(
            "set_event_sets",
            py::overload_cast<const std::vector<eventBelongs> &>(&eventSorter::set_event_sets)
        )
        .def(
            "set_bools", py::overload_cast<const event_bool_fn>(&eventSorter::set_bools)
        )
        .def(
            "set_bools",
            py::overload_cast<const std::vector<event_bool_fn>>(&eventSorter::set_bools)
        )
        .def("size", &eventSorter::size)
        .def("position", py::overload_cast<event &>(&eventSorter::position))
        .def(
            "position",
            py::overload_cast<std::vector<event> &>(&eventSorter::position)
        )
        .def("sort", py::overload_cast<std::vector<event> &>(&eventSorter::sort));
    m.def(
        "make_sample_sorter",
        py::overload_cast<const std::vector<event> &, event_equal_fn>(&make_sample_sorter),
        py::arg("sample"),
        py::arg("comparator") = event_equal_fn(external_legs_comparator)
    );
    m.def(
        "make_sample_sorter",
        py::overload_cast<std::vector<std::shared_ptr<event>>, event_equal_fn>(
            &make_sample_sorter
        ),
        py::arg("sample"),
        py::arg("comparator") = event_equal_fn(external_legs_comparator)
    );

    // ---- process ----
    auto process_cls =
        py::classh<process>(m, "Process")
            .def(
                py::init<std::vector<std::shared_ptr<event>>, bool, bool>(),
                py::arg("events"),
                py::arg("filter_partons") = false,
                py::arg("column_major") = false
            )
            .def_readwrite("n", &process::n_)
            .def_readonly("n_summed", &process::n_summed)
            .def_readwrite("proc_id", &process::proc_id_)
            .def_readwrite("weight", &process::weight_)
            .def_readwrite("scale", &process::scale_)
            .def_readwrite("muF", &process::muF_)
            .def_readwrite("muR", &process::muR_)
            .def_readwrite("muPS", &process::muPS_)
            .def_readwrite("alphaEW", &process::alphaEW_)
            .def_readwrite("alphaS", &process::alphaS_)
            .def_readwrite("mass", &process::mass_)
            .def_readwrite("vtim", &process::vtim_)
            .def_readwrite("spin", &process::spin_)
            .def_readwrite("pdg", &process::pdg_)
            .def_readwrite("status", &process::status_)
            .def_readwrite("wgts", &process::wgts_)
            .def_readwrite("helicity", &process::helicity_)
            .def_readwrite("flavor", &process::flavor_)
            .def_readwrite("column_major", &process::column_major)
            .def_readwrite("filter", &process::filter)
            .def_readonly("umami_momenta", &process::umami_momenta_)
            .def_readonly("umami_mass", &process::umami_mass_)
            .def_readonly("umami_vtim", &process::umami_vtim_)
            .def_readonly("umami_spin", &process::umami_spin_)
            .def_readonly("umami_pdg", &process::umami_pdg_)
            .def_readonly("umami_status", &process::umami_status_)
            .def_readonly("umami_mother", &process::umami_mother_)
            .def_readonly("umami_icol", &process::umami_icol_)
            .def_readwrite("events", &process::events)
            .def("add_event", py::overload_cast<const event &>(&process::add_event))
            .def(
                "add_event",
                py::overload_cast<const std::vector<event> &>(&process::add_event)
            )
            .def(
                "add_event_raw", py::overload_cast<const event &>(&process::add_event_raw)
            )
            .def(
                "add_event_filtered",
                py::overload_cast<const event &>(&process::add_event_filtered)
            )
            .def(
                "add_event_umami_filtered",
                py::overload_cast<const std::vector<event> &>(
                    &process::add_event_umami_filtered
                )
            )
            .def("to_umami", &process::to_umami)
            .def("from_umami", &process::from_umami)
            .def("validate", &process::validate)
            .def("transpose", &process::transpose)
            // Unlike the other 25+ transpose_*() partials transpose_wgts()
            // isn't just a redundant narrower transpose(): transpose()
            // rebuilds process::events as brand-new Event objects, while
            // transpose_wgts() writes wgts_ into the *existing* Event
            // objects in place -- the ones a Reweightor's own .events (and
            // an xSecAccumulator reading through it) still points to.
            // Needed for callers driving Reweightor.run_streaming()'s loop
            // by hand
            .def("transpose_wgts", &process::transpose_wgts)
            .def("size", &process::size)
            .def("__len__", &process::size)
            .def("gS", &process::gS)
            .def("set_gS", &process::set_gS)
            .def("get_muF", &process::get_muF)
            .def("get_muR", &process::get_muR)
            .def("get_muPS", &process::get_muPS)
            .def("append_wgts", &process::append_wgts, py::arg("wgts"))
            // Specific momentum components
            .def("E", &process::E)
            .def("x", &process::x)
            .def("y", &process::y)
            .def("z", &process::z)
            .def("set_E", &process::set_E, py::arg("E"))
            .def("set_x", &process::set_x, py::arg("x"))
            .def("set_y", &process::set_y, py::arg("y"))
            .def("set_z", &process::set_z, py::arg("z"));
    bind_vecArrN_property(process_cls, "momenta", &process::momenta_);
    bind_vecArrN_property(process_cls, "mother", &process::mother_);
    bind_vecArrN_property(process_cls, "icol", &process::icol_);

    // ---- initNode ----
    auto initnode_cls =
        py::classh<initNode>(m, "InitNode")
            .def(py::init<>())
            .def_readwrite("idWgt", &initNode::idWgt_)
            .def_readwrite("nProc", &initNode::nProc_)
            .def_readwrite("xSec", &initNode::xSec_)
            .def_readwrite("xSecErr", &initNode::xSecErr_)
            .def_readwrite("xMax", &initNode::xMax_)
            .def_readwrite("lProc", &initNode::lProc_)
            .def("validate_init", &initNode::validate_init);
    bind_arrN_property(initnode_cls, "idBm", &initNode::idBm_);
    bind_arrN_property(initnode_cls, "eBm", &initNode::eBm_);
    bind_arrN_property(initnode_cls, "pdfG", &initNode::pdfG_);
    bind_arrN_property(initnode_cls, "pdfS", &initNode::pdfS_);

    // weight_norm's `verbose` parameter is left unbound (defaults to
    // nullptr, ie no warning printed)
    m.def(
        "weight_norm",
        [](const initNode &init, const std::vector<std::shared_ptr<event>> &evts) {
            return weight_norm(init, evts);
        },
        py::arg("init"),
        py::arg("events")
    );

    // ---- histogram::entry ----
    py::classh<histogram::entry>(m, "HistogramEntry")
        .def(py::init<>())
        .def_readwrite("evt", &histogram::entry::evt)
        .def_readwrite("index", &histogram::entry::index)
        .def_readwrite("base_weight", &histogram::entry::base_weight);

    // ---- histogram ----
    // As with EventSorter above, only the explicit-indices/EventSorter/
    // event_hash_fn overloads are bound for the constructors and
    // add_event(s) -- the EventBelongs/event_bool_fn/cevent_bool_fn/vector
    // forms in Rex.h are pure convenience wrappers that build an
    // EventSorter internally, so nothing is lost by asking Python callers
    // to do that EventSorter(...) wrapping themselves; skipping them also
    // sidesteps the same event_bool_fn/cevent_bool_fn ambiguity noted for
    // EventSorter's own bindings
    //
    // nominal/mult/n_wgts_synced/entries are derived/internal state --
    // rebuilt by rebuild()/sync_wgts(), not meant to be hand-edited --
    // so, like ProcReweightor.normalisation/proc above, they're exposed
    // read-only. `norm` is likewise read-only: mutating it has the side
    // effect of forcing a full re-sync (see the C++ comment on
    // histogram::set_norm()), which a plain attribute write can't trigger;
    // use set_norm()/normalise() instead. n_bins stays read-write, same as
    // in C++, since setting it has no side effect until the next
    // rebuild()/sync_wgts() call
    py::classh<histogram>(m, "Histogram")
        .def(py::init<>())
        .def(
            py::init<const std::vector<std::shared_ptr<event>> &, const std::vector<std::size_t> &>(),
            py::arg("events"),
            py::arg("indices")
        )
        .def(
            py::init<const std::vector<std::shared_ptr<event>> &, eventSorter>(),
            py::arg("events"),
            py::arg("sorter")
        )
        .def(
            py::init<const std::vector<std::shared_ptr<event>> &, event_hash_fn>(),
            py::arg("events"),
            py::arg("hash")
        )
        .def_readonly("entries", &histogram::entries)
        .def_readonly("nominal", &histogram::nominal)
        .def_readonly("mult", &histogram::mult)
        .def_readonly("n_wgts_synced", &histogram::n_wgts_synced)
        .def_readonly("norm", &histogram::norm)
        .def_readwrite("n_bins", &histogram::n_bins)
        .def(
            "add_event",
            py::overload_cast<std::shared_ptr<event>, std::size_t>(&histogram::add_event),
            py::arg("event"),
            py::arg("index")
        )
        .def(
            "add_event",
            py::overload_cast<std::shared_ptr<event>, eventSorter>(&histogram::add_event),
            py::arg("event"),
            py::arg("sorter")
        )
        .def(
            "add_event",
            py::overload_cast<std::shared_ptr<event>, event_hash_fn>(&histogram::add_event),
            py::arg("event"),
            py::arg("hash")
        )
        .def(
            "add_events",
            py::overload_cast<const std::vector<std::shared_ptr<event>> &, const std::vector<std::size_t> &>(
                &histogram::add_events
            ),
            py::arg("events"),
            py::arg("indices")
        )
        .def(
            "add_events",
            py::overload_cast<const std::vector<std::shared_ptr<event>> &, eventSorter>(&histogram::add_events),
            py::arg("events"),
            py::arg("sorter")
        )
        .def(
            "add_events",
            py::overload_cast<const std::vector<std::shared_ptr<event>> &, event_hash_fn>(&histogram::add_events),
            py::arg("events"),
            py::arg("hash")
        )
        .def("rebuild", &histogram::rebuild)
        .def("normalise", &histogram::normalise, py::arg("init"))
        .def("set_norm", &histogram::set_norm, py::arg("norm"))
        .def("sync_wgts", &histogram::sync_wgts)
        .def("size", &histogram::size)
        .def("__len__", &histogram::size)
        .def("bins", &histogram::bins)
        .def("rows", &histogram::rows)
        .def("counts", &histogram::counts)
        .def(
            "heights",
            py::overload_cast<std::size_t>(&histogram::heights, py::const_),
            py::arg("row")
        )
        .def("heights", py::overload_cast<>(&histogram::heights, py::const_));

    // ---- lhe ----
    // Real inheritance from InitNode (not a flattened mirror), so REX::tea's
    // reweightor/param_rwgt (bound in tearex.cpp, and themselves derived from
    // lhe in C++) can be registered as proper Python subclasses too.
    // As with eventBelongs/eventSorter above, the vector<event>-by-value
    // constructor is skipped in favour of the vector<shared_ptr<event>>
    // form: a Python list of Event objects converts to either one, so
    // binding both would leave one of them permanently shadowed
    py::classh<lhe, initNode>(m, "Lhe")
        .def(py::init<>())
        .def(py::init<const initNode &>(), py::arg("init"))
        .def(
            py::init<std::vector<std::shared_ptr<event>>>(), py::arg("events")
        )
        .def(
            py::init<const initNode &, std::vector<std::shared_ptr<event>>>(),
            py::arg("init"),
            py::arg("events")
        )
        .def_readwrite("filter_processes", &lhe::filter_processes)
        .def_readwrite("column_major", &lhe::column_major)
        .def_readwrite("max_batch", &lhe::max_batch)
        .def_readwrite("events", &lhe::events)
        .def_readwrite("processes", &lhe::processes)
        .def_readonly("sorted_events", &lhe::sorted_events)
        .def_readonly("process_order", &lhe::process_order)
        .def_readonly("process_group", &lhe::process_group)
        .def_property(
            "weight_ids",
            [](const lhe &self) {
                return self.weight_ids ? *self.weight_ids : std::vector<std::string>{};
            },
            [](lhe &self, const std::vector<std::string> &ids) {
                self.weight_ids = std::make_shared<std::vector<std::string>>(ids);
            }
        )
        .def_readwrite("weight_context", &lhe::weight_context)
        // ---- streaming (see open_lhe_source()/open_lhe_sink() below) ----
        // `source` is read-only: it's only ever meant to be set via
        // open_streaming(), which also eagerly reads init/header off of it --
        // assigning one directly would leave those unset
        .def_readonly("source", &lhe::source)
        .def_readwrite("batch_size", &lhe::batch_size)
        .def_readonly("events_pulled_total", &lhe::events_pulled_total)
        .def_property_readonly("streaming", &lhe::streaming)
        .def_static(
            "open_streaming",
            &lhe::open_streaming,
            py::arg("source"),
            py::arg("batch_size") = 10000
        )
        .def("pull_batch", &lhe::pull_batch)
        .def("add_event", py::overload_cast<std::shared_ptr<event>>(&lhe::add_event))
        .def("add_event", py::overload_cast<const event &>(&lhe::add_event))
        .def("set_sorter", py::overload_cast<>(&lhe::set_sorter))
        .def("set_sorter", py::overload_cast<const eventSorter &>(&lhe::set_sorter))
        .def("set_sorter", py::overload_cast<event_equal_fn>(&lhe::set_sorter))
        .def("sort_events", &lhe::sort_events)
        .def("unsort_events", &lhe::unsort_events)
        .def("events_to_processes", &lhe::events_to_processes)
        .def("processes_to_events", &lhe::processes_to_events)
        .def("transpose", py::overload_cast<>(&lhe::transpose))
        .def(
            "transpose",
            py::overload_cast<std::string>(&lhe::transpose),
            py::arg("direction")
        )
        .def("set_filter", &lhe::set_filter)
        .def("set_max_batch", &lhe::set_max_batch)
        .def(
            "add_weight_id",
            py::overload_cast<const std::string &>(&lhe::add_weight_id),
            py::arg("id")
        )
        .def("extract_weight_ids", &lhe::extract_weight_ids)
        .def("sync_weight_ids", &lhe::sync_weight_ids)
        .def(
            "append_weight_ids", &lhe::append_weight_ids, py::arg("include") = false
        )
        // ---- histogram management ----
        // Owned histograms, built against self.events. As with add_event
        // above, add_histogram is only bound in its shared_ptr<Histogram>
        // form -- a Python Histogram converts to that transparently via
        // classh, so the Histogram-by-value overload would just be
        // permanently shadowed. make_histogram mirrors the same
        // explicit-indices/EventSorter/event_hash_fn subset Histogram
        // itself binds above, for the same reason
        .def_readonly("histograms", &lhe::histograms)
        .def(
            "add_histogram",
            py::overload_cast<const std::string &, std::shared_ptr<histogram>>(&lhe::add_histogram),
            py::arg("name"),
            py::arg("histogram")
        )
        .def(
            "make_histogram",
            py::overload_cast<const std::string &, const std::vector<std::size_t> &>(&lhe::make_histogram),
            py::arg("name"),
            py::arg("indices")
        )
        .def(
            "make_histogram",
            py::overload_cast<const std::string &, eventSorter>(&lhe::make_histogram),
            py::arg("name"),
            py::arg("sorter")
        )
        .def(
            "make_histogram",
            py::overload_cast<const std::string &, event_hash_fn>(&lhe::make_histogram),
            py::arg("name"),
            py::arg("hash")
        )
        .def("has_histogram", &lhe::has_histogram, py::arg("name"))
        // Bound as a lambda returning the map's own shared_ptr<histogram>
        // rather than directly to lhe::hist() (which returns a plain
        // histogram&): a bare C++ reference return only resolves back to
        // the same live Python object if that particular instance has
        // already been seen through a shared_ptr/holder pathway (eg via
        // .histograms[name] below) -- the very first access otherwise
        // silently hands back a disconnected copy, so mutating it (or
        // syncing it later via sync_histogram()) wouldn't touch the
        // histogram actually stored on this Lhe. Returning the shared_ptr
        // directly, exactly as .histograms[name] does, sidesteps that
        // regardless of access order
        .def(
            "hist",
            [](lhe &self, const std::string &name) -> std::shared_ptr<histogram> {
                auto it = self.histograms.find(name);
                if (it == self.histograms.end())
                    throw std::out_of_range("Lhe.hist: no histogram named '" + name + "'");
                return it->second;
            },
            py::arg("name")
        )
        .def("remove_histogram", &lhe::remove_histogram, py::arg("name"))
        .def("sync_histograms", &lhe::sync_histograms)
        .def("sync_histogram", &lhe::sync_histogram, py::arg("name"))
        // In-memory string round-trip, mirroring Slha.write() below: only
        // write_lhef(lhe&, ostream&, bool)/write_lhef(lhe&, filename, bool)
        // exist in C++, so file-path I/O was the only Python-reachable
        // form; wrapping the ostream overload in an ostringstream gives a
        // string result without needing to touch disk
        .def(
            "to_string",
            [](lhe &self, bool include_ids) {
                std::ostringstream ss;
                write_lhef(self, ss, include_ids);
                return ss.str();
            },
            py::arg("include_ids") = false
        );
    m.def(
        "load_lhef",
        py::overload_cast<const std::string &>(&load_lhef),
        py::arg("filename")
    );
    m.def(
        "write_lhef",
        py::overload_cast<lhe &, const std::string &, bool>(&write_lhef),
        py::arg("doc"),
        py::arg("filename"),
        py::arg("include_ids") = false
    );
    // to_lhe(const string&) parses an XML string directly into an Lhe --
    // the in-memory counterpart to load_lhef(filename); the
    // shared_ptr<xmlNode>-taking overload stays unbound (xmlNode itself
    // isn't exposed to Python)
    m.def("to_lhe", py::overload_cast<const std::string &>(&to_lhe), py::arg("xml"));

    // ---- eventSource / eventSink (streaming) ----
    // Both are otherwise-opaque handles from Python: their std::function
    // members exist to be invoked internally by Lhe.pull_batch()/
    // Reweightor.run_streaming(), not called directly. Concrete instances
    // come from open_lhe_source()/open_lhe_sink() below -- the only
    // constructors bound, matching how xmlNode/xmlDoc stay unexposed and
    // load_lhef()/write_lhef() are the practical entry points for the
    // non-streaming path. Only the filename-taking open_lhe_source()
    // overload is bound; the std::istream& overload has no Python-side
    // counterpart to pass in
    py::classh<eventSource>(m, "EventSource")
        .def_property_readonly("seekable", &eventSource::seekable);
    py::classh<eventSink>(m, "EventSink");
    m.def(
        "open_lhe_source",
        py::overload_cast<const std::string &>(&open_lhe_source),
        py::arg("filename")
    );
    m.def(
        "open_lhe_sink", &open_lhe_sink, py::arg("filename"), py::arg("include_ids") = false
    );

    // ---- slha ----
    // Only the single-index get/set overloads are bound: the multi-index
    // overloads take std::initializer_list<int>, which pybind11 has no
    // caster for (it can't be built from a runtime-sized Python list), and
    // has_entry only exists in that form, so it has no bindable overload at
    // all
    py::classh<slha>(m, "Slha")
        .def(py::init<>())
        .def(
            "get",
            py::overload_cast<const std::string &, int, double>(&slha::get, py::const_),
            py::arg("block"),
            py::arg("index"),
            py::arg("fallback") = 0.0
        )
        .def(
            "set",
            py::overload_cast<const std::string &, int, double>(&slha::set),
            py::arg("block"),
            py::arg("index"),
            py::arg("value")
        )
        .def(
            "get_decay", &slha::get_decay, py::arg("pid"), py::arg("fallback") = 0.0
        )
        .def("set_decay", &slha::set_decay, py::arg("pid"), py::arg("width"))
        .def("has_block", &slha::has_block, py::arg("block"))
        .def(
            "write",
            [](const slha &self, int value_precision, bool scientific, const std::string &indent) {
                std::ostringstream ss;
                self.write(ss, value_precision, scientific, indent);
                return ss.str();
            },
            py::arg("value_precision") = 6,
            py::arg("scientific") = true,
            py::arg("indent") = "      "
        );
    m.def(
        "load_slha", py::overload_cast<const std::string &>(&load_slha), py::arg("filename")
    );
    m.def("to_slha", py::overload_cast<const std::string &>(&to_slha), py::arg("text"));
    // Extracts an already-loaded Lhe doc's SLHA param card (eg from its
    // <slha> header block), distinct from to_slha(text) above which parses
    // a raw SLHA-format string
    m.def("to_slha", py::overload_cast<const lhe &>(&to_slha), py::arg("doc"));

    bind_tearex(m.def_submodule("tea"));
}
