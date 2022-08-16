#include <Pythia8/Basics.h>
#include <Pythia8/Event.h>
#include <Pythia8/Info.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ResonanceWidths.h>
#include <iterator>
#include <memory>
#include <sstream> // __str__
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <Pythia8/UserHooks.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/BeamShape.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// Pythia8::Particle file:Pythia8/Event.h line:32
struct PyCallBack_Pythia8_Particle : public Pythia8::Particle {
	using Pythia8::Particle::Particle;

	int index() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::Particle *>(this), "index");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return Particle::index();
	}
};

void bind_Pythia8_Event(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::Particle file:Pythia8/Event.h line:32
		pybind11::class_<Pythia8::Particle, std::shared_ptr<Pythia8::Particle>, PyCallBack_Pythia8_Particle> cl(M("Pythia8"), "Particle", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Particle(); }, [](){ return new PyCallBack_Pythia8_Particle(); } ) );
		cl.def( pybind11::init( [](int const & a0){ return new Pythia8::Particle(a0); }, [](int const & a0){ return new PyCallBack_Pythia8_Particle(a0); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1){ return new Pythia8::Particle(a0, a1); }, [](int const & a0, int const & a1){ return new PyCallBack_Pythia8_Particle(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2){ return new Pythia8::Particle(a0, a1, a2); }, [](int const & a0, int const & a1, int const & a2){ return new PyCallBack_Pythia8_Particle(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3){ return new Pythia8::Particle(a0, a1, a2, a3); }, [](int const & a0, int const & a1, int const & a2, int const & a3){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4){ return new Pythia8::Particle(a0, a1, a2, a3, a4); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12, double const & a13){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12, double const & a13){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13); } ), "doc");
		cl.def( pybind11::init<int, int, int, int, int, int, int, int, double, double, double, double, double, double, double>(), pybind11::arg("idIn"), pybind11::arg("statusIn"), pybind11::arg("mother1In"), pybind11::arg("mother2In"), pybind11::arg("daughter1In"), pybind11::arg("daughter2In"), pybind11::arg("colIn"), pybind11::arg("acolIn"), pybind11::arg("pxIn"), pybind11::arg("pyIn"), pybind11::arg("pzIn"), pybind11::arg("eIn"), pybind11::arg("mIn"), pybind11::arg("scaleIn"), pybind11::arg("polIn") );

		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); } ), "doc");
		cl.def( pybind11::init( [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9, double const & a10){ return new Pythia8::Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, [](int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9, double const & a10){ return new PyCallBack_Pythia8_Particle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); } ), "doc");
		cl.def( pybind11::init<int, int, int, int, int, int, int, int, class Pythia8::Vec4, double, double, double>(), pybind11::arg("idIn"), pybind11::arg("statusIn"), pybind11::arg("mother1In"), pybind11::arg("mother2In"), pybind11::arg("daughter1In"), pybind11::arg("daughter2In"), pybind11::arg("colIn"), pybind11::arg("acolIn"), pybind11::arg("pIn"), pybind11::arg("mIn"), pybind11::arg("scaleIn"), pybind11::arg("polIn") );

		cl.def( pybind11::init( [](PyCallBack_Pythia8_Particle const &o){ return new PyCallBack_Pythia8_Particle(o); } ) );
		cl.def( pybind11::init( [](Pythia8::Particle const &o){ return new Pythia8::Particle(o); } ) );
		cl.def_readwrite("idSave", &Pythia8::Particle::idSave);
		cl.def_readwrite("statusSave", &Pythia8::Particle::statusSave);
		cl.def_readwrite("mother1Save", &Pythia8::Particle::mother1Save);
		cl.def_readwrite("mother2Save", &Pythia8::Particle::mother2Save);
		cl.def_readwrite("daughter1Save", &Pythia8::Particle::daughter1Save);
		cl.def_readwrite("daughter2Save", &Pythia8::Particle::daughter2Save);
		cl.def_readwrite("colSave", &Pythia8::Particle::colSave);
		cl.def_readwrite("acolSave", &Pythia8::Particle::acolSave);
		cl.def_readwrite("pSave", &Pythia8::Particle::pSave);
		cl.def_readwrite("mSave", &Pythia8::Particle::mSave);
		cl.def_readwrite("scaleSave", &Pythia8::Particle::scaleSave);
		cl.def_readwrite("polSave", &Pythia8::Particle::polSave);
		cl.def_readwrite("hasVertexSave", &Pythia8::Particle::hasVertexSave);
		cl.def_readwrite("vProdSave", &Pythia8::Particle::vProdSave);
		cl.def_readwrite("tauSave", &Pythia8::Particle::tauSave);
		cl.def_readwrite("pdePtr", &Pythia8::Particle::pdePtr);
		cl.def("assign", (class Pythia8::Particle & (Pythia8::Particle::*)(const class Pythia8::Particle &)) &Pythia8::Particle::operator=, "C++: Pythia8::Particle::operator=(const class Pythia8::Particle &) --> class Pythia8::Particle &", pybind11::return_value_policy::reference, pybind11::arg("pt"));
		cl.def("setEvtPtr", (void (Pythia8::Particle::*)(class Pythia8::Event *)) &Pythia8::Particle::setEvtPtr, "C++: Pythia8::Particle::setEvtPtr(class Pythia8::Event *) --> void", pybind11::arg("evtPtrIn"));
		cl.def("setPDEPtr", [](Pythia8::Particle &o) -> void { return o.setPDEPtr(); }, "");
		cl.def("setPDEPtr", (void (Pythia8::Particle::*)(class std::shared_ptr<class Pythia8::ParticleDataEntry>)) &Pythia8::Particle::setPDEPtr, "C++: Pythia8::Particle::setPDEPtr(class std::shared_ptr<class Pythia8::ParticleDataEntry>) --> void", pybind11::arg("pdePtrIn"));
		cl.def("id", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::id, "C++: Pythia8::Particle::id(int) --> void", pybind11::arg("idIn"));
		cl.def("status", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::status, "C++: Pythia8::Particle::status(int) --> void", pybind11::arg("statusIn"));
		cl.def("statusPos", (void (Pythia8::Particle::*)()) &Pythia8::Particle::statusPos, "C++: Pythia8::Particle::statusPos() --> void");
		cl.def("statusNeg", (void (Pythia8::Particle::*)()) &Pythia8::Particle::statusNeg, "C++: Pythia8::Particle::statusNeg() --> void");
		cl.def("statusCode", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::statusCode, "C++: Pythia8::Particle::statusCode(int) --> void", pybind11::arg("statusIn"));
		cl.def("mother1", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::mother1, "C++: Pythia8::Particle::mother1(int) --> void", pybind11::arg("mother1In"));
		cl.def("mother2", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::mother2, "C++: Pythia8::Particle::mother2(int) --> void", pybind11::arg("mother2In"));
		cl.def("mothers", [](Pythia8::Particle &o) -> void { return o.mothers(); }, "");
		cl.def("mothers", [](Pythia8::Particle &o, int const & a0) -> void { return o.mothers(a0); }, "", pybind11::arg("mother1In"));
		cl.def("mothers", (void (Pythia8::Particle::*)(int, int)) &Pythia8::Particle::mothers, "C++: Pythia8::Particle::mothers(int, int) --> void", pybind11::arg("mother1In"), pybind11::arg("mother2In"));
		cl.def("daughter1", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::daughter1, "C++: Pythia8::Particle::daughter1(int) --> void", pybind11::arg("daughter1In"));
		cl.def("daughter2", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::daughter2, "C++: Pythia8::Particle::daughter2(int) --> void", pybind11::arg("daughter2In"));
		cl.def("daughters", [](Pythia8::Particle &o) -> void { return o.daughters(); }, "");
		cl.def("daughters", [](Pythia8::Particle &o, int const & a0) -> void { return o.daughters(a0); }, "", pybind11::arg("daughter1In"));
		cl.def("daughters", (void (Pythia8::Particle::*)(int, int)) &Pythia8::Particle::daughters, "C++: Pythia8::Particle::daughters(int, int) --> void", pybind11::arg("daughter1In"), pybind11::arg("daughter2In"));
		cl.def("col", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::col, "C++: Pythia8::Particle::col(int) --> void", pybind11::arg("colIn"));
		cl.def("acol", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::acol, "C++: Pythia8::Particle::acol(int) --> void", pybind11::arg("acolIn"));
		cl.def("cols", [](Pythia8::Particle &o) -> void { return o.cols(); }, "");
		cl.def("cols", [](Pythia8::Particle &o, int const & a0) -> void { return o.cols(a0); }, "", pybind11::arg("colIn"));
		cl.def("cols", (void (Pythia8::Particle::*)(int, int)) &Pythia8::Particle::cols, "C++: Pythia8::Particle::cols(int, int) --> void", pybind11::arg("colIn"), pybind11::arg("acolIn"));
		cl.def("p", (void (Pythia8::Particle::*)(class Pythia8::Vec4)) &Pythia8::Particle::p, "C++: Pythia8::Particle::p(class Pythia8::Vec4) --> void", pybind11::arg("pIn"));
		cl.def("p", (void (Pythia8::Particle::*)(double, double, double, double)) &Pythia8::Particle::p, "C++: Pythia8::Particle::p(double, double, double, double) --> void", pybind11::arg("pxIn"), pybind11::arg("pyIn"), pybind11::arg("pzIn"), pybind11::arg("eIn"));
		cl.def("px", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::px, "C++: Pythia8::Particle::px(double) --> void", pybind11::arg("pxIn"));
		cl.def("py", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::py, "C++: Pythia8::Particle::py(double) --> void", pybind11::arg("pyIn"));
		cl.def("pz", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::pz, "C++: Pythia8::Particle::pz(double) --> void", pybind11::arg("pzIn"));
		cl.def("e", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::e, "C++: Pythia8::Particle::e(double) --> void", pybind11::arg("eIn"));
		cl.def("m", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::m, "C++: Pythia8::Particle::m(double) --> void", pybind11::arg("mIn"));
		cl.def("scale", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::scale, "C++: Pythia8::Particle::scale(double) --> void", pybind11::arg("scaleIn"));
		cl.def("pol", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::pol, "C++: Pythia8::Particle::pol(double) --> void", pybind11::arg("polIn"));
		cl.def("vProd", (void (Pythia8::Particle::*)(class Pythia8::Vec4)) &Pythia8::Particle::vProd, "C++: Pythia8::Particle::vProd(class Pythia8::Vec4) --> void", pybind11::arg("vProdIn"));
		cl.def("vProd", (void (Pythia8::Particle::*)(double, double, double, double)) &Pythia8::Particle::vProd, "C++: Pythia8::Particle::vProd(double, double, double, double) --> void", pybind11::arg("xProdIn"), pybind11::arg("yProdIn"), pybind11::arg("zProdIn"), pybind11::arg("tProdIn"));
		cl.def("xProd", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::xProd, "C++: Pythia8::Particle::xProd(double) --> void", pybind11::arg("xProdIn"));
		cl.def("yProd", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::yProd, "C++: Pythia8::Particle::yProd(double) --> void", pybind11::arg("yProdIn"));
		cl.def("zProd", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::zProd, "C++: Pythia8::Particle::zProd(double) --> void", pybind11::arg("zProdIn"));
		cl.def("tProd", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::tProd, "C++: Pythia8::Particle::tProd(double) --> void", pybind11::arg("tProdIn"));
		cl.def("vProdAdd", (void (Pythia8::Particle::*)(class Pythia8::Vec4)) &Pythia8::Particle::vProdAdd, "C++: Pythia8::Particle::vProdAdd(class Pythia8::Vec4) --> void", pybind11::arg("vProdIn"));
		cl.def("tau", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::tau, "C++: Pythia8::Particle::tau(double) --> void", pybind11::arg("tauIn"));
		cl.def("id", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::id, "C++: Pythia8::Particle::id() const --> int");
		cl.def("status", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::status, "C++: Pythia8::Particle::status() const --> int");
		cl.def("mother1", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::mother1, "C++: Pythia8::Particle::mother1() const --> int");
		cl.def("mother2", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::mother2, "C++: Pythia8::Particle::mother2() const --> int");
		cl.def("daughter1", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::daughter1, "C++: Pythia8::Particle::daughter1() const --> int");
		cl.def("daughter2", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::daughter2, "C++: Pythia8::Particle::daughter2() const --> int");
		cl.def("col", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::col, "C++: Pythia8::Particle::col() const --> int");
		cl.def("acol", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::acol, "C++: Pythia8::Particle::acol() const --> int");
		cl.def("p", (class Pythia8::Vec4 (Pythia8::Particle::*)() const) &Pythia8::Particle::p, "C++: Pythia8::Particle::p() const --> class Pythia8::Vec4");
		cl.def("px", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::px, "C++: Pythia8::Particle::px() const --> double");
		cl.def("py", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::py, "C++: Pythia8::Particle::py() const --> double");
		cl.def("pz", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pz, "C++: Pythia8::Particle::pz() const --> double");
		cl.def("e", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::e, "C++: Pythia8::Particle::e() const --> double");
		cl.def("m", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::m, "C++: Pythia8::Particle::m() const --> double");
		cl.def("scale", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::scale, "C++: Pythia8::Particle::scale() const --> double");
		cl.def("pol", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pol, "C++: Pythia8::Particle::pol() const --> double");
		cl.def("hasVertex", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::hasVertex, "C++: Pythia8::Particle::hasVertex() const --> bool");
		cl.def("vProd", (class Pythia8::Vec4 (Pythia8::Particle::*)() const) &Pythia8::Particle::vProd, "C++: Pythia8::Particle::vProd() const --> class Pythia8::Vec4");
		cl.def("xProd", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::xProd, "C++: Pythia8::Particle::xProd() const --> double");
		cl.def("yProd", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::yProd, "C++: Pythia8::Particle::yProd() const --> double");
		cl.def("zProd", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::zProd, "C++: Pythia8::Particle::zProd() const --> double");
		cl.def("tProd", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::tProd, "C++: Pythia8::Particle::tProd() const --> double");
		cl.def("tau", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::tau, "C++: Pythia8::Particle::tau() const --> double");
		cl.def("idAbs", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::idAbs, "C++: Pythia8::Particle::idAbs() const --> int");
		cl.def("statusAbs", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::statusAbs, "C++: Pythia8::Particle::statusAbs() const --> int");
		cl.def("isFinal", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isFinal, "C++: Pythia8::Particle::isFinal() const --> bool");
		cl.def("intPol", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::intPol, "C++: Pythia8::Particle::intPol() const --> int");
		cl.def("isRescatteredIncoming", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isRescatteredIncoming, "C++: Pythia8::Particle::isRescatteredIncoming() const --> bool");
		cl.def("m2", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::m2, "C++: Pythia8::Particle::m2() const --> double");
		cl.def("mCalc", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mCalc, "C++: Pythia8::Particle::mCalc() const --> double");
		cl.def("m2Calc", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::m2Calc, "C++: Pythia8::Particle::m2Calc() const --> double");
		cl.def("eCalc", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::eCalc, "C++: Pythia8::Particle::eCalc() const --> double");
		cl.def("pT", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pT, "C++: Pythia8::Particle::pT() const --> double");
		cl.def("pT2", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pT2, "C++: Pythia8::Particle::pT2() const --> double");
		cl.def("mT", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mT, "C++: Pythia8::Particle::mT() const --> double");
		cl.def("mT2", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mT2, "C++: Pythia8::Particle::mT2() const --> double");
		cl.def("pAbs", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pAbs, "C++: Pythia8::Particle::pAbs() const --> double");
		cl.def("pAbs2", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pAbs2, "C++: Pythia8::Particle::pAbs2() const --> double");
		cl.def("eT", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::eT, "C++: Pythia8::Particle::eT() const --> double");
		cl.def("eT2", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::eT2, "C++: Pythia8::Particle::eT2() const --> double");
		cl.def("theta", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::theta, "C++: Pythia8::Particle::theta() const --> double");
		cl.def("phi", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::phi, "C++: Pythia8::Particle::phi() const --> double");
		cl.def("thetaXZ", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::thetaXZ, "C++: Pythia8::Particle::thetaXZ() const --> double");
		cl.def("pPos", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pPos, "C++: Pythia8::Particle::pPos() const --> double");
		cl.def("pNeg", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::pNeg, "C++: Pythia8::Particle::pNeg() const --> double");
		cl.def("y", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::y, "C++: Pythia8::Particle::y() const --> double");
		cl.def("eta", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::eta, "C++: Pythia8::Particle::eta() const --> double");
		cl.def("y", (double (Pythia8::Particle::*)(double) const) &Pythia8::Particle::y, "C++: Pythia8::Particle::y(double) const --> double", pybind11::arg("mCut"));
		cl.def("y", (double (Pythia8::Particle::*)(double, class Pythia8::RotBstMatrix &) const) &Pythia8::Particle::y, "C++: Pythia8::Particle::y(double, class Pythia8::RotBstMatrix &) const --> double", pybind11::arg("mCut"), pybind11::arg("M"));
		cl.def("vDec", (class Pythia8::Vec4 (Pythia8::Particle::*)() const) &Pythia8::Particle::vDec, "C++: Pythia8::Particle::vDec() const --> class Pythia8::Vec4");
		cl.def("xDec", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::xDec, "C++: Pythia8::Particle::xDec() const --> double");
		cl.def("yDec", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::yDec, "C++: Pythia8::Particle::yDec() const --> double");
		cl.def("zDec", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::zDec, "C++: Pythia8::Particle::zDec() const --> double");
		cl.def("tDec", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::tDec, "C++: Pythia8::Particle::tDec() const --> double");
		cl.def("index", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::index, "C++: Pythia8::Particle::index() const --> int");
		cl.def("iTopCopy", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::iTopCopy, "C++: Pythia8::Particle::iTopCopy() const --> int");
		cl.def("iBotCopy", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::iBotCopy, "C++: Pythia8::Particle::iBotCopy() const --> int");
		cl.def("iTopCopyId", [](Pythia8::Particle const &o) -> int { return o.iTopCopyId(); }, "");
		cl.def("iTopCopyId", (int (Pythia8::Particle::*)(bool) const) &Pythia8::Particle::iTopCopyId, "C++: Pythia8::Particle::iTopCopyId(bool) const --> int", pybind11::arg("simplify"));
		cl.def("iBotCopyId", [](Pythia8::Particle const &o) -> int { return o.iBotCopyId(); }, "");
		cl.def("iBotCopyId", (int (Pythia8::Particle::*)(bool) const) &Pythia8::Particle::iBotCopyId, "C++: Pythia8::Particle::iBotCopyId(bool) const --> int", pybind11::arg("simplify"));
		cl.def("motherList", (class std::vector<int, class std::allocator<int> > (Pythia8::Particle::*)() const) &Pythia8::Particle::motherList, "C++: Pythia8::Particle::motherList() const --> class std::vector<int, class std::allocator<int> >");
		cl.def("daughterList", (class std::vector<int, class std::allocator<int> > (Pythia8::Particle::*)() const) &Pythia8::Particle::daughterList, "C++: Pythia8::Particle::daughterList() const --> class std::vector<int, class std::allocator<int> >");
		cl.def("daughterListRecursive", (class std::vector<int, class std::allocator<int> > (Pythia8::Particle::*)() const) &Pythia8::Particle::daughterListRecursive, "C++: Pythia8::Particle::daughterListRecursive() const --> class std::vector<int, class std::allocator<int> >");
		cl.def("sisterList", [](Pythia8::Particle const &o) -> std::vector<int, class std::allocator<int> > { return o.sisterList(); }, "");
		cl.def("sisterList", (class std::vector<int, class std::allocator<int> > (Pythia8::Particle::*)(bool) const) &Pythia8::Particle::sisterList, "C++: Pythia8::Particle::sisterList(bool) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("traceTopBot"));
		cl.def("isAncestor", (bool (Pythia8::Particle::*)(int) const) &Pythia8::Particle::isAncestor, "C++: Pythia8::Particle::isAncestor(int) const --> bool", pybind11::arg("iAncestor"));
		cl.def("statusHepMC", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::statusHepMC, "C++: Pythia8::Particle::statusHepMC() const --> int");
		cl.def("isFinalPartonLevel", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isFinalPartonLevel, "C++: Pythia8::Particle::isFinalPartonLevel() const --> bool");
		cl.def("undoDecay", (bool (Pythia8::Particle::*)()) &Pythia8::Particle::undoDecay, "C++: Pythia8::Particle::undoDecay() --> bool");
		cl.def("name", (std::string (Pythia8::Particle::*)() const) &Pythia8::Particle::name, "C++: Pythia8::Particle::name() const --> std::string");
		cl.def("nameWithStatus", [](Pythia8::Particle const &o) -> std::string { return o.nameWithStatus(); }, "");
		cl.def("nameWithStatus", (std::string (Pythia8::Particle::*)(int) const) &Pythia8::Particle::nameWithStatus, "C++: Pythia8::Particle::nameWithStatus(int) const --> std::string", pybind11::arg("maxLen"));
		cl.def("spinType", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::spinType, "C++: Pythia8::Particle::spinType() const --> int");
		cl.def("chargeType", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::chargeType, "C++: Pythia8::Particle::chargeType() const --> int");
		cl.def("charge", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::charge, "C++: Pythia8::Particle::charge() const --> double");
		cl.def("isCharged", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isCharged, "C++: Pythia8::Particle::isCharged() const --> bool");
		cl.def("isNeutral", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isNeutral, "C++: Pythia8::Particle::isNeutral() const --> bool");
		cl.def("colType", (int (Pythia8::Particle::*)() const) &Pythia8::Particle::colType, "C++: Pythia8::Particle::colType() const --> int");
		cl.def("m0", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::m0, "C++: Pythia8::Particle::m0() const --> double");
		cl.def("mWidth", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mWidth, "C++: Pythia8::Particle::mWidth() const --> double");
		cl.def("mMin", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mMin, "C++: Pythia8::Particle::mMin() const --> double");
		cl.def("mMax", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mMax, "C++: Pythia8::Particle::mMax() const --> double");
		cl.def("mSel", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::mSel, "C++: Pythia8::Particle::mSel() const --> double");
		cl.def("constituentMass", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::constituentMass, "C++: Pythia8::Particle::constituentMass() const --> double");
		cl.def("tau0", (double (Pythia8::Particle::*)() const) &Pythia8::Particle::tau0, "C++: Pythia8::Particle::tau0() const --> double");
		cl.def("mayDecay", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::mayDecay, "C++: Pythia8::Particle::mayDecay() const --> bool");
		cl.def("canDecay", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::canDecay, "C++: Pythia8::Particle::canDecay() const --> bool");
		cl.def("doExternalDecay", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::doExternalDecay, "C++: Pythia8::Particle::doExternalDecay() const --> bool");
		cl.def("isResonance", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isResonance, "C++: Pythia8::Particle::isResonance() const --> bool");
		cl.def("isVisible", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isVisible, "C++: Pythia8::Particle::isVisible() const --> bool");
		cl.def("isLepton", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isLepton, "C++: Pythia8::Particle::isLepton() const --> bool");
		cl.def("isQuark", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isQuark, "C++: Pythia8::Particle::isQuark() const --> bool");
		cl.def("isGluon", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isGluon, "C++: Pythia8::Particle::isGluon() const --> bool");
		cl.def("isDiquark", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isDiquark, "C++: Pythia8::Particle::isDiquark() const --> bool");
		cl.def("isParton", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isParton, "C++: Pythia8::Particle::isParton() const --> bool");
		cl.def("isHadron", (bool (Pythia8::Particle::*)() const) &Pythia8::Particle::isHadron, "C++: Pythia8::Particle::isHadron() const --> bool");
		cl.def("particleDataEntry", (class Pythia8::ParticleDataEntry & (Pythia8::Particle::*)() const) &Pythia8::Particle::particleDataEntry, "C++: Pythia8::Particle::particleDataEntry() const --> class Pythia8::ParticleDataEntry &", pybind11::return_value_policy::reference);
		cl.def("rescale3", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::rescale3, "C++: Pythia8::Particle::rescale3(double) --> void", pybind11::arg("fac"));
		cl.def("rescale4", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::rescale4, "C++: Pythia8::Particle::rescale4(double) --> void", pybind11::arg("fac"));
		cl.def("rescale5", (void (Pythia8::Particle::*)(double)) &Pythia8::Particle::rescale5, "C++: Pythia8::Particle::rescale5(double) --> void", pybind11::arg("fac"));
		cl.def("rot", (void (Pythia8::Particle::*)(double, double)) &Pythia8::Particle::rot, "C++: Pythia8::Particle::rot(double, double) --> void", pybind11::arg("thetaIn"), pybind11::arg("phiIn"));
		cl.def("bst", (void (Pythia8::Particle::*)(double, double, double)) &Pythia8::Particle::bst, "C++: Pythia8::Particle::bst(double, double, double) --> void", pybind11::arg("betaX"), pybind11::arg("betaY"), pybind11::arg("betaZ"));
		cl.def("bst", (void (Pythia8::Particle::*)(double, double, double, double)) &Pythia8::Particle::bst, "C++: Pythia8::Particle::bst(double, double, double, double) --> void", pybind11::arg("betaX"), pybind11::arg("betaY"), pybind11::arg("betaZ"), pybind11::arg("gamma"));
		cl.def("bst", (void (Pythia8::Particle::*)(const class Pythia8::Vec4 &)) &Pythia8::Particle::bst, "C++: Pythia8::Particle::bst(const class Pythia8::Vec4 &) --> void", pybind11::arg("pBst"));
		cl.def("bst", (void (Pythia8::Particle::*)(const class Pythia8::Vec4 &, double)) &Pythia8::Particle::bst, "C++: Pythia8::Particle::bst(const class Pythia8::Vec4 &, double) --> void", pybind11::arg("pBst"), pybind11::arg("mBst"));
		cl.def("bstback", (void (Pythia8::Particle::*)(const class Pythia8::Vec4 &)) &Pythia8::Particle::bstback, "C++: Pythia8::Particle::bstback(const class Pythia8::Vec4 &) --> void", pybind11::arg("pBst"));
		cl.def("bstback", (void (Pythia8::Particle::*)(const class Pythia8::Vec4 &, double)) &Pythia8::Particle::bstback, "C++: Pythia8::Particle::bstback(const class Pythia8::Vec4 &, double) --> void", pybind11::arg("pBst"), pybind11::arg("mBst"));
		cl.def("rotbst", [](Pythia8::Particle &o, const class Pythia8::RotBstMatrix & a0) -> void { return o.rotbst(a0); }, "", pybind11::arg("M"));
		cl.def("rotbst", (void (Pythia8::Particle::*)(const class Pythia8::RotBstMatrix &, bool)) &Pythia8::Particle::rotbst, "C++: Pythia8::Particle::rotbst(const class Pythia8::RotBstMatrix &, bool) --> void", pybind11::arg("M"), pybind11::arg("boostVertex"));
		cl.def("offsetHistory", (void (Pythia8::Particle::*)(int, int, int, int)) &Pythia8::Particle::offsetHistory, "C++: Pythia8::Particle::offsetHistory(int, int, int, int) --> void", pybind11::arg("minMother"), pybind11::arg("addMother"), pybind11::arg("minDaughter"), pybind11::arg("addDaughter"));
		cl.def("offsetCol", (void (Pythia8::Particle::*)(int)) &Pythia8::Particle::offsetCol, "C++: Pythia8::Particle::offsetCol(int) --> void", pybind11::arg("addCol"));
	}
}
