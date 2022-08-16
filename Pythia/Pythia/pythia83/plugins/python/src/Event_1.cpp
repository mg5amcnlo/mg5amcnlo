#include <Pythia8/Basics.h>
#include <Pythia8/Event.h>
#include <Pythia8/Info.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ResonanceWidths.h>
#include <istream>
#include <iterator>
#include <memory>
#include <ostream>
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

void bind_Pythia8_Event_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// Pythia8::m(const class Pythia8::Particle &, const class Pythia8::Particle &) file:Pythia8/Event.h line:318
	M("Pythia8").def("m", (double (*)(const class Pythia8::Particle &, const class Pythia8::Particle &)) &Pythia8::m, "C++: Pythia8::m(const class Pythia8::Particle &, const class Pythia8::Particle &) --> double", pybind11::arg("pp1"), pybind11::arg("pp2"));

	// Pythia8::m2(const class Pythia8::Particle &, const class Pythia8::Particle &) file:Pythia8/Event.h line:319
	M("Pythia8").def("m2", (double (*)(const class Pythia8::Particle &, const class Pythia8::Particle &)) &Pythia8::m2, "C++: Pythia8::m2(const class Pythia8::Particle &, const class Pythia8::Particle &) --> double", pybind11::arg("pp1"), pybind11::arg("pp2"));

	// Pythia8::m2(const class Pythia8::Particle &, const class Pythia8::Particle &, const class Pythia8::Particle &) file:Pythia8/Event.h line:320
	M("Pythia8").def("m2", (double (*)(const class Pythia8::Particle &, const class Pythia8::Particle &, const class Pythia8::Particle &)) &Pythia8::m2, "C++: Pythia8::m2(const class Pythia8::Particle &, const class Pythia8::Particle &, const class Pythia8::Particle &) --> double", pybind11::arg("pp1"), pybind11::arg("pp2"), pybind11::arg("pp3"));

	{ // Pythia8::Event file:Pythia8/Event.h line:381
		pybind11::class_<Pythia8::Event, std::shared_ptr<Pythia8::Event>> cl(M("Pythia8"), "Event", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Event(); } ), "doc" );
		cl.def( pybind11::init<int>(), pybind11::arg("capacity") );

		cl.def( pybind11::init( [](Pythia8::Event const &o){ return new Pythia8::Event(o); } ) );
		cl.def("assign", (class Pythia8::Event & (Pythia8::Event::*)(const class Pythia8::Event &)) &Pythia8::Event::operator=, "C++: Pythia8::Event::operator=(const class Pythia8::Event &) --> class Pythia8::Event &", pybind11::return_value_policy::reference, pybind11::arg("oldEvent"));
		cl.def("init", [](Pythia8::Event &o) -> void { return o.init(); }, "");
		cl.def("init", [](Pythia8::Event &o, class std::basic_string<char> const & a0) -> void { return o.init(a0); }, "", pybind11::arg("headerIn"));
		cl.def("init", [](Pythia8::Event &o, class std::basic_string<char> const & a0, class Pythia8::ParticleData * a1) -> void { return o.init(a0, a1); }, "", pybind11::arg("headerIn"), pybind11::arg("particleDataPtrIn"));
		cl.def("init", (void (Pythia8::Event::*)(std::string, class Pythia8::ParticleData *, int)) &Pythia8::Event::init, "C++: Pythia8::Event::init(std::string, class Pythia8::ParticleData *, int) --> void", pybind11::arg("headerIn"), pybind11::arg("particleDataPtrIn"), pybind11::arg("startColTagIn"));
		cl.def("clear", (void (Pythia8::Event::*)()) &Pythia8::Event::clear, "C++: Pythia8::Event::clear() --> void");
		cl.def("free", (void (Pythia8::Event::*)()) &Pythia8::Event::free, "C++: Pythia8::Event::free() --> void");
		cl.def("reset", (void (Pythia8::Event::*)()) &Pythia8::Event::reset, "C++: Pythia8::Event::reset() --> void");
		cl.def("__getitem__", (class Pythia8::Particle & (Pythia8::Event::*)(int)) &Pythia8::Event::operator[], "C++: Pythia8::Event::operator[](int) --> class Pythia8::Particle &", pybind11::return_value_policy::reference, pybind11::arg("i"));
		cl.def("front", (class Pythia8::Particle & (Pythia8::Event::*)()) &Pythia8::Event::front, "C++: Pythia8::Event::front() --> class Pythia8::Particle &", pybind11::return_value_policy::reference);
		cl.def("at", (class Pythia8::Particle & (Pythia8::Event::*)(int)) &Pythia8::Event::at, "C++: Pythia8::Event::at(int) --> class Pythia8::Particle &", pybind11::return_value_policy::reference, pybind11::arg("i"));
		cl.def("back", (class Pythia8::Particle & (Pythia8::Event::*)()) &Pythia8::Event::back, "C++: Pythia8::Event::back() --> class Pythia8::Particle &", pybind11::return_value_policy::reference);
		cl.def("size", (int (Pythia8::Event::*)() const) &Pythia8::Event::size, "C++: Pythia8::Event::size() const --> int");
		cl.def("append", (int (Pythia8::Event::*)(class Pythia8::Particle)) &Pythia8::Event::append, "C++: Pythia8::Event::append(class Pythia8::Particle) --> int", pybind11::arg("entryIn"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, double const & a8, double const & a9, double const & a10, double const & a11, double const & a12, double const & a13) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"), pybind11::arg("scaleIn"));
		cl.def("append", (int (Pythia8::Event::*)(int, int, int, int, int, int, int, int, double, double, double, double, double, double, double)) &Pythia8::Event::append, "C++: Pythia8::Event::append(int, int, int, int, int, int, int, int, double, double, double, double, double, double, double) --> int", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"), pybind11::arg("scaleIn"), pybind11::arg("polIn"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, class Pythia8::Vec4 const & a8, double const & a9, double const & a10) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"), pybind11::arg("scaleIn"));
		cl.def("append", (int (Pythia8::Event::*)(int, int, int, int, int, int, int, int, class Pythia8::Vec4, double, double, double)) &Pythia8::Event::append, "C++: Pythia8::Event::append(int, int, int, int, int, int, int, int, class Pythia8::Vec4, double, double, double) --> int", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("mother1"), pybind11::arg("mother2"), pybind11::arg("daughter1"), pybind11::arg("daughter2"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"), pybind11::arg("scaleIn"), pybind11::arg("polIn"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, double const & a4, double const & a5, double const & a6, double const & a7) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, double const & a4, double const & a5, double const & a6, double const & a7, double const & a8) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, double const & a4, double const & a5, double const & a6, double const & a7, double const & a8, double const & a9) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"), pybind11::arg("scaleIn"));
		cl.def("append", (int (Pythia8::Event::*)(int, int, int, int, double, double, double, double, double, double, double)) &Pythia8::Event::append, "C++: Pythia8::Event::append(int, int, int, int, double, double, double, double, double, double, double) --> int", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("px"), pybind11::arg("py"), pybind11::arg("pz"), pybind11::arg("e"), pybind11::arg("m"), pybind11::arg("scaleIn"), pybind11::arg("polIn"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, class Pythia8::Vec4 const & a4) -> int { return o.append(a0, a1, a2, a3, a4); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, class Pythia8::Vec4 const & a4, double const & a5) -> int { return o.append(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"));
		cl.def("append", [](Pythia8::Event &o, int const & a0, int const & a1, int const & a2, int const & a3, class Pythia8::Vec4 const & a4, double const & a5, double const & a6) -> int { return o.append(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"), pybind11::arg("scaleIn"));
		cl.def("append", (int (Pythia8::Event::*)(int, int, int, int, class Pythia8::Vec4, double, double, double)) &Pythia8::Event::append, "C++: Pythia8::Event::append(int, int, int, int, class Pythia8::Vec4, double, double, double) --> int", pybind11::arg("id"), pybind11::arg("status"), pybind11::arg("col"), pybind11::arg("acol"), pybind11::arg("p"), pybind11::arg("m"), pybind11::arg("scaleIn"), pybind11::arg("polIn"));
		cl.def("setEvtPtr", [](Pythia8::Event &o) -> void { return o.setEvtPtr(); }, "");
		cl.def("setEvtPtr", (void (Pythia8::Event::*)(int)) &Pythia8::Event::setEvtPtr, "C++: Pythia8::Event::setEvtPtr(int) --> void", pybind11::arg("iSet"));
		cl.def("copy", [](Pythia8::Event &o, int const & a0) -> int { return o.copy(a0); }, "", pybind11::arg("iCopy"));
		cl.def("copy", (int (Pythia8::Event::*)(int, int)) &Pythia8::Event::copy, "C++: Pythia8::Event::copy(int, int) --> int", pybind11::arg("iCopy"), pybind11::arg("newStatus"));
		cl.def("list", [](Pythia8::Event const &o) -> void { return o.list(); }, "");
		cl.def("list", [](Pythia8::Event const &o, bool const & a0) -> void { return o.list(a0); }, "", pybind11::arg("showScaleAndVertex"));
		cl.def("list", [](Pythia8::Event const &o, bool const & a0, bool const & a1) -> void { return o.list(a0, a1); }, "", pybind11::arg("showScaleAndVertex"), pybind11::arg("showMothersAndDaughters"));
		cl.def("list", (void (Pythia8::Event::*)(bool, bool, int) const) &Pythia8::Event::list, "C++: Pythia8::Event::list(bool, bool, int) const --> void", pybind11::arg("showScaleAndVertex"), pybind11::arg("showMothersAndDaughters"), pybind11::arg("precision"));
		cl.def("popBack", [](Pythia8::Event &o) -> void { return o.popBack(); }, "");
		cl.def("popBack", (void (Pythia8::Event::*)(int)) &Pythia8::Event::popBack, "C++: Pythia8::Event::popBack(int) --> void", pybind11::arg("nRemove"));
		cl.def("remove", [](Pythia8::Event &o, int const & a0, int const & a1) -> void { return o.remove(a0, a1); }, "", pybind11::arg("iFirst"), pybind11::arg("iLast"));
		cl.def("remove", (void (Pythia8::Event::*)(int, int, bool)) &Pythia8::Event::remove, "C++: Pythia8::Event::remove(int, int, bool) --> void", pybind11::arg("iFirst"), pybind11::arg("iLast"), pybind11::arg("shiftHistory"));
		cl.def("restorePtrs", (void (Pythia8::Event::*)()) &Pythia8::Event::restorePtrs, "C++: Pythia8::Event::restorePtrs() --> void");
		cl.def("saveSize", (void (Pythia8::Event::*)()) &Pythia8::Event::saveSize, "C++: Pythia8::Event::saveSize() --> void");
		cl.def("restoreSize", (void (Pythia8::Event::*)()) &Pythia8::Event::restoreSize, "C++: Pythia8::Event::restoreSize() --> void");
		cl.def("savedSizeValue", (int (Pythia8::Event::*)()) &Pythia8::Event::savedSizeValue, "C++: Pythia8::Event::savedSizeValue() --> int");
		cl.def("initColTag", [](Pythia8::Event &o) -> void { return o.initColTag(); }, "");
		cl.def("initColTag", (void (Pythia8::Event::*)(int)) &Pythia8::Event::initColTag, "C++: Pythia8::Event::initColTag(int) --> void", pybind11::arg("colTag"));
		cl.def("lastColTag", (int (Pythia8::Event::*)() const) &Pythia8::Event::lastColTag, "C++: Pythia8::Event::lastColTag() const --> int");
		cl.def("nextColTag", (int (Pythia8::Event::*)()) &Pythia8::Event::nextColTag, "C++: Pythia8::Event::nextColTag() --> int");
		cl.def("scale", (void (Pythia8::Event::*)(double)) &Pythia8::Event::scale, "C++: Pythia8::Event::scale(double) --> void", pybind11::arg("scaleIn"));
		cl.def("scale", (double (Pythia8::Event::*)() const) &Pythia8::Event::scale, "C++: Pythia8::Event::scale() const --> double");
		cl.def("scaleSecond", (void (Pythia8::Event::*)(double)) &Pythia8::Event::scaleSecond, "C++: Pythia8::Event::scaleSecond(double) --> void", pybind11::arg("scaleSecondIn"));
		cl.def("scaleSecond", (double (Pythia8::Event::*)() const) &Pythia8::Event::scaleSecond, "C++: Pythia8::Event::scaleSecond() const --> double");
		cl.def("daughterList", (class std::vector<int, class std::allocator<int> > (Pythia8::Event::*)(int) const) &Pythia8::Event::daughterList, "C++: Pythia8::Event::daughterList(int) const --> class std::vector<int, class std::allocator<int> >", pybind11::arg("i"));
		cl.def("nFinal", [](Pythia8::Event const &o) -> int { return o.nFinal(); }, "");
		cl.def("nFinal", (int (Pythia8::Event::*)(bool) const) &Pythia8::Event::nFinal, "C++: Pythia8::Event::nFinal(bool) const --> int", pybind11::arg("chargedOnly"));
		cl.def("dyAbs", (double (Pythia8::Event::*)(int, int) const) &Pythia8::Event::dyAbs, "C++: Pythia8::Event::dyAbs(int, int) const --> double", pybind11::arg("i1"), pybind11::arg("i2"));
		cl.def("detaAbs", (double (Pythia8::Event::*)(int, int) const) &Pythia8::Event::detaAbs, "C++: Pythia8::Event::detaAbs(int, int) const --> double", pybind11::arg("i1"), pybind11::arg("i2"));
		cl.def("dphiAbs", (double (Pythia8::Event::*)(int, int) const) &Pythia8::Event::dphiAbs, "C++: Pythia8::Event::dphiAbs(int, int) const --> double", pybind11::arg("i1"), pybind11::arg("i2"));
		cl.def("RRapPhi", (double (Pythia8::Event::*)(int, int) const) &Pythia8::Event::RRapPhi, "C++: Pythia8::Event::RRapPhi(int, int) const --> double", pybind11::arg("i1"), pybind11::arg("i2"));
		cl.def("REtaPhi", (double (Pythia8::Event::*)(int, int) const) &Pythia8::Event::REtaPhi, "C++: Pythia8::Event::REtaPhi(int, int) const --> double", pybind11::arg("i1"), pybind11::arg("i2"));
		cl.def("rot", (void (Pythia8::Event::*)(double, double)) &Pythia8::Event::rot, "C++: Pythia8::Event::rot(double, double) --> void", pybind11::arg("theta"), pybind11::arg("phi"));
		cl.def("bst", (void (Pythia8::Event::*)(double, double, double)) &Pythia8::Event::bst, "C++: Pythia8::Event::bst(double, double, double) --> void", pybind11::arg("betaX"), pybind11::arg("betaY"), pybind11::arg("betaZ"));
		cl.def("bst", (void (Pythia8::Event::*)(double, double, double, double)) &Pythia8::Event::bst, "C++: Pythia8::Event::bst(double, double, double, double) --> void", pybind11::arg("betaX"), pybind11::arg("betaY"), pybind11::arg("betaZ"), pybind11::arg("gamma"));
		cl.def("bst", (void (Pythia8::Event::*)(const class Pythia8::Vec4 &)) &Pythia8::Event::bst, "C++: Pythia8::Event::bst(const class Pythia8::Vec4 &) --> void", pybind11::arg("vec"));
		cl.def("rotbst", [](Pythia8::Event &o, const class Pythia8::RotBstMatrix & a0) -> void { return o.rotbst(a0); }, "", pybind11::arg("M"));
		cl.def("rotbst", (void (Pythia8::Event::*)(const class Pythia8::RotBstMatrix &, bool)) &Pythia8::Event::rotbst, "C++: Pythia8::Event::rotbst(const class Pythia8::RotBstMatrix &, bool) --> void", pybind11::arg("M"), pybind11::arg("boostVertices"));
		cl.def("clearJunctions", (void (Pythia8::Event::*)()) &Pythia8::Event::clearJunctions, "C++: Pythia8::Event::clearJunctions() --> void");
		cl.def("appendJunction", (int (Pythia8::Event::*)(int, int, int, int)) &Pythia8::Event::appendJunction, "C++: Pythia8::Event::appendJunction(int, int, int, int) --> int", pybind11::arg("kind"), pybind11::arg("col0"), pybind11::arg("col1"), pybind11::arg("col2"));
		cl.def("sizeJunction", (int (Pythia8::Event::*)() const) &Pythia8::Event::sizeJunction, "C++: Pythia8::Event::sizeJunction() const --> int");
		cl.def("remainsJunction", (bool (Pythia8::Event::*)(int) const) &Pythia8::Event::remainsJunction, "C++: Pythia8::Event::remainsJunction(int) const --> bool", pybind11::arg("i"));
		cl.def("remainsJunction", (void (Pythia8::Event::*)(int, bool)) &Pythia8::Event::remainsJunction, "C++: Pythia8::Event::remainsJunction(int, bool) --> void", pybind11::arg("i"), pybind11::arg("remainsIn"));
		cl.def("kindJunction", (int (Pythia8::Event::*)(int) const) &Pythia8::Event::kindJunction, "C++: Pythia8::Event::kindJunction(int) const --> int", pybind11::arg("i"));
		cl.def("colJunction", (int (Pythia8::Event::*)(int, int) const) &Pythia8::Event::colJunction, "C++: Pythia8::Event::colJunction(int, int) const --> int", pybind11::arg("i"), pybind11::arg("j"));
		cl.def("colJunction", (void (Pythia8::Event::*)(int, int, int)) &Pythia8::Event::colJunction, "C++: Pythia8::Event::colJunction(int, int, int) --> void", pybind11::arg("i"), pybind11::arg("j"), pybind11::arg("colIn"));
		cl.def("endColJunction", (int (Pythia8::Event::*)(int, int) const) &Pythia8::Event::endColJunction, "C++: Pythia8::Event::endColJunction(int, int) const --> int", pybind11::arg("i"), pybind11::arg("j"));
		cl.def("endColJunction", (void (Pythia8::Event::*)(int, int, int)) &Pythia8::Event::endColJunction, "C++: Pythia8::Event::endColJunction(int, int, int) --> void", pybind11::arg("i"), pybind11::arg("j"), pybind11::arg("endColIn"));
		cl.def("statusJunction", (int (Pythia8::Event::*)(int, int) const) &Pythia8::Event::statusJunction, "C++: Pythia8::Event::statusJunction(int, int) const --> int", pybind11::arg("i"), pybind11::arg("j"));
		cl.def("statusJunction", (void (Pythia8::Event::*)(int, int, int)) &Pythia8::Event::statusJunction, "C++: Pythia8::Event::statusJunction(int, int, int) --> void", pybind11::arg("i"), pybind11::arg("j"), pybind11::arg("statusIn"));
		cl.def("eraseJunction", (void (Pythia8::Event::*)(int)) &Pythia8::Event::eraseJunction, "C++: Pythia8::Event::eraseJunction(int) --> void", pybind11::arg("i"));
		cl.def("saveJunctionSize", (void (Pythia8::Event::*)()) &Pythia8::Event::saveJunctionSize, "C++: Pythia8::Event::saveJunctionSize() --> void");
		cl.def("restoreJunctionSize", (void (Pythia8::Event::*)()) &Pythia8::Event::restoreJunctionSize, "C++: Pythia8::Event::restoreJunctionSize() --> void");
		cl.def("listJunctions", (void (Pythia8::Event::*)() const) &Pythia8::Event::listJunctions, "C++: Pythia8::Event::listJunctions() const --> void");
		cl.def("savePartonLevelSize", (void (Pythia8::Event::*)()) &Pythia8::Event::savePartonLevelSize, "C++: Pythia8::Event::savePartonLevelSize() --> void");
		cl.def("__iadd__", (class Pythia8::Event & (Pythia8::Event::*)(const class Pythia8::Event &)) &Pythia8::Event::operator+=, "C++: Pythia8::Event::operator+=(const class Pythia8::Event &) --> class Pythia8::Event &", pybind11::return_value_policy::reference, pybind11::arg("addEvent"));
	}
}
