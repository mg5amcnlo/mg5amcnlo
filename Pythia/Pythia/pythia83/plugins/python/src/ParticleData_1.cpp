#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SigmaLowEnergy.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/Weights.h>
#include <functional>
#include <ios>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <sstream> // __str__
#include <streambuf>
#include <string>
#include <utility>
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

void bind_Pythia8_ParticleData_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::ParticleData file:Pythia8/ParticleData.h line:423
		pybind11::class_<Pythia8::ParticleData, std::shared_ptr<Pythia8::ParticleData>> cl(M("Pythia8"), "ParticleData", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::ParticleData(); } ) );
		cl.def( pybind11::init( [](Pythia8::ParticleData const &o){ return new Pythia8::ParticleData(o); } ) );
		cl.def("assign", (class Pythia8::ParticleData & (Pythia8::ParticleData::*)(const class Pythia8::ParticleData &)) &Pythia8::ParticleData::operator=, "C++: Pythia8::ParticleData::operator=(const class Pythia8::ParticleData &) --> class Pythia8::ParticleData &", pybind11::return_value_policy::reference, pybind11::arg("oldPD"));
		cl.def("initPtrs", (void (Pythia8::ParticleData::*)(class Pythia8::Info *)) &Pythia8::ParticleData::initPtrs, "C++: Pythia8::ParticleData::initPtrs(class Pythia8::Info *) --> void", pybind11::arg("infoPtrIn"));
		cl.def("init", [](Pythia8::ParticleData &o) -> bool { return o.init(); }, "");
		cl.def("init", (bool (Pythia8::ParticleData::*)(std::string)) &Pythia8::ParticleData::init, "C++: Pythia8::ParticleData::init(std::string) --> bool", pybind11::arg("startFile"));
		cl.def("init", (bool (Pythia8::ParticleData::*)(const class Pythia8::ParticleData &)) &Pythia8::ParticleData::init, "C++: Pythia8::ParticleData::init(const class Pythia8::ParticleData &) --> bool", pybind11::arg("particleDataIn"));
		cl.def("init", (bool (Pythia8::ParticleData::*)(class std::basic_istream<char> &)) &Pythia8::ParticleData::init, "C++: Pythia8::ParticleData::init(class std::basic_istream<char> &) --> bool", pybind11::arg("is"));
		cl.def("reInit", [](Pythia8::ParticleData &o, class std::basic_string<char> const & a0) -> bool { return o.reInit(a0); }, "", pybind11::arg("startFile"));
		cl.def("reInit", (bool (Pythia8::ParticleData::*)(std::string, bool)) &Pythia8::ParticleData::reInit, "C++: Pythia8::ParticleData::reInit(std::string, bool) --> bool", pybind11::arg("startFile"), pybind11::arg("xmlFormat"));
		cl.def("readXML", [](Pythia8::ParticleData &o, class std::basic_string<char> const & a0) -> bool { return o.readXML(a0); }, "", pybind11::arg("inFile"));
		cl.def("readXML", (bool (Pythia8::ParticleData::*)(std::string, bool)) &Pythia8::ParticleData::readXML, "C++: Pythia8::ParticleData::readXML(std::string, bool) --> bool", pybind11::arg("inFile"), pybind11::arg("reset"));
		cl.def("listXML", (void (Pythia8::ParticleData::*)(std::string)) &Pythia8::ParticleData::listXML, "C++: Pythia8::ParticleData::listXML(std::string) --> void", pybind11::arg("outFile"));
		cl.def("readXML", [](Pythia8::ParticleData &o, class std::basic_istream<char> & a0) -> bool { return o.readXML(a0); }, "", pybind11::arg("is"));
		cl.def("readXML", (bool (Pythia8::ParticleData::*)(class std::basic_istream<char> &, bool)) &Pythia8::ParticleData::readXML, "C++: Pythia8::ParticleData::readXML(class std::basic_istream<char> &, bool) --> bool", pybind11::arg("is"), pybind11::arg("reset"));
		cl.def("copyXML", (bool (Pythia8::ParticleData::*)(const class Pythia8::ParticleData &)) &Pythia8::ParticleData::copyXML, "C++: Pythia8::ParticleData::copyXML(const class Pythia8::ParticleData &) --> bool", pybind11::arg("particleDataIn"));
		cl.def("loadXML", [](Pythia8::ParticleData &o, class std::basic_string<char> const & a0) -> bool { return o.loadXML(a0); }, "", pybind11::arg("inFile"));
		cl.def("loadXML", (bool (Pythia8::ParticleData::*)(std::string, bool)) &Pythia8::ParticleData::loadXML, "C++: Pythia8::ParticleData::loadXML(std::string, bool) --> bool", pybind11::arg("inFile"), pybind11::arg("reset"));
		cl.def("loadXML", [](Pythia8::ParticleData &o, class std::basic_istream<char> & a0) -> bool { return o.loadXML(a0); }, "", pybind11::arg("is"));
		cl.def("loadXML", (bool (Pythia8::ParticleData::*)(class std::basic_istream<char> &, bool)) &Pythia8::ParticleData::loadXML, "C++: Pythia8::ParticleData::loadXML(class std::basic_istream<char> &, bool) --> bool", pybind11::arg("is"), pybind11::arg("reset"));
		cl.def("processXML", [](Pythia8::ParticleData &o) -> bool { return o.processXML(); }, "");
		cl.def("processXML", (bool (Pythia8::ParticleData::*)(bool)) &Pythia8::ParticleData::processXML, "C++: Pythia8::ParticleData::processXML(bool) --> bool", pybind11::arg("reset"));
		cl.def("readFF", [](Pythia8::ParticleData &o, class std::basic_string<char> const & a0) -> bool { return o.readFF(a0); }, "", pybind11::arg("inFile"));
		cl.def("readFF", (bool (Pythia8::ParticleData::*)(std::string, bool)) &Pythia8::ParticleData::readFF, "C++: Pythia8::ParticleData::readFF(std::string, bool) --> bool", pybind11::arg("inFile"), pybind11::arg("reset"));
		cl.def("readFF", [](Pythia8::ParticleData &o, class std::basic_istream<char> & a0) -> bool { return o.readFF(a0); }, "", pybind11::arg("is"));
		cl.def("readFF", (bool (Pythia8::ParticleData::*)(class std::basic_istream<char> &, bool)) &Pythia8::ParticleData::readFF, "C++: Pythia8::ParticleData::readFF(class std::basic_istream<char> &, bool) --> bool", pybind11::arg("is"), pybind11::arg("reset"));
		cl.def("listFF", (void (Pythia8::ParticleData::*)(std::string)) &Pythia8::ParticleData::listFF, "C++: Pythia8::ParticleData::listFF(std::string) --> void", pybind11::arg("outFile"));
		cl.def("readString", [](Pythia8::ParticleData &o, class std::basic_string<char> const & a0) -> bool { return o.readString(a0); }, "", pybind11::arg("lineIn"));
		cl.def("readString", (bool (Pythia8::ParticleData::*)(std::string, bool)) &Pythia8::ParticleData::readString, "C++: Pythia8::ParticleData::readString(std::string, bool) --> bool", pybind11::arg("lineIn"), pybind11::arg("warn"));
		cl.def("readingFailed", (bool (Pythia8::ParticleData::*)()) &Pythia8::ParticleData::readingFailed, "C++: Pythia8::ParticleData::readingFailed() --> bool");
		cl.def("listAll", (void (Pythia8::ParticleData::*)(std::ostream &)) &Pythia8::ParticleData::listAll, "C++: Pythia8::ParticleData::listAll(std::ostream &) --> void", pybind11::arg("stream"));
		cl.def("listAll", (void (Pythia8::ParticleData::*)()) &Pythia8::ParticleData::listAll, "C++: Pythia8::ParticleData::listAll() --> void");
		cl.def("listChanged", [](Pythia8::ParticleData &o) -> void { return o.listChanged(); }, "");
		cl.def("listChanged", (void (Pythia8::ParticleData::*)(bool)) &Pythia8::ParticleData::listChanged, "C++: Pythia8::ParticleData::listChanged(bool) --> void", pybind11::arg("changedRes"));
		cl.def("list", [](Pythia8::ParticleData &o, class std::basic_ostream<char> & a0) -> void { return o.list(a0); }, "", pybind11::arg("stream"));
		cl.def("list", [](Pythia8::ParticleData &o, class std::basic_ostream<char> & a0, bool const & a1) -> void { return o.list(a0, a1); }, "", pybind11::arg("stream"), pybind11::arg("chargedOnly"));
		cl.def("list", (void (Pythia8::ParticleData::*)(std::ostream &, bool, bool)) &Pythia8::ParticleData::list, "C++: Pythia8::ParticleData::list(std::ostream &, bool, bool) --> void", pybind11::arg("stream"), pybind11::arg("chargedOnly"), pybind11::arg("changedRes"));
		cl.def("list", [](Pythia8::ParticleData &o) -> void { return o.list(); }, "");
		cl.def("list", [](Pythia8::ParticleData &o, bool const & a0) -> void { return o.list(a0); }, "", pybind11::arg("changedOnly"));
		cl.def("list", (void (Pythia8::ParticleData::*)(bool, bool)) &Pythia8::ParticleData::list, "C++: Pythia8::ParticleData::list(bool, bool) --> void", pybind11::arg("changedOnly"), pybind11::arg("changedRes"));
		cl.def("list", (void (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::list, "C++: Pythia8::ParticleData::list(int) --> void", pybind11::arg("idList"));
		cl.def("list", (void (Pythia8::ParticleData::*)(class std::vector<int, class std::allocator<int> >)) &Pythia8::ParticleData::list, "C++: Pythia8::ParticleData::list(class std::vector<int, class std::allocator<int> >) --> void", pybind11::arg("idList"));
		cl.def("getReadHistory", [](Pythia8::ParticleData &o) -> std::vector<std::string, class std::allocator<std::string > > { return o.getReadHistory(); }, "");
		cl.def("getReadHistory", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::getReadHistory, "C++: Pythia8::ParticleData::getReadHistory(int) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("subrun"));
		cl.def("checkTable", [](Pythia8::ParticleData &o) -> void { return o.checkTable(); }, "");
		cl.def("checkTable", (void (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::checkTable, "C++: Pythia8::ParticleData::checkTable(int) --> void", pybind11::arg("verbosity"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0) -> void { return o.addParticle(a0); }, "", pybind11::arg("idIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1) -> void { return o.addParticle(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2) -> void { return o.addParticle(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3) -> void { return o.addParticle(a0, a1, a2, a3); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4) -> void { return o.addParticle(a0, a1, a2, a3, a4); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8, double const & a9) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def("addParticle", (void (Pythia8::ParticleData::*)(int, std::string, int, int, int, double, double, double, double, double, bool)) &Pythia8::ParticleData::addParticle, "C++: Pythia8::ParticleData::addParticle(int, std::string, int, int, int, double, double, double, double, double, bool) --> void", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2) -> void { return o.addParticle(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3) -> void { return o.addParticle(a0, a1, a2, a3); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4) -> void { return o.addParticle(a0, a1, a2, a3, a4); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def("addParticle", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9, double const & a10) -> void { return o.addParticle(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def("addParticle", (void (Pythia8::ParticleData::*)(int, std::string, std::string, int, int, int, double, double, double, double, double, bool)) &Pythia8::ParticleData::addParticle, "C++: Pythia8::ParticleData::addParticle(int, std::string, std::string, int, int, int, double, double, double, double, double, bool) --> void", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2) -> void { return o.setAll(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3) -> void { return o.setAll(a0, a1, a2, a3); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4) -> void { return o.setAll(a0, a1, a2, a3, a4); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5) -> void { return o.setAll(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def("setAll", [](Pythia8::ParticleData &o, int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9, double const & a10) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, "", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def("setAll", (void (Pythia8::ParticleData::*)(int, std::string, std::string, int, int, int, double, double, double, double, double, bool)) &Pythia8::ParticleData::setAll, "C++: Pythia8::ParticleData::setAll(int, std::string, std::string, int, int, int, double, double, double, double, double, bool) --> void", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn"));
		cl.def("isParticle", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isParticle, "C++: Pythia8::ParticleData::isParticle(int) const --> bool", pybind11::arg("idIn"));
		cl.def("findParticle", (class std::shared_ptr<class Pythia8::ParticleDataEntry> (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::findParticle, "C++: Pythia8::ParticleData::findParticle(int) --> class std::shared_ptr<class Pythia8::ParticleDataEntry>", pybind11::arg("idIn"));
		cl.def("nextId", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::nextId, "C++: Pythia8::ParticleData::nextId(int) const --> int", pybind11::arg("idIn"));
		cl.def("name", (void (Pythia8::ParticleData::*)(int, std::string)) &Pythia8::ParticleData::name, "C++: Pythia8::ParticleData::name(int, std::string) --> void", pybind11::arg("idIn"), pybind11::arg("nameIn"));
		cl.def("antiName", (void (Pythia8::ParticleData::*)(int, std::string)) &Pythia8::ParticleData::antiName, "C++: Pythia8::ParticleData::antiName(int, std::string) --> void", pybind11::arg("idIn"), pybind11::arg("antiNameIn"));
		cl.def("names", (void (Pythia8::ParticleData::*)(int, std::string, std::string)) &Pythia8::ParticleData::names, "C++: Pythia8::ParticleData::names(int, std::string, std::string) --> void", pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def("spinType", (void (Pythia8::ParticleData::*)(int, int)) &Pythia8::ParticleData::spinType, "C++: Pythia8::ParticleData::spinType(int, int) --> void", pybind11::arg("idIn"), pybind11::arg("spinTypeIn"));
		cl.def("chargeType", (void (Pythia8::ParticleData::*)(int, int)) &Pythia8::ParticleData::chargeType, "C++: Pythia8::ParticleData::chargeType(int, int) --> void", pybind11::arg("idIn"), pybind11::arg("chargeTypeIn"));
		cl.def("colType", (void (Pythia8::ParticleData::*)(int, int)) &Pythia8::ParticleData::colType, "C++: Pythia8::ParticleData::colType(int, int) --> void", pybind11::arg("idIn"), pybind11::arg("colTypeIn"));
		cl.def("m0", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::m0, "C++: Pythia8::ParticleData::m0(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("m0In"));
		cl.def("mWidth", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::mWidth, "C++: Pythia8::ParticleData::mWidth(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("mWidthIn"));
		cl.def("mMin", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::mMin, "C++: Pythia8::ParticleData::mMin(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("mMinIn"));
		cl.def("mMax", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::mMax, "C++: Pythia8::ParticleData::mMax(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("mMaxIn"));
		cl.def("tau0", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::tau0, "C++: Pythia8::ParticleData::tau0(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("tau0In"));
		cl.def("isResonance", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::isResonance, "C++: Pythia8::ParticleData::isResonance(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("isResonanceIn"));
		cl.def("mayDecay", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::mayDecay, "C++: Pythia8::ParticleData::mayDecay(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("mayDecayIn"));
		cl.def("tauCalc", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::tauCalc, "C++: Pythia8::ParticleData::tauCalc(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("tauCalcIn"));
		cl.def("doExternalDecay", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::doExternalDecay, "C++: Pythia8::ParticleData::doExternalDecay(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("doExternalDecayIn"));
		cl.def("varWidth", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::varWidth, "C++: Pythia8::ParticleData::varWidth(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("varWidthIn"));
		cl.def("isVisible", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::isVisible, "C++: Pythia8::ParticleData::isVisible(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("isVisibleIn"));
		cl.def("doForceWidth", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::doForceWidth, "C++: Pythia8::ParticleData::doForceWidth(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("doForceWidthIn"));
		cl.def("hasChanged", (void (Pythia8::ParticleData::*)(int, bool)) &Pythia8::ParticleData::hasChanged, "C++: Pythia8::ParticleData::hasChanged(int, bool) --> void", pybind11::arg("idIn"), pybind11::arg("hasChangedIn"));
		cl.def("hasAnti", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::hasAnti, "C++: Pythia8::ParticleData::hasAnti(int) const --> bool", pybind11::arg("idIn"));
		cl.def("antiId", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::antiId, "C++: Pythia8::ParticleData::antiId(int) const --> int", pybind11::arg("idIn"));
		cl.def("name", (std::string (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::name, "C++: Pythia8::ParticleData::name(int) const --> std::string", pybind11::arg("idIn"));
		cl.def("spinType", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::spinType, "C++: Pythia8::ParticleData::spinType(int) const --> int", pybind11::arg("idIn"));
		cl.def("chargeType", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::chargeType, "C++: Pythia8::ParticleData::chargeType(int) const --> int", pybind11::arg("idIn"));
		cl.def("charge", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::charge, "C++: Pythia8::ParticleData::charge(int) const --> double", pybind11::arg("idIn"));
		cl.def("colType", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::colType, "C++: Pythia8::ParticleData::colType(int) const --> int", pybind11::arg("idIn"));
		cl.def("m0", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::m0, "C++: Pythia8::ParticleData::m0(int) const --> double", pybind11::arg("idIn"));
		cl.def("mWidth", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::mWidth, "C++: Pythia8::ParticleData::mWidth(int) const --> double", pybind11::arg("idIn"));
		cl.def("mMin", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::mMin, "C++: Pythia8::ParticleData::mMin(int) const --> double", pybind11::arg("idIn"));
		cl.def("m0Min", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::m0Min, "C++: Pythia8::ParticleData::m0Min(int) const --> double", pybind11::arg("idIn"));
		cl.def("mMax", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::mMax, "C++: Pythia8::ParticleData::mMax(int) const --> double", pybind11::arg("idIn"));
		cl.def("m0Max", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::m0Max, "C++: Pythia8::ParticleData::m0Max(int) const --> double", pybind11::arg("idIn"));
		cl.def("tau0", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::tau0, "C++: Pythia8::ParticleData::tau0(int) const --> double", pybind11::arg("idIn"));
		cl.def("isResonance", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isResonance, "C++: Pythia8::ParticleData::isResonance(int) const --> bool", pybind11::arg("idIn"));
		cl.def("mayDecay", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::mayDecay, "C++: Pythia8::ParticleData::mayDecay(int) const --> bool", pybind11::arg("idIn"));
		cl.def("tauCalc", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::tauCalc, "C++: Pythia8::ParticleData::tauCalc(int) const --> bool", pybind11::arg("idIn"));
		cl.def("doExternalDecay", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::doExternalDecay, "C++: Pythia8::ParticleData::doExternalDecay(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isVisible", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isVisible, "C++: Pythia8::ParticleData::isVisible(int) const --> bool", pybind11::arg("idIn"));
		cl.def("doForceWidth", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::doForceWidth, "C++: Pythia8::ParticleData::doForceWidth(int) const --> bool", pybind11::arg("idIn"));
		cl.def("hasChanged", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::hasChanged, "C++: Pythia8::ParticleData::hasChanged(int) const --> bool", pybind11::arg("idIn"));
		cl.def("hasChangedMMin", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::hasChangedMMin, "C++: Pythia8::ParticleData::hasChangedMMin(int) const --> bool", pybind11::arg("idIn"));
		cl.def("hasChangedMMax", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::hasChangedMMax, "C++: Pythia8::ParticleData::hasChangedMMax(int) const --> bool", pybind11::arg("idIn"));
		cl.def("useBreitWigner", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::useBreitWigner, "C++: Pythia8::ParticleData::useBreitWigner(int) const --> bool", pybind11::arg("idIn"));
		cl.def("varWidth", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::varWidth, "C++: Pythia8::ParticleData::varWidth(int) const --> bool", pybind11::arg("idIn"));
		cl.def("constituentMass", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::constituentMass, "C++: Pythia8::ParticleData::constituentMass(int) const --> double", pybind11::arg("idIn"));
		cl.def("mSel", (double (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::mSel, "C++: Pythia8::ParticleData::mSel(int) const --> double", pybind11::arg("idIn"));
		cl.def("mRun", (double (Pythia8::ParticleData::*)(int, double) const) &Pythia8::ParticleData::mRun, "C++: Pythia8::ParticleData::mRun(int, double) const --> double", pybind11::arg("idIn"), pybind11::arg("mH"));
		cl.def("canDecay", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::canDecay, "C++: Pythia8::ParticleData::canDecay(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isLepton", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isLepton, "C++: Pythia8::ParticleData::isLepton(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isQuark", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isQuark, "C++: Pythia8::ParticleData::isQuark(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isGluon", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isGluon, "C++: Pythia8::ParticleData::isGluon(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isDiquark", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isDiquark, "C++: Pythia8::ParticleData::isDiquark(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isParton", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isParton, "C++: Pythia8::ParticleData::isParton(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isHadron", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isHadron, "C++: Pythia8::ParticleData::isHadron(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isMeson", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isMeson, "C++: Pythia8::ParticleData::isMeson(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isBaryon", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isBaryon, "C++: Pythia8::ParticleData::isBaryon(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isOnium", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isOnium, "C++: Pythia8::ParticleData::isOnium(int) const --> bool", pybind11::arg("idIn"));
		cl.def("isOctetHadron", (bool (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::isOctetHadron, "C++: Pythia8::ParticleData::isOctetHadron(int) const --> bool", pybind11::arg("idIn"));
		cl.def("heaviestQuark", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::heaviestQuark, "C++: Pythia8::ParticleData::heaviestQuark(int) const --> int", pybind11::arg("idIn"));
		cl.def("baryonNumberType", (int (Pythia8::ParticleData::*)(int) const) &Pythia8::ParticleData::baryonNumberType, "C++: Pythia8::ParticleData::baryonNumberType(int) const --> int", pybind11::arg("idIn"));
		cl.def("nQuarksInCode", (int (Pythia8::ParticleData::*)(int, int) const) &Pythia8::ParticleData::nQuarksInCode, "C++: Pythia8::ParticleData::nQuarksInCode(int, int) const --> int", pybind11::arg("idIn"), pybind11::arg("idQIn"));
		cl.def("rescaleBR", [](Pythia8::ParticleData &o, int const & a0) -> void { return o.rescaleBR(a0); }, "", pybind11::arg("idIn"));
		cl.def("rescaleBR", (void (Pythia8::ParticleData::*)(int, double)) &Pythia8::ParticleData::rescaleBR, "C++: Pythia8::ParticleData::rescaleBR(int, double) --> void", pybind11::arg("idIn"), pybind11::arg("newSumBR"));
		cl.def("resInit", (void (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::resInit, "C++: Pythia8::ParticleData::resInit(int) --> void", pybind11::arg("idIn"));
		cl.def("resWidth", [](Pythia8::ParticleData &o, int const & a0, double const & a1) -> double { return o.resWidth(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"));
		cl.def("resWidth", [](Pythia8::ParticleData &o, int const & a0, double const & a1, int const & a2) -> double { return o.resWidth(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"));
		cl.def("resWidth", [](Pythia8::ParticleData &o, int const & a0, double const & a1, int const & a2, bool const & a3) -> double { return o.resWidth(a0, a1, a2, a3); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"), pybind11::arg("openOnly"));
		cl.def("resWidth", (double (Pythia8::ParticleData::*)(int, double, int, bool, bool)) &Pythia8::ParticleData::resWidth, "C++: Pythia8::ParticleData::resWidth(int, double, int, bool, bool) --> double", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"), pybind11::arg("openOnly"), pybind11::arg("setBR"));
		cl.def("resWidthOpen", [](Pythia8::ParticleData &o, int const & a0, double const & a1) -> double { return o.resWidthOpen(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"));
		cl.def("resWidthOpen", (double (Pythia8::ParticleData::*)(int, double, int)) &Pythia8::ParticleData::resWidthOpen, "C++: Pythia8::ParticleData::resWidthOpen(int, double, int) --> double", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"));
		cl.def("resWidthStore", [](Pythia8::ParticleData &o, int const & a0, double const & a1) -> double { return o.resWidthStore(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"));
		cl.def("resWidthStore", (double (Pythia8::ParticleData::*)(int, double, int)) &Pythia8::ParticleData::resWidthStore, "C++: Pythia8::ParticleData::resWidthStore(int, double, int) --> double", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"));
		cl.def("resOpenFrac", [](Pythia8::ParticleData &o, int const & a0) -> double { return o.resOpenFrac(a0); }, "", pybind11::arg("id1In"));
		cl.def("resOpenFrac", [](Pythia8::ParticleData &o, int const & a0, int const & a1) -> double { return o.resOpenFrac(a0, a1); }, "", pybind11::arg("id1In"), pybind11::arg("id2In"));
		cl.def("resOpenFrac", (double (Pythia8::ParticleData::*)(int, int, int)) &Pythia8::ParticleData::resOpenFrac, "C++: Pythia8::ParticleData::resOpenFrac(int, int, int) --> double", pybind11::arg("id1In"), pybind11::arg("id2In"), pybind11::arg("id3In"));
		cl.def("resWidthRescaleFactor", (double (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::resWidthRescaleFactor, "C++: Pythia8::ParticleData::resWidthRescaleFactor(int) --> double", pybind11::arg("idIn"));
		cl.def("resWidthChan", [](Pythia8::ParticleData &o, int const & a0, double const & a1) -> double { return o.resWidthChan(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"));
		cl.def("resWidthChan", [](Pythia8::ParticleData &o, int const & a0, double const & a1, int const & a2) -> double { return o.resWidthChan(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idAbs1"));
		cl.def("resWidthChan", (double (Pythia8::ParticleData::*)(int, double, int, int)) &Pythia8::ParticleData::resWidthChan, "C++: Pythia8::ParticleData::resWidthChan(int, double, int, int) --> double", pybind11::arg("idIn"), pybind11::arg("mHat"), pybind11::arg("idAbs1"), pybind11::arg("idAbs2"));
		cl.def("particleDataEntryPtr", (class std::shared_ptr<class Pythia8::ParticleDataEntry> (Pythia8::ParticleData::*)(int)) &Pythia8::ParticleData::particleDataEntryPtr, "C++: Pythia8::ParticleData::particleDataEntryPtr(int) --> class std::shared_ptr<class Pythia8::ParticleDataEntry>", pybind11::arg("idIn"));
		cl.def("getIsInit", (bool (Pythia8::ParticleData::*)()) &Pythia8::ParticleData::getIsInit, "C++: Pythia8::ParticleData::getIsInit() --> bool");
	}
}
