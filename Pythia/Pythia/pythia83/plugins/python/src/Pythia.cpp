#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/Event.h>
#include <Pythia8/GammaKinematics.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/LesHouches.h>
#include <Pythia8/Merging.h>
#include <Pythia8/MergingHooks.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ParticleDecays.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/PartonVertex.h>
#include <Pythia8/PhaseSpace.h>
#include <Pythia8/Pythia.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/SLHAinterface.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SharedPointers.h>
#include <Pythia8/ShowerModel.h>
#include <Pythia8/SigmaLowEnergy.h>
#include <Pythia8/SigmaProcess.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/UserHooks.h>
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

void bind_Pythia8_Pythia(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::Pythia file:Pythia8/Pythia.h line:69
		pybind11::class_<Pythia8::Pythia, std::shared_ptr<Pythia8::Pythia>> cl(M("Pythia8"), "Pythia", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Pythia(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::Pythia(a0); } ), "doc" , pybind11::arg("xmlDir"));
		cl.def( pybind11::init<std::string, bool>(), pybind11::arg("xmlDir"), pybind11::arg("printBanner") );

		cl.def( pybind11::init( [](class Pythia8::Settings & a0, class Pythia8::ParticleData & a1){ return new Pythia8::Pythia(a0, a1); } ), "doc" , pybind11::arg("settingsIn"), pybind11::arg("particleDataIn"));
		cl.def( pybind11::init<class Pythia8::Settings &, class Pythia8::ParticleData &, bool>(), pybind11::arg("settingsIn"), pybind11::arg("particleDataIn"), pybind11::arg("printBanner") );

		cl.def( pybind11::init( [](class std::basic_istream<char> & a0, class std::basic_istream<char> & a1){ return new Pythia8::Pythia(a0, a1); } ), "doc" , pybind11::arg("settingsStrings"), pybind11::arg("particleDataStrings"));
		cl.def( pybind11::init<class std::basic_istream<char> &, class std::basic_istream<char> &, bool>(), pybind11::arg("settingsStrings"), pybind11::arg("particleDataStrings"), pybind11::arg("printBanner") );

		cl.def_readwrite("process", &Pythia8::Pythia::process);
		cl.def_readwrite("event", &Pythia8::Pythia::event);
		cl.def_readwrite("settings", &Pythia8::Pythia::settings);
		cl.def_readwrite("particleData", &Pythia8::Pythia::particleData);
		cl.def_readwrite("rndm", &Pythia8::Pythia::rndm);
		cl.def_readwrite("coupSM", &Pythia8::Pythia::coupSM);
		cl.def_readwrite("coupSUSY", &Pythia8::Pythia::coupSUSY);
		cl.def_readwrite("slhaInterface", &Pythia8::Pythia::slhaInterface);
		cl.def_readwrite("partonSystems", &Pythia8::Pythia::partonSystems);
		cl.def_readwrite("mergingPtr", &Pythia8::Pythia::mergingPtr);
		cl.def_readwrite("mergingHooksPtr", &Pythia8::Pythia::mergingHooksPtr);
		cl.def_readwrite("heavyIonsPtr", &Pythia8::Pythia::heavyIonsPtr);
		cl.def_readwrite("hiHooksPtr", &Pythia8::Pythia::hiHooksPtr);
		cl.def_readwrite("hadronWidths", &Pythia8::Pythia::hadronWidths);
		cl.def_readwrite("beamA", &Pythia8::Pythia::beamA);
		cl.def_readwrite("beamB", &Pythia8::Pythia::beamB);
		cl.def("checkVersion", (bool (Pythia8::Pythia::*)()) &Pythia8::Pythia::checkVersion, "C++: Pythia8::Pythia::checkVersion() --> bool");
		cl.def("readString", [](Pythia8::Pythia &o, class std::basic_string<char> const & a0) -> bool { return o.readString(a0); }, "", pybind11::arg(""));
		cl.def("readString", (bool (Pythia8::Pythia::*)(std::string, bool)) &Pythia8::Pythia::readString, "C++: Pythia8::Pythia::readString(std::string, bool) --> bool", pybind11::arg(""), pybind11::arg("warn"));
		cl.def("readFile", [](Pythia8::Pythia &o, class std::basic_string<char> const & a0) -> bool { return o.readFile(a0); }, "", pybind11::arg("fileName"));
		cl.def("readFile", [](Pythia8::Pythia &o, class std::basic_string<char> const & a0, bool const & a1) -> bool { return o.readFile(a0, a1); }, "", pybind11::arg("fileName"), pybind11::arg("warn"));
		cl.def("readFile", (bool (Pythia8::Pythia::*)(std::string, bool, int)) &Pythia8::Pythia::readFile, "C++: Pythia8::Pythia::readFile(std::string, bool, int) --> bool", pybind11::arg("fileName"), pybind11::arg("warn"), pybind11::arg("subrun"));
		cl.def("readFile", (bool (Pythia8::Pythia::*)(std::string, int)) &Pythia8::Pythia::readFile, "C++: Pythia8::Pythia::readFile(std::string, int) --> bool", pybind11::arg("fileName"), pybind11::arg("subrun"));
		cl.def("readFile", [](Pythia8::Pythia &o) -> bool { return o.readFile(); }, "");
		cl.def("readFile", [](Pythia8::Pythia &o, class std::basic_istream<char> & a0) -> bool { return o.readFile(a0); }, "", pybind11::arg("is"));
		cl.def("readFile", [](Pythia8::Pythia &o, class std::basic_istream<char> & a0, bool const & a1) -> bool { return o.readFile(a0, a1); }, "", pybind11::arg("is"), pybind11::arg("warn"));
		cl.def("readFile", (bool (Pythia8::Pythia::*)(class std::basic_istream<char> &, bool, int)) &Pythia8::Pythia::readFile, "C++: Pythia8::Pythia::readFile(class std::basic_istream<char> &, bool, int) --> bool", pybind11::arg("is"), pybind11::arg("warn"), pybind11::arg("subrun"));
		cl.def("readFile", (bool (Pythia8::Pythia::*)(class std::basic_istream<char> &, int)) &Pythia8::Pythia::readFile, "C++: Pythia8::Pythia::readFile(class std::basic_istream<char> &, int) --> bool", pybind11::arg("is"), pybind11::arg("subrun"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1) -> bool { return o.setPDFPtr(a0, a1); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2) -> bool { return o.setPDFPtr(a0, a1, a2); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3) -> bool { return o.setPDFPtr(a0, a1, a2, a3); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9, class std::shared_ptr<class Pythia8::PDF> const & a10) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9, class std::shared_ptr<class Pythia8::PDF> const & a10, class std::shared_ptr<class Pythia8::PDF> const & a11) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"), pybind11::arg("pdfUnresBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9, class std::shared_ptr<class Pythia8::PDF> const & a10, class std::shared_ptr<class Pythia8::PDF> const & a11, class std::shared_ptr<class Pythia8::PDF> const & a12) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"), pybind11::arg("pdfUnresBPtrIn"), pybind11::arg("pdfUnresGamAPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9, class std::shared_ptr<class Pythia8::PDF> const & a10, class std::shared_ptr<class Pythia8::PDF> const & a11, class std::shared_ptr<class Pythia8::PDF> const & a12, class std::shared_ptr<class Pythia8::PDF> const & a13) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"), pybind11::arg("pdfUnresBPtrIn"), pybind11::arg("pdfUnresGamAPtrIn"), pybind11::arg("pdfUnresGamBPtrIn"));
		cl.def("setPDFPtr", [](Pythia8::Pythia &o, class std::shared_ptr<class Pythia8::PDF> const & a0, class std::shared_ptr<class Pythia8::PDF> const & a1, class std::shared_ptr<class Pythia8::PDF> const & a2, class std::shared_ptr<class Pythia8::PDF> const & a3, class std::shared_ptr<class Pythia8::PDF> const & a4, class std::shared_ptr<class Pythia8::PDF> const & a5, class std::shared_ptr<class Pythia8::PDF> const & a6, class std::shared_ptr<class Pythia8::PDF> const & a7, class std::shared_ptr<class Pythia8::PDF> const & a8, class std::shared_ptr<class Pythia8::PDF> const & a9, class std::shared_ptr<class Pythia8::PDF> const & a10, class std::shared_ptr<class Pythia8::PDF> const & a11, class std::shared_ptr<class Pythia8::PDF> const & a12, class std::shared_ptr<class Pythia8::PDF> const & a13, class std::shared_ptr<class Pythia8::PDF> const & a14) -> bool { return o.setPDFPtr(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14); }, "", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"), pybind11::arg("pdfUnresBPtrIn"), pybind11::arg("pdfUnresGamAPtrIn"), pybind11::arg("pdfUnresGamBPtrIn"), pybind11::arg("pdfVMDAPtrIn"));
		cl.def("setPDFPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>)) &Pythia8::Pythia::setPDFPtr, "C++: Pythia8::Pythia::setPDFPtr(class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>) --> bool", pybind11::arg("pdfAPtrIn"), pybind11::arg("pdfBPtrIn"), pybind11::arg("pdfHardAPtrIn"), pybind11::arg("pdfHardBPtrIn"), pybind11::arg("pdfPomAPtrIn"), pybind11::arg("pdfPomBPtrIn"), pybind11::arg("pdfGamAPtrIn"), pybind11::arg("pdfGamBPtrIn"), pybind11::arg("pdfHardGamAPtrIn"), pybind11::arg("pdfHardGamBPtrIn"), pybind11::arg("pdfUnresAPtrIn"), pybind11::arg("pdfUnresBPtrIn"), pybind11::arg("pdfUnresGamAPtrIn"), pybind11::arg("pdfUnresGamBPtrIn"), pybind11::arg("pdfVMDAPtrIn"), pybind11::arg("pdfVMDBPtrIn"));
		cl.def("setPDFAPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::PDF>)) &Pythia8::Pythia::setPDFAPtr, "C++: Pythia8::Pythia::setPDFAPtr(class std::shared_ptr<class Pythia8::PDF>) --> bool", pybind11::arg("pdfAPtrIn"));
		cl.def("setPDFBPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::PDF>)) &Pythia8::Pythia::setPDFBPtr, "C++: Pythia8::Pythia::setPDFBPtr(class std::shared_ptr<class Pythia8::PDF>) --> bool", pybind11::arg("pdfBPtrIn"));
		cl.def("setPhotonFluxPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>)) &Pythia8::Pythia::setPhotonFluxPtr, "C++: Pythia8::Pythia::setPhotonFluxPtr(class std::shared_ptr<class Pythia8::PDF>, class std::shared_ptr<class Pythia8::PDF>) --> bool", pybind11::arg("photonFluxAIn"), pybind11::arg("photonFluxBIn"));
		cl.def("setLHAupPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::LHAup>)) &Pythia8::Pythia::setLHAupPtr, "C++: Pythia8::Pythia::setLHAupPtr(class std::shared_ptr<class Pythia8::LHAup>) --> bool", pybind11::arg("lhaUpPtrIn"));
		cl.def("setDecayPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::DecayHandler>, class std::vector<int, class std::allocator<int> >)) &Pythia8::Pythia::setDecayPtr, "C++: Pythia8::Pythia::setDecayPtr(class std::shared_ptr<class Pythia8::DecayHandler>, class std::vector<int, class std::allocator<int> >) --> bool", pybind11::arg("decayHandlePtrIn"), pybind11::arg("handledParticlesIn"));
		cl.def("setRndmEnginePtr", (bool (Pythia8::Pythia::*)(class Pythia8::RndmEngine *)) &Pythia8::Pythia::setRndmEnginePtr, "C++: Pythia8::Pythia::setRndmEnginePtr(class Pythia8::RndmEngine *) --> bool", pybind11::arg("rndmEnginePtrIn"));
		cl.def("setUserHooksPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::UserHooks>)) &Pythia8::Pythia::setUserHooksPtr, "C++: Pythia8::Pythia::setUserHooksPtr(class std::shared_ptr<class Pythia8::UserHooks>) --> bool", pybind11::arg("userHooksPtrIn"));
		cl.def("addUserHooksPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::UserHooks>)) &Pythia8::Pythia::addUserHooksPtr, "C++: Pythia8::Pythia::addUserHooksPtr(class std::shared_ptr<class Pythia8::UserHooks>) --> bool", pybind11::arg("userHooksPtrIn"));
		cl.def("setMergingPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::Merging>)) &Pythia8::Pythia::setMergingPtr, "C++: Pythia8::Pythia::setMergingPtr(class std::shared_ptr<class Pythia8::Merging>) --> bool", pybind11::arg("mergingPtrIn"));
		cl.def("setMergingHooksPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::MergingHooks>)) &Pythia8::Pythia::setMergingHooksPtr, "C++: Pythia8::Pythia::setMergingHooksPtr(class std::shared_ptr<class Pythia8::MergingHooks>) --> bool", pybind11::arg("mergingHooksPtrIn"));
		cl.def("setBeamShapePtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::BeamShape>)) &Pythia8::Pythia::setBeamShapePtr, "C++: Pythia8::Pythia::setBeamShapePtr(class std::shared_ptr<class Pythia8::BeamShape>) --> bool", pybind11::arg("beamShapePtrIn"));
		cl.def("setShowerModelPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::ShowerModel>)) &Pythia8::Pythia::setShowerModelPtr, "C++: Pythia8::Pythia::setShowerModelPtr(class std::shared_ptr<class Pythia8::ShowerModel>) --> bool", pybind11::arg("showerModelPtrIn"));
		cl.def("setHeavyIonsPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::HeavyIons>)) &Pythia8::Pythia::setHeavyIonsPtr, "C++: Pythia8::Pythia::setHeavyIonsPtr(class std::shared_ptr<class Pythia8::HeavyIons>) --> bool", pybind11::arg("heavyIonsPtrIn"));
		cl.def("setHIHooks", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::HIUserHooks>)) &Pythia8::Pythia::setHIHooks, "C++: Pythia8::Pythia::setHIHooks(class std::shared_ptr<class Pythia8::HIUserHooks>) --> bool", pybind11::arg("hiHooksPtrIn"));
		cl.def("getHeavyIonsPtr", (class std::shared_ptr<class Pythia8::HeavyIons> (Pythia8::Pythia::*)()) &Pythia8::Pythia::getHeavyIonsPtr, "C++: Pythia8::Pythia::getHeavyIonsPtr() --> class std::shared_ptr<class Pythia8::HeavyIons>");
		cl.def("getBeamShapePtr", (class std::shared_ptr<class Pythia8::BeamShape> (Pythia8::Pythia::*)()) &Pythia8::Pythia::getBeamShapePtr, "C++: Pythia8::Pythia::getBeamShapePtr() --> class std::shared_ptr<class Pythia8::BeamShape>");
		cl.def("getShowerModelPtr", (class std::shared_ptr<class Pythia8::ShowerModel> (Pythia8::Pythia::*)()) &Pythia8::Pythia::getShowerModelPtr, "C++: Pythia8::Pythia::getShowerModelPtr() --> class std::shared_ptr<class Pythia8::ShowerModel>");
		cl.def("setPartonVertexPtr", (bool (Pythia8::Pythia::*)(class std::shared_ptr<class Pythia8::PartonVertex>)) &Pythia8::Pythia::setPartonVertexPtr, "C++: Pythia8::Pythia::setPartonVertexPtr(class std::shared_ptr<class Pythia8::PartonVertex>) --> bool", pybind11::arg("partonVertexPtrIn"));
		cl.def("init", (bool (Pythia8::Pythia::*)()) &Pythia8::Pythia::init, "C++: Pythia8::Pythia::init() --> bool");
		cl.def("next", (bool (Pythia8::Pythia::*)()) &Pythia8::Pythia::next, "C++: Pythia8::Pythia::next() --> bool");
		cl.def("next", (bool (Pythia8::Pythia::*)(int)) &Pythia8::Pythia::next, "C++: Pythia8::Pythia::next(int) --> bool", pybind11::arg("procTypeIn"));
		cl.def("setBeamIDs", [](Pythia8::Pythia &o, int const & a0) -> bool { return o.setBeamIDs(a0); }, "", pybind11::arg("idAin"));
		cl.def("setBeamIDs", (bool (Pythia8::Pythia::*)(int, int)) &Pythia8::Pythia::setBeamIDs, "C++: Pythia8::Pythia::setBeamIDs(int, int) --> bool", pybind11::arg("idAin"), pybind11::arg("idBin"));
		cl.def("setKinematics", (bool (Pythia8::Pythia::*)(double)) &Pythia8::Pythia::setKinematics, "C++: Pythia8::Pythia::setKinematics(double) --> bool", pybind11::arg("eCMIn"));
		cl.def("setKinematics", (bool (Pythia8::Pythia::*)(double, double)) &Pythia8::Pythia::setKinematics, "C++: Pythia8::Pythia::setKinematics(double, double) --> bool", pybind11::arg("eAIn"), pybind11::arg("eBIn"));
		cl.def("setKinematics", (bool (Pythia8::Pythia::*)(double, double, double, double, double, double)) &Pythia8::Pythia::setKinematics, "C++: Pythia8::Pythia::setKinematics(double, double, double, double, double, double) --> bool", pybind11::arg("pxAIn"), pybind11::arg("pyAIn"), pybind11::arg("pzAIn"), pybind11::arg("pxBIn"), pybind11::arg("pyBIn"), pybind11::arg("pzBIn"));
		cl.def("setKinematics", (bool (Pythia8::Pythia::*)(class Pythia8::Vec4, class Pythia8::Vec4)) &Pythia8::Pythia::setKinematics, "C++: Pythia8::Pythia::setKinematics(class Pythia8::Vec4, class Pythia8::Vec4) --> bool", pybind11::arg("pAIn"), pybind11::arg("pBIn"));
		cl.def("forceTimeShower", [](Pythia8::Pythia &o, int const & a0, int const & a1, double const & a2) -> int { return o.forceTimeShower(a0, a1, a2); }, "", pybind11::arg("iBeg"), pybind11::arg("iEnd"), pybind11::arg("pTmax"));
		cl.def("forceTimeShower", (int (Pythia8::Pythia::*)(int, int, double, int)) &Pythia8::Pythia::forceTimeShower, "C++: Pythia8::Pythia::forceTimeShower(int, int, double, int) --> int", pybind11::arg("iBeg"), pybind11::arg("iEnd"), pybind11::arg("pTmax"), pybind11::arg("nBranchMax"));
		cl.def("forceHadronLevel", [](Pythia8::Pythia &o) -> bool { return o.forceHadronLevel(); }, "");
		cl.def("forceHadronLevel", (bool (Pythia8::Pythia::*)(bool)) &Pythia8::Pythia::forceHadronLevel, "C++: Pythia8::Pythia::forceHadronLevel(bool) --> bool", pybind11::arg("findJunctions"));
		cl.def("moreDecays", (bool (Pythia8::Pythia::*)()) &Pythia8::Pythia::moreDecays, "C++: Pythia8::Pythia::moreDecays() --> bool");
		cl.def("moreDecays", (bool (Pythia8::Pythia::*)(int)) &Pythia8::Pythia::moreDecays, "C++: Pythia8::Pythia::moreDecays(int) --> bool", pybind11::arg("index"));
		cl.def("forceRHadronDecays", (bool (Pythia8::Pythia::*)()) &Pythia8::Pythia::forceRHadronDecays, "C++: Pythia8::Pythia::forceRHadronDecays() --> bool");
		cl.def("doLowEnergyProcess", (bool (Pythia8::Pythia::*)(int, int, int)) &Pythia8::Pythia::doLowEnergyProcess, "C++: Pythia8::Pythia::doLowEnergyProcess(int, int, int) --> bool", pybind11::arg("i1"), pybind11::arg("i2"), pybind11::arg("procTypeIn"));
		cl.def("getSigmaTotal", (double (Pythia8::Pythia::*)()) &Pythia8::Pythia::getSigmaTotal, "C++: Pythia8::Pythia::getSigmaTotal() --> double");
		cl.def("getSigmaTotal", [](Pythia8::Pythia &o, double const & a0) -> double { return o.getSigmaTotal(a0); }, "", pybind11::arg("eCM12"));
		cl.def("getSigmaTotal", (double (Pythia8::Pythia::*)(double, int)) &Pythia8::Pythia::getSigmaTotal, "C++: Pythia8::Pythia::getSigmaTotal(double, int) --> double", pybind11::arg("eCM12"), pybind11::arg("mixLoHi"));
		cl.def("getSigmaTotal", [](Pythia8::Pythia &o, int const & a0, int const & a1, double const & a2) -> double { return o.getSigmaTotal(a0, a1, a2); }, "", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"));
		cl.def("getSigmaTotal", (double (Pythia8::Pythia::*)(int, int, double, int)) &Pythia8::Pythia::getSigmaTotal, "C++: Pythia8::Pythia::getSigmaTotal(int, int, double, int) --> double", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("mixLoHi"));
		cl.def("getSigmaTotal", [](Pythia8::Pythia &o, int const & a0, int const & a1, double const & a2, double const & a3, double const & a4) -> double { return o.getSigmaTotal(a0, a1, a2, a3, a4); }, "", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("m1"), pybind11::arg("m2"));
		cl.def("getSigmaTotal", (double (Pythia8::Pythia::*)(int, int, double, double, double, int)) &Pythia8::Pythia::getSigmaTotal, "C++: Pythia8::Pythia::getSigmaTotal(int, int, double, double, double, int) --> double", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("m1"), pybind11::arg("m2"), pybind11::arg("mixLoHi"));
		cl.def("getSigmaPartial", (double (Pythia8::Pythia::*)(int)) &Pythia8::Pythia::getSigmaPartial, "C++: Pythia8::Pythia::getSigmaPartial(int) --> double", pybind11::arg("procTypeIn"));
		cl.def("getSigmaPartial", [](Pythia8::Pythia &o, double const & a0, int const & a1) -> double { return o.getSigmaPartial(a0, a1); }, "", pybind11::arg("eCM12"), pybind11::arg("procTypeIn"));
		cl.def("getSigmaPartial", (double (Pythia8::Pythia::*)(double, int, int)) &Pythia8::Pythia::getSigmaPartial, "C++: Pythia8::Pythia::getSigmaPartial(double, int, int) --> double", pybind11::arg("eCM12"), pybind11::arg("procTypeIn"), pybind11::arg("mixLoHi"));
		cl.def("getSigmaPartial", [](Pythia8::Pythia &o, int const & a0, int const & a1, double const & a2, int const & a3) -> double { return o.getSigmaPartial(a0, a1, a2, a3); }, "", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("procTypeIn"));
		cl.def("getSigmaPartial", (double (Pythia8::Pythia::*)(int, int, double, int, int)) &Pythia8::Pythia::getSigmaPartial, "C++: Pythia8::Pythia::getSigmaPartial(int, int, double, int, int) --> double", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("procTypeIn"), pybind11::arg("mixLoHi"));
		cl.def("getSigmaPartial", [](Pythia8::Pythia &o, int const & a0, int const & a1, double const & a2, double const & a3, double const & a4, int const & a5) -> double { return o.getSigmaPartial(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("m1"), pybind11::arg("m2"), pybind11::arg("procTypeIn"));
		cl.def("getSigmaPartial", (double (Pythia8::Pythia::*)(int, int, double, double, double, int, int)) &Pythia8::Pythia::getSigmaPartial, "C++: Pythia8::Pythia::getSigmaPartial(int, int, double, double, double, int, int) --> double", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("eCM12"), pybind11::arg("m1"), pybind11::arg("m2"), pybind11::arg("procTypeIn"), pybind11::arg("mixLoHi"));
		cl.def("LHAeventList", (void (Pythia8::Pythia::*)()) &Pythia8::Pythia::LHAeventList, "C++: Pythia8::Pythia::LHAeventList() --> void");
		cl.def("LHAeventSkip", (bool (Pythia8::Pythia::*)(int)) &Pythia8::Pythia::LHAeventSkip, "C++: Pythia8::Pythia::LHAeventSkip(int) --> bool", pybind11::arg("nSkip"));
		cl.def("stat", (void (Pythia8::Pythia::*)()) &Pythia8::Pythia::stat, "C++: Pythia8::Pythia::stat() --> void");
		cl.def("flag", (bool (Pythia8::Pythia::*)(std::string)) &Pythia8::Pythia::flag, "C++: Pythia8::Pythia::flag(std::string) --> bool", pybind11::arg("key"));
		cl.def("mode", (int (Pythia8::Pythia::*)(std::string)) &Pythia8::Pythia::mode, "C++: Pythia8::Pythia::mode(std::string) --> int", pybind11::arg("key"));
		cl.def("parm", (double (Pythia8::Pythia::*)(std::string)) &Pythia8::Pythia::parm, "C++: Pythia8::Pythia::parm(std::string) --> double", pybind11::arg("key"));
		cl.def("word", (std::string (Pythia8::Pythia::*)(std::string)) &Pythia8::Pythia::word, "C++: Pythia8::Pythia::word(std::string) --> std::string", pybind11::arg("key"));
		cl.def("getPDFPtr", [](Pythia8::Pythia &o, int const & a0) -> std::shared_ptr<class Pythia8::PDF> { return o.getPDFPtr(a0); }, "", pybind11::arg("idIn"));
		cl.def("getPDFPtr", [](Pythia8::Pythia &o, int const & a0, int const & a1) -> std::shared_ptr<class Pythia8::PDF> { return o.getPDFPtr(a0, a1); }, "", pybind11::arg("idIn"), pybind11::arg("sequence"));
		cl.def("getPDFPtr", [](Pythia8::Pythia &o, int const & a0, int const & a1, class std::basic_string<char> const & a2) -> std::shared_ptr<class Pythia8::PDF> { return o.getPDFPtr(a0, a1, a2); }, "", pybind11::arg("idIn"), pybind11::arg("sequence"), pybind11::arg("beam"));
		cl.def("getPDFPtr", (class std::shared_ptr<class Pythia8::PDF> (Pythia8::Pythia::*)(int, int, std::string, bool)) &Pythia8::Pythia::getPDFPtr, "C++: Pythia8::Pythia::getPDFPtr(int, int, std::string, bool) --> class std::shared_ptr<class Pythia8::PDF>", pybind11::arg("idIn"), pybind11::arg("sequence"), pybind11::arg("beam"), pybind11::arg("resolved"));
		cl.def("infoPython", (class Pythia8::Info (Pythia8::Pythia::*)()) &Pythia8::Pythia::infoPython, "C++: Pythia8::Pythia::infoPython() --> class Pythia8::Info");
	}
}
