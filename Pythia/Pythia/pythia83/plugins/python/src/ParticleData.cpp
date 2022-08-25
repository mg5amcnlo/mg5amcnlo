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
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <sstream> // __str__
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

void bind_Pythia8_ParticleData(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::DecayChannel file:Pythia8/ParticleData.h line:35
		pybind11::class_<Pythia8::DecayChannel, std::shared_ptr<Pythia8::DecayChannel>> cl(M("Pythia8"), "DecayChannel", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::DecayChannel(); } ), "doc" );
		cl.def( pybind11::init( [](int const & a0){ return new Pythia8::DecayChannel(a0); } ), "doc" , pybind11::arg("onModeIn"));
		cl.def( pybind11::init( [](int const & a0, double const & a1){ return new Pythia8::DecayChannel(a0, a1); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2){ return new Pythia8::DecayChannel(a0, a1, a2); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3){ return new Pythia8::DecayChannel(a0, a1, a2, a3); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4, a5); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4, a5, a6); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4, a5, a6, a7); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, int const & a8){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4, a5, a6, a7, a8); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"));
		cl.def( pybind11::init( [](int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, int const & a8, int const & a9){ return new Pythia8::DecayChannel(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); } ), "doc" , pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"), pybind11::arg("prod6"));
		cl.def( pybind11::init<int, double, int, int, int, int, int, int, int, int, int>(), pybind11::arg("onModeIn"), pybind11::arg("bRatioIn"), pybind11::arg("meModeIn"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"), pybind11::arg("prod6"), pybind11::arg("prod7") );

		cl.def( pybind11::init( [](Pythia8::DecayChannel const &o){ return new Pythia8::DecayChannel(o); } ) );
		cl.def("assign", (class Pythia8::DecayChannel & (Pythia8::DecayChannel::*)(const class Pythia8::DecayChannel &)) &Pythia8::DecayChannel::operator=, "C++: Pythia8::DecayChannel::operator=(const class Pythia8::DecayChannel &) --> class Pythia8::DecayChannel &", pybind11::return_value_policy::reference, pybind11::arg("oldDC"));
		cl.def("onMode", (void (Pythia8::DecayChannel::*)(int)) &Pythia8::DecayChannel::onMode, "C++: Pythia8::DecayChannel::onMode(int) --> void", pybind11::arg("onModeIn"));
		cl.def("bRatio", [](Pythia8::DecayChannel &o, double const & a0) -> void { return o.bRatio(a0); }, "", pybind11::arg("bRatioIn"));
		cl.def("bRatio", (void (Pythia8::DecayChannel::*)(double, bool)) &Pythia8::DecayChannel::bRatio, "C++: Pythia8::DecayChannel::bRatio(double, bool) --> void", pybind11::arg("bRatioIn"), pybind11::arg("countAsChanged"));
		cl.def("rescaleBR", (void (Pythia8::DecayChannel::*)(double)) &Pythia8::DecayChannel::rescaleBR, "C++: Pythia8::DecayChannel::rescaleBR(double) --> void", pybind11::arg("fac"));
		cl.def("meMode", (void (Pythia8::DecayChannel::*)(int)) &Pythia8::DecayChannel::meMode, "C++: Pythia8::DecayChannel::meMode(int) --> void", pybind11::arg("meModeIn"));
		cl.def("multiplicity", (void (Pythia8::DecayChannel::*)(int)) &Pythia8::DecayChannel::multiplicity, "C++: Pythia8::DecayChannel::multiplicity(int) --> void", pybind11::arg("multIn"));
		cl.def("product", (void (Pythia8::DecayChannel::*)(int, int)) &Pythia8::DecayChannel::product, "C++: Pythia8::DecayChannel::product(int, int) --> void", pybind11::arg("i"), pybind11::arg("prodIn"));
		cl.def("setHasChanged", (void (Pythia8::DecayChannel::*)(bool)) &Pythia8::DecayChannel::setHasChanged, "C++: Pythia8::DecayChannel::setHasChanged(bool) --> void", pybind11::arg("hasChangedIn"));
		cl.def("onMode", (int (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::onMode, "C++: Pythia8::DecayChannel::onMode() const --> int");
		cl.def("bRatio", (double (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::bRatio, "C++: Pythia8::DecayChannel::bRatio() const --> double");
		cl.def("meMode", (int (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::meMode, "C++: Pythia8::DecayChannel::meMode() const --> int");
		cl.def("multiplicity", (int (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::multiplicity, "C++: Pythia8::DecayChannel::multiplicity() const --> int");
		cl.def("product", (int (Pythia8::DecayChannel::*)(int) const) &Pythia8::DecayChannel::product, "C++: Pythia8::DecayChannel::product(int) const --> int", pybind11::arg("i"));
		cl.def("hasChanged", (bool (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::hasChanged, "C++: Pythia8::DecayChannel::hasChanged() const --> bool");
		cl.def("contains", (bool (Pythia8::DecayChannel::*)(int) const) &Pythia8::DecayChannel::contains, "C++: Pythia8::DecayChannel::contains(int) const --> bool", pybind11::arg("id1"));
		cl.def("contains", (bool (Pythia8::DecayChannel::*)(int, int) const) &Pythia8::DecayChannel::contains, "C++: Pythia8::DecayChannel::contains(int, int) const --> bool", pybind11::arg("id1"), pybind11::arg("id2"));
		cl.def("contains", (bool (Pythia8::DecayChannel::*)(int, int, int) const) &Pythia8::DecayChannel::contains, "C++: Pythia8::DecayChannel::contains(int, int, int) const --> bool", pybind11::arg("id1"), pybind11::arg("id2"), pybind11::arg("id3"));
		cl.def("currentBR", (void (Pythia8::DecayChannel::*)(double)) &Pythia8::DecayChannel::currentBR, "C++: Pythia8::DecayChannel::currentBR(double) --> void", pybind11::arg("currentBRIn"));
		cl.def("currentBR", (double (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::currentBR, "C++: Pythia8::DecayChannel::currentBR() const --> double");
		cl.def("onShellWidth", (void (Pythia8::DecayChannel::*)(double)) &Pythia8::DecayChannel::onShellWidth, "C++: Pythia8::DecayChannel::onShellWidth(double) --> void", pybind11::arg("onShellWidthIn"));
		cl.def("onShellWidth", (double (Pythia8::DecayChannel::*)() const) &Pythia8::DecayChannel::onShellWidth, "C++: Pythia8::DecayChannel::onShellWidth() const --> double");
		cl.def("onShellWidthFactor", (void (Pythia8::DecayChannel::*)(double)) &Pythia8::DecayChannel::onShellWidthFactor, "C++: Pythia8::DecayChannel::onShellWidthFactor(double) --> void", pybind11::arg("factor"));
		cl.def("openSec", (void (Pythia8::DecayChannel::*)(int, double)) &Pythia8::DecayChannel::openSec, "C++: Pythia8::DecayChannel::openSec(int, double) --> void", pybind11::arg("idSgn"), pybind11::arg("openSecIn"));
		cl.def("openSec", (double (Pythia8::DecayChannel::*)(int) const) &Pythia8::DecayChannel::openSec, "C++: Pythia8::DecayChannel::openSec(int) const --> double", pybind11::arg("idSgn"));
	}
	{ // Pythia8::ParticleDataEntry file:Pythia8/ParticleData.h line:125
		pybind11::class_<Pythia8::ParticleDataEntry, std::shared_ptr<Pythia8::ParticleDataEntry>> cl(M("Pythia8"), "ParticleDataEntry", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::ParticleDataEntry(); } ), "doc" );
		cl.def( pybind11::init( [](int const & a0){ return new Pythia8::ParticleDataEntry(a0); } ), "doc" , pybind11::arg("idIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1){ return new Pythia8::ParticleDataEntry(a0, a1); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2){ return new Pythia8::ParticleDataEntry(a0, a1, a2); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7, a8); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8, double const & a9){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def( pybind11::init<int, std::string, int, int, int, double, double, double, double, double, bool>(), pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn") );

		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2){ return new Pythia8::ParticleDataEntry(a0, a1, a2); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7, a8); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def( pybind11::init( [](int const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, int const & a3, int const & a4, int const & a5, double const & a6, double const & a7, double const & a8, double const & a9, double const & a10){ return new Pythia8::ParticleDataEntry(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); } ), "doc" , pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def( pybind11::init<int, std::string, std::string, int, int, int, double, double, double, double, double, bool>(), pybind11::arg("idIn"), pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn") );

		cl.def( pybind11::init( [](Pythia8::ParticleDataEntry const &o){ return new Pythia8::ParticleDataEntry(o); } ) );
		cl.def("assign", (class Pythia8::ParticleDataEntry & (Pythia8::ParticleDataEntry::*)(const class Pythia8::ParticleDataEntry &)) &Pythia8::ParticleDataEntry::operator=, "C++: Pythia8::ParticleDataEntry::operator=(const class Pythia8::ParticleDataEntry &) --> class Pythia8::ParticleDataEntry &", pybind11::return_value_policy::reference, pybind11::arg("oldPDE"));
		cl.def("setDefaults", (void (Pythia8::ParticleDataEntry::*)()) &Pythia8::ParticleDataEntry::setDefaults, "C++: Pythia8::ParticleDataEntry::setDefaults() --> void");
		cl.def("initPtr", (void (Pythia8::ParticleDataEntry::*)(class Pythia8::ParticleData *)) &Pythia8::ParticleDataEntry::initPtr, "C++: Pythia8::ParticleDataEntry::initPtr(class Pythia8::ParticleData *) --> void", pybind11::arg("particleDataPtrIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.setAll(a0, a1); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2) -> void { return o.setAll(a0, a1, a2); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3) -> void { return o.setAll(a0, a1, a2, a3); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4) -> void { return o.setAll(a0, a1, a2, a3, a4); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5) -> void { return o.setAll(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"));
		cl.def("setAll", [](Pythia8::ParticleDataEntry &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, int const & a2, int const & a3, int const & a4, double const & a5, double const & a6, double const & a7, double const & a8, double const & a9) -> void { return o.setAll(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"));
		cl.def("setAll", (void (Pythia8::ParticleDataEntry::*)(std::string, std::string, int, int, int, double, double, double, double, double, bool)) &Pythia8::ParticleDataEntry::setAll, "C++: Pythia8::ParticleDataEntry::setAll(std::string, std::string, int, int, int, double, double, double, double, double, bool) --> void", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"), pybind11::arg("spinTypeIn"), pybind11::arg("chargeTypeIn"), pybind11::arg("colTypeIn"), pybind11::arg("m0In"), pybind11::arg("mWidthIn"), pybind11::arg("mMinIn"), pybind11::arg("mMaxIn"), pybind11::arg("tau0In"), pybind11::arg("varWidthIn"));
		cl.def("setName", (void (Pythia8::ParticleDataEntry::*)(std::string)) &Pythia8::ParticleDataEntry::setName, "C++: Pythia8::ParticleDataEntry::setName(std::string) --> void", pybind11::arg("nameIn"));
		cl.def("setAntiName", (void (Pythia8::ParticleDataEntry::*)(std::string)) &Pythia8::ParticleDataEntry::setAntiName, "C++: Pythia8::ParticleDataEntry::setAntiName(std::string) --> void", pybind11::arg("antiNameIn"));
		cl.def("setNames", (void (Pythia8::ParticleDataEntry::*)(std::string, std::string)) &Pythia8::ParticleDataEntry::setNames, "C++: Pythia8::ParticleDataEntry::setNames(std::string, std::string) --> void", pybind11::arg("nameIn"), pybind11::arg("antiNameIn"));
		cl.def("setSpinType", (void (Pythia8::ParticleDataEntry::*)(int)) &Pythia8::ParticleDataEntry::setSpinType, "C++: Pythia8::ParticleDataEntry::setSpinType(int) --> void", pybind11::arg("spinTypeIn"));
		cl.def("setChargeType", (void (Pythia8::ParticleDataEntry::*)(int)) &Pythia8::ParticleDataEntry::setChargeType, "C++: Pythia8::ParticleDataEntry::setChargeType(int) --> void", pybind11::arg("chargeTypeIn"));
		cl.def("setColType", (void (Pythia8::ParticleDataEntry::*)(int)) &Pythia8::ParticleDataEntry::setColType, "C++: Pythia8::ParticleDataEntry::setColType(int) --> void", pybind11::arg("colTypeIn"));
		cl.def("setM0", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::setM0, "C++: Pythia8::ParticleDataEntry::setM0(double) --> void", pybind11::arg("m0In"));
		cl.def("setMWidth", [](Pythia8::ParticleDataEntry &o, double const & a0) -> void { return o.setMWidth(a0); }, "", pybind11::arg("mWidthIn"));
		cl.def("setMWidth", (void (Pythia8::ParticleDataEntry::*)(double, bool)) &Pythia8::ParticleDataEntry::setMWidth, "C++: Pythia8::ParticleDataEntry::setMWidth(double, bool) --> void", pybind11::arg("mWidthIn"), pybind11::arg("countAsChanged"));
		cl.def("setMMin", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::setMMin, "C++: Pythia8::ParticleDataEntry::setMMin(double) --> void", pybind11::arg("mMinIn"));
		cl.def("setMMax", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::setMMax, "C++: Pythia8::ParticleDataEntry::setMMax(double) --> void", pybind11::arg("mMaxIn"));
		cl.def("setMMinNoChange", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::setMMinNoChange, "C++: Pythia8::ParticleDataEntry::setMMinNoChange(double) --> void", pybind11::arg("mMinIn"));
		cl.def("setMMaxNoChange", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::setMMaxNoChange, "C++: Pythia8::ParticleDataEntry::setMMaxNoChange(double) --> void", pybind11::arg("mMaxIn"));
		cl.def("setTau0", [](Pythia8::ParticleDataEntry &o, double const & a0) -> void { return o.setTau0(a0); }, "", pybind11::arg("tau0In"));
		cl.def("setTau0", (void (Pythia8::ParticleDataEntry::*)(double, bool)) &Pythia8::ParticleDataEntry::setTau0, "C++: Pythia8::ParticleDataEntry::setTau0(double, bool) --> void", pybind11::arg("tau0In"), pybind11::arg("countAsChanged"));
		cl.def("setVarWidth", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setVarWidth, "C++: Pythia8::ParticleDataEntry::setVarWidth(bool) --> void", pybind11::arg("varWidthIn"));
		cl.def("setIsResonance", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setIsResonance, "C++: Pythia8::ParticleDataEntry::setIsResonance(bool) --> void", pybind11::arg("isResonanceIn"));
		cl.def("setMayDecay", [](Pythia8::ParticleDataEntry &o, bool const & a0) -> void { return o.setMayDecay(a0); }, "", pybind11::arg("mayDecayIn"));
		cl.def("setMayDecay", (void (Pythia8::ParticleDataEntry::*)(bool, bool)) &Pythia8::ParticleDataEntry::setMayDecay, "C++: Pythia8::ParticleDataEntry::setMayDecay(bool, bool) --> void", pybind11::arg("mayDecayIn"), pybind11::arg("countAsChanged"));
		cl.def("setTauCalc", [](Pythia8::ParticleDataEntry &o, bool const & a0) -> void { return o.setTauCalc(a0); }, "", pybind11::arg("tauCalcIn"));
		cl.def("setTauCalc", (void (Pythia8::ParticleDataEntry::*)(bool, bool)) &Pythia8::ParticleDataEntry::setTauCalc, "C++: Pythia8::ParticleDataEntry::setTauCalc(bool, bool) --> void", pybind11::arg("tauCalcIn"), pybind11::arg("countAsChanged"));
		cl.def("setDoExternalDecay", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setDoExternalDecay, "C++: Pythia8::ParticleDataEntry::setDoExternalDecay(bool) --> void", pybind11::arg("doExternalDecayIn"));
		cl.def("setIsVisible", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setIsVisible, "C++: Pythia8::ParticleDataEntry::setIsVisible(bool) --> void", pybind11::arg("isVisibleIn"));
		cl.def("setDoForceWidth", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setDoForceWidth, "C++: Pythia8::ParticleDataEntry::setDoForceWidth(bool) --> void", pybind11::arg("doForceWidthIn"));
		cl.def("setHasChanged", (void (Pythia8::ParticleDataEntry::*)(bool)) &Pythia8::ParticleDataEntry::setHasChanged, "C++: Pythia8::ParticleDataEntry::setHasChanged(bool) --> void", pybind11::arg("hasChangedIn"));
		cl.def("id", (int (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::id, "C++: Pythia8::ParticleDataEntry::id() const --> int");
		cl.def("antiId", (int (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::antiId, "C++: Pythia8::ParticleDataEntry::antiId() const --> int");
		cl.def("hasAnti", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::hasAnti, "C++: Pythia8::ParticleDataEntry::hasAnti() const --> bool");
		cl.def("name", [](Pythia8::ParticleDataEntry const &o) -> std::string { return o.name(); }, "");
		cl.def("name", (std::string (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::name, "C++: Pythia8::ParticleDataEntry::name(int) const --> std::string", pybind11::arg("idIn"));
		cl.def("spinType", (int (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::spinType, "C++: Pythia8::ParticleDataEntry::spinType() const --> int");
		cl.def("chargeType", [](Pythia8::ParticleDataEntry const &o) -> int { return o.chargeType(); }, "");
		cl.def("chargeType", (int (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::chargeType, "C++: Pythia8::ParticleDataEntry::chargeType(int) const --> int", pybind11::arg("idIn"));
		cl.def("charge", [](Pythia8::ParticleDataEntry const &o) -> double { return o.charge(); }, "");
		cl.def("charge", (double (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::charge, "C++: Pythia8::ParticleDataEntry::charge(int) const --> double", pybind11::arg("idIn"));
		cl.def("colType", [](Pythia8::ParticleDataEntry const &o) -> int { return o.colType(); }, "");
		cl.def("colType", (int (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::colType, "C++: Pythia8::ParticleDataEntry::colType(int) const --> int", pybind11::arg("idIn"));
		cl.def("m0", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::m0, "C++: Pythia8::ParticleDataEntry::m0() const --> double");
		cl.def("mWidth", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::mWidth, "C++: Pythia8::ParticleDataEntry::mWidth() const --> double");
		cl.def("mMin", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::mMin, "C++: Pythia8::ParticleDataEntry::mMin() const --> double");
		cl.def("mMax", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::mMax, "C++: Pythia8::ParticleDataEntry::mMax() const --> double");
		cl.def("m0Min", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::m0Min, "C++: Pythia8::ParticleDataEntry::m0Min() const --> double");
		cl.def("m0Max", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::m0Max, "C++: Pythia8::ParticleDataEntry::m0Max() const --> double");
		cl.def("tau0", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::tau0, "C++: Pythia8::ParticleDataEntry::tau0() const --> double");
		cl.def("isResonance", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isResonance, "C++: Pythia8::ParticleDataEntry::isResonance() const --> bool");
		cl.def("varWidth", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::varWidth, "C++: Pythia8::ParticleDataEntry::varWidth() const --> bool");
		cl.def("mayDecay", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::mayDecay, "C++: Pythia8::ParticleDataEntry::mayDecay() const --> bool");
		cl.def("tauCalc", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::tauCalc, "C++: Pythia8::ParticleDataEntry::tauCalc() const --> bool");
		cl.def("doExternalDecay", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::doExternalDecay, "C++: Pythia8::ParticleDataEntry::doExternalDecay() const --> bool");
		cl.def("isVisible", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isVisible, "C++: Pythia8::ParticleDataEntry::isVisible() const --> bool");
		cl.def("doForceWidth", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::doForceWidth, "C++: Pythia8::ParticleDataEntry::doForceWidth() const --> bool");
		cl.def("hasChanged", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::hasChanged, "C++: Pythia8::ParticleDataEntry::hasChanged() const --> bool");
		cl.def("hasChangedMMin", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::hasChangedMMin, "C++: Pythia8::ParticleDataEntry::hasChangedMMin() const --> bool");
		cl.def("hasChangedMMax", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::hasChangedMMax, "C++: Pythia8::ParticleDataEntry::hasChangedMMax() const --> bool");
		cl.def("initBWmass", (void (Pythia8::ParticleDataEntry::*)()) &Pythia8::ParticleDataEntry::initBWmass, "C++: Pythia8::ParticleDataEntry::initBWmass() --> void");
		cl.def("constituentMass", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::constituentMass, "C++: Pythia8::ParticleDataEntry::constituentMass() const --> double");
		cl.def("mSel", (double (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::mSel, "C++: Pythia8::ParticleDataEntry::mSel() const --> double");
		cl.def("mRun", (double (Pythia8::ParticleDataEntry::*)(double) const) &Pythia8::ParticleDataEntry::mRun, "C++: Pythia8::ParticleDataEntry::mRun(double) const --> double", pybind11::arg("mH"));
		cl.def("useBreitWigner", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::useBreitWigner, "C++: Pythia8::ParticleDataEntry::useBreitWigner() const --> bool");
		cl.def("canDecay", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::canDecay, "C++: Pythia8::ParticleDataEntry::canDecay() const --> bool");
		cl.def("isLepton", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isLepton, "C++: Pythia8::ParticleDataEntry::isLepton() const --> bool");
		cl.def("isQuark", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isQuark, "C++: Pythia8::ParticleDataEntry::isQuark() const --> bool");
		cl.def("isGluon", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isGluon, "C++: Pythia8::ParticleDataEntry::isGluon() const --> bool");
		cl.def("isDiquark", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isDiquark, "C++: Pythia8::ParticleDataEntry::isDiquark() const --> bool");
		cl.def("isParton", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isParton, "C++: Pythia8::ParticleDataEntry::isParton() const --> bool");
		cl.def("isHadron", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isHadron, "C++: Pythia8::ParticleDataEntry::isHadron() const --> bool");
		cl.def("isMeson", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isMeson, "C++: Pythia8::ParticleDataEntry::isMeson() const --> bool");
		cl.def("isBaryon", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isBaryon, "C++: Pythia8::ParticleDataEntry::isBaryon() const --> bool");
		cl.def("isOnium", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isOnium, "C++: Pythia8::ParticleDataEntry::isOnium() const --> bool");
		cl.def("isOctetHadron", (bool (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::isOctetHadron, "C++: Pythia8::ParticleDataEntry::isOctetHadron() const --> bool");
		cl.def("heaviestQuark", [](Pythia8::ParticleDataEntry const &o) -> int { return o.heaviestQuark(); }, "");
		cl.def("heaviestQuark", (int (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::heaviestQuark, "C++: Pythia8::ParticleDataEntry::heaviestQuark(int) const --> int", pybind11::arg("idIn"));
		cl.def("baryonNumberType", [](Pythia8::ParticleDataEntry const &o) -> int { return o.baryonNumberType(); }, "");
		cl.def("baryonNumberType", (int (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::baryonNumberType, "C++: Pythia8::ParticleDataEntry::baryonNumberType(int) const --> int", pybind11::arg("idIn"));
		cl.def("nQuarksInCode", (int (Pythia8::ParticleDataEntry::*)(int) const) &Pythia8::ParticleDataEntry::nQuarksInCode, "C++: Pythia8::ParticleDataEntry::nQuarksInCode(int) const --> int", pybind11::arg("idQIn"));
		cl.def("clearChannels", (void (Pythia8::ParticleDataEntry::*)()) &Pythia8::ParticleDataEntry::clearChannels, "C++: Pythia8::ParticleDataEntry::clearChannels() --> void");
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o) -> void { return o.addChannel(); }, "");
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0) -> void { return o.addChannel(a0); }, "", pybind11::arg("onMode"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1) -> void { return o.addChannel(a0, a1); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2) -> void { return o.addChannel(a0, a1, a2); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3) -> void { return o.addChannel(a0, a1, a2, a3); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4) -> void { return o.addChannel(a0, a1, a2, a3, a4); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5) -> void { return o.addChannel(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6) -> void { return o.addChannel(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7) -> void { return o.addChannel(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, int const & a8) -> void { return o.addChannel(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"));
		cl.def("addChannel", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, int const & a8, int const & a9) -> void { return o.addChannel(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }, "", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"), pybind11::arg("prod6"));
		cl.def("addChannel", (void (Pythia8::ParticleDataEntry::*)(int, double, int, int, int, int, int, int, int, int, int)) &Pythia8::ParticleDataEntry::addChannel, "C++: Pythia8::ParticleDataEntry::addChannel(int, double, int, int, int, int, int, int, int, int, int) --> void", pybind11::arg("onMode"), pybind11::arg("bRatio"), pybind11::arg("meMode"), pybind11::arg("prod0"), pybind11::arg("prod1"), pybind11::arg("prod2"), pybind11::arg("prod3"), pybind11::arg("prod4"), pybind11::arg("prod5"), pybind11::arg("prod6"), pybind11::arg("prod7"));
		cl.def("sizeChannels", (int (Pythia8::ParticleDataEntry::*)() const) &Pythia8::ParticleDataEntry::sizeChannels, "C++: Pythia8::ParticleDataEntry::sizeChannels() const --> int");
		cl.def("channel", (class Pythia8::DecayChannel & (Pythia8::ParticleDataEntry::*)(int)) &Pythia8::ParticleDataEntry::channel, "C++: Pythia8::ParticleDataEntry::channel(int) --> class Pythia8::DecayChannel &", pybind11::return_value_policy::reference, pybind11::arg("i"));
		cl.def("rescaleBR", [](Pythia8::ParticleDataEntry &o) -> void { return o.rescaleBR(); }, "");
		cl.def("rescaleBR", (void (Pythia8::ParticleDataEntry::*)(double)) &Pythia8::ParticleDataEntry::rescaleBR, "C++: Pythia8::ParticleDataEntry::rescaleBR(double) --> void", pybind11::arg("newSumBR"));
		cl.def("preparePick", [](Pythia8::ParticleDataEntry &o, int const & a0) -> bool { return o.preparePick(a0); }, "", pybind11::arg("idSgn"));
		cl.def("preparePick", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1) -> bool { return o.preparePick(a0, a1); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"));
		cl.def("preparePick", (bool (Pythia8::ParticleDataEntry::*)(int, double, int)) &Pythia8::ParticleDataEntry::preparePick, "C++: Pythia8::ParticleDataEntry::preparePick(int, double, int) --> bool", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idInFlav"));
		cl.def("pickChannel", (class Pythia8::DecayChannel & (Pythia8::ParticleDataEntry::*)()) &Pythia8::ParticleDataEntry::pickChannel, "C++: Pythia8::ParticleDataEntry::pickChannel() --> class Pythia8::DecayChannel &", pybind11::return_value_policy::reference);
		cl.def("resInit", (void (Pythia8::ParticleDataEntry::*)(class Pythia8::Info *)) &Pythia8::ParticleDataEntry::resInit, "C++: Pythia8::ParticleDataEntry::resInit(class Pythia8::Info *) --> void", pybind11::arg("infoPtrIn"));
		cl.def("resWidth", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1) -> double { return o.resWidth(a0, a1); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"));
		cl.def("resWidth", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2) -> double { return o.resWidth(a0, a1, a2); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idIn"));
		cl.def("resWidth", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1, int const & a2, bool const & a3) -> double { return o.resWidth(a0, a1, a2, a3); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idIn"), pybind11::arg("openOnly"));
		cl.def("resWidth", (double (Pythia8::ParticleDataEntry::*)(int, double, int, bool, bool)) &Pythia8::ParticleDataEntry::resWidth, "C++: Pythia8::ParticleDataEntry::resWidth(int, double, int, bool, bool) --> double", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idIn"), pybind11::arg("openOnly"), pybind11::arg("setBR"));
		cl.def("resWidthOpen", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1) -> double { return o.resWidthOpen(a0, a1); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"));
		cl.def("resWidthOpen", (double (Pythia8::ParticleDataEntry::*)(int, double, int)) &Pythia8::ParticleDataEntry::resWidthOpen, "C++: Pythia8::ParticleDataEntry::resWidthOpen(int, double, int) --> double", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idIn"));
		cl.def("resWidthStore", [](Pythia8::ParticleDataEntry &o, int const & a0, double const & a1) -> double { return o.resWidthStore(a0, a1); }, "", pybind11::arg("idSgn"), pybind11::arg("mHat"));
		cl.def("resWidthStore", (double (Pythia8::ParticleDataEntry::*)(int, double, int)) &Pythia8::ParticleDataEntry::resWidthStore, "C++: Pythia8::ParticleDataEntry::resWidthStore(int, double, int) --> double", pybind11::arg("idSgn"), pybind11::arg("mHat"), pybind11::arg("idIn"));
		cl.def("resOpenFrac", (double (Pythia8::ParticleDataEntry::*)(int)) &Pythia8::ParticleDataEntry::resOpenFrac, "C++: Pythia8::ParticleDataEntry::resOpenFrac(int) --> double", pybind11::arg("idSgn"));
		cl.def("resWidthRescaleFactor", (double (Pythia8::ParticleDataEntry::*)()) &Pythia8::ParticleDataEntry::resWidthRescaleFactor, "C++: Pythia8::ParticleDataEntry::resWidthRescaleFactor() --> double");
		cl.def("resWidthChan", [](Pythia8::ParticleDataEntry &o, double const & a0) -> double { return o.resWidthChan(a0); }, "", pybind11::arg("mHat"));
		cl.def("resWidthChan", [](Pythia8::ParticleDataEntry &o, double const & a0, int const & a1) -> double { return o.resWidthChan(a0, a1); }, "", pybind11::arg("mHat"), pybind11::arg("idAbs1"));
		cl.def("resWidthChan", (double (Pythia8::ParticleDataEntry::*)(double, int, int)) &Pythia8::ParticleDataEntry::resWidthChan, "C++: Pythia8::ParticleDataEntry::resWidthChan(double, int, int) --> double", pybind11::arg("mHat"), pybind11::arg("idAbs1"), pybind11::arg("idAbs2"));
	}
}
