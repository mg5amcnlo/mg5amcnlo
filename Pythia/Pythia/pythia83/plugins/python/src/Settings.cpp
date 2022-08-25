#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/PartonSystems.h>
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

void bind_Pythia8_Settings(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::Flag file:Pythia8/Settings.h line:29
		pybind11::class_<Pythia8::Flag, std::shared_ptr<Pythia8::Flag>> cl(M("Pythia8"), "Flag", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Flag(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::Flag(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init<std::string, bool>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn") );

		cl.def( pybind11::init( [](Pythia8::Flag const &o){ return new Pythia8::Flag(o); } ) );
		cl.def_readwrite("name", &Pythia8::Flag::name);
		cl.def_readwrite("valNow", &Pythia8::Flag::valNow);
		cl.def_readwrite("valDefault", &Pythia8::Flag::valDefault);
		cl.def("assign", (class Pythia8::Flag & (Pythia8::Flag::*)(const class Pythia8::Flag &)) &Pythia8::Flag::operator=, "C++: Pythia8::Flag::operator=(const class Pythia8::Flag &) --> class Pythia8::Flag &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::Mode file:Pythia8/Settings.h line:47
		pybind11::class_<Pythia8::Mode, std::shared_ptr<Pythia8::Mode>> cl(M("Pythia8"), "Mode", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Mode(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::Mode(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, int const & a1){ return new Pythia8::Mode(a0, a1); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, int const & a1, bool const & a2){ return new Pythia8::Mode(a0, a1, a2); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, int const & a1, bool const & a2, bool const & a3){ return new Pythia8::Mode(a0, a1, a2, a3); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, int const & a1, bool const & a2, bool const & a3, int const & a4){ return new Pythia8::Mode(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, int const & a1, bool const & a2, bool const & a3, int const & a4, int const & a5){ return new Pythia8::Mode(a0, a1, a2, a3, a4, a5); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"));
		cl.def( pybind11::init<std::string, int, bool, bool, int, int, bool>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"), pybind11::arg("optOnlyIn") );

		cl.def( pybind11::init( [](Pythia8::Mode const &o){ return new Pythia8::Mode(o); } ) );
		cl.def_readwrite("name", &Pythia8::Mode::name);
		cl.def_readwrite("valNow", &Pythia8::Mode::valNow);
		cl.def_readwrite("valDefault", &Pythia8::Mode::valDefault);
		cl.def_readwrite("hasMin", &Pythia8::Mode::hasMin);
		cl.def_readwrite("hasMax", &Pythia8::Mode::hasMax);
		cl.def_readwrite("valMin", &Pythia8::Mode::valMin);
		cl.def_readwrite("valMax", &Pythia8::Mode::valMax);
		cl.def_readwrite("optOnly", &Pythia8::Mode::optOnly);
		cl.def("assign", (class Pythia8::Mode & (Pythia8::Mode::*)(const class Pythia8::Mode &)) &Pythia8::Mode::operator=, "C++: Pythia8::Mode::operator=(const class Pythia8::Mode &) --> class Pythia8::Mode &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::Parm file:Pythia8/Settings.h line:71
		pybind11::class_<Pythia8::Parm, std::shared_ptr<Pythia8::Parm>> cl(M("Pythia8"), "Parm", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Parm(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::Parm(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, double const & a1){ return new Pythia8::Parm(a0, a1); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, double const & a1, bool const & a2){ return new Pythia8::Parm(a0, a1, a2); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, double const & a1, bool const & a2, bool const & a3){ return new Pythia8::Parm(a0, a1, a2, a3); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, double const & a1, bool const & a2, bool const & a3, double const & a4){ return new Pythia8::Parm(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"));
		cl.def( pybind11::init<std::string, double, bool, bool, double, double>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn") );

		cl.def( pybind11::init( [](Pythia8::Parm const &o){ return new Pythia8::Parm(o); } ) );
		cl.def_readwrite("name", &Pythia8::Parm::name);
		cl.def_readwrite("valNow", &Pythia8::Parm::valNow);
		cl.def_readwrite("valDefault", &Pythia8::Parm::valDefault);
		cl.def_readwrite("hasMin", &Pythia8::Parm::hasMin);
		cl.def_readwrite("hasMax", &Pythia8::Parm::hasMax);
		cl.def_readwrite("valMin", &Pythia8::Parm::valMin);
		cl.def_readwrite("valMax", &Pythia8::Parm::valMax);
		cl.def("assign", (class Pythia8::Parm & (Pythia8::Parm::*)(const class Pythia8::Parm &)) &Pythia8::Parm::operator=, "C++: Pythia8::Parm::operator=(const class Pythia8::Parm &) --> class Pythia8::Parm &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::Word file:Pythia8/Settings.h line:94
		pybind11::class_<Pythia8::Word, std::shared_ptr<Pythia8::Word>> cl(M("Pythia8"), "Word", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Word(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::Word(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init<std::string, std::string>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn") );

		cl.def( pybind11::init( [](Pythia8::Word const &o){ return new Pythia8::Word(o); } ) );
		cl.def_readwrite("name", &Pythia8::Word::name);
		cl.def_readwrite("valNow", &Pythia8::Word::valNow);
		cl.def_readwrite("valDefault", &Pythia8::Word::valDefault);
		cl.def("assign", (class Pythia8::Word & (Pythia8::Word::*)(const class Pythia8::Word &)) &Pythia8::Word::operator=, "C++: Pythia8::Word::operator=(const class Pythia8::Word &) --> class Pythia8::Word &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::FVec file:Pythia8/Settings.h line:111
		pybind11::class_<Pythia8::FVec, std::shared_ptr<Pythia8::FVec>> cl(M("Pythia8"), "FVec", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::FVec(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::FVec(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init<std::string, class std::vector<bool, class std::allocator<bool> >>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn") );

		cl.def( pybind11::init( [](Pythia8::FVec const &o){ return new Pythia8::FVec(o); } ) );
		cl.def_readwrite("name", &Pythia8::FVec::name);
		cl.def_readwrite("valNow", &Pythia8::FVec::valNow);
		cl.def_readwrite("valDefault", &Pythia8::FVec::valDefault);
		cl.def("assign", (class Pythia8::FVec & (Pythia8::FVec::*)(const class Pythia8::FVec &)) &Pythia8::FVec::operator=, "C++: Pythia8::FVec::operator=(const class Pythia8::FVec &) --> class Pythia8::FVec &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::MVec file:Pythia8/Settings.h line:129
		pybind11::class_<Pythia8::MVec, std::shared_ptr<Pythia8::MVec>> cl(M("Pythia8"), "MVec", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::MVec(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::MVec(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<int, class std::allocator<int> > const & a1){ return new Pythia8::MVec(a0, a1); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<int, class std::allocator<int> > const & a1, bool const & a2){ return new Pythia8::MVec(a0, a1, a2); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<int, class std::allocator<int> > const & a1, bool const & a2, bool const & a3){ return new Pythia8::MVec(a0, a1, a2, a3); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<int, class std::allocator<int> > const & a1, bool const & a2, bool const & a3, int const & a4){ return new Pythia8::MVec(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"));
		cl.def( pybind11::init<std::string, class std::vector<int, class std::allocator<int> >, bool, bool, int, int>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn") );

		cl.def( pybind11::init( [](Pythia8::MVec const &o){ return new Pythia8::MVec(o); } ) );
		cl.def_readwrite("name", &Pythia8::MVec::name);
		cl.def_readwrite("valNow", &Pythia8::MVec::valNow);
		cl.def_readwrite("valDefault", &Pythia8::MVec::valDefault);
		cl.def_readwrite("hasMin", &Pythia8::MVec::hasMin);
		cl.def_readwrite("hasMax", &Pythia8::MVec::hasMax);
		cl.def_readwrite("valMin", &Pythia8::MVec::valMin);
		cl.def_readwrite("valMax", &Pythia8::MVec::valMax);
		cl.def("assign", (class Pythia8::MVec & (Pythia8::MVec::*)(const class Pythia8::MVec &)) &Pythia8::MVec::operator=, "C++: Pythia8::MVec::operator=(const class Pythia8::MVec &) --> class Pythia8::MVec &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::PVec file:Pythia8/Settings.h line:152
		pybind11::class_<Pythia8::PVec, std::shared_ptr<Pythia8::PVec>> cl(M("Pythia8"), "PVec", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::PVec(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::PVec(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<double, class std::allocator<double> > const & a1){ return new Pythia8::PVec(a0, a1); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<double, class std::allocator<double> > const & a1, bool const & a2){ return new Pythia8::PVec(a0, a1, a2); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<double, class std::allocator<double> > const & a1, bool const & a2, bool const & a3){ return new Pythia8::PVec(a0, a1, a2, a3); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"));
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0, class std::vector<double, class std::allocator<double> > const & a1, bool const & a2, bool const & a3, double const & a4){ return new Pythia8::PVec(a0, a1, a2, a3, a4); } ), "doc" , pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"));
		cl.def( pybind11::init<std::string, class std::vector<double, class std::allocator<double> >, bool, bool, double, double>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn") );

		cl.def( pybind11::init( [](Pythia8::PVec const &o){ return new Pythia8::PVec(o); } ) );
		cl.def_readwrite("name", &Pythia8::PVec::name);
		cl.def_readwrite("valNow", &Pythia8::PVec::valNow);
		cl.def_readwrite("valDefault", &Pythia8::PVec::valDefault);
		cl.def_readwrite("hasMin", &Pythia8::PVec::hasMin);
		cl.def_readwrite("hasMax", &Pythia8::PVec::hasMax);
		cl.def_readwrite("valMin", &Pythia8::PVec::valMin);
		cl.def_readwrite("valMax", &Pythia8::PVec::valMax);
		cl.def("assign", (class Pythia8::PVec & (Pythia8::PVec::*)(const class Pythia8::PVec &)) &Pythia8::PVec::operator=, "C++: Pythia8::PVec::operator=(const class Pythia8::PVec &) --> class Pythia8::PVec &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::WVec file:Pythia8/Settings.h line:175
		pybind11::class_<Pythia8::WVec, std::shared_ptr<Pythia8::WVec>> cl(M("Pythia8"), "WVec", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::WVec(); } ), "doc" );
		cl.def( pybind11::init( [](class std::basic_string<char> const & a0){ return new Pythia8::WVec(a0); } ), "doc" , pybind11::arg("nameIn"));
		cl.def( pybind11::init<std::string, class std::vector<std::string, class std::allocator<std::string > >>(), pybind11::arg("nameIn"), pybind11::arg("defaultIn") );

		cl.def( pybind11::init( [](Pythia8::WVec const &o){ return new Pythia8::WVec(o); } ) );
		cl.def_readwrite("name", &Pythia8::WVec::name);
		cl.def_readwrite("valNow", &Pythia8::WVec::valNow);
		cl.def_readwrite("valDefault", &Pythia8::WVec::valDefault);
		cl.def("assign", (class Pythia8::WVec & (Pythia8::WVec::*)(const class Pythia8::WVec &)) &Pythia8::WVec::operator=, "C++: Pythia8::WVec::operator=(const class Pythia8::WVec &) --> class Pythia8::WVec &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::Settings file:Pythia8/Settings.h line:195
		pybind11::class_<Pythia8::Settings, std::shared_ptr<Pythia8::Settings>> cl(M("Pythia8"), "Settings", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::Settings(); } ) );
		cl.def( pybind11::init( [](Pythia8::Settings const &o){ return new Pythia8::Settings(o); } ) );
		cl.def("initPtrs", (void (Pythia8::Settings::*)(class Pythia8::Info *)) &Pythia8::Settings::initPtrs, "C++: Pythia8::Settings::initPtrs(class Pythia8::Info *) --> void", pybind11::arg("infoPtrIn"));
		cl.def("init", [](Pythia8::Settings &o) -> bool { return o.init(); }, "");
		cl.def("init", [](Pythia8::Settings &o, class std::basic_string<char> const & a0) -> bool { return o.init(a0); }, "", pybind11::arg("startFile"));
		cl.def("init", (bool (Pythia8::Settings::*)(std::string, bool)) &Pythia8::Settings::init, "C++: Pythia8::Settings::init(std::string, bool) --> bool", pybind11::arg("startFile"), pybind11::arg("append"));
		cl.def("init", [](Pythia8::Settings &o, class std::basic_istream<char> & a0) -> bool { return o.init(a0); }, "", pybind11::arg("is"));
		cl.def("init", (bool (Pythia8::Settings::*)(class std::basic_istream<char> &, bool)) &Pythia8::Settings::init, "C++: Pythia8::Settings::init(class std::basic_istream<char> &, bool) --> bool", pybind11::arg("is"), pybind11::arg("append"));
		cl.def("reInit", [](Pythia8::Settings &o) -> bool { return o.reInit(); }, "");
		cl.def("reInit", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::reInit, "C++: Pythia8::Settings::reInit(std::string) --> bool", pybind11::arg("startFile"));
		cl.def("readString", [](Pythia8::Settings &o, class std::basic_string<char> const & a0) -> bool { return o.readString(a0); }, "", pybind11::arg("line"));
		cl.def("readString", (bool (Pythia8::Settings::*)(std::string, bool)) &Pythia8::Settings::readString, "C++: Pythia8::Settings::readString(std::string, bool) --> bool", pybind11::arg("line"), pybind11::arg("warn"));
		cl.def("writeFile", [](Pythia8::Settings &o, class std::basic_string<char> const & a0) -> bool { return o.writeFile(a0); }, "", pybind11::arg("toFile"));
		cl.def("writeFile", (bool (Pythia8::Settings::*)(std::string, bool)) &Pythia8::Settings::writeFile, "C++: Pythia8::Settings::writeFile(std::string, bool) --> bool", pybind11::arg("toFile"), pybind11::arg("writeAll"));
		cl.def("writeFile", [](Pythia8::Settings &o) -> bool { return o.writeFile(); }, "");
		cl.def("writeFile", [](Pythia8::Settings &o, class std::basic_ostream<char> & a0) -> bool { return o.writeFile(a0); }, "", pybind11::arg("os"));
		cl.def("writeFile", (bool (Pythia8::Settings::*)(std::ostream &, bool)) &Pythia8::Settings::writeFile, "C++: Pythia8::Settings::writeFile(std::ostream &, bool) --> bool", pybind11::arg("os"), pybind11::arg("writeAll"));
		cl.def("writeFileXML", [](Pythia8::Settings &o) -> bool { return o.writeFileXML(); }, "");
		cl.def("writeFileXML", (bool (Pythia8::Settings::*)(std::ostream &)) &Pythia8::Settings::writeFileXML, "C++: Pythia8::Settings::writeFileXML(std::ostream &) --> bool", pybind11::arg("os"));
		cl.def("listAll", (void (Pythia8::Settings::*)()) &Pythia8::Settings::listAll, "C++: Pythia8::Settings::listAll() --> void");
		cl.def("listChanged", (void (Pythia8::Settings::*)()) &Pythia8::Settings::listChanged, "C++: Pythia8::Settings::listChanged() --> void");
		cl.def("list", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::list, "C++: Pythia8::Settings::list(std::string) --> void", pybind11::arg("match"));
		cl.def("output", [](Pythia8::Settings &o, class std::basic_string<char> const & a0) -> std::string { return o.output(a0); }, "", pybind11::arg("keyIn"));
		cl.def("output", (std::string (Pythia8::Settings::*)(std::string, bool)) &Pythia8::Settings::output, "C++: Pythia8::Settings::output(std::string, bool) --> std::string", pybind11::arg("keyIn"), pybind11::arg("fullLine"));
		cl.def("getReadHistory", [](Pythia8::Settings &o) -> std::vector<std::string, class std::allocator<std::string > > { return o.getReadHistory(); }, "");
		cl.def("getReadHistory", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::Settings::*)(int)) &Pythia8::Settings::getReadHistory, "C++: Pythia8::Settings::getReadHistory(int) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("subrun"));
		cl.def("resetAll", (void (Pythia8::Settings::*)()) &Pythia8::Settings::resetAll, "C++: Pythia8::Settings::resetAll() --> void");
		cl.def("isFlag", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isFlag, "C++: Pythia8::Settings::isFlag(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isMode", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isMode, "C++: Pythia8::Settings::isMode(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isParm", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isParm, "C++: Pythia8::Settings::isParm(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isWord", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isWord, "C++: Pythia8::Settings::isWord(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isFVec", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isFVec, "C++: Pythia8::Settings::isFVec(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isMVec", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isMVec, "C++: Pythia8::Settings::isMVec(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isPVec", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isPVec, "C++: Pythia8::Settings::isPVec(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("isWVec", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::isWVec, "C++: Pythia8::Settings::isWVec(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("addFlag", (void (Pythia8::Settings::*)(std::string, bool)) &Pythia8::Settings::addFlag, "C++: Pythia8::Settings::addFlag(std::string, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"));
		cl.def("addMode", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, int const & a1, bool const & a2, bool const & a3, int const & a4, int const & a5) -> void { return o.addMode(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("keyIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"));
		cl.def("addMode", (void (Pythia8::Settings::*)(std::string, int, bool, bool, int, int, bool)) &Pythia8::Settings::addMode, "C++: Pythia8::Settings::addMode(std::string, int, bool, bool, int, int, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"), pybind11::arg("optOnlyIn"));
		cl.def("addParm", (void (Pythia8::Settings::*)(std::string, double, bool, bool, double, double)) &Pythia8::Settings::addParm, "C++: Pythia8::Settings::addParm(std::string, double, bool, bool, double, double) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"));
		cl.def("addWord", (void (Pythia8::Settings::*)(std::string, std::string)) &Pythia8::Settings::addWord, "C++: Pythia8::Settings::addWord(std::string, std::string) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"));
		cl.def("addFVec", (void (Pythia8::Settings::*)(std::string, class std::vector<bool, class std::allocator<bool> >)) &Pythia8::Settings::addFVec, "C++: Pythia8::Settings::addFVec(std::string, class std::vector<bool, class std::allocator<bool> >) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"));
		cl.def("addMVec", (void (Pythia8::Settings::*)(std::string, class std::vector<int, class std::allocator<int> >, bool, bool, int, int)) &Pythia8::Settings::addMVec, "C++: Pythia8::Settings::addMVec(std::string, class std::vector<int, class std::allocator<int> >, bool, bool, int, int) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"));
		cl.def("addPVec", (void (Pythia8::Settings::*)(std::string, class std::vector<double, class std::allocator<double> >, bool, bool, double, double)) &Pythia8::Settings::addPVec, "C++: Pythia8::Settings::addPVec(std::string, class std::vector<double, class std::allocator<double> >, bool, bool, double, double) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"), pybind11::arg("hasMinIn"), pybind11::arg("hasMaxIn"), pybind11::arg("minIn"), pybind11::arg("maxIn"));
		cl.def("addWVec", (void (Pythia8::Settings::*)(std::string, class std::vector<std::string, class std::allocator<std::string > >)) &Pythia8::Settings::addWVec, "C++: Pythia8::Settings::addWVec(std::string, class std::vector<std::string, class std::allocator<std::string > >) --> void", pybind11::arg("keyIn"), pybind11::arg("defaultIn"));
		cl.def("flag", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::flag, "C++: Pythia8::Settings::flag(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("mode", (int (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::mode, "C++: Pythia8::Settings::mode(std::string) --> int", pybind11::arg("keyIn"));
		cl.def("parm", (double (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::parm, "C++: Pythia8::Settings::parm(std::string) --> double", pybind11::arg("keyIn"));
		cl.def("word", (std::string (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::word, "C++: Pythia8::Settings::word(std::string) --> std::string", pybind11::arg("keyIn"));
		cl.def("fvec", (class std::vector<bool, class std::allocator<bool> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::fvec, "C++: Pythia8::Settings::fvec(std::string) --> class std::vector<bool, class std::allocator<bool> >", pybind11::arg("keyIn"));
		cl.def("mvec", (class std::vector<int, class std::allocator<int> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::mvec, "C++: Pythia8::Settings::mvec(std::string) --> class std::vector<int, class std::allocator<int> >", pybind11::arg("keyIn"));
		cl.def("pvec", (class std::vector<double, class std::allocator<double> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::pvec, "C++: Pythia8::Settings::pvec(std::string) --> class std::vector<double, class std::allocator<double> >", pybind11::arg("keyIn"));
		cl.def("wvec", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::wvec, "C++: Pythia8::Settings::wvec(std::string) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("keyIn"));
		cl.def("flagDefault", (bool (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::flagDefault, "C++: Pythia8::Settings::flagDefault(std::string) --> bool", pybind11::arg("keyIn"));
		cl.def("modeDefault", (int (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::modeDefault, "C++: Pythia8::Settings::modeDefault(std::string) --> int", pybind11::arg("keyIn"));
		cl.def("parmDefault", (double (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::parmDefault, "C++: Pythia8::Settings::parmDefault(std::string) --> double", pybind11::arg("keyIn"));
		cl.def("wordDefault", (std::string (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::wordDefault, "C++: Pythia8::Settings::wordDefault(std::string) --> std::string", pybind11::arg("keyIn"));
		cl.def("fvecDefault", (class std::vector<bool, class std::allocator<bool> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::fvecDefault, "C++: Pythia8::Settings::fvecDefault(std::string) --> class std::vector<bool, class std::allocator<bool> >", pybind11::arg("keyIn"));
		cl.def("mvecDefault", (class std::vector<int, class std::allocator<int> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::mvecDefault, "C++: Pythia8::Settings::mvecDefault(std::string) --> class std::vector<int, class std::allocator<int> >", pybind11::arg("keyIn"));
		cl.def("pvecDefault", (class std::vector<double, class std::allocator<double> > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::pvecDefault, "C++: Pythia8::Settings::pvecDefault(std::string) --> class std::vector<double, class std::allocator<double> >", pybind11::arg("keyIn"));
		cl.def("wvecDefault", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::wvecDefault, "C++: Pythia8::Settings::wvecDefault(std::string) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg("keyIn"));
		cl.def("getFlagMap", (class std::map<std::string, class Pythia8::Flag, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Flag> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getFlagMap, "C++: Pythia8::Settings::getFlagMap(std::string) --> class std::map<std::string, class Pythia8::Flag, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Flag> > >", pybind11::arg("match"));
		cl.def("getModeMap", (class std::map<std::string, class Pythia8::Mode, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Mode> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getModeMap, "C++: Pythia8::Settings::getModeMap(std::string) --> class std::map<std::string, class Pythia8::Mode, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Mode> > >", pybind11::arg("match"));
		cl.def("getParmMap", (class std::map<std::string, class Pythia8::Parm, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Parm> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getParmMap, "C++: Pythia8::Settings::getParmMap(std::string) --> class std::map<std::string, class Pythia8::Parm, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Parm> > >", pybind11::arg("match"));
		cl.def("getWordMap", (class std::map<std::string, class Pythia8::Word, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Word> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getWordMap, "C++: Pythia8::Settings::getWordMap(std::string) --> class std::map<std::string, class Pythia8::Word, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::Word> > >", pybind11::arg("match"));
		cl.def("getFVecMap", (class std::map<std::string, class Pythia8::FVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::FVec> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getFVecMap, "C++: Pythia8::Settings::getFVecMap(std::string) --> class std::map<std::string, class Pythia8::FVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::FVec> > >", pybind11::arg("match"));
		cl.def("getMVecMap", (class std::map<std::string, class Pythia8::MVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::MVec> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getMVecMap, "C++: Pythia8::Settings::getMVecMap(std::string) --> class std::map<std::string, class Pythia8::MVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::MVec> > >", pybind11::arg("match"));
		cl.def("getPVecMap", (class std::map<std::string, class Pythia8::PVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::PVec> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getPVecMap, "C++: Pythia8::Settings::getPVecMap(std::string) --> class std::map<std::string, class Pythia8::PVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::PVec> > >", pybind11::arg("match"));
		cl.def("getWVecMap", (class std::map<std::string, class Pythia8::WVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::WVec> > > (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::getWVecMap, "C++: Pythia8::Settings::getWVecMap(std::string) --> class std::map<std::string, class Pythia8::WVec, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, class Pythia8::WVec> > >", pybind11::arg("match"));
		cl.def("flag", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, bool const & a1) -> void { return o.flag(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("flag", (void (Pythia8::Settings::*)(std::string, bool, bool)) &Pythia8::Settings::flag, "C++: Pythia8::Settings::flag(std::string, bool, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("mode", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, int const & a1) -> bool { return o.mode(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("mode", (bool (Pythia8::Settings::*)(std::string, int, bool)) &Pythia8::Settings::mode, "C++: Pythia8::Settings::mode(std::string, int, bool) --> bool", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("parm", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, double const & a1) -> void { return o.parm(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("parm", (void (Pythia8::Settings::*)(std::string, double, bool)) &Pythia8::Settings::parm, "C++: Pythia8::Settings::parm(std::string, double, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("word", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.word(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("word", (void (Pythia8::Settings::*)(std::string, std::string, bool)) &Pythia8::Settings::word, "C++: Pythia8::Settings::word(std::string, std::string, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("fvec", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, class std::vector<bool, class std::allocator<bool> > const & a1) -> void { return o.fvec(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("fvec", (void (Pythia8::Settings::*)(std::string, class std::vector<bool, class std::allocator<bool> >, bool)) &Pythia8::Settings::fvec, "C++: Pythia8::Settings::fvec(std::string, class std::vector<bool, class std::allocator<bool> >, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("mvec", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, class std::vector<int, class std::allocator<int> > const & a1) -> void { return o.mvec(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("mvec", (void (Pythia8::Settings::*)(std::string, class std::vector<int, class std::allocator<int> >, bool)) &Pythia8::Settings::mvec, "C++: Pythia8::Settings::mvec(std::string, class std::vector<int, class std::allocator<int> >, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("pvec", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, class std::vector<double, class std::allocator<double> > const & a1) -> void { return o.pvec(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("pvec", (void (Pythia8::Settings::*)(std::string, class std::vector<double, class std::allocator<double> >, bool)) &Pythia8::Settings::pvec, "C++: Pythia8::Settings::pvec(std::string, class std::vector<double, class std::allocator<double> >, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("wvec", [](Pythia8::Settings &o, class std::basic_string<char> const & a0, class std::vector<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > > const & a1) -> void { return o.wvec(a0, a1); }, "", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("wvec", (void (Pythia8::Settings::*)(std::string, class std::vector<std::string, class std::allocator<std::string > >, bool)) &Pythia8::Settings::wvec, "C++: Pythia8::Settings::wvec(std::string, class std::vector<std::string, class std::allocator<std::string > >, bool) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"), pybind11::arg("force"));
		cl.def("forceMode", (void (Pythia8::Settings::*)(std::string, int)) &Pythia8::Settings::forceMode, "C++: Pythia8::Settings::forceMode(std::string, int) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("forceParm", (void (Pythia8::Settings::*)(std::string, double)) &Pythia8::Settings::forceParm, "C++: Pythia8::Settings::forceParm(std::string, double) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("forceMVec", (void (Pythia8::Settings::*)(std::string, class std::vector<int, class std::allocator<int> >)) &Pythia8::Settings::forceMVec, "C++: Pythia8::Settings::forceMVec(std::string, class std::vector<int, class std::allocator<int> >) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("forcePVec", (void (Pythia8::Settings::*)(std::string, class std::vector<double, class std::allocator<double> >)) &Pythia8::Settings::forcePVec, "C++: Pythia8::Settings::forcePVec(std::string, class std::vector<double, class std::allocator<double> >) --> void", pybind11::arg("keyIn"), pybind11::arg("nowIn"));
		cl.def("resetFlag", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetFlag, "C++: Pythia8::Settings::resetFlag(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetMode", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetMode, "C++: Pythia8::Settings::resetMode(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetParm", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetParm, "C++: Pythia8::Settings::resetParm(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetWord", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetWord, "C++: Pythia8::Settings::resetWord(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetFVec", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetFVec, "C++: Pythia8::Settings::resetFVec(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetMVec", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetMVec, "C++: Pythia8::Settings::resetMVec(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetPVec", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetPVec, "C++: Pythia8::Settings::resetPVec(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("resetWVec", (void (Pythia8::Settings::*)(std::string)) &Pythia8::Settings::resetWVec, "C++: Pythia8::Settings::resetWVec(std::string) --> void", pybind11::arg("keyIn"));
		cl.def("getIsInit", (bool (Pythia8::Settings::*)()) &Pythia8::Settings::getIsInit, "C++: Pythia8::Settings::getIsInit() --> bool");
		cl.def("readingFailed", (bool (Pythia8::Settings::*)()) &Pythia8::Settings::readingFailed, "C++: Pythia8::Settings::readingFailed() --> bool");
		cl.def("unfinishedInput", (bool (Pythia8::Settings::*)()) &Pythia8::Settings::unfinishedInput, "C++: Pythia8::Settings::unfinishedInput() --> bool");
		cl.def("hasHardProc", (bool (Pythia8::Settings::*)()) &Pythia8::Settings::hasHardProc, "C++: Pythia8::Settings::hasHardProc() --> bool");
	}
}
