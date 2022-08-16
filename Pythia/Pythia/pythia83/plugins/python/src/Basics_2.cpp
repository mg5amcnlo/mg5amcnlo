#include <Pythia8/Basics.h>
#include <functional>
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

void bind_Pythia8_Basics_2(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::HistPlot file:Pythia8/Basics.h line:630
		pybind11::class_<Pythia8::HistPlot, std::shared_ptr<Pythia8::HistPlot>> cl(M("Pythia8"), "HistPlot", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init<std::string>(), pybind11::arg("pythonName") );

		cl.def("frame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0) -> void { return o.frame(a0); }, "", pybind11::arg("frameIn"));
		cl.def("frame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.frame(a0, a1); }, "", pybind11::arg("frameIn"), pybind11::arg("titleIn"));
		cl.def("frame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2) -> void { return o.frame(a0, a1, a2); }, "", pybind11::arg("frameIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"));
		cl.def("frame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3) -> void { return o.frame(a0, a1, a2, a3); }, "", pybind11::arg("frameIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"));
		cl.def("frame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3, double const & a4) -> void { return o.frame(a0, a1, a2, a3, a4); }, "", pybind11::arg("frameIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"), pybind11::arg("xSizeIn"));
		cl.def("frame", (void (Pythia8::HistPlot::*)(std::string, std::string, std::string, std::string, double, double)) &Pythia8::HistPlot::frame, "C++: Pythia8::HistPlot::frame(std::string, std::string, std::string, std::string, double, double) --> void", pybind11::arg("frameIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"), pybind11::arg("xSizeIn"), pybind11::arg("ySizeIn"));
		cl.def("add", [](Pythia8::HistPlot &o, const class Pythia8::Hist & a0) -> void { return o.add(a0); }, "", pybind11::arg("histIn"));
		cl.def("add", [](Pythia8::HistPlot &o, const class Pythia8::Hist & a0, class std::basic_string<char> const & a1) -> void { return o.add(a0, a1); }, "", pybind11::arg("histIn"), pybind11::arg("styleIn"));
		cl.def("add", (void (Pythia8::HistPlot::*)(const class Pythia8::Hist &, std::string, std::string)) &Pythia8::HistPlot::add, "C++: Pythia8::HistPlot::add(const class Pythia8::Hist &, std::string, std::string) --> void", pybind11::arg("histIn"), pybind11::arg("styleIn"), pybind11::arg("legendIn"));
		cl.def("addFile", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0) -> void { return o.addFile(a0); }, "", pybind11::arg("fileIn"));
		cl.def("addFile", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1) -> void { return o.addFile(a0, a1); }, "", pybind11::arg("fileIn"), pybind11::arg("styleIn"));
		cl.def("addFile", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, class std::basic_string<char> const & a1, class std::basic_string<char> const & a2) -> void { return o.addFile(a0, a1, a2); }, "", pybind11::arg("fileIn"), pybind11::arg("styleIn"), pybind11::arg("legendIn"));
		cl.def("addFile", (void (Pythia8::HistPlot::*)(std::string, std::string, std::string, std::string)) &Pythia8::HistPlot::addFile, "C++: Pythia8::HistPlot::addFile(std::string, std::string, std::string, std::string) --> void", pybind11::arg("fileIn"), pybind11::arg("styleIn"), pybind11::arg("legendIn"), pybind11::arg("xyerrIn"));
		cl.def("plot", [](Pythia8::HistPlot &o) -> void { return o.plot(); }, "");
		cl.def("plot", [](Pythia8::HistPlot &o, bool const & a0) -> void { return o.plot(a0); }, "", pybind11::arg("logY"));
		cl.def("plot", [](Pythia8::HistPlot &o, bool const & a0, bool const & a1) -> void { return o.plot(a0, a1); }, "", pybind11::arg("logY"), pybind11::arg("logX"));
		cl.def("plot", (void (Pythia8::HistPlot::*)(bool, bool, bool)) &Pythia8::HistPlot::plot, "C++: Pythia8::HistPlot::plot(bool, bool, bool) --> void", pybind11::arg("logY"), pybind11::arg("logX"), pybind11::arg("userBorders"));
		cl.def("plot", [](Pythia8::HistPlot &o, double const & a0, double const & a1, double const & a2, double const & a3) -> void { return o.plot(a0, a1, a2, a3); }, "", pybind11::arg("xMinUserIn"), pybind11::arg("xMaxUserIn"), pybind11::arg("yMinUserIn"), pybind11::arg("yMaxUserIn"));
		cl.def("plot", [](Pythia8::HistPlot &o, double const & a0, double const & a1, double const & a2, double const & a3, bool const & a4) -> void { return o.plot(a0, a1, a2, a3, a4); }, "", pybind11::arg("xMinUserIn"), pybind11::arg("xMaxUserIn"), pybind11::arg("yMinUserIn"), pybind11::arg("yMaxUserIn"), pybind11::arg("logY"));
		cl.def("plot", (void (Pythia8::HistPlot::*)(double, double, double, double, bool, bool)) &Pythia8::HistPlot::plot, "C++: Pythia8::HistPlot::plot(double, double, double, double, bool, bool) --> void", pybind11::arg("xMinUserIn"), pybind11::arg("xMaxUserIn"), pybind11::arg("yMinUserIn"), pybind11::arg("yMaxUserIn"), pybind11::arg("logY"), pybind11::arg("logX"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1) -> void { return o.plotFrame(a0, a1); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1, class std::basic_string<char> const & a2) -> void { return o.plotFrame(a0, a1, a2); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3) -> void { return o.plotFrame(a0, a1, a2, a3); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3, class std::basic_string<char> const & a4) -> void { return o.plotFrame(a0, a1, a2, a3, a4); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3, class std::basic_string<char> const & a4, class std::basic_string<char> const & a5) -> void { return o.plotFrame(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"), pybind11::arg("styleIn"));
		cl.def("plotFrame", [](Pythia8::HistPlot &o, class std::basic_string<char> const & a0, const class Pythia8::Hist & a1, class std::basic_string<char> const & a2, class std::basic_string<char> const & a3, class std::basic_string<char> const & a4, class std::basic_string<char> const & a5, class std::basic_string<char> const & a6) -> void { return o.plotFrame(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"), pybind11::arg("styleIn"), pybind11::arg("legendIn"));
		cl.def("plotFrame", (void (Pythia8::HistPlot::*)(std::string, const class Pythia8::Hist &, std::string, std::string, std::string, std::string, std::string, bool)) &Pythia8::HistPlot::plotFrame, "C++: Pythia8::HistPlot::plotFrame(std::string, const class Pythia8::Hist &, std::string, std::string, std::string, std::string, std::string, bool) --> void", pybind11::arg("frameIn"), pybind11::arg("histIn"), pybind11::arg("titleIn"), pybind11::arg("xLabIn"), pybind11::arg("yLabIn"), pybind11::arg("styleIn"), pybind11::arg("legendIn"), pybind11::arg("logY"));
	}
}
