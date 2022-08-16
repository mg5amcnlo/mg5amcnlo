#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_std_exception(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_stdexcept(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_std_stl_vector(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Basics(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Basics_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Basics_2(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Info(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Settings(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_ParticleData(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_ParticleData_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Event(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Event_1(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Analysis(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_SigmaProcess(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_UserHooks(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_TimeShower(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_Pythia(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_Pythia8_HeavyIons(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(pythia8, root_module) {
	root_module.doc() = "* Documentation through the built-in help function in Python provides\n  the relevant mappings between the Python interface and the C++\n  code. For documentation on the purpose of the code, see the user\n  HTML manual and Doxygen.\n* Templated C++ types are returned as native Python types,\n  e.g. vectors are returned as lists, etc. Similarly, native Python\n  types can be passed as arguments and are converted to their\n  equivalent C++ templates.\n* No difference is made between passing by reference or pointer;\n  methods in C++ which take a pointer as an argument in Python simply\n  take the object, e.g. foo(Vec4*) and foo(Vec4&) are the same in the\n  Python interface.\n* All operators defined in C++, e.g. Vec4*double are available. Note\n  that reverse operators are not, e.g. double*Vec4.\n* Classes with defined [] operators are iterable, using standard\n  Python iteration, e.g. for prt in pythia.event.\n* Classes with a << operator can be printed via the built-in print\n  function in Python. Note this means that a string representation via\n  str is also available for these classes in Python.\n* Derived classes in Python can be passed back to PYTHIA, and should\n  be available for all classes. See main10.py for a direct Python\n  translation of the C++ main10.cc example which uses a derived class\n  from the UserHooks class to veto events.\n* The constant Pythia::info member is available as\n  Pythia::infoPython. Note, this creates a new instance of the Info\n  class, and so needs to be called each time the information might be\n  updated.\n";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_ != "std" ? "" : "std");
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	std::vector< std::pair<std::string, std::string> > sub_modules {
		{"", "std"},
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule(p.second.c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_std_exception(M);
	bind_std_stdexcept(M);
	bind_std_stl_vector(M);
	bind_Pythia8_Basics(M);
	bind_Pythia8_Basics_1(M);
	bind_Pythia8_Basics_2(M);
	bind_Pythia8_Info(M);
	bind_Pythia8_Settings(M);
	bind_Pythia8_ParticleData(M);
	bind_Pythia8_ParticleData_1(M);
	bind_Pythia8_Event(M);
	bind_Pythia8_Event_1(M);
	bind_Pythia8_Analysis(M);
	bind_Pythia8_SigmaProcess(M);
	bind_Pythia8_UserHooks(M);
	bind_Pythia8_TimeShower(M);
	bind_Pythia8_Pythia(M);
	bind_Pythia8_HeavyIons(M);

}
