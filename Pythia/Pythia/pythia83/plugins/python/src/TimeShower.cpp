#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/BeamRemnants.h>
#include <Pythia8/ColourReconnection.h>
#include <Pythia8/Event.h>
#include <Pythia8/FragmentationFlavZpT.h>
#include <Pythia8/FragmentationSystems.h>
#include <Pythia8/HadronLevel.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/HelicityBasics.h>
#include <Pythia8/HelicityMatrixElements.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/LesHouches.h>
#include <Pythia8/MergingHooks.h>
#include <Pythia8/MultipartonInteractions.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ParticleDecays.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/PartonVertex.h>
#include <Pythia8/PhysicsBase.h>
#include <Pythia8/RHadrons.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/SLHAinterface.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SigmaLowEnergy.h>
#include <Pythia8/SigmaProcess.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/SpaceShower.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/StringInteractions.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/TauDecays.h>
#include <Pythia8/TimeShower.h>
#include <Pythia8/Weights.h>
#include <complex>
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

// Pythia8::TimeShower file:Pythia8/TimeShower.h line:33
struct PyCallBack_Pythia8_TimeShower : public Pythia8::TimeShower {
	using Pythia8::TimeShower::TimeShower;

	bool limitPTmax(class Pythia8::Event & a0, double a1, double a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "limitPTmax");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::limitPTmax(a0, a1, a2);
	}
	int shower(int a0, int a1, class Pythia8::Event & a2, double a3, int a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "shower");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return TimeShower::shower(a0, a1, a2, a3, a4);
	}
	int showerQED(int a0, int a1, class Pythia8::Event & a2, double a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "showerQED");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return TimeShower::showerQED(a0, a1, a2, a3);
	}
	int showerQEDafterRemnants(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "showerQEDafterRemnants");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return TimeShower::showerQEDafterRemnants(a0);
	}
	void prepareProcess(class Pythia8::Event & a0, class Pythia8::Event & a1, class std::vector<int, class std::allocator<int> > & a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "prepareProcess");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::prepareProcess(a0, a1, a2);
	}
	void prepareGlobal(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "prepareGlobal");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::prepareGlobal(a0);
	}
	void prepare(int a0, class Pythia8::Event & a1, bool a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "prepare");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::prepare(a0, a1, a2);
	}
	void rescatterUpdate(int a0, class Pythia8::Event & a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "rescatterUpdate");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::rescatterUpdate(a0, a1);
	}
	void update(int a0, class Pythia8::Event & a1, bool a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "update");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::update(a0, a1, a2);
	}
	double pTnext(class Pythia8::Event & a0, double a1, double a2, bool a3, bool a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "pTnext");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::pTnext(a0, a1, a2, a3, a4);
	}
	double pTnextResDec() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "pTnextResDec");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::pTnextResDec();
	}
	bool branch(class Pythia8::Event & a0, bool a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "branch");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::branch(a0, a1);
	}
	bool resonanceShower(class Pythia8::Event & a0, class Pythia8::Event & a1, class std::vector<int, class std::allocator<int> > & a2, double a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "resonanceShower");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::resonanceShower(a0, a1, a2, a3);
	}
	void list() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "list");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TimeShower::list();
	}
	bool initUncertainties() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "initUncertainties");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::initUncertainties();
	}
	bool initEnhancements() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "initEnhancements");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::initEnhancements();
	}
	bool getHasWeaklyRadiated() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "getHasWeaklyRadiated");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::getHasWeaklyRadiated();
	}
	int system() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "system");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return TimeShower::system();
	}
	double enhancePTmax() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "enhancePTmax");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::enhancePTmax();
	}
	double pTLastInShower() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "pTLastInShower");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::pTLastInShower();
	}
	class Pythia8::Event clustered(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "clustered");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<class Pythia8::Event>::value) {
				static pybind11::detail::overload_caster_t<class Pythia8::Event> caster;
				return pybind11::detail::cast_ref<class Pythia8::Event>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Pythia8::Event>(std::move(o));
		}
		return TimeShower::clustered(a0, a1, a2, a3, a4);
	}
	using _binder_ret_0 = class std::map<class std::basic_string<char>, double, struct std::less<class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, double> > >;
	_binder_ret_0 getStateVariables(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "getStateVariables");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return TimeShower::getStateVariables(a0, a1, a2, a3, a4);
	}
	bool isTimelike(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "isTimelike");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::isTimelike(a0, a1, a2, a3, a4);
	}
	using _binder_ret_1 = class std::vector<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >;
	_binder_ret_1 getSplittingName(const class Pythia8::Event & a0, int a1, int a2, int a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "getSplittingName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return TimeShower::getSplittingName(a0, a1, a2, a3);
	}
	double getSplittingProb(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "getSplittingProb");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::getSplittingProb(a0, a1, a2, a3, a4);
	}
	bool allowedSplitting(const class Pythia8::Event & a0, int a1, int a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "allowedSplitting");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return TimeShower::allowedSplitting(a0, a1, a2);
	}
	using _binder_ret_2 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_2 getRecoilers(const class Pythia8::Event & a0, int a1, int a2, class std::basic_string<char> a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "getRecoilers");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return TimeShower::getRecoilers(a0, a1, a2, a3);
	}
	double enhanceFactor(const class std::basic_string<char> & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "enhanceFactor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return TimeShower::enhanceFactor(a0);
	}
	void onInitInfoPtr() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "onInitInfoPtr");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onInitInfoPtr();
	}
	void onBeginEvent() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "onBeginEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onBeginEvent();
	}
	void onEndEvent(enum Pythia8::PhysicsBase::Status a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "onEndEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onEndEvent(a0);
	}
	void onStat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::TimeShower *>(this), "onStat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onStat();
	}
};

// Pythia8::DecayHandler file:Pythia8/ParticleDecays.h line:31
struct PyCallBack_Pythia8_DecayHandler : public Pythia8::DecayHandler {
	using Pythia8::DecayHandler::DecayHandler;

	bool decay(class std::vector<int, class std::allocator<int> > & a0, class std::vector<double, class std::allocator<double> > & a1, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > & a2, int a3, const class Pythia8::Event & a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::DecayHandler *>(this), "decay");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DecayHandler::decay(a0, a1, a2, a3, a4);
	}
	bool chainDecay(class std::vector<int, class std::allocator<int> > & a0, class std::vector<int, class std::allocator<int> > & a1, class std::vector<double, class std::allocator<double> > & a2, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > & a3, int a4, const class Pythia8::Event & a5) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::DecayHandler *>(this), "chainDecay");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4, a5);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return DecayHandler::chainDecay(a0, a1, a2, a3, a4, a5);
	}
};

// Pythia8::SpaceShower file:Pythia8/SpaceShower.h line:33
struct PyCallBack_Pythia8_SpaceShower : public Pythia8::SpaceShower {
	using Pythia8::SpaceShower::SpaceShower;

	bool limitPTmax(class Pythia8::Event & a0, double a1, double a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "limitPTmax");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::limitPTmax(a0, a1, a2);
	}
	void prepare(int a0, class Pythia8::Event & a1, bool a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "prepare");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SpaceShower::prepare(a0, a1, a2);
	}
	void update(int a0, class Pythia8::Event & a1, bool a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "update");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SpaceShower::update(a0, a1, a2);
	}
	double pTnext(class Pythia8::Event & a0, double a1, double a2, int a3, bool a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "pTnext");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SpaceShower::pTnext(a0, a1, a2, a3, a4);
	}
	bool branch(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "branch");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::branch(a0);
	}
	void list() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "list");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SpaceShower::list();
	}
	bool initUncertainties() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "initUncertainties");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::initUncertainties();
	}
	bool initEnhancements() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "initEnhancements");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::initEnhancements();
	}
	bool doRestart() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "doRestart");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::doRestart();
	}
	bool wasGamma2qqbar() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "wasGamma2qqbar");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::wasGamma2qqbar();
	}
	bool getHasWeaklyRadiated() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "getHasWeaklyRadiated");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::getHasWeaklyRadiated();
	}
	int system() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "system");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SpaceShower::system();
	}
	double enhancePTmax() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "enhancePTmax");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SpaceShower::enhancePTmax();
	}
	class Pythia8::Event clustered(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "clustered");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<class Pythia8::Event>::value) {
				static pybind11::detail::overload_caster_t<class Pythia8::Event> caster;
				return pybind11::detail::cast_ref<class Pythia8::Event>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Pythia8::Event>(std::move(o));
		}
		return SpaceShower::clustered(a0, a1, a2, a3, a4);
	}
	using _binder_ret_0 = class std::map<class std::basic_string<char>, double, struct std::less<class std::basic_string<char> >, class std::allocator<struct std::pair<const class std::basic_string<char>, double> > >;
	_binder_ret_0 getStateVariables(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "getStateVariables");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_0>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_0> caster;
				return pybind11::detail::cast_ref<_binder_ret_0>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_0>(std::move(o));
		}
		return SpaceShower::getStateVariables(a0, a1, a2, a3, a4);
	}
	bool isSpacelike(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "isSpacelike");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::isSpacelike(a0, a1, a2, a3, a4);
	}
	using _binder_ret_1 = class std::vector<class std::basic_string<char>, class std::allocator<class std::basic_string<char> > >;
	_binder_ret_1 getSplittingName(const class Pythia8::Event & a0, int a1, int a2, int a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "getSplittingName");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_1>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_1> caster;
				return pybind11::detail::cast_ref<_binder_ret_1>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_1>(std::move(o));
		}
		return SpaceShower::getSplittingName(a0, a1, a2, a3);
	}
	double getSplittingProb(const class Pythia8::Event & a0, int a1, int a2, int a3, class std::basic_string<char> a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "getSplittingProb");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SpaceShower::getSplittingProb(a0, a1, a2, a3, a4);
	}
	bool allowedSplitting(const class Pythia8::Event & a0, int a1, int a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "allowedSplitting");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SpaceShower::allowedSplitting(a0, a1, a2);
	}
	using _binder_ret_2 = class std::vector<int, class std::allocator<int> >;
	_binder_ret_2 getRecoilers(const class Pythia8::Event & a0, int a1, int a2, class std::basic_string<char> a3) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "getRecoilers");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3);
			if (pybind11::detail::cast_is_temporary_value_reference<_binder_ret_2>::value) {
				static pybind11::detail::overload_caster_t<_binder_ret_2> caster;
				return pybind11::detail::cast_ref<_binder_ret_2>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<_binder_ret_2>(std::move(o));
		}
		return SpaceShower::getRecoilers(a0, a1, a2, a3);
	}
	double enhanceFactor(const class std::basic_string<char> & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "enhanceFactor");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SpaceShower::enhanceFactor(a0);
	}
	void onInitInfoPtr() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "onInitInfoPtr");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onInitInfoPtr();
	}
	void onBeginEvent() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "onBeginEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onBeginEvent();
	}
	void onEndEvent(enum Pythia8::PhysicsBase::Status a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "onEndEvent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onEndEvent(a0);
	}
	void onStat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SpaceShower *>(this), "onStat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return PhysicsBase::onStat();
	}
};

void bind_Pythia8_TimeShower(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::TimeShower file:Pythia8/TimeShower.h line:33
		pybind11::class_<Pythia8::TimeShower, std::shared_ptr<Pythia8::TimeShower>, PyCallBack_Pythia8_TimeShower> cl(M("Pythia8"), "TimeShower", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::TimeShower(); }, [](){ return new PyCallBack_Pythia8_TimeShower(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Pythia8_TimeShower const &o){ return new PyCallBack_Pythia8_TimeShower(o); } ) );
		cl.def( pybind11::init( [](Pythia8::TimeShower const &o){ return new Pythia8::TimeShower(o); } ) );
		cl.def_readwrite("mergingHooksPtr", &Pythia8::TimeShower::mergingHooksPtr);
		cl.def_readwrite("beamOffset", &Pythia8::TimeShower::beamOffset);
		cl.def_readwrite("partonVertexPtr", &Pythia8::TimeShower::partonVertexPtr);
		cl.def_readwrite("doUncertainties", &Pythia8::TimeShower::doUncertainties);
		cl.def_readwrite("uVarMuSoftCorr", &Pythia8::TimeShower::uVarMuSoftCorr);
		cl.def_readwrite("uVarMPIshowers", &Pythia8::TimeShower::uVarMPIshowers);
		cl.def_readwrite("noResVariations", &Pythia8::TimeShower::noResVariations);
		cl.def_readwrite("noProcVariations", &Pythia8::TimeShower::noProcVariations);
		cl.def_readwrite("nUncertaintyVariations", &Pythia8::TimeShower::nUncertaintyVariations);
		cl.def_readwrite("nVarQCD", &Pythia8::TimeShower::nVarQCD);
		cl.def_readwrite("uVarNflavQ", &Pythia8::TimeShower::uVarNflavQ);
		cl.def_readwrite("dASmax", &Pythia8::TimeShower::dASmax);
		cl.def_readwrite("cNSpTmin", &Pythia8::TimeShower::cNSpTmin);
		cl.def_readwrite("uVarpTmin2", &Pythia8::TimeShower::uVarpTmin2);
		cl.def_readwrite("overFactor", &Pythia8::TimeShower::overFactor);
		cl.def_readwrite("varG2GGmuRfac", &Pythia8::TimeShower::varG2GGmuRfac);
		cl.def_readwrite("varQ2QGmuRfac", &Pythia8::TimeShower::varQ2QGmuRfac);
		cl.def_readwrite("varG2QQmuRfac", &Pythia8::TimeShower::varG2QQmuRfac);
		cl.def_readwrite("varX2XGmuRfac", &Pythia8::TimeShower::varX2XGmuRfac);
		cl.def_readwrite("varG2GGcNS", &Pythia8::TimeShower::varG2GGcNS);
		cl.def_readwrite("varQ2QGcNS", &Pythia8::TimeShower::varQ2QGcNS);
		cl.def_readwrite("varG2QQcNS", &Pythia8::TimeShower::varG2QQcNS);
		cl.def_readwrite("varX2XGcNS", &Pythia8::TimeShower::varX2XGcNS);
		cl.def_readwrite("enhanceFSR", &Pythia8::TimeShower::enhanceFSR);
		cl.def("limitPTmax", [](Pythia8::TimeShower &o, class Pythia8::Event & a0) -> bool { return o.limitPTmax(a0); }, "", pybind11::arg(""));
		cl.def("limitPTmax", [](Pythia8::TimeShower &o, class Pythia8::Event & a0, double const & a1) -> bool { return o.limitPTmax(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("limitPTmax", (bool (Pythia8::TimeShower::*)(class Pythia8::Event &, double, double)) &Pythia8::TimeShower::limitPTmax, "C++: Pythia8::TimeShower::limitPTmax(class Pythia8::Event &, double, double) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("shower", [](Pythia8::TimeShower &o, int const & a0, int const & a1, class Pythia8::Event & a2, double const & a3) -> int { return o.shower(a0, a1, a2, a3); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("shower", (int (Pythia8::TimeShower::*)(int, int, class Pythia8::Event &, double, int)) &Pythia8::TimeShower::shower, "C++: Pythia8::TimeShower::shower(int, int, class Pythia8::Event &, double, int) --> int", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("showerQED", (int (Pythia8::TimeShower::*)(int, int, class Pythia8::Event &, double)) &Pythia8::TimeShower::showerQED, "C++: Pythia8::TimeShower::showerQED(int, int, class Pythia8::Event &, double) --> int", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("showerQEDafterRemnants", (int (Pythia8::TimeShower::*)(class Pythia8::Event &)) &Pythia8::TimeShower::showerQEDafterRemnants, "C++: Pythia8::TimeShower::showerQEDafterRemnants(class Pythia8::Event &) --> int", pybind11::arg(""));
		cl.def("prepareProcess", (void (Pythia8::TimeShower::*)(class Pythia8::Event &, class Pythia8::Event &, class std::vector<int, class std::allocator<int> > &)) &Pythia8::TimeShower::prepareProcess, "C++: Pythia8::TimeShower::prepareProcess(class Pythia8::Event &, class Pythia8::Event &, class std::vector<int, class std::allocator<int> > &) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("prepareGlobal", (void (Pythia8::TimeShower::*)(class Pythia8::Event &)) &Pythia8::TimeShower::prepareGlobal, "C++: Pythia8::TimeShower::prepareGlobal(class Pythia8::Event &) --> void", pybind11::arg(""));
		cl.def("prepare", [](Pythia8::TimeShower &o, int const & a0, class Pythia8::Event & a1) -> void { return o.prepare(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("prepare", (void (Pythia8::TimeShower::*)(int, class Pythia8::Event &, bool)) &Pythia8::TimeShower::prepare, "C++: Pythia8::TimeShower::prepare(int, class Pythia8::Event &, bool) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("rescatterUpdate", (void (Pythia8::TimeShower::*)(int, class Pythia8::Event &)) &Pythia8::TimeShower::rescatterUpdate, "C++: Pythia8::TimeShower::rescatterUpdate(int, class Pythia8::Event &) --> void", pybind11::arg(""), pybind11::arg(""));
		cl.def("update", [](Pythia8::TimeShower &o, int const & a0, class Pythia8::Event & a1) -> void { return o.update(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("update", (void (Pythia8::TimeShower::*)(int, class Pythia8::Event &, bool)) &Pythia8::TimeShower::update, "C++: Pythia8::TimeShower::update(int, class Pythia8::Event &, bool) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", [](Pythia8::TimeShower &o, class Pythia8::Event & a0, double const & a1, double const & a2) -> double { return o.pTnext(a0, a1, a2); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", [](Pythia8::TimeShower &o, class Pythia8::Event & a0, double const & a1, double const & a2, bool const & a3) -> double { return o.pTnext(a0, a1, a2, a3); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", (double (Pythia8::TimeShower::*)(class Pythia8::Event &, double, double, bool, bool)) &Pythia8::TimeShower::pTnext, "C++: Pythia8::TimeShower::pTnext(class Pythia8::Event &, double, double, bool, bool) --> double", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnextResDec", (double (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::pTnextResDec, "C++: Pythia8::TimeShower::pTnextResDec() --> double");
		cl.def("branch", [](Pythia8::TimeShower &o, class Pythia8::Event & a0) -> bool { return o.branch(a0); }, "", pybind11::arg(""));
		cl.def("branch", (bool (Pythia8::TimeShower::*)(class Pythia8::Event &, bool)) &Pythia8::TimeShower::branch, "C++: Pythia8::TimeShower::branch(class Pythia8::Event &, bool) --> bool", pybind11::arg(""), pybind11::arg(""));
		cl.def("resonanceShower", [](Pythia8::TimeShower &o, class Pythia8::Event & a0, class Pythia8::Event & a1, class std::vector<int, class std::allocator<int> > & a2) -> bool { return o.resonanceShower(a0, a1, a2); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("resonanceShower", (bool (Pythia8::TimeShower::*)(class Pythia8::Event &, class Pythia8::Event &, class std::vector<int, class std::allocator<int> > &, double)) &Pythia8::TimeShower::resonanceShower, "C++: Pythia8::TimeShower::resonanceShower(class Pythia8::Event &, class Pythia8::Event &, class std::vector<int, class std::allocator<int> > &, double) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("list", (void (Pythia8::TimeShower::*)() const) &Pythia8::TimeShower::list, "C++: Pythia8::TimeShower::list() const --> void");
		cl.def("initUncertainties", (bool (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::initUncertainties, "C++: Pythia8::TimeShower::initUncertainties() --> bool");
		cl.def("initEnhancements", (bool (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::initEnhancements, "C++: Pythia8::TimeShower::initEnhancements() --> bool");
		cl.def("getHasWeaklyRadiated", (bool (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::getHasWeaklyRadiated, "C++: Pythia8::TimeShower::getHasWeaklyRadiated() --> bool");
		cl.def("system", (int (Pythia8::TimeShower::*)() const) &Pythia8::TimeShower::system, "C++: Pythia8::TimeShower::system() const --> int");
		cl.def("enhancePTmax", (double (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::enhancePTmax, "C++: Pythia8::TimeShower::enhancePTmax() --> double");
		cl.def("pTLastInShower", (double (Pythia8::TimeShower::*)()) &Pythia8::TimeShower::pTLastInShower, "C++: Pythia8::TimeShower::pTLastInShower() --> double");
		cl.def("clustered", (class Pythia8::Event (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::TimeShower::clustered, "C++: Pythia8::TimeShower::clustered(const class Pythia8::Event &, int, int, int, std::string) --> class Pythia8::Event", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getStateVariables", (class std::map<std::string, double, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, double> > > (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::TimeShower::getStateVariables, "C++: Pythia8::TimeShower::getStateVariables(const class Pythia8::Event &, int, int, int, std::string) --> class std::map<std::string, double, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, double> > >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("isTimelike", (bool (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::TimeShower::isTimelike, "C++: Pythia8::TimeShower::isTimelike(const class Pythia8::Event &, int, int, int, std::string) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getSplittingName", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, int)) &Pythia8::TimeShower::getSplittingName, "C++: Pythia8::TimeShower::getSplittingName(const class Pythia8::Event &, int, int, int) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getSplittingProb", (double (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::TimeShower::getSplittingProb, "C++: Pythia8::TimeShower::getSplittingProb(const class Pythia8::Event &, int, int, int, std::string) --> double", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("allowedSplitting", (bool (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int)) &Pythia8::TimeShower::allowedSplitting, "C++: Pythia8::TimeShower::allowedSplitting(const class Pythia8::Event &, int, int) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getRecoilers", (class std::vector<int, class std::allocator<int> > (Pythia8::TimeShower::*)(const class Pythia8::Event &, int, int, std::string)) &Pythia8::TimeShower::getRecoilers, "C++: Pythia8::TimeShower::getRecoilers(const class Pythia8::Event &, int, int, std::string) --> class std::vector<int, class std::allocator<int> >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("enhanceFactor", (double (Pythia8::TimeShower::*)(const std::string &)) &Pythia8::TimeShower::enhanceFactor, "C++: Pythia8::TimeShower::enhanceFactor(const std::string &) --> double", pybind11::arg("name"));
		cl.def("assign", (class Pythia8::TimeShower & (Pythia8::TimeShower::*)(const class Pythia8::TimeShower &)) &Pythia8::TimeShower::operator=, "C++: Pythia8::TimeShower::operator=(const class Pythia8::TimeShower &) --> class Pythia8::TimeShower &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::DecayHandler file:Pythia8/ParticleDecays.h line:31
		pybind11::class_<Pythia8::DecayHandler, std::shared_ptr<Pythia8::DecayHandler>, PyCallBack_Pythia8_DecayHandler> cl(M("Pythia8"), "DecayHandler", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::DecayHandler(); }, [](){ return new PyCallBack_Pythia8_DecayHandler(); } ) );
		cl.def("decay", (bool (Pythia8::DecayHandler::*)(class std::vector<int, class std::allocator<int> > &, class std::vector<double, class std::allocator<double> > &, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > &, int, const class Pythia8::Event &)) &Pythia8::DecayHandler::decay, "C++: Pythia8::DecayHandler::decay(class std::vector<int, class std::allocator<int> > &, class std::vector<double, class std::allocator<double> > &, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > &, int, const class Pythia8::Event &) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("chainDecay", (bool (Pythia8::DecayHandler::*)(class std::vector<int, class std::allocator<int> > &, class std::vector<int, class std::allocator<int> > &, class std::vector<double, class std::allocator<double> > &, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > &, int, const class Pythia8::Event &)) &Pythia8::DecayHandler::chainDecay, "C++: Pythia8::DecayHandler::chainDecay(class std::vector<int, class std::allocator<int> > &, class std::vector<int, class std::allocator<int> > &, class std::vector<double, class std::allocator<double> > &, class std::vector<class Pythia8::Vec4, class std::allocator<class Pythia8::Vec4> > &, int, const class Pythia8::Event &) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("assign", (class Pythia8::DecayHandler & (Pythia8::DecayHandler::*)(const class Pythia8::DecayHandler &)) &Pythia8::DecayHandler::operator=, "C++: Pythia8::DecayHandler::operator=(const class Pythia8::DecayHandler &) --> class Pythia8::DecayHandler &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
	{ // Pythia8::SpaceShower file:Pythia8/SpaceShower.h line:33
		pybind11::class_<Pythia8::SpaceShower, std::shared_ptr<Pythia8::SpaceShower>, PyCallBack_Pythia8_SpaceShower> cl(M("Pythia8"), "SpaceShower", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::SpaceShower(); }, [](){ return new PyCallBack_Pythia8_SpaceShower(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Pythia8_SpaceShower const &o){ return new PyCallBack_Pythia8_SpaceShower(o); } ) );
		cl.def( pybind11::init( [](Pythia8::SpaceShower const &o){ return new Pythia8::SpaceShower(o); } ) );
		cl.def_readwrite("mergingHooksPtr", &Pythia8::SpaceShower::mergingHooksPtr);
		cl.def_readwrite("beamOffset", &Pythia8::SpaceShower::beamOffset);
		cl.def_readwrite("partonVertexPtr", &Pythia8::SpaceShower::partonVertexPtr);
		cl.def_readwrite("doUncertainties", &Pythia8::SpaceShower::doUncertainties);
		cl.def_readwrite("uVarMuSoftCorr", &Pythia8::SpaceShower::uVarMuSoftCorr);
		cl.def_readwrite("uVarMPIshowers", &Pythia8::SpaceShower::uVarMPIshowers);
		cl.def_readwrite("nUncertaintyVariations", &Pythia8::SpaceShower::nUncertaintyVariations);
		cl.def_readwrite("nVarQCD", &Pythia8::SpaceShower::nVarQCD);
		cl.def_readwrite("uVarNflavQ", &Pythia8::SpaceShower::uVarNflavQ);
		cl.def_readwrite("dASmax", &Pythia8::SpaceShower::dASmax);
		cl.def_readwrite("cNSpTmin", &Pythia8::SpaceShower::cNSpTmin);
		cl.def_readwrite("uVarpTmin2", &Pythia8::SpaceShower::uVarpTmin2);
		cl.def_readwrite("overFactor", &Pythia8::SpaceShower::overFactor);
		cl.def_readwrite("varG2GGmuRfac", &Pythia8::SpaceShower::varG2GGmuRfac);
		cl.def_readwrite("varQ2QGmuRfac", &Pythia8::SpaceShower::varQ2QGmuRfac);
		cl.def_readwrite("varQ2GQmuRfac", &Pythia8::SpaceShower::varQ2GQmuRfac);
		cl.def_readwrite("varG2QQmuRfac", &Pythia8::SpaceShower::varG2QQmuRfac);
		cl.def_readwrite("varX2XGmuRfac", &Pythia8::SpaceShower::varX2XGmuRfac);
		cl.def_readwrite("varG2GGcNS", &Pythia8::SpaceShower::varG2GGcNS);
		cl.def_readwrite("varQ2QGcNS", &Pythia8::SpaceShower::varQ2QGcNS);
		cl.def_readwrite("varQ2GQcNS", &Pythia8::SpaceShower::varQ2GQcNS);
		cl.def_readwrite("varG2QQcNS", &Pythia8::SpaceShower::varG2QQcNS);
		cl.def_readwrite("varX2XGcNS", &Pythia8::SpaceShower::varX2XGcNS);
		cl.def_readwrite("enhanceISR", &Pythia8::SpaceShower::enhanceISR);
		cl.def("limitPTmax", [](Pythia8::SpaceShower &o, class Pythia8::Event & a0) -> bool { return o.limitPTmax(a0); }, "", pybind11::arg(""));
		cl.def("limitPTmax", [](Pythia8::SpaceShower &o, class Pythia8::Event & a0, double const & a1) -> bool { return o.limitPTmax(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("limitPTmax", (bool (Pythia8::SpaceShower::*)(class Pythia8::Event &, double, double)) &Pythia8::SpaceShower::limitPTmax, "C++: Pythia8::SpaceShower::limitPTmax(class Pythia8::Event &, double, double) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("prepare", [](Pythia8::SpaceShower &o, int const & a0, class Pythia8::Event & a1) -> void { return o.prepare(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("prepare", (void (Pythia8::SpaceShower::*)(int, class Pythia8::Event &, bool)) &Pythia8::SpaceShower::prepare, "C++: Pythia8::SpaceShower::prepare(int, class Pythia8::Event &, bool) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("update", [](Pythia8::SpaceShower &o, int const & a0, class Pythia8::Event & a1) -> void { return o.update(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("update", (void (Pythia8::SpaceShower::*)(int, class Pythia8::Event &, bool)) &Pythia8::SpaceShower::update, "C++: Pythia8::SpaceShower::update(int, class Pythia8::Event &, bool) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", [](Pythia8::SpaceShower &o, class Pythia8::Event & a0, double const & a1, double const & a2) -> double { return o.pTnext(a0, a1, a2); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", [](Pythia8::SpaceShower &o, class Pythia8::Event & a0, double const & a1, double const & a2, int const & a3) -> double { return o.pTnext(a0, a1, a2, a3); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("pTnext", (double (Pythia8::SpaceShower::*)(class Pythia8::Event &, double, double, int, bool)) &Pythia8::SpaceShower::pTnext, "C++: Pythia8::SpaceShower::pTnext(class Pythia8::Event &, double, double, int, bool) --> double", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("branch", (bool (Pythia8::SpaceShower::*)(class Pythia8::Event &)) &Pythia8::SpaceShower::branch, "C++: Pythia8::SpaceShower::branch(class Pythia8::Event &) --> bool", pybind11::arg(""));
		cl.def("list", (void (Pythia8::SpaceShower::*)() const) &Pythia8::SpaceShower::list, "C++: Pythia8::SpaceShower::list() const --> void");
		cl.def("initUncertainties", (bool (Pythia8::SpaceShower::*)()) &Pythia8::SpaceShower::initUncertainties, "C++: Pythia8::SpaceShower::initUncertainties() --> bool");
		cl.def("initEnhancements", (bool (Pythia8::SpaceShower::*)()) &Pythia8::SpaceShower::initEnhancements, "C++: Pythia8::SpaceShower::initEnhancements() --> bool");
		cl.def("doRestart", (bool (Pythia8::SpaceShower::*)() const) &Pythia8::SpaceShower::doRestart, "C++: Pythia8::SpaceShower::doRestart() const --> bool");
		cl.def("wasGamma2qqbar", (bool (Pythia8::SpaceShower::*)()) &Pythia8::SpaceShower::wasGamma2qqbar, "C++: Pythia8::SpaceShower::wasGamma2qqbar() --> bool");
		cl.def("getHasWeaklyRadiated", (bool (Pythia8::SpaceShower::*)()) &Pythia8::SpaceShower::getHasWeaklyRadiated, "C++: Pythia8::SpaceShower::getHasWeaklyRadiated() --> bool");
		cl.def("system", (int (Pythia8::SpaceShower::*)() const) &Pythia8::SpaceShower::system, "C++: Pythia8::SpaceShower::system() const --> int");
		cl.def("enhancePTmax", (double (Pythia8::SpaceShower::*)() const) &Pythia8::SpaceShower::enhancePTmax, "C++: Pythia8::SpaceShower::enhancePTmax() const --> double");
		cl.def("clustered", (class Pythia8::Event (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::SpaceShower::clustered, "C++: Pythia8::SpaceShower::clustered(const class Pythia8::Event &, int, int, int, std::string) --> class Pythia8::Event", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getStateVariables", (class std::map<std::string, double, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, double> > > (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::SpaceShower::getStateVariables, "C++: Pythia8::SpaceShower::getStateVariables(const class Pythia8::Event &, int, int, int, std::string) --> class std::map<std::string, double, struct std::less<std::string >, class std::allocator<struct std::pair<const std::string, double> > >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("isSpacelike", (bool (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::SpaceShower::isSpacelike, "C++: Pythia8::SpaceShower::isSpacelike(const class Pythia8::Event &, int, int, int, std::string) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getSplittingName", (class std::vector<std::string, class std::allocator<std::string > > (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, int)) &Pythia8::SpaceShower::getSplittingName, "C++: Pythia8::SpaceShower::getSplittingName(const class Pythia8::Event &, int, int, int) --> class std::vector<std::string, class std::allocator<std::string > >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getSplittingProb", (double (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, int, std::string)) &Pythia8::SpaceShower::getSplittingProb, "C++: Pythia8::SpaceShower::getSplittingProb(const class Pythia8::Event &, int, int, int, std::string) --> double", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("allowedSplitting", (bool (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int)) &Pythia8::SpaceShower::allowedSplitting, "C++: Pythia8::SpaceShower::allowedSplitting(const class Pythia8::Event &, int, int) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("getRecoilers", (class std::vector<int, class std::allocator<int> > (Pythia8::SpaceShower::*)(const class Pythia8::Event &, int, int, std::string)) &Pythia8::SpaceShower::getRecoilers, "C++: Pythia8::SpaceShower::getRecoilers(const class Pythia8::Event &, int, int, std::string) --> class std::vector<int, class std::allocator<int> >", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("enhanceFactor", (double (Pythia8::SpaceShower::*)(const std::string &)) &Pythia8::SpaceShower::enhanceFactor, "C++: Pythia8::SpaceShower::enhanceFactor(const std::string &) --> double", pybind11::arg("name"));
		cl.def("assign", (class Pythia8::SpaceShower & (Pythia8::SpaceShower::*)(const class Pythia8::SpaceShower &)) &Pythia8::SpaceShower::operator=, "C++: Pythia8::SpaceShower::operator=(const class Pythia8::SpaceShower &) --> class Pythia8::SpaceShower &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
}
