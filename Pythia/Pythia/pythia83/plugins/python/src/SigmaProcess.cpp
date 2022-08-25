#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/Event.h>
#include <Pythia8/FragmentationFlavZpT.h>
#include <Pythia8/Info.h>
#include <Pythia8/LesHouches.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PhysicsBase.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/SLHAinterface.h>
#include <Pythia8/SigmaProcess.h>
#include <istream>
#include <iterator>
#include <memory>
#include <ostream>
#include <sstream>
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

// Pythia8::SigmaProcess file:Pythia8/SigmaProcess.h line:86
struct PyCallBack_Pythia8_SigmaProcess : public Pythia8::SigmaProcess {
	using Pythia8::SigmaProcess::SigmaProcess;

	void initProc() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "initProc");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::initProc();
	}
	bool initFlux() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "initFlux");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::initFlux();
	}
	void set1Kin(double a0, double a1, double a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "set1Kin");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::set1Kin(a0, a1, a2);
	}
	void set2Kin(double a0, double a1, double a2, double a3, double a4, double a5, double a6, double a7) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "set2Kin");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4, a5, a6, a7);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::set2Kin(a0, a1, a2, a3, a4, a5, a6, a7);
	}
	void set2KinMPI(double a0, double a1, double a2, double a3, double a4, double a5, double a6, bool a7, double a8, double a9) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "set2KinMPI");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::set2KinMPI(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
	}
	void set3Kin(double a0, double a1, double a2, class Pythia8::Vec4 a3, class Pythia8::Vec4 a4, class Pythia8::Vec4 a5, double a6, double a7, double a8, double a9, double a10, double a11) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "set3Kin");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::set3Kin(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
	}
	void sigmaKin() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "sigmaKin");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::sigmaKin();
	}
	double sigmaHat() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "sigmaHat");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::sigmaHat();
	}
	double sigmaHatWrap(int a0, int a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "sigmaHatWrap");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::sigmaHatWrap(a0, a1);
	}
	double sigmaPDF(bool a0, bool a1, bool a2, double a3, double a4) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "sigmaPDF");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::sigmaPDF(a0, a1, a2, a3, a4);
	}
	void setIdColAcol() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "setIdColAcol");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::setIdColAcol();
	}
	bool final2KinMPI(int a0, int a1, class Pythia8::Vec4 a2, class Pythia8::Vec4 a3, double a4, double a5) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "final2KinMPI");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2, a3, a4, a5);
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::final2KinMPI(a0, a1, a2, a3, a4, a5);
	}
	double weightDecayFlav(class Pythia8::Event & a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "weightDecayFlav");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::weightDecayFlav(a0);
	}
	double weightDecay(class Pythia8::Event & a0, int a1, int a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "weightDecay");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::weightDecay(a0, a1, a2);
	}
	void setScale() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "setScale");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::setScale();
	}
	class std::basic_string<char> name() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "name");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class std::basic_string<char>>::value) {
				static pybind11::detail::overload_caster_t<class std::basic_string<char>> caster;
				return pybind11::detail::cast_ref<class std::basic_string<char>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::basic_string<char>>(std::move(o));
		}
		return SigmaProcess::name();
	}
	int code() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "code");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::code();
	}
	int nFinal() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "nFinal");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::nFinal();
	}
	class std::basic_string<char> inFlux() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "inFlux");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class std::basic_string<char>>::value) {
				static pybind11::detail::overload_caster_t<class std::basic_string<char>> caster;
				return pybind11::detail::cast_ref<class std::basic_string<char>>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class std::basic_string<char>>(std::move(o));
		}
		return SigmaProcess::inFlux();
	}
	bool convert2mb() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "convert2mb");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::convert2mb();
	}
	bool convertM2() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "convertM2");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::convertM2();
	}
	bool isLHA() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isLHA");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isLHA();
	}
	bool isNonDiff() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isNonDiff");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isNonDiff();
	}
	bool isResolved() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isResolved");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isResolved();
	}
	bool isDiffA() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isDiffA");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isDiffA();
	}
	bool isDiffB() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isDiffB");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isDiffB();
	}
	bool isDiffC() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isDiffC");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isDiffC();
	}
	bool isSUSY() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isSUSY");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isSUSY();
	}
	bool allowNegativeSigma() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "allowNegativeSigma");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::allowNegativeSigma();
	}
	int id3Mass() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "id3Mass");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::id3Mass();
	}
	int id4Mass() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "id4Mass");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::id4Mass();
	}
	int id5Mass() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "id5Mass");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::id5Mass();
	}
	int resonanceA() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "resonanceA");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::resonanceA();
	}
	int resonanceB() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "resonanceB");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::resonanceB();
	}
	bool isSChannel() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isSChannel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isSChannel();
	}
	int idSChannel() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "idSChannel");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::idSChannel();
	}
	bool isQCD3body() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "isQCD3body");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::isQCD3body();
	}
	int idTchan1() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "idTchan1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::idTchan1();
	}
	int idTchan2() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "idTchan2");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::idTchan2();
	}
	double tChanFracPow1() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "tChanFracPow1");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::tChanFracPow1();
	}
	double tChanFracPow2() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "tChanFracPow2");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<double>::value) {
				static pybind11::detail::overload_caster_t<double> caster;
				return pybind11::detail::cast_ref<double>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<double>(std::move(o));
		}
		return SigmaProcess::tChanFracPow2();
	}
	bool useMirrorWeight() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "useMirrorWeight");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::useMirrorWeight();
	}
	int gmZmode() const override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "gmZmode");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return SigmaProcess::gmZmode();
	}
	void setIdInDiff(int a0, int a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "setIdInDiff");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SigmaProcess::setIdInDiff(a0, a1);
	}
	bool setupForME() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "setupForME");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<bool>::value) {
				static pybind11::detail::overload_caster_t<bool> caster;
				return pybind11::detail::cast_ref<bool>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<bool>(std::move(o));
		}
		return SigmaProcess::setupForME();
	}
	void onInitInfoPtr() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "onInitInfoPtr");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "onBeginEvent");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "onEndEvent");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Pythia8::SigmaProcess *>(this), "onStat");
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

void bind_Pythia8_SigmaProcess(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Pythia8::SigmaProcess file:Pythia8/SigmaProcess.h line:86
		pybind11::class_<Pythia8::SigmaProcess, std::shared_ptr<Pythia8::SigmaProcess>, PyCallBack_Pythia8_SigmaProcess> cl(M("Pythia8"), "SigmaProcess", "");
		pybind11::handle cl_type = cl;

		cl.def( pybind11::init( [](){ return new Pythia8::SigmaProcess(); }, [](){ return new PyCallBack_Pythia8_SigmaProcess(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Pythia8_SigmaProcess const &o){ return new PyCallBack_Pythia8_SigmaProcess(o); } ) );
		cl.def( pybind11::init( [](Pythia8::SigmaProcess const &o){ return new Pythia8::SigmaProcess(o); } ) );
		cl.def_readwrite("lhaUpPtr", &Pythia8::SigmaProcess::lhaUpPtr);
		cl.def_readwrite("doVarE", &Pythia8::SigmaProcess::doVarE);
		cl.def_readwrite("nQuarkIn", &Pythia8::SigmaProcess::nQuarkIn);
		cl.def_readwrite("renormScale1", &Pythia8::SigmaProcess::renormScale1);
		cl.def_readwrite("renormScale2", &Pythia8::SigmaProcess::renormScale2);
		cl.def_readwrite("renormScale3", &Pythia8::SigmaProcess::renormScale3);
		cl.def_readwrite("renormScale3VV", &Pythia8::SigmaProcess::renormScale3VV);
		cl.def_readwrite("factorScale1", &Pythia8::SigmaProcess::factorScale1);
		cl.def_readwrite("factorScale2", &Pythia8::SigmaProcess::factorScale2);
		cl.def_readwrite("factorScale3", &Pythia8::SigmaProcess::factorScale3);
		cl.def_readwrite("factorScale3VV", &Pythia8::SigmaProcess::factorScale3VV);
		cl.def_readwrite("Kfactor", &Pythia8::SigmaProcess::Kfactor);
		cl.def_readwrite("mcME", &Pythia8::SigmaProcess::mcME);
		cl.def_readwrite("mbME", &Pythia8::SigmaProcess::mbME);
		cl.def_readwrite("mmuME", &Pythia8::SigmaProcess::mmuME);
		cl.def_readwrite("mtauME", &Pythia8::SigmaProcess::mtauME);
		cl.def_readwrite("renormMultFac", &Pythia8::SigmaProcess::renormMultFac);
		cl.def_readwrite("renormFixScale", &Pythia8::SigmaProcess::renormFixScale);
		cl.def_readwrite("factorMultFac", &Pythia8::SigmaProcess::factorMultFac);
		cl.def_readwrite("factorFixScale", &Pythia8::SigmaProcess::factorFixScale);
		cl.def_readwrite("higgsH1parity", &Pythia8::SigmaProcess::higgsH1parity);
		cl.def_readwrite("higgsH2parity", &Pythia8::SigmaProcess::higgsH2parity);
		cl.def_readwrite("higgsA3parity", &Pythia8::SigmaProcess::higgsA3parity);
		cl.def_readwrite("higgsH1eta", &Pythia8::SigmaProcess::higgsH1eta);
		cl.def_readwrite("higgsH2eta", &Pythia8::SigmaProcess::higgsH2eta);
		cl.def_readwrite("higgsA3eta", &Pythia8::SigmaProcess::higgsA3eta);
		cl.def_readwrite("higgsH1phi", &Pythia8::SigmaProcess::higgsH1phi);
		cl.def_readwrite("higgsH2phi", &Pythia8::SigmaProcess::higgsH2phi);
		cl.def_readwrite("higgsA3phi", &Pythia8::SigmaProcess::higgsA3phi);
		cl.def_readwrite("idA", &Pythia8::SigmaProcess::idA);
		cl.def_readwrite("idB", &Pythia8::SigmaProcess::idB);
		cl.def_readwrite("mA", &Pythia8::SigmaProcess::mA);
		cl.def_readwrite("mB", &Pythia8::SigmaProcess::mB);
		cl.def_readwrite("isLeptonA", &Pythia8::SigmaProcess::isLeptonA);
		cl.def_readwrite("isLeptonB", &Pythia8::SigmaProcess::isLeptonB);
		cl.def_readwrite("hasLeptonBeams", &Pythia8::SigmaProcess::hasLeptonBeams);
		cl.def_readwrite("beamA2gamma", &Pythia8::SigmaProcess::beamA2gamma);
		cl.def_readwrite("beamB2gamma", &Pythia8::SigmaProcess::beamB2gamma);
		cl.def_readwrite("hasGamma", &Pythia8::SigmaProcess::hasGamma);
		cl.def_readwrite("inBeamA", &Pythia8::SigmaProcess::inBeamA);
		cl.def_readwrite("inBeamB", &Pythia8::SigmaProcess::inBeamB);
		cl.def_readwrite("inPair", &Pythia8::SigmaProcess::inPair);
		cl.def_readwrite("mH", &Pythia8::SigmaProcess::mH);
		cl.def_readwrite("sH", &Pythia8::SigmaProcess::sH);
		cl.def_readwrite("sH2", &Pythia8::SigmaProcess::sH2);
		cl.def_readwrite("Q2RenSave", &Pythia8::SigmaProcess::Q2RenSave);
		cl.def_readwrite("alpEM", &Pythia8::SigmaProcess::alpEM);
		cl.def_readwrite("alpS", &Pythia8::SigmaProcess::alpS);
		cl.def_readwrite("Q2FacSave", &Pythia8::SigmaProcess::Q2FacSave);
		cl.def_readwrite("x1Save", &Pythia8::SigmaProcess::x1Save);
		cl.def_readwrite("x2Save", &Pythia8::SigmaProcess::x2Save);
		cl.def_readwrite("pdf1Save", &Pythia8::SigmaProcess::pdf1Save);
		cl.def_readwrite("pdf2Save", &Pythia8::SigmaProcess::pdf2Save);
		cl.def_readwrite("sigmaSumSave", &Pythia8::SigmaProcess::sigmaSumSave);
		cl.def_readwrite("id1", &Pythia8::SigmaProcess::id1);
		cl.def_readwrite("id2", &Pythia8::SigmaProcess::id2);
		cl.def_readwrite("id3", &Pythia8::SigmaProcess::id3);
		cl.def_readwrite("id4", &Pythia8::SigmaProcess::id4);
		cl.def_readwrite("id5", &Pythia8::SigmaProcess::id5);
		cl.def_readwrite("cosTheta", &Pythia8::SigmaProcess::cosTheta);
		cl.def_readwrite("sinTheta", &Pythia8::SigmaProcess::sinTheta);
		cl.def_readwrite("phi", &Pythia8::SigmaProcess::phi);
		cl.def_readwrite("sHMass", &Pythia8::SigmaProcess::sHMass);
		cl.def_readwrite("sHBeta", &Pythia8::SigmaProcess::sHBeta);
		cl.def_readwrite("pT2Mass", &Pythia8::SigmaProcess::pT2Mass);
		cl.def_readwrite("pTFin", &Pythia8::SigmaProcess::pTFin);
		cl.def_readwrite("pTFinT", &Pythia8::SigmaProcess::pTFinT);
		cl.def_readwrite("cosThetaT", &Pythia8::SigmaProcess::cosThetaT);
		cl.def_readwrite("sinThetaT", &Pythia8::SigmaProcess::sinThetaT);
		cl.def_readwrite("phiT", &Pythia8::SigmaProcess::phiT);
		cl.def_readwrite("swapTU", &Pythia8::SigmaProcess::swapTU);
		cl.def("setLHAPtr", (void (Pythia8::SigmaProcess::*)(class std::shared_ptr<class Pythia8::LHAup>)) &Pythia8::SigmaProcess::setLHAPtr, "C++: Pythia8::SigmaProcess::setLHAPtr(class std::shared_ptr<class Pythia8::LHAup>) --> void", pybind11::arg("lhaUpPtrIn"));
		cl.def("updateBeamIDs", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::updateBeamIDs, "C++: Pythia8::SigmaProcess::updateBeamIDs() --> void");
		cl.def("initProc", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::initProc, "C++: Pythia8::SigmaProcess::initProc() --> void");
		cl.def("initFlux", (bool (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::initFlux, "C++: Pythia8::SigmaProcess::initFlux() --> bool");
		cl.def("set1Kin", (void (Pythia8::SigmaProcess::*)(double, double, double)) &Pythia8::SigmaProcess::set1Kin, "C++: Pythia8::SigmaProcess::set1Kin(double, double, double) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("set2Kin", (void (Pythia8::SigmaProcess::*)(double, double, double, double, double, double, double, double)) &Pythia8::SigmaProcess::set2Kin, "C++: Pythia8::SigmaProcess::set2Kin(double, double, double, double, double, double, double, double) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("set2KinMPI", (void (Pythia8::SigmaProcess::*)(double, double, double, double, double, double, double, bool, double, double)) &Pythia8::SigmaProcess::set2KinMPI, "C++: Pythia8::SigmaProcess::set2KinMPI(double, double, double, double, double, double, double, bool, double, double) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("set3Kin", (void (Pythia8::SigmaProcess::*)(double, double, double, class Pythia8::Vec4, class Pythia8::Vec4, class Pythia8::Vec4, double, double, double, double, double, double)) &Pythia8::SigmaProcess::set3Kin, "C++: Pythia8::SigmaProcess::set3Kin(double, double, double, class Pythia8::Vec4, class Pythia8::Vec4, class Pythia8::Vec4, double, double, double, double, double, double) --> void", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("sigmaKin", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::sigmaKin, "C++: Pythia8::SigmaProcess::sigmaKin() --> void");
		cl.def("sigmaHat", (double (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::sigmaHat, "C++: Pythia8::SigmaProcess::sigmaHat() --> double");
		cl.def("sigmaHatWrap", [](Pythia8::SigmaProcess &o) -> double { return o.sigmaHatWrap(); }, "");
		cl.def("sigmaHatWrap", [](Pythia8::SigmaProcess &o, int const & a0) -> double { return o.sigmaHatWrap(a0); }, "", pybind11::arg("id1in"));
		cl.def("sigmaHatWrap", (double (Pythia8::SigmaProcess::*)(int, int)) &Pythia8::SigmaProcess::sigmaHatWrap, "C++: Pythia8::SigmaProcess::sigmaHatWrap(int, int) --> double", pybind11::arg("id1in"), pybind11::arg("id2in"));
		cl.def("sigmaPDF", [](Pythia8::SigmaProcess &o) -> double { return o.sigmaPDF(); }, "");
		cl.def("sigmaPDF", [](Pythia8::SigmaProcess &o, bool const & a0) -> double { return o.sigmaPDF(a0); }, "", pybind11::arg("initPS"));
		cl.def("sigmaPDF", [](Pythia8::SigmaProcess &o, bool const & a0, bool const & a1) -> double { return o.sigmaPDF(a0, a1); }, "", pybind11::arg("initPS"), pybind11::arg("samexGamma"));
		cl.def("sigmaPDF", [](Pythia8::SigmaProcess &o, bool const & a0, bool const & a1, bool const & a2) -> double { return o.sigmaPDF(a0, a1, a2); }, "", pybind11::arg("initPS"), pybind11::arg("samexGamma"), pybind11::arg("useNewXvalues"));
		cl.def("sigmaPDF", [](Pythia8::SigmaProcess &o, bool const & a0, bool const & a1, bool const & a2, double const & a3) -> double { return o.sigmaPDF(a0, a1, a2, a3); }, "", pybind11::arg("initPS"), pybind11::arg("samexGamma"), pybind11::arg("useNewXvalues"), pybind11::arg("x1New"));
		cl.def("sigmaPDF", (double (Pythia8::SigmaProcess::*)(bool, bool, bool, double, double)) &Pythia8::SigmaProcess::sigmaPDF, "C++: Pythia8::SigmaProcess::sigmaPDF(bool, bool, bool, double, double) --> double", pybind11::arg("initPS"), pybind11::arg("samexGamma"), pybind11::arg("useNewXvalues"), pybind11::arg("x1New"), pybind11::arg("x2New"));
		cl.def("pickInState", [](Pythia8::SigmaProcess &o) -> void { return o.pickInState(); }, "");
		cl.def("pickInState", [](Pythia8::SigmaProcess &o, int const & a0) -> void { return o.pickInState(a0); }, "", pybind11::arg("id1in"));
		cl.def("pickInState", (void (Pythia8::SigmaProcess::*)(int, int)) &Pythia8::SigmaProcess::pickInState, "C++: Pythia8::SigmaProcess::pickInState(int, int) --> void", pybind11::arg("id1in"), pybind11::arg("id2in"));
		cl.def("setIdColAcol", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::setIdColAcol, "C++: Pythia8::SigmaProcess::setIdColAcol() --> void");
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o) -> bool { return o.final2KinMPI(); }, "");
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o, int const & a0) -> bool { return o.final2KinMPI(a0); }, "", pybind11::arg(""));
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1) -> bool { return o.final2KinMPI(a0, a1); }, "", pybind11::arg(""), pybind11::arg(""));
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, class Pythia8::Vec4 const & a2) -> bool { return o.final2KinMPI(a0, a1, a2); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, class Pythia8::Vec4 const & a2, class Pythia8::Vec4 const & a3) -> bool { return o.final2KinMPI(a0, a1, a2, a3); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("final2KinMPI", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, class Pythia8::Vec4 const & a2, class Pythia8::Vec4 const & a3, double const & a4) -> bool { return o.final2KinMPI(a0, a1, a2, a3, a4); }, "", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("final2KinMPI", (bool (Pythia8::SigmaProcess::*)(int, int, class Pythia8::Vec4, class Pythia8::Vec4, double, double)) &Pythia8::SigmaProcess::final2KinMPI, "C++: Pythia8::SigmaProcess::final2KinMPI(int, int, class Pythia8::Vec4, class Pythia8::Vec4, double, double) --> bool", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("weightDecayFlav", (double (Pythia8::SigmaProcess::*)(class Pythia8::Event &)) &Pythia8::SigmaProcess::weightDecayFlav, "C++: Pythia8::SigmaProcess::weightDecayFlav(class Pythia8::Event &) --> double", pybind11::arg(""));
		cl.def("weightDecay", (double (Pythia8::SigmaProcess::*)(class Pythia8::Event &, int, int)) &Pythia8::SigmaProcess::weightDecay, "C++: Pythia8::SigmaProcess::weightDecay(class Pythia8::Event &, int, int) --> double", pybind11::arg(""), pybind11::arg(""), pybind11::arg(""));
		cl.def("setScale", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::setScale, "C++: Pythia8::SigmaProcess::setScale() --> void");
		cl.def("name", (std::string (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::name, "C++: Pythia8::SigmaProcess::name() const --> std::string");
		cl.def("code", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::code, "C++: Pythia8::SigmaProcess::code() const --> int");
		cl.def("nFinal", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::nFinal, "C++: Pythia8::SigmaProcess::nFinal() const --> int");
		cl.def("inFlux", (std::string (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::inFlux, "C++: Pythia8::SigmaProcess::inFlux() const --> std::string");
		cl.def("convert2mb", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::convert2mb, "C++: Pythia8::SigmaProcess::convert2mb() const --> bool");
		cl.def("convertM2", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::convertM2, "C++: Pythia8::SigmaProcess::convertM2() const --> bool");
		cl.def("isLHA", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isLHA, "C++: Pythia8::SigmaProcess::isLHA() const --> bool");
		cl.def("isNonDiff", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isNonDiff, "C++: Pythia8::SigmaProcess::isNonDiff() const --> bool");
		cl.def("isResolved", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isResolved, "C++: Pythia8::SigmaProcess::isResolved() const --> bool");
		cl.def("isDiffA", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isDiffA, "C++: Pythia8::SigmaProcess::isDiffA() const --> bool");
		cl.def("isDiffB", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isDiffB, "C++: Pythia8::SigmaProcess::isDiffB() const --> bool");
		cl.def("isDiffC", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isDiffC, "C++: Pythia8::SigmaProcess::isDiffC() const --> bool");
		cl.def("isSUSY", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isSUSY, "C++: Pythia8::SigmaProcess::isSUSY() const --> bool");
		cl.def("allowNegativeSigma", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::allowNegativeSigma, "C++: Pythia8::SigmaProcess::allowNegativeSigma() const --> bool");
		cl.def("id3Mass", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::id3Mass, "C++: Pythia8::SigmaProcess::id3Mass() const --> int");
		cl.def("id4Mass", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::id4Mass, "C++: Pythia8::SigmaProcess::id4Mass() const --> int");
		cl.def("id5Mass", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::id5Mass, "C++: Pythia8::SigmaProcess::id5Mass() const --> int");
		cl.def("resonanceA", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::resonanceA, "C++: Pythia8::SigmaProcess::resonanceA() const --> int");
		cl.def("resonanceB", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::resonanceB, "C++: Pythia8::SigmaProcess::resonanceB() const --> int");
		cl.def("isSChannel", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isSChannel, "C++: Pythia8::SigmaProcess::isSChannel() const --> bool");
		cl.def("idSChannel", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::idSChannel, "C++: Pythia8::SigmaProcess::idSChannel() const --> int");
		cl.def("isQCD3body", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::isQCD3body, "C++: Pythia8::SigmaProcess::isQCD3body() const --> bool");
		cl.def("idTchan1", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::idTchan1, "C++: Pythia8::SigmaProcess::idTchan1() const --> int");
		cl.def("idTchan2", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::idTchan2, "C++: Pythia8::SigmaProcess::idTchan2() const --> int");
		cl.def("tChanFracPow1", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::tChanFracPow1, "C++: Pythia8::SigmaProcess::tChanFracPow1() const --> double");
		cl.def("tChanFracPow2", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::tChanFracPow2, "C++: Pythia8::SigmaProcess::tChanFracPow2() const --> double");
		cl.def("useMirrorWeight", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::useMirrorWeight, "C++: Pythia8::SigmaProcess::useMirrorWeight() const --> bool");
		cl.def("gmZmode", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::gmZmode, "C++: Pythia8::SigmaProcess::gmZmode() const --> int");
		cl.def("swappedTU", (bool (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::swappedTU, "C++: Pythia8::SigmaProcess::swappedTU() const --> bool");
		cl.def("id", (int (Pythia8::SigmaProcess::*)(int) const) &Pythia8::SigmaProcess::id, "C++: Pythia8::SigmaProcess::id(int) const --> int", pybind11::arg("i"));
		cl.def("col", (int (Pythia8::SigmaProcess::*)(int) const) &Pythia8::SigmaProcess::col, "C++: Pythia8::SigmaProcess::col(int) const --> int", pybind11::arg("i"));
		cl.def("acol", (int (Pythia8::SigmaProcess::*)(int) const) &Pythia8::SigmaProcess::acol, "C++: Pythia8::SigmaProcess::acol(int) const --> int", pybind11::arg("i"));
		cl.def("m", (double (Pythia8::SigmaProcess::*)(int) const) &Pythia8::SigmaProcess::m, "C++: Pythia8::SigmaProcess::m(int) const --> double", pybind11::arg("i"));
		cl.def("getParton", (class Pythia8::Particle (Pythia8::SigmaProcess::*)(int) const) &Pythia8::SigmaProcess::getParton, "C++: Pythia8::SigmaProcess::getParton(int) const --> class Pythia8::Particle", pybind11::arg("i"));
		cl.def("Q2Ren", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::Q2Ren, "C++: Pythia8::SigmaProcess::Q2Ren() const --> double");
		cl.def("alphaEMRen", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::alphaEMRen, "C++: Pythia8::SigmaProcess::alphaEMRen() const --> double");
		cl.def("alphaSRen", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::alphaSRen, "C++: Pythia8::SigmaProcess::alphaSRen() const --> double");
		cl.def("Q2Fac", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::Q2Fac, "C++: Pythia8::SigmaProcess::Q2Fac() const --> double");
		cl.def("pdf1", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::pdf1, "C++: Pythia8::SigmaProcess::pdf1() const --> double");
		cl.def("pdf2", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::pdf2, "C++: Pythia8::SigmaProcess::pdf2() const --> double");
		cl.def("thetaMPI", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::thetaMPI, "C++: Pythia8::SigmaProcess::thetaMPI() const --> double");
		cl.def("phiMPI", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::phiMPI, "C++: Pythia8::SigmaProcess::phiMPI() const --> double");
		cl.def("sHBetaMPI", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::sHBetaMPI, "C++: Pythia8::SigmaProcess::sHBetaMPI() const --> double");
		cl.def("pT2MPI", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::pT2MPI, "C++: Pythia8::SigmaProcess::pT2MPI() const --> double");
		cl.def("pTMPIFin", (double (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::pTMPIFin, "C++: Pythia8::SigmaProcess::pTMPIFin() const --> double");
		cl.def("saveKin", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::saveKin, "C++: Pythia8::SigmaProcess::saveKin() --> void");
		cl.def("loadKin", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::loadKin, "C++: Pythia8::SigmaProcess::loadKin() --> void");
		cl.def("swapKin", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::swapKin, "C++: Pythia8::SigmaProcess::swapKin() --> void");
		cl.def("setIdInDiff", (void (Pythia8::SigmaProcess::*)(int, int)) &Pythia8::SigmaProcess::setIdInDiff, "C++: Pythia8::SigmaProcess::setIdInDiff(int, int) --> void", pybind11::arg(""), pybind11::arg(""));
		cl.def("addBeamA", (void (Pythia8::SigmaProcess::*)(int)) &Pythia8::SigmaProcess::addBeamA, "C++: Pythia8::SigmaProcess::addBeamA(int) --> void", pybind11::arg("idIn"));
		cl.def("addBeamB", (void (Pythia8::SigmaProcess::*)(int)) &Pythia8::SigmaProcess::addBeamB, "C++: Pythia8::SigmaProcess::addBeamB(int) --> void", pybind11::arg("idIn"));
		cl.def("sizeBeamA", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::sizeBeamA, "C++: Pythia8::SigmaProcess::sizeBeamA() const --> int");
		cl.def("sizeBeamB", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::sizeBeamB, "C++: Pythia8::SigmaProcess::sizeBeamB() const --> int");
		cl.def("addPair", (void (Pythia8::SigmaProcess::*)(int, int)) &Pythia8::SigmaProcess::addPair, "C++: Pythia8::SigmaProcess::addPair(int, int) --> void", pybind11::arg("idAIn"), pybind11::arg("idBIn"));
		cl.def("sizePair", (int (Pythia8::SigmaProcess::*)() const) &Pythia8::SigmaProcess::sizePair, "C++: Pythia8::SigmaProcess::sizePair() const --> int");
		cl.def("setupForME", (bool (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::setupForME, "C++: Pythia8::SigmaProcess::setupForME() --> bool");
		cl.def("setupForMEin", (bool (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::setupForMEin, "C++: Pythia8::SigmaProcess::setupForMEin() --> bool");
		cl.def("setId", [](Pythia8::SigmaProcess &o) -> void { return o.setId(); }, "");
		cl.def("setId", [](Pythia8::SigmaProcess &o, int const & a0) -> void { return o.setId(a0); }, "", pybind11::arg("id1in"));
		cl.def("setId", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1) -> void { return o.setId(a0, a1); }, "", pybind11::arg("id1in"), pybind11::arg("id2in"));
		cl.def("setId", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2) -> void { return o.setId(a0, a1, a2); }, "", pybind11::arg("id1in"), pybind11::arg("id2in"), pybind11::arg("id3in"));
		cl.def("setId", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3) -> void { return o.setId(a0, a1, a2, a3); }, "", pybind11::arg("id1in"), pybind11::arg("id2in"), pybind11::arg("id3in"), pybind11::arg("id4in"));
		cl.def("setId", (void (Pythia8::SigmaProcess::*)(int, int, int, int, int)) &Pythia8::SigmaProcess::setId, "C++: Pythia8::SigmaProcess::setId(int, int, int, int, int) --> void", pybind11::arg("id1in"), pybind11::arg("id2in"), pybind11::arg("id3in"), pybind11::arg("id4in"), pybind11::arg("id5in"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o) -> void { return o.setColAcol(); }, "");
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0) -> void { return o.setColAcol(a0); }, "", pybind11::arg("col1"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1) -> void { return o.setColAcol(a0, a1); }, "", pybind11::arg("col1"), pybind11::arg("acol1"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2) -> void { return o.setColAcol(a0, a1, a2); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3) -> void { return o.setColAcol(a0, a1, a2, a3); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4) -> void { return o.setColAcol(a0, a1, a2, a3, a4); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5) -> void { return o.setColAcol(a0, a1, a2, a3, a4, a5); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"), pybind11::arg("acol3"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6) -> void { return o.setColAcol(a0, a1, a2, a3, a4, a5, a6); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"), pybind11::arg("acol3"), pybind11::arg("col4"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7) -> void { return o.setColAcol(a0, a1, a2, a3, a4, a5, a6, a7); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"), pybind11::arg("acol3"), pybind11::arg("col4"), pybind11::arg("acol4"));
		cl.def("setColAcol", [](Pythia8::SigmaProcess &o, int const & a0, int const & a1, int const & a2, int const & a3, int const & a4, int const & a5, int const & a6, int const & a7, int const & a8) -> void { return o.setColAcol(a0, a1, a2, a3, a4, a5, a6, a7, a8); }, "", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"), pybind11::arg("acol3"), pybind11::arg("col4"), pybind11::arg("acol4"), pybind11::arg("col5"));
		cl.def("setColAcol", (void (Pythia8::SigmaProcess::*)(int, int, int, int, int, int, int, int, int, int)) &Pythia8::SigmaProcess::setColAcol, "C++: Pythia8::SigmaProcess::setColAcol(int, int, int, int, int, int, int, int, int, int) --> void", pybind11::arg("col1"), pybind11::arg("acol1"), pybind11::arg("col2"), pybind11::arg("acol2"), pybind11::arg("col3"), pybind11::arg("acol3"), pybind11::arg("col4"), pybind11::arg("acol4"), pybind11::arg("col5"), pybind11::arg("acol5"));
		cl.def("swapColAcol", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::swapColAcol, "C++: Pythia8::SigmaProcess::swapColAcol() --> void");
		cl.def("swapCol1234", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::swapCol1234, "C++: Pythia8::SigmaProcess::swapCol1234() --> void");
		cl.def("swapCol12", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::swapCol12, "C++: Pythia8::SigmaProcess::swapCol12() --> void");
		cl.def("swapCol34", (void (Pythia8::SigmaProcess::*)()) &Pythia8::SigmaProcess::swapCol34, "C++: Pythia8::SigmaProcess::swapCol34() --> void");
		cl.def("weightTopDecay", (double (Pythia8::SigmaProcess::*)(class Pythia8::Event &, int, int)) &Pythia8::SigmaProcess::weightTopDecay, "C++: Pythia8::SigmaProcess::weightTopDecay(class Pythia8::Event &, int, int) --> double", pybind11::arg("process"), pybind11::arg("iResBeg"), pybind11::arg("iResEnd"));
		cl.def("weightHiggsDecay", (double (Pythia8::SigmaProcess::*)(class Pythia8::Event &, int, int)) &Pythia8::SigmaProcess::weightHiggsDecay, "C++: Pythia8::SigmaProcess::weightHiggsDecay(class Pythia8::Event &, int, int) --> double", pybind11::arg("process"), pybind11::arg("iResBeg"), pybind11::arg("iResEnd"));
		cl.def("assign", (class Pythia8::SigmaProcess & (Pythia8::SigmaProcess::*)(const class Pythia8::SigmaProcess &)) &Pythia8::SigmaProcess::operator=, "C++: Pythia8::SigmaProcess::operator=(const class Pythia8::SigmaProcess &) --> class Pythia8::SigmaProcess &", pybind11::return_value_policy::reference, pybind11::arg(""));
	}
}
