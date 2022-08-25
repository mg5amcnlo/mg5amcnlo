#include <Pythia8/Analysis.h>
#include <Pythia8/Basics.h>
#include <Pythia8/BeamParticle.h>
#include <Pythia8/BoseEinstein.h>
#include <Pythia8/ColourReconnection.h>
#include <Pythia8/Event.h>
#include <Pythia8/FragmentationFlavZpT.h>
#include <Pythia8/FragmentationSystems.h>
#include <Pythia8/GammaKinematics.h>
#include <Pythia8/HIUserHooks.h>
#include <Pythia8/HadronLevel.h>
#include <Pythia8/HadronWidths.h>
#include <Pythia8/HeavyIons.h>
#include <Pythia8/HelicityBasics.h>
#include <Pythia8/History.h>
#include <Pythia8/Info.h>
#include <Pythia8/LHEF3.h>
#include <Pythia8/LesHouches.h>
#include <Pythia8/Merging.h>
#include <Pythia8/MergingHooks.h>
#include <Pythia8/MultipartonInteractions.h>
#include <Pythia8/NucleonExcitations.h>
#include <Pythia8/ParticleData.h>
#include <Pythia8/ParticleDecays.h>
#include <Pythia8/PartonDistributions.h>
#include <Pythia8/PartonLevel.h>
#include <Pythia8/PartonSystems.h>
#include <Pythia8/PartonVertex.h>
#include <Pythia8/PhaseSpace.h>
#include <Pythia8/PhysicsBase.h>
#include <Pythia8/ProcessContainer.h>
#include <Pythia8/Pythia.h>
#include <Pythia8/ResonanceDecays.h>
#include <Pythia8/ResonanceWidths.h>
#include <Pythia8/Ropewalk.h>
#include <Pythia8/SLHAinterface.h>
#include <Pythia8/Settings.h>
#include <Pythia8/SharedPointers.h>
#include <Pythia8/ShowerModel.h>
#include <Pythia8/SigmaLowEnergy.h>
#include <Pythia8/SigmaProcess.h>
#include <Pythia8/SigmaTotal.h>
#include <Pythia8/SimpleSpaceShower.h>
#include <Pythia8/SimpleTimeShower.h>
#include <Pythia8/StandardModel.h>
#include <Pythia8/SusyCouplings.h>
#include <Pythia8/SusyLesHouches.h>
#include <Pythia8/UserHooks.h>
#include <Pythia8/Weights.h>
#include <array>
#include <complex>
#include <deque>
#include <functional>
#include <istream>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <sstream> // __str__
#include <string>
#include <unordered_map>
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

void bind_std_stl_vector(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// std::vector file:bits/stl_vector.h line:214

	// std::vector file:bits/stl_vector.h line:214

	// std::vector file:bits/stl_vector.h line:214

	// std::vector file:bits/stl_vector.h line:214

	// std::vector file:bits/stl_vector.h line:214

	// std::vector file:bits/stl_bvector.h line:541

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

	// std::map file:bits/stl_map.h line:96

}
