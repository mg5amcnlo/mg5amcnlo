// FxFx classification for LHE events carrying MC@NLO-Delta dipole scales.
// Include after Pythia8Plugins/CombineMatchingInput.h, so this derives from
// the installed Pythia version's MadGraph matching implementation.
#ifndef MG5_FXFX_DELTA_MATCHING_H
#define MG5_FXFX_DELTA_MATCHING_H

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace Pythia8 {

class FxFxDeltaMatching : public JetMatchingMadgraph {
public:
  void sortIncomingProcess(const Event& event) override {
    // Ordinary FxFx keeps the installed hook's production-scale convention.
    if (!infoPtr->scales ||
        infoPtr->scales->attributes.find("fxfxdelta") ==
        infoPtr->scales->attributes.end()) {
      JetMatchingMadgraph::sortIncomingProcess(event);
      return;
    }
    const std::map<std::string, double>& attributes =
      infoPtr->scales->attributes;
    if (attributes.find("fxfxdelta")->second != 1.)
      throw std::runtime_error("Invalid FxFx Delta event marker");

    // Pythia's LHE reader minimizes each particle's production scale over
    // scalup_i_j and resets SCALUP. Neither is a merging label for Delta.
    // Encode the explicit flags only in matching's private process copy;
    // the physical process and its shower starting scales remain intact.
    const double reference = infoPtr->scalup();
    if (!std::isfinite(reference))
      throw std::runtime_error("Nonfinite FxFx Delta matching reference");
    std::vector<double> saved(eventProcessOrig.size());
    for (int i = 0; i < eventProcessOrig.size(); ++i)
      saved[i] = eventProcessOrig[i].scale();
    const int nQmatch = settingsPtr->mode("JetMatching:nQmatch");
    // Use the base hook's own selection. Pythia can append resonance decay
    // products that have no LHE label and must not enter FxFx matching.
    omitResonanceDecays(eventProcessOrig, true);
    for (int i = 0; i < workEvent.size(); ++i) {
      if (!workEvent[i].isFinal()) continue;
      if (!workEvent[i].isGluon() && workEvent[i].idAbs() > nQmatch) continue;
      const int original = workEvent[i].daughter1();
      if (original < 3 || original >= eventProcessOrig.size())
        throw std::runtime_error("Invalid original FxFx Delta particle index");
      // Pythia's incoming partons are 3,4; the LHE entries are 1,2.
      const std::string key = "fxfxmatch" + std::to_string(original - 2);
      const auto flag = attributes.find(key);
      if (flag == attributes.end() ||
          (flag->second != 0. && flag->second != 1.))
        throw std::runtime_error("Missing or invalid FxFx Delta flag: " + key);
      eventProcessOrig[original].scale(
        flag->second == 1. ? reference : reference + 1.);
    }
    JetMatchingMadgraph::sortIncomingProcess(event);
    for (int i = 0; i < eventProcessOrig.size(); ++i)
      eventProcessOrig[i].scale(saved[i]);
  }
};

} // namespace Pythia8

#endif
