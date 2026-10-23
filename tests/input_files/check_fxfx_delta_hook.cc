// Minimal Pythia interface fixture: the base sorter retains Pythia's actual
// production-scale comparison and omits resonance decay products.
// Physical scales have deliberately lost the old pt_clust/SCALUP equality.
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace Pythia8 {
struct Particle {
  int id, status, parent, original;
  double value;
  Particle(int idIn = 0, int statusIn = 0, double scaleIn = 0., int parentIn = 0)
    : id(idIn), status(statusIn), parent(parentIn), original(0), value(scaleIn) {}
  double scale() const { return value; }
  void scale(double input) { value = input; }
  bool isFinal() const { return status == 1; }
  bool isGluon() const { return id == 21; }
  int idAbs() const { return std::abs(id); }
  int daughter1() const { return original; }
};
struct Event {
  std::vector<Particle> particles;
  int size() const { return particles.size(); }
  Particle& operator[](int i) { return particles.at(i); }
  const Particle& operator[](int i) const { return particles.at(i); }
};
struct Scales { std::map<std::string, double> attributes; };
struct Info {
  Scales* scales;
  double reference;
  double scalup() const { return reference; }
};
struct Settings { int mode(const std::string&) const { return 5; } };
class JetMatchingMadgraph {
public:
  Event eventProcessOrig, workEvent;
  Info* infoPtr;
  Settings* settingsPtr;
  std::vector<int> matched, excluded;
  virtual ~JetMatchingMadgraph() {}
  void omitResonanceDecays(const Event& input, bool) {
    workEvent.particles.clear();
    for (int i = 0; i < input.size(); ++i) {
      if (input[i].parent > 4) continue;
      Particle retained = input[i];
      retained.original = i;
      workEvent.particles.push_back(retained);
    }
  }
  virtual void sortIncomingProcess(const Event&) {
    matched.clear();
    excluded.clear();
    omitResonanceDecays(eventProcessOrig, true);
    for (int i = 0; i < workEvent.size(); ++i) {
      const Particle& parton = workEvent[i];
      if (!parton.isFinal()) continue;
      if (!parton.isGluon() && parton.idAbs() > 5) continue;
      if (std::trunc(1000. * parton.scale()) ==
          std::trunc(1000. * infoPtr->scalup())) matched.push_back(parton.original - 2);
      else excluded.push_back(parton.original - 2);
    }
  }
};
} // namespace Pythia8

#include "FxFxDeltaMatching.h"

int main() {
  using namespace Pythia8;
  Event process;
  process.particles.resize(11);
  process[3] = Particle(1, -1, 18.);
  process[4] = Particle(-1, -1, 22.);
  process[5] = Particle(21, 1, 7.);  // LHE 3: matched despite a low Delta scale.
  process[6] = Particle(23, 2, 90.); // LHE 4: inserted resonance, omitted by sorter.
  process[7] = Particle(1, 1, 22.);  // LHE 5: excluded despite equality with SCALUP.
  process[8] = Particle(21, 1, 3.);  // LHE 6: another matched parton.
  // Pythia-generated resonance decays have no flags or original LHE labels.
  process[9] = Particle(1, 1, 45., 6);
  process[10] = Particle(-1, 1, 45., 6);
  Scales scales;
  scales.attributes["fxfxdelta"] = 1.;
  scales.attributes["fxfxmatch3"] = 1.;
  scales.attributes["fxfxmatch5"] = 0.;
  scales.attributes["fxfxmatch6"] = 1.;
  scales.attributes["scalup_3_1"] = 7.;
  scales.attributes["scalup_5_2"] = 22.;
  Info info = {&scales, 22.};
  Settings settings;
  FxFxDeltaMatching hook;
  hook.eventProcessOrig = process;
  hook.infoPtr = &info;
  hook.settingsPtr = &settings;
  const std::map<std::string, double> originalAttributes = scales.attributes;
  for (int repeat = 0; repeat < 2; ++repeat) {
    hook.sortIncomingProcess(process);
    assert(hook.matched == std::vector<int>({3,6}));
    assert(hook.excluded == std::vector<int>({5}));
    for (int i = 0; i < process.size(); ++i)
      assert(hook.eventProcessOrig[i].scale() == process[i].scale());
    assert(scales.attributes == originalAttributes);
    assert(info.scalup() == 22.);
  }
  // No matched parton is still meaningful; never derive the reference from
  // the largest final-state marker, which could describe an excluded leg.
  scales.attributes["fxfxmatch3"] = 0.;
  scales.attributes["fxfxmatch6"] = 0.;
  hook.sortIncomingProcess(process);
  assert(hook.matched.empty());
  assert(hook.excluded == std::vector<int>({3,5,6}));

  // Ordinary FxFx and events without a scales block use the base sorter.
  scales.attributes.erase("fxfxdelta");
  hook.sortIncomingProcess(process);
  assert(hook.matched == std::vector<int>({5}));
  info.scales = NULL;
  hook.sortIncomingProcess(process);
  assert(hook.matched == std::vector<int>({5}));
  info.scales = &scales;

  // A malformed combined event must not silently fall back to scale labels.
  scales.attributes["fxfxdelta"] = 1.;
  scales.attributes.erase("fxfxmatch3");
  bool rejected = false;
  try { hook.sortIncomingProcess(process); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  scales.attributes["fxfxmatch3"] = 0.5;
  rejected = false;
  try { hook.sortIncomingProcess(process); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
  std::cout << "PASS FxFx Delta matching flags\n";
}
