// Exercise the stock matching hook through Pythia's actual LHE reordering.
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/CombineMatchingInput.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace Pythia8;
  try {
    if (argc != 3) throw std::runtime_error("usage: check_matching events.lhe matched");
    const int expected = std::stoi(argv[2]);
    Pythia pythia;
    const char* settings[] = {
      "Beams:frameType = 4", "Beams:setProductionScalesFromLHEF = on",
      "PartonLevel:all = off", "HadronLevel:all = off", "23:mayDecay = off",
      "JetMatching:merge = on", "JetMatching:doFxFx = on",
      "JetMatching:scheme = 1", "JetMatching:setMad = off",
      "JetMatching:nJetMax = 1", "JetMatching:nPartonsNow = 1",
      "JetMatching:qCut = 30", "JetMatching:qCutME = 20",
      "Init:showChangedSettings = off", "Init:showChangedParticleData = off",
      "Next:numberShowInfo = 0", "Next:numberShowProcess = 0",
      "Next:numberShowEvent = 0"};
    for (const char* setting : settings)
      if (!pythia.readString(setting)) throw std::runtime_error(setting);
    if (!pythia.readString(std::string("Beams:LHEF = ") + argv[1]))
      throw std::runtime_error("Invalid LHE path");
    CombineMatchingInput combined;
#if PYTHIA_VERSION_INTEGER < 8300
    UserHooks* hook = combined.getHook(pythia);
    pythia.setUserHooksPtr(hook);
    JetMatchingMadgraph* matching = dynamic_cast<JetMatchingMadgraph*>(hook);
#else
    combined.setHook(pythia);
    auto matching = std::dynamic_pointer_cast<JetMatchingMadgraph>(combined.hook);
#endif
    if (!matching || !pythia.init() || !pythia.next())
      throw std::runtime_error("Could not process FxFx event");
    int gluon = 0;
    for (int i = 5; i < pythia.process.size(); ++i)
      if (pythia.process[i].id() == 21) gluon = i;
    const Event matched = matching->getProcessSubset();
    if (!gluon || matched.size() != expected ||
        (expected && (matched[0].id() != 21 ||
         std::abs(matched[0].px() - pythia.process[gluon].px()) > 1.e-10)))
      throw std::runtime_error("Incorrect production-gluon matching classification");
    if (std::abs(pythia.info.scalup() - 60.) > 1.e-10 ||
        std::abs(pythia.process[gluon].scale() - (expected ? 60. : 59.999)) > 1.e-10)
      throw std::runtime_error("FxFx scalar scales were not retained");
    if (gluon != 6)
      throw std::runtime_error("Unexpected production-gluon process position");
    std::cout << "PASS stock FxFx matching " << expected << " gluon\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
