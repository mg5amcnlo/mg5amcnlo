// Exercise the actual PYTHIA massive-emitter kernels with MECs disabled.
// The companion Python runner instruments an isolated SimpleTimeShower.cc.
// This checks shower kernels and emission routing, not complete NLO matching.
#include "Pythia8/Pythia.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

using namespace Pythia8;
extern "C" int mg5_massive_probe_report(int pdg);

class MassivePair : public LHAup {
public:
  explicit MassivePair(int id) : pdg(id) {}
  bool setInit() override {
    setBeamA(11, 500.);
    setBeamB(-11, 500.);
    setStrategy(3);
    addProcess(1, 1., 0., 1.);
    return true;
  }
  bool setEvent(int = 0) override {
    setProcess(1, 1., 500., 1./137., 0.118);
    addParticle(11, -1, 0, 0, 0, 0, 0., 0., 500., 500.);
    addParticle(-11, -1, 0, 0, 0, 0, 0., 0., -500., 500.);
    const double p = std::sqrt(500.*500.-100.*100.);
    const int col = pdg == 24 ? 0 : 501;
    const int acol = pdg == 1000021 ? 502 : 0;
    addParticle(pdg, 1, 1, 2, col, acol, p, 0., 0., 500., 100.);
    addParticle(pdg == 1000021 ? pdg : -pdg, 1, 1, 2, acol, col,
                -p, 0., 0., 500., 100.);
    return true;
  }
private:
  int pdg;
};

class EmissionCheck : public UserHooks {
public:
  explicit EmissionCheck(int id) : pdg(id) {}
  int qcd = 0, qed = 0;
  bool canVetoFSREmission() override { return true; }
  bool doVetoFSREmission(int oldSize, const Event& event, int,
                         bool = false) override {
    if (event[oldSize].idAbs() == pdg) {
      if (event[oldSize+1].id() == 21) ++qcd;
      if (event[oldSize+1].id() == 22) ++qed;
    }
    return false;
  }
private:
  int pdg;
};

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int pdg = std::atoi(argv[2]);
  if (pdg != 24 && pdg != 1000002 && pdg != 1000021) return 2;
  Pythia pythia(argv[1], false);
  pythia.setLHAupPtr(std::make_shared<MassivePair>(pdg));
  auto check = std::make_shared<EmissionCheck>(pdg);
  pythia.setUserHooksPtr(check);
  const char* settings[] = {
    "Beams:frameType = 5", "PartonLevel:ISR = off",
    "PartonLevel:MPI = off", "HadronLevel:all = off", "Check:event = on",
    "TimeShower:QCDshower = on", "TimeShower:QEDshowerByQ = off",
    "TimeShower:QEDshowerByL = off", "TimeShower:QEDshowerByOther = on",
    "TimeShower:QEDshowerByGamma = off", "TimeShower:alphaEMorder = 0",
    "TimeShower:MEcorrections = off", "TimeShower:MEextended = off",
    "TimeShower:globalRecoil = on", "TimeShower:nMaxGlobalBranch = 1",
    "TimeShower:pTmaxMatch = 1", "TimeShower:limitPTmaxGlobal = on",
    "Random:setSeed = on", "Random:seed = 734612",
    "Next:numberShowLHA = 0", "Next:numberShowEvent = 0",
    "Next:numberShowProcess = 0", "Next:numberShowInfo = 0",
    "Init:showChangedSettings = off", "Init:showChangedParticleData = off",
    "Init:showProcesses = off"};
  for (auto setting : settings) if (!pythia.readString(setting)) return 3;
  // Colourless QED uses the documented patched mode-0 setup; QCD keeps mode 2.
  for (const std::string setting : {
      "TimeShower:globalRecoilMode = " + std::to_string(pdg == 24 ? 0 : 2),
      "TimeShower:nMaxGlobalRecoil = " + std::to_string(pdg == 24 ? 2 : 1),
      "TimeShower:nPartonsInBorn = " + std::to_string(pdg == 24 ? -1 : 2),
      std::to_string(pdg)+":m0 = 100.",
      std::to_string(pdg)+":mWidth = 0.",
      std::to_string(pdg)+":mayDecay = off"})
    if (!pythia.readString(setting)) return 3;
  if (!pythia.init()) return 4;
  int failed = 0;
  for (int i=0; i<500; ++i) if (!pythia.next()) ++failed;
  const int kernelFailure = mg5_massive_probe_report(pdg);
  std::cout << "MASSIVE_EMISSIONS pdg=" << pdg << " qcd=" << check->qcd
            << " qed=" << check->qed << " failed_events=" << failed << '\n';
  return kernelFailure || failed || (pdg != 24 && check->qcd == 0)
    || (pdg != 1000021 && check->qed == 0);
}
