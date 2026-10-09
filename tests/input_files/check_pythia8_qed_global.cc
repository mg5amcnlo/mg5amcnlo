// Smoke test against the actual PYTHIA 8.318 Simple showers. Compile with
// the patched SimpleTimeShower.cc before libpythia8 on the linker command.
// Tests the first QED emission from colourless S and H hard states and
// checks that later emissions use local recoil. No MC@NLO S/H weights are
// used here: this tests the configured shower contract, not NLO accuracy.
#include "Pythia8/Pythia.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace Pythia8;

class HardLeptons : public LHAup {
public:
  explicit HardLeptons(bool hard) : hardEvent(hard) {}
  bool setInit() override {
    setBeamA(11, 100.);
    setBeamB(-11, 100.);
    setStrategy(3);
    addProcess(1, 1., 0., 1.);
    return true;
  }
  bool setEvent(int = 0) override {
    setProcess(1, 1., 100., 1./137., 0.118);
    addParticle(11, -1, 0, 0, 0, 0, 0., 0., 100., 100.);
    addParticle(-11, -1, 0, 0, 0, 0, 0., 0., -100., 100.);
    const double e = hardEvent ? 45. : 50.;
    const double pz = hardEvent ? -5. : 0.;
    const double pt = std::sqrt(e*e-pz*pz);
    addParticle(13, 1, 1, 2, 0, 0, pt, 0., pz, e);
    addParticle(-13, 1, 1, 2, 0, 0, -pt, 0., pz, e);
    addParticle(15, 1, 1, 2, 0, 0, 0., pt, pz, e);
    addParticle(-15, 1, 1, 2, 0, 0, 0., -pt, pz, e);
    if (hardEvent) addParticle(22, 1, 1, 2, 0, 0, 0., 0., 20., 20.);
    return true;
  }
private:
  bool hardEvent;
};

class RecoilCheck : public UserHooks {
public:
  explicit RecoilCheck(bool hard) : hardEvent(hard) {}
  int firstEmissions = 0, laterEmissions = 0, failures = 0;
  double largestResidual = 0.;
  bool first = true;
  bool canVetoFSREmission() override { return true; }
  bool doVetoFSREmission(int oldSize, const Event& event, int,
                         bool inResonance = false) override {
    // A local map appends radiator, photon, one recoiler; the first
    // global map appends radiator, photon and all three spectators.
    const int expected = first && !hardEvent ? 5 : 3;
    if (event.size()-oldSize != expected || inResonance) ++failures;
    if (first) ++firstEmissions;
    else ++laterEmissions;
    first = false;
    Vec4 total;
    for (int i=0; i<event.size(); ++i)
      if (event[i].isFinal()) total += event[i].p();
    const double residual = std::abs(total.e()-200.)+std::abs(total.px())
      +std::abs(total.py())+std::abs(total.pz());
    largestResidual = std::max(largestResidual, residual);
    if (!std::isfinite(residual) || residual>2.e-7) ++failures;
    return false;
  }
private:
  bool hardEvent;
};

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  bool hard = std::atoi(argv[2]) != 0;
  Pythia pythia(argv[1], false);
  auto lha = std::make_shared<HardLeptons>(hard);
  auto check = std::make_shared<RecoilCheck>(hard);
  pythia.setLHAupPtr(lha);
  pythia.setUserHooksPtr(check);
  const char* settings[] = {
    "Beams:frameType = 5", "Beams:allowMomentumSpread = off",
    "PartonLevel:ISR = off", "PartonLevel:MPI = off",
    "HadronLevel:all = off", "Check:event = on",
    "TimeShower:QCDshower = off", "TimeShower:QEDshowerByQ = off",
    "TimeShower:QEDshowerByL = on", "TimeShower:QEDshowerByGamma = off",
    "TimeShower:MEcorrections = off", "TimeShower:MEextended = off",
    "TimeShower:globalRecoil = on", "TimeShower:globalRecoilMode = 0",
    "TimeShower:nMaxGlobalRecoil = 4", "TimeShower:nMaxGlobalBranch = 1",
    "TimeShower:nPartonsInBorn = -1", "TimeShower:pTmaxMatch = 1",
    "TimeShower:limitPTmaxGlobal = on", "TimeShower:pTminChgL = 0.001",
    "TimeShower:alphaEMorder = 0", "13:m0 = 0.", "15:m0 = 0.",
    "13:mayDecay = off", "15:mayDecay = off",
    "Random:setSeed = on", "Random:seed = 734612",
    "Next:numberShowLHA = 0",
    "Next:numberShowEvent = 0", "Next:numberShowProcess = 0",
    "Next:numberShowInfo = 0", "Init:showChangedSettings = off",
    "Init:showChangedParticleData = off", "Init:showProcesses = off"};
  for (auto setting : settings) if (!pythia.readString(setting)) return 3;
  if (!pythia.init()) return 4;
  int failedEvents = 0;
  for (int i=0; i<500; ++i) {
    check->first = true;
    if (!pythia.next()) ++failedEvents;
  }
  std::cout << "QED_GLOBAL_CHECK " << (hard ? "H" : "S")
            << " first=" << check->firstEmissions
            << " later=" << check->laterEmissions
            << " failures=" << check->failures
            << " failed_events=" << failedEvents
            << " momentum_residual=" << std::scientific << std::setprecision(6)
            << check->largestResidual << '\n';
  return check->firstEmissions>0 && check->laterEmissions>0 &&
    check->failures==0 && failedEvents==0 ? 0 : 1;
}
