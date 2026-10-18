// Direct numerical control of the same Pythia API used by vendor/SudGen.
// argv: XML directory, alpha_s(mZ), alpha order, CMW (0/1), ISR pdfMode.
// stdin: stop start dipole_mass pdg dipole_type seed, one point per line.
#include "Pythia8/Pythia.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  Pythia8::Pythia pythia(argv[1], false);
  auto setting = [&](const std::string& value) {
    if (!pythia.readString(value)) throw std::runtime_error(value);
  };
  for (const auto& line : {
      "HardQCD:all = on", "Check:epTolErr = 0.01",
      "Merging:doPTLundMerging = on", "Merging:Process = pp>jj",
      "Merging:TMS = -1", "Merging:includeWeightInXSection = off",
      "Merging:nJetMax = 1000", "Merging:applyVeto = off",
      "Merging:useShowerPlugin = on", "Merging:nRequested = 0",
      "Merging:muRfactors = {}", "Merging:doSchemeVariation = off",
      "PartonLevel:MPI = off", "Print:quiet = on",
      "UncertaintyBands:doVariations = off",
      "Beams:setProductionScalesFromLHEF = off",
      "TimeShower:QEDshowerByQ = off", "TimeShower:QEDshowerByL = off",
      "TimeShower:QEDshowerByOther = off", "Check:abortIfVeto = on",
      "Merging:mayRemoveDecayProducts = on",
      "1:m0 = 0", "2:m0 = 0", "3:m0 = 0", "4:m0 = 0", "5:m0 = 0",
      "TimeShower:MEcorrections = off", "TimeShower:PhiPolAsym = off",
      "TimeShower:PhiPolAsymHard = off", "SpaceShower:MEcorrections = off",
      "SpaceShower:PhiPolAsym = off", "SpaceShower:PhiPolAsymHard = off",
      "SpaceShower:PhiIntAsym = off", "SpaceShower:RapidityOrder = off",
      "SpaceShower:nQuarkIn = 5", "StandardModel:alphaSnfmax = 5",
      "Random:setSeed = on",
      "Random:seed = 290311"}) setting(line);
  pythia.settings.forceParm("SpaceShower:pT0Ref", 0.);
  for (const auto& sector : {"SpaceShower:", "TimeShower:"}) {
    setting(std::string(sector) + "alphaSvalue = " + argv[2]);
    setting(std::string(sector) + "alphaSorder = " + argv[3]);
    setting(std::string(sector) + "alphaSuseCMW = "
            + (std::stoi(argv[4]) ? "on" : "off"));
  }
  setting(std::string("SpaceShower:pdfMode = ") + argv[5]);
  if (!pythia.init() || !pythia.next()) return 3;
  if (pythia.settings.flag("UncertaintyBands:doVariations")) return 4;
  Pythia8::AlphaStrong coupling;
  coupling.init(std::stod(argv[2]), std::stoi(argv[3]), 5, std::stoi(argv[4]));
  const int pdfMode = std::stoi(argv[5]);
  std::cout << std::setprecision(12);
  double stop, start, mass;
  int pdg, type, seed;
  while (std::cin >> stop >> start >> mass >> pdg >> type >> seed) {
    pythia.rndm.init(seed);
    auto begin = std::chrono::steady_clock::now();
    const double probability = pythia.mergingPtr->generateSingleSudakov(
        start, stop, mass*mass, pdg, type, 7000.*7000., 0.1);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    // These trial counts are hard-coded in Pythia 8.318's shower routines.
    const int trials = (type == 1 || type == 3) ? 500000 : 10000;
    // Independent integral of the massless forward kernels, with precisely
    // the same Pythia coupling. No A2 addition beyond an optional CMW coupling.
    auto kernel = [&](double z) {
      if (pdg != 21) return (4./3.)*(-2.*std::log(1.-z)
          +2.*std::log(z)-3.*z);
      return 6.*(-std::log(1.-z)+std::log(z)-2.*z+0.5*z*z-z*z*z/3.)
          +5.*(2.*z*z*z/3.-z*z+z);
    };
    auto rate = [&](double logk) {
      const double k = std::exp(logk), r = k/mass;
      const double eps = 2.*r/(std::sqrt(4.+r*r)+r), b = 1.-eps;
      double integral;
      if (pdfMode == 1) {
        integral = (pdg != 21) ? (8./3.)*(-std::log(eps)-.75*b*b)
            : 6.*(-std::log(eps)-b*b+b*b*b/3.-std::pow(b,4)/4.)
              +5.*(b*b/2.-2.*b*b*b/3.+std::pow(b,4)/2.);
      } else {
        integral = b > 0.5 ? kernel(b)-kernel(0.5) : 0.;
      }
      return coupling.alphaS(k*k)*integral/std::acos(-1.);
    };
    const int n = 8192;
    // prepare() also bounds the dummy radiator's production scale by mDip.
    const double upper = std::min(start,mass);
    const double step = std::log(upper/stop)/n, logStop = std::log(stop);
    double exponent = rate(logStop)+rate(std::log(upper));
    for (int i=1; i<n; ++i) exponent += (i%2 ? 4. : 2.)*rate(logStop+i*step);
    const double matchedAnalytic = std::exp(-step*exponent/3.);
    std::cout << "RESULT " << stop << ' ' << start << ' ' << mass << ' '
              << pdg << ' ' << type << ' ' << seed << ' ' << probability
              << ' ' << trials << ' ' << seconds << ' ' << matchedAnalytic
              << ' ' << coupling.alphaS(stop*stop) << ' '
              << coupling.alphaS(start*start) << std::endl;
  }
}
