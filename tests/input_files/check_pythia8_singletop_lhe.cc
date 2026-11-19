// Complete showers of matched stable-top LHE events with stock Pythia.
// Use the production launcher's steering file; no hooks or shower changes.
#include "Pythia8/Pythia.h"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  Pythia8::Pythia pythia(argv[1], false);
  if (!pythia.readFile(argv[2]) ||
      !pythia.readString("Beams:frameType=4") ||
      !pythia.readString(std::string("Beams:LHEF=")+argv[3])) return 3;
  if (!pythia.init()) return 4;
  int events = 0, negative = 0;
  const double ptEdges[] = {30., 60., 100., 150., 250., 400., 600.};
  const double yEdges[] = {-3., -2., -1., 0., 1., 2., 3.};
  std::array<double, 8> ptSum{}, ptSum2{}, ySum{}, ySum2{};
  while (true) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      pythia.stat();
      return 5;
    }
    if (!std::isfinite(pythia.info.weight())) return 6;
    int tops = 0;
    for (int j = 0; j < pythia.event.size(); ++j) {
      const auto& particle = pythia.event[j];
      if (particle.idAbs() != 6 || !particle.isFinal()) continue;
      ++tops;
      if (std::abs(particle.m()-pythia.particleData.m0(6)) > 1.e-6) return 7;
      int ptBin = 0, yBin = 0;
      while (ptBin < 7 && particle.pT() >= ptEdges[ptBin]) ++ptBin;
      while (yBin < 7 && particle.y() >= yEdges[yBin]) ++yBin;
      double weight = pythia.info.weight();
      ptSum[ptBin] += weight; ptSum2[ptBin] += weight*weight;
      ySum[yBin] += weight; ySum2[yBin] += weight*weight;
    }
    if (tops != 1) return 8;
    ++events;
    if (pythia.info.weight() < 0.) ++negative;
  }
  pythia.stat();
  std::cout << "LOCAL_MATCHED_LHE events=" << events << " negative=" << negative
            << " stable_top=passed\n";
  std::cout << std::setprecision(17);
  for (int j = 0; j < 8; ++j) {
    std::cout << "LOCAL_TOP_BIN pt " << j << ' ' << ptSum[j] << ' ' << ptSum2[j] << '\n';
    std::cout << "LOCAL_TOP_BIN y " << j << ' ' << ySum[j] << ' ' << ySum2[j] << '\n';
  }
  return events == 0 ? 9 : 0;
}
