// Exercise the real SudGen bridge against an installed Pythia library.
#include "Pythia8/Pythia.h"
#include <iomanip>
#include <iostream>

extern "C" {
  extern Pythia8::Pythia pythia;
  void pythia_init_(double*);
  void pythia_get_no_emission_prob_(double&, double&, double&, double&,
                                   int&, int&, int&, double&);
  void pythia_get_no_emission_prob_x_(double&, double&, double&, double&,
                                     int&, int&, int&, double&, double&);
}

int main() {
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 290311");
  // The bridge must override variations, including automatic merging ones.
  pythia.readString("UncertaintyBands:doVariations = on");
  if (pythia.settings.isPVec("Merging:muRfactors"))
    pythia.readString("Merging:muRfactors = {0.5, 2.0}");
  double masses[21] = {.33, .33, .5, 1.5, 4.8, 0, 0, 0, 0, 0,
                      .00051099893, 0, .10565837, 0, 1.77682,
                      0, 0, 0, 0, 0, 0};
  pythia_init_(masses);
  std::cout << "VARIATIONS "
            << pythia.settings.flag("UncertaintyBands:doVariations") << ' '
            << (pythia.settings.isPVec("Merging:muRfactors")
                ? pythia.settings.pvec("Merging:muRfactors").size() : 0)
            << ' ' << (pythia.settings.isFlag("Merging:doSchemeVariation")
                ? pythia.settings.flag("Merging:doSchemeVariation") : false)
            << '\n' << std::setprecision(12);
  double start = 80.419002445756163, mass = start, cutoff = 0., x = .1;
  for (int type : {1, 2}) {
    for (double stop : {20., 23., start}) {
      double probability = -1.;
      int pdg = 2, seed = 310117;
      pythia_get_no_emission_prob_(probability, start, stop, mass,
                                   pdg, type, seed, cutoff);
      std::cout << "PROBABILITY " << type << ' ' << stop << ' '
                << probability << '\n';
      if (type == 1 && stop == 20.) {
        double explicit_x = -1.;
        pythia_get_no_emission_prob_x_(explicit_x, start, stop, mass,
                                       pdg, type, seed, cutoff, x);
        std::cout << "EXPLICIT_X " << explicit_x << '\n';
      }
    }
  }
}
