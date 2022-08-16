// NucleonExcitations.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for computing mass-dependent widths and branching ratios

#ifndef Pythia8_NucleonExcitations_H
#define Pythia8_NucleonExcitations_H

#include "Pythia8/HadronWidths.h"
#include "Pythia8/MathTools.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PhysicsBase.h"

namespace Pythia8 {

//==========================================================================

// The NucleonExcitations class is used to calculate cross sections for
// explicit nucleon excitation channels, e.g. p p -> p p(1520).

class NucleonExcitations : public PhysicsBase {

public:

  // Constructor.
  NucleonExcitations() = default;

  // Objects of this class should only be passed as references.
  NucleonExcitations(const NucleonExcitations&) = delete;
  NucleonExcitations(NucleonExcitations&&) = delete;
  NucleonExcitations& operator=(const NucleonExcitations&) = delete;
  NucleonExcitations& operator=(NucleonExcitations&&) = delete;

  // Read in excitation data from the specified file.
  bool init(string path);

  // Read in excitation data from the specified stream.
  bool init(istream& stream);

  // Validate that the loaded data makes sense.
  bool check();

  // Get all nucleon excitations from particle data.
  vector<int> getExcitationMasks() const;

  // Get masks (ids without quark content) for all implemented cross sections.
  vector<pair<int, int>> getChannels() const;

  // Get total excitation cross sections for NN at the specified energy.
  double sigmaExTotal(double eCM) const;

  // Get cross section for NN -> CD. Quark content in masks is ignored.
  double sigmaExPartial(double eCM, int maskC, int maskD) const;

  // Pick excited particles and their masses.
  bool pickExcitation(int idA, int idB, double eCM,
    int& idCOut, double& mCOut, int& idDOut, double& mDOut);

  // Calculate the total excitation cross section without using interpolation.
  double sigmaCalc(double eCM) const {
    double sig = 0.;
    for (int maskEx : getExcitationMasks())
      sig += sigmaCalc(eCM, 0002, maskEx) + sigmaCalc(eCM, 0004, maskEx);
    return sig;
  }

  // Calculate partial excitation cross section without using interpolation.
  double sigmaCalc(double eCM, int maskC, int maskD) const;

  // Regenerate parameterization for all cross sections.
  bool parameterizeAll(int precision, double threshold = 8.);

  // Write all cross section data to an xml file.
  bool save(ostream& stream) const;
  bool save(string file = "NucleonExcitations.dat") const {
    ofstream stream(file); return save(stream); }

private:

  // Struct for storing parameterized sigma for each excitation channel.
  struct ExcitationChannel {
    LinearInterpolator sigma;

    // The particle ids without quark content (e.g. 0002 for p and n).
    int maskA, maskB;

    // Scale factor used at high energies.
    double scaleFactor;
  };

  // The available excitation channels.
  vector<ExcitationChannel> excitationChannels;

  // Total excitation cross section, precalculated for efficiency.
  LinearInterpolator sigmaTotal;

  // Get total available phase space.
  double psSize(double eCM, ParticleDataEntry& prodA,
    ParticleDataEntry& prodB) const;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HadronWidths_H
