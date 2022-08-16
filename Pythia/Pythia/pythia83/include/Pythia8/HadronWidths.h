// HadronWidths.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for computing mass-dependent widths and branching ratios

#ifndef Pythia8_HadronWidths_H
#define Pythia8_HadronWidths_H

#include "Pythia8/MathTools.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/PhysicsBase.h"

namespace Pythia8 {

//==========================================================================

// The HadronWidths class is used to compute mass-dependent widths
// and branching ratios of hadrons.

class HadronWidths : public PhysicsBase {

public:

  // Load hadron widths data from an xml file.
  bool init(string path);
  bool init(istream& stream);

  // Check whether input data is valid and matches particle data.
  bool check();

  // Get a list of all implemented resonances.
  vector<int> getResonances() const;

  // Get whether the specified incoming particles can form a resonance.
  bool hasResonances(int idA, int idB) const;

  // Get resonances that can be formed by the specified incoming particles.
  vector<int> possibleResonances(int idA, int idB) const;

  // Returns whether the specified particle is handled by HadronWidths.
  bool hasData(int id) const {
    auto iter = entries.find(abs(id));
    return iter != entries.end();
  }

  // Get whether the resonance can decay into the specified products.
  bool canDecay(int id, int prodA, int prodB) const;

  // Get the total width of the specified particle at the specified mass.
  double width(int id, double m) const;

  // Get the partial width for the specified decay channel of the particle.
  double partialWidth(int idR, int prodA, int prodB, double m) const;

  // Get the branching ratio for the specified decay channel of the particle.
  double br(int idR, int prodA, int prodB, double m) const;

  // Get the mass distribution density for the particle at the specified mass.
  double mDistr(int id, double m) const;

  // Sample masses for the outgoing system with a given eCM.
  bool pickMasses(int idA, int idB, double eCM,
    double& mAOut, double& mBOut, int lType = 1);

  // Pick a decay channel for the specified particle, together with phase
  // space configuration. Returns whether successful.
  bool pickDecay(int idDec, double m, int& idAOut, int& idBOut,
    double& mAOut, double& mBOut);

  // Calculate the total width of the particle without using interpolation.
  double widthCalc(int id, double m) const;

  // Calculate partial width of the particle without using interpolation.
  double widthCalc(int id, int prodA, int prodB, double m) const;

  // Regenerate parameterization for the specified particle.
  bool parameterize(int id, int precision);

  // Regenerate parameterization for all particles.
  void parameterizeAll(int precision);

  // Write all width data to an xml file.
  bool save(ostream& stream) const;
  bool save(string file = "HadronWidths.dat") const {
    ofstream stream(file); return save(stream); }

private:

  // Struct for mass dependent partial width and other decay channel info.
  struct ResonanceDecayChannel {
    LinearInterpolator partialWidth;
    int prodA, prodB;

    // 2l+1, where l is the angular momentum of the outgoing two-body system.
    int lType;

    // Minimum mass for this channel.
    double mThreshold;
  };

  // Structure for total width parameterization and map to decay channels.
  struct HadronWidthEntry {
    LinearInterpolator width;
    map<pair<int, int>, ResonanceDecayChannel> decayChannels;
    bool isUserDefined;
  };

  // Map from particle id to corresponding HadronWidthEntry.
  map<int, HadronWidthEntry> entries;

  // Gets key for the decay and flips idR if necessary
  pair<int, int> getKey(int& idR, int idA, int idB) const;

  // Map from signatures to candidate resonances. Used for optimization.
  map<int, vector<int>> signatureToParticles;

  // Get signature of system based on total baryon number and electric charge.
  int getSignature(int baryonNumber, int charge) const;

  // Get total available phase space.
  double psSize(double eCM, ParticleDataEntryPtr prodA,
    ParticleDataEntryPtr prodB, double lType) const;

  // Calculate partial width of the particle without using interpolation.
  double widthCalc(int id, DecayChannel& channel, double m) const;

  // Generate parameterization for particle and its decay products if needed.
  bool parameterizeRecursive(int id, int precision);

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HadronWidths_H
