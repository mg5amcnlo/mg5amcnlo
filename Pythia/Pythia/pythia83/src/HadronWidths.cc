// HadronWidths.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the HadronWidths class.

#include "Pythia8/HadronWidths.h"

namespace Pythia8 {

//--------------------------------------------------------------------------

// Static methods for reading xml files.

static string attributeValue(string line, string attribute) {
  if (line.find(attribute) == string::npos) return "";
  int iBegAttri = line.find(attribute);
  int iBegQuote = line.find("\"", iBegAttri + 1);
  int iEndQuote = line.find("\"", iBegQuote + 1);
  return line.substr(iBegQuote + 1, iEndQuote - iBegQuote - 1);
}

static int intAttributeValue(string line, string attribute) {
  string valString = attributeValue(line, attribute);
  if (valString == "") return 0;
  istringstream valStream(valString);
  int intVal;
  valStream >> intVal;
  return intVal;
}

static double doubleAttributeValue(string line, string attribute) {
  string valString = attributeValue(line, attribute);
  if (valString == "") return 0.;
  istringstream valStream(valString);
  double doubleVal;
  valStream >> doubleVal;
  return doubleVal;
}

static void completeTag(istream& stream, string& line) {
  while (line.find(">") == string::npos) {
    string addLine;
    if (!getline(stream, addLine)) break;
    line += " " + addLine;
  }
}

//==========================================================================

// The HadronWidths class.

//--------------------------------------------------------------------------

// Initialize.

bool HadronWidths::init(string path) {

  ifstream stream(path);
  if (!stream.is_open()) {
    infoPtr->errorMsg( "Error in HadronWidths::init: "
        "unable to open file");
    return false;
  }

  return init(stream);
}

//--------------------------------------------------------------------------

// Initialize.

bool HadronWidths::init(istream& stream) {

  string line;

  while (getline(stream, line)) {

    string word1;
    if (!(istringstream(line) >> word1))
      continue;

    if (word1 == "<width") {
      completeTag(stream, line);

      int id = intAttributeValue(line, "id");
      auto entryIter = entries.find(id);
      if (entryIter != entries.end() && entryIter->second.isUserDefined) {
        infoPtr->errorMsg( "Error in HadronWidths::init: "
          "resonance is defined more than once",
          std::to_string(id));
        continue;
      }

      double left  = doubleAttributeValue(line, "left");
      double right = doubleAttributeValue(line, "right");

      istringstream dataStr(attributeValue(line, "data"));
      vector<double> data;
      double currentData;
      while (dataStr >> currentData)
        data.push_back(currentData);

      // Insert resonance in entries
      LinearInterpolator widths(left, right, data);
      entries.emplace(id, HadronWidthEntry{ widths, {}, false });

      // Insert resonance in signature map
      int signature = getSignature(particleDataPtr->isBaryon(id),
                                   particleDataPtr->chargeType(id));

      auto iter = signatureToParticles.find(signature);
      if (iter == signatureToParticles.end())
        // If signature has not been used yet, insert a new vector into the map
        signatureToParticles.emplace(signature, vector<int> { id });
      else
        // If signature has been used already, add id to the existing vector
        iter->second.push_back(id);
    }
    else if (word1 == "<partialWidth") {
      completeTag(stream, line);

      int id = intAttributeValue(line, "id");

      auto entryIter = entries.find(id);
      if (entryIter == entries.end()) {
        infoPtr->errorMsg( "Error in HadronWidths::readXML: "
          "got partial width for a particle with undefined total width",
          std::to_string(id));
        continue;
      }

      int lType = intAttributeValue(line, "lType");

      istringstream productStr(attributeValue(line, "products"));
      int prod1, prod2;
      productStr >> prod1;
      productStr >> prod2;

      istringstream dataStr(attributeValue(line, "data"));
      vector<double> data;
      double currentData;
      while (dataStr >> currentData)
        data.push_back(currentData);

      HadronWidthEntry& entry = entryIter->second;
      LinearInterpolator widths(entry.width.left(), entry.width.right(), data);

      // Generate key to ensure canonical ordering of decay products.
      pair<int, int> key = getKey(id, prod1, prod2);
      double mThreshold = particleDataPtr->mMin(key.first)
                        + particleDataPtr->mMin(key.second);

      // Insert new decay channel.
      entry.decayChannels.emplace(key, ResonanceDecayChannel {
        widths, key.first, key.second, lType, mThreshold });
    }
  }

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Check whether input data is valid and matches particle data.

bool HadronWidths::check() {

  // Check that all resonance entries make sense.
  for (auto& entryPair : entries) {
    int id = entryPair.first;
    HadronWidthEntry& entry = entryPair.second;

    // Check that entry id actually corresponds to a particle.
    if (!particleDataPtr->isParticle(id)) {
      infoPtr->errorMsg("Error in HadronWidths::check: "
        "resonance is not a particle", std::to_string(id));
      return false;
    }

    // Check that entry id is positive (antiparticles are handled by symmetry).
    if (id < 0) {
      infoPtr->errorMsg("Error in HadronWidths::check: "
        "resonance is an antiparticle", std::to_string(id));
      return false;
    }

    // Check that entry id is hadron.
    if (!particleDataPtr->isHadron(id)) {
      infoPtr->errorMsg("Error in HadronWidths::check: "
        "resonance is not a hadron", std::to_string(id));
      return false;
    }

    // Check that mass boundaries are same in particle entry and widths entry.
    if (particleDataPtr->mMin(id) < entry.width.left()) {
      infoPtr->errorMsg("Warning in HadronWidths::check: "
        "inconsistent lower mass bound", std::to_string(id));
    }
    if (particleDataPtr->mMax(id) > entry.width.right()) {
      infoPtr->errorMsg("Warning in HadronWidths::check: "
        "inconsistent upper mass bound", std::to_string(id));
    }

    // Check that all decay channels make sense.
    for (auto channelPair : entry.decayChannels) {
      ResonanceDecayChannel& channel = channelPair.second;
      int idA = channel.prodA, idB = channel.prodB;
      string channelStr = std::to_string(id) + " --> "
          + std::to_string(idA) + " + " + std::to_string(idB);

      // Check that decay product ids actually correspond to particles.
      for (int idProd : { idA, idB })
      if (!particleDataPtr->isParticle(idProd)) {
        infoPtr->errorMsg("Error in HadronWidths::check: "
          "decay product is not a particle", std::to_string(idProd));
        return false;
      }
      // Check that lType makes sense.
      if (channel.lType <= 0) {
        infoPtr->errorMsg("Error in HadronWidths::check: "
          "decay channel does not specify a valid lType", channelStr);
        return false;
      }

      // Check that decay conserves charge.
      if (particleDataPtr->chargeType(idA) + particleDataPtr->chargeType(idB)
        != particleDataPtr->chargeType(id)) {
        infoPtr->errorMsg("Error in HadronWidths::check: "
          "decay does not conserve charge", channelStr);
        return false;
      }
    }
  }

  for (auto& entry : *particleDataPtr) {
    if (entry.second->varWidth() && !hasData(entry.first)) {
      infoPtr->errorMsg("Warning in HadronWidths::check: "
        "particle uses mass dependent width, but width is not defined",
        std::to_string(entry.first));
    }
  }

  return true;
}

//--------------------------------------------------------------------------

// Gets key for the decay and flips idR if necessary

pair<int, int> HadronWidths::getKey(int& idR, int idA, int idB) const {

  if (idR < 0) {
    idR = -idR;
    idA = particleDataPtr->antiId(idA);
    idB = particleDataPtr->antiId(idB);
  }

  if (abs(idA) < abs(idB))
    return { idB, idA };
  else
    return { idA, idB };
}

//--------------------------------------------------------------------------

// Get signature of system based on total baryon number and electric charge.

int HadronWidths::getSignature(int baryonNumber, int charge) const {
  return 100 * baryonNumber
       +  10 * ((charge >= 0) ? charge : (10 + charge));
}

//--------------------------------------------------------------------------

// Get a list of all implemented resonances.

vector<int> HadronWidths::getResonances() const {
  vector<int> resonances;
  for (auto& p : entries)
    resonances.push_back(p.first);
  return resonances;
}
//--------------------------------------------------------------------------

// Get whether the specified incoming particles can form a resonance.

bool HadronWidths::hasResonances(int idA, int idB) const {

  ParticleDataEntryPtr entryA = particleDataPtr->findParticle(idA);
  ParticleDataEntryPtr entryB = particleDataPtr->findParticle(idB);
  if (!entryA || !entryB) {
    infoPtr->errorMsg("Error in HadronWidths::possibleResonances: "
      "invalid input particle ids");
    return false;
  }

  // Get signature for system and look only for resonances that matches it.
  int baryonNumber = entryA->isBaryon() + entryB->isBaryon();
  int charge = entryA->chargeType(idA) + entryB->chargeType(idB);
  int signature = getSignature(baryonNumber, charge);
  auto iter = signatureToParticles.find(signature);
  if (iter == signatureToParticles.end())
    return false;

  // For resonances that matches signature, check that decay channel exists.
  for (int res : iter->second)
    if (canDecay(res, idA, idB))
      return true;

  // No resonances found.
  return false;
}

//--------------------------------------------------------------------------

// Get resonances that can be formed by the specified incoming particles.

vector<int> HadronWidths::possibleResonances(int idA, int idB) const {

  vector<int> resonances;
  ParticleDataEntryPtr entryA = particleDataPtr->findParticle(idA);
  ParticleDataEntryPtr entryB = particleDataPtr->findParticle(idB);
  if (!entryA || !entryB) {
    infoPtr->errorMsg("Error in HadronWidths::possibleResonances: "
      "invalid input particle ids");
    return resonances;
  }

  // Get signature for system and look only for resonances that matches it.
  int baryonNumber = entryA->isBaryon() + entryB->isBaryon();
  int charge = entryA->chargeType(idA) + entryB->chargeType(idB);
  int signature = getSignature(baryonNumber, charge);
  auto iter = signatureToParticles.find(signature);
  if (iter == signatureToParticles.end())
    return vector<int>();

  // For resonances that matches signature, check that decay channel exists.
  for (int res : iter->second)
    if (canDecay(res, idA, idB))
      resonances.push_back(res);

  // For pi0pi0 and pi+pi-, add f0(500) explicitly.
  if ( (idA == 111 && idB == 111)
    || (abs(idA) == 211 && abs(idB) == 211 && idA * idB < 0) )
    resonances.push_back(9000221);

  // Done.
  return resonances;
}

//--------------------------------------------------------------------------

// Get whether the resonance can decay into the specified products.

bool HadronWidths::canDecay(int idR, int idA, int idB) const {

  auto entryIter = entries.find(idR);
  if (entryIter == entries.end())
    return false;

  pair<int, int> key = getKey(idR, idA, idB);
  auto channelIter = entryIter->second.decayChannels.find(key);
  return channelIter != entryIter->second.decayChannels.end();
}

//--------------------------------------------------------------------------

// Get the total width of the specified particle at the specified mass.

double HadronWidths::width(int id, double m) const {
  auto iter = entries.find(abs(id));
  return (iter != entries.end()) ? iter->second.width(m)
       : particleDataPtr->mWidth(id);
}

//--------------------------------------------------------------------------

// Get the partial width for the specified decay channel of the particle.

double HadronWidths::partialWidth(int idR, int idA, int idB, double m) const {

  auto entryIter = entries.find(idR);
  if (entryIter == entries.end())
    return 0.;

  pair<int, int> key = getKey(idR, idA, idB);
  auto channelIter = entryIter->second.decayChannels.find(key);
  if (channelIter == entryIter->second.decayChannels.end())
    return 0.;

  return (m <= channelIter->second.mThreshold) ? 0.
        : channelIter->second.partialWidth(m);
}

//--------------------------------------------------------------------------

// Get the branching ratio for the specified decay channel of the particle.

double HadronWidths::br(int idR, int idA, int idB, double m) const {

  auto entryIter = entries.find(idR);
  if (entryIter == entries.end())
    return 0.;

  pair<int, int> key = getKey(idR, idA, idB);
  auto channelIter = entryIter->second.decayChannels.find(key);
  if (channelIter == entryIter->second.decayChannels.end())
    return 0.;

  double widthNow = entryIter->second.width(m);
  if (widthNow == 0.)
    return 0.;
  else
    return (m <= channelIter->second.mThreshold) ? 0.
          : channelIter->second.partialWidth(m) / widthNow;
}

//--------------------------------------------------------------------------

// Get the mass distribution density for the particle at the specified mass.

double HadronWidths::mDistr(int id, double m) const  {
  auto iter = entries.find(abs(id));
  double w = (iter == entries.end()) ? particleDataPtr->mWidth(id)
           : iter->second.width(m);
  double m0 = particleDataPtr->m0(id);
  return 0.5 / M_PI * w / (pow2(m - m0) + 0.25 * w * w);
}

//--------------------------------------------------------------------------

// Pick a decay channel for the specified particle, together with phase
// space configuration. Returns whether successful.

bool HadronWidths::pickDecay(int idDec, double m, int& idAOut, int& idBOut,
    double& mAOut, double& mBOut) {

  // Find table entry for decaying particle.
  bool isAnti = (idDec < 0);
  if (isAnti) idDec = -idDec;
  auto entriesIter = entries.find(idDec);
  if (entriesIter == entries.end()) {
    infoPtr->errorMsg("Error in HadronWidths::pickDecay: "
      "particle not found", std::to_string(idDec));
    return false;
  }
  HadronWidthEntry& entry = entriesIter->second;

  // Pick decay channel.
  vector<pair<int, int>> prodsList;
  vector<double> sigmas;
  bool gotAny = false;
  for (auto& channel : entry.decayChannels) {
    if (m <= channel.second.mThreshold)
      continue;
    double sigma = channel.second.partialWidth(m);
    if (sigma > 0.) {
      gotAny = true;
      prodsList.push_back(channel.first);
      sigmas.push_back(sigma);
    }
  }
  if (!gotAny) {
     infoPtr->errorMsg("Error in HadronWidths::pickDecay: "
       "no channels have positive widths",
       "for " + to_string(idDec) + " @ " + to_string(m) + " GeV");
    return false;
  }

  // Select decay products. Check spin type of decay.
  pair<int, int> prods = prodsList[rndmPtr->pick(sigmas)];
  int idA = prods.first;
  int idB = prods.second;
  int lType = entry.decayChannels.at(prods).lType;

  // Select masses of decay products.
  double mA, mB;
  if (!pickMasses(idA, idB, m, mA, mB, lType)) {
    infoPtr->errorMsg("Error in HadronWidths::pickDecay: failed to pick "
      "masses", "for " + to_string(idDec) + " --> " + to_string(idA)
      + " + " + to_string(idB) + " @ " + to_string(m));
    return false;
  }

  // Done.
  idAOut = isAnti ? particleDataPtr->antiId(idA) : idA;
  idBOut = isAnti ? particleDataPtr->antiId(idB) : idB;
  mAOut = mA;
  mBOut = mB;
  return true;

}

//--------------------------------------------------------------------------

// Constants used when sampling masses.
static constexpr int    MAXLOOP        = 100;
static constexpr double MINWIDTH       = 0.001;
static constexpr double MAXWIDTHGROWTH = 2.;

//--------------------------------------------------------------------------

// Pick a pair of masses given pair invariant mass and angular momentum.

bool HadronWidths::pickMasses(int idA, int idB, double eCM,
  double& mAOut, double& mBOut, int lType) {

  // Minimal masses must be a possible choice.
  double mAMin = particleDataPtr->mMin(idA);
  double mBMin = particleDataPtr->mMin(idB);
  if (mAMin + mBMin >=  eCM) {
    infoPtr->errorMsg("Error in HadronWidths::pickMasses: "
      "energy is smaller than minimum masses");
    return false;
  }

  if (lType <= 0) {
    infoPtr->errorMsg("Error in HadronWidths::pickMasses: "
      "invalid angular momentum", "2l+1 = " + to_string(lType));
    return false;
  }

  // Done if none of the daughters have a width.
  double mAFix      = particleDataPtr->m0(idA);
  double gammaAFix  = particleDataPtr->mWidth(idA);
  bool hasFixWidthA = (gammaAFix > MINWIDTH);
  double mBFix      = particleDataPtr->m0(idB);
  double gammaBFix  = particleDataPtr->mWidth(idB);
  bool hasFixWidthB = (gammaBFix > MINWIDTH);
  mAOut             = mAFix;
  mBOut             = mBFix;
  if (!hasFixWidthA && !hasFixWidthB) return true;

  // Get width entries for particles with mass-depedent widths.
  bool hasVarWidthA = hasData(idA) && particleDataPtr->varWidth(idA);
  bool hasWidthA    = hasFixWidthA || hasVarWidthA;
  HadronWidthEntry* entryA = nullptr;
  if (hasVarWidthA) {
    auto iterA = entries.find( abs(idA) );
    if (iterA == entries.end()) {
      infoPtr->errorMsg("Error in HadronWidths::pickMasses: "
        "mass distribution for particle is not defined", std::to_string(idA));
      return false;
    }
    entryA = &iterA->second;
  }
  bool hasVarWidthB = hasData(idB) && particleDataPtr->varWidth(idB);
  bool hasWidthB    = hasFixWidthB || hasVarWidthB;
  HadronWidthEntry* entryB = nullptr;
  if (hasVarWidthB) {
    auto iterB = entries.find( abs(idB) );
    if (iterB == entries.end()) {
      infoPtr->errorMsg("Error in HadronWidths::pickMasses: "
        "mass distribution for particle is not defined", std::to_string(idB));
      return false;
    }
    entryB = &iterB->second;
  }

  // Parameters for mass selection.
  double mAMax  = min(particleDataPtr->mMax(idA), eCM - mBMin);
  if (hasVarWidthA) gammaAFix = entryA->width(mAFix);
  double bwAMin = (hasWidthA) ? atan(2. * (mAMin - mAFix) / gammaAFix) : 0.;
  double bwAMax = (hasWidthA) ? atan(2. * (mAMax - mAFix) / gammaAFix) : 0.;
  double mBMax  = min(particleDataPtr->mMax(idB), eCM - mAMin);
  if (hasVarWidthB) gammaBFix = entryB->width(mBFix);
  double bwBMin = (hasWidthB) ? atan(2. * (mBMin - mBFix) / gammaBFix) : 0.;
  double bwBMax = (hasWidthB) ? atan(2. * (mBMax - mBFix) / gammaBFix) : 0.;
  double p2Max  = (eCM*eCM - pow2(mAMin + mBMin))
                * (eCM*eCM - pow2(mAMin - mBMin));

  // Loop over attempts to pick the two masses simultaneously.
  double wtTot, gammaAVar, gammaBVar, bwAFix, bwBFix, bwAVar, bwBVar;
  for (int i = 0; i < MAXLOOP; ++i) {
    wtTot = 1.;

    // Simplify handling if full procedure does not seem to work.
    if (2 * i > MAXLOOP) {
      hasVarWidthA = false;
      hasVarWidthB = false;
    }
    if (4 * i > 3 * MAXLOOP) lType = 0;

    // Initially pick according to simple Breit-Wigner.
    if (hasWidthA) mAOut = mAFix + 0.5 * gammaAFix * tan(bwAMin
      + rndmPtr->flat() * (bwAMax - bwAMin));
    if (hasWidthB) mBOut = mBFix + 0.5 * gammaBFix * tan(bwBMin
      + rndmPtr->flat() * (bwBMax - bwBMin));

    // Correction given by BW(Gamma_now)/BW(Gamma_fix) for variable width.
    // Note: width not allowed to explode at large masses.
    if (hasVarWidthA) {
      gammaAVar = min(entryA->width(mAOut), MAXWIDTHGROWTH * gammaAFix);
      bwAVar    = gammaAVar / (pow2( mAOut - mAFix) + 0.25 * pow2(gammaAVar));
      bwAFix    = gammaAFix / (pow2( mAOut - mAFix) + 0.25 * pow2(gammaAFix));
      wtTot    *= bwAVar / (bwAFix * MAXWIDTHGROWTH);
    }
    if (hasVarWidthB) {
      gammaBVar = min(entryB->width(mBOut), MAXWIDTHGROWTH * gammaBFix);
      bwBVar    = gammaBVar / (pow2( mBOut - mBFix) + 0.25 * pow2(gammaBVar));
      bwBFix    = gammaBFix / (pow2( mBOut - mBFix) + 0.25 * pow2(gammaBFix));
      wtTot    *= bwBVar / (bwBFix * MAXWIDTHGROWTH);
    }

    // Weight by (p/p_max)^lType.
    if (mAOut + mBOut >= eCM) continue;
    double p2Ratio = (eCM*eCM - pow2(mAOut + mBOut))
                   * (eCM*eCM - pow2(mAOut - mBOut)) / p2Max;
    if (lType > 0) wtTot *= pow(p2Ratio, 0.5 * lType);
    if (wtTot > rndmPtr->flat()) {
      // Give warning message only for more severe cases
      if (4 * i > 3 * MAXLOOP) infoPtr->errorMsg("Warning in HadronWidths::"
        "pickMasses: angular momentum and running widths not used");
      return true;
    }
  }

  // Last resort: pick masses within limits known to work, without weight.
  infoPtr->errorMsg("Warning in HadronWidths::pickMasses: "
   "using last-resort simplified description");
  double mSpanNorm = (eCM - mAMin - mBMin) / (gammaAFix + gammaBFix);
  mAOut = mAMin + rndmPtr->flat() * mSpanNorm * gammaAFix;
  mBOut = mBMin + rndmPtr->flat() * mSpanNorm * gammaBFix;

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Calculate the total width of the particle without using interpolation.

double HadronWidths::widthCalc(int id, double m) const {

  // Get particle entry.
  ParticleDataEntryPtr entry = particleDataPtr->findParticle(id);
  if (entry == nullptr) {
    infoPtr->errorMsg("Error in HadronWidths::widthCalc: "
      "particle not found", to_string(id));
    return 0.;
  }

  // Sum contributions from all channels.
  double w = 0.;
  for (int iChan = 0; iChan < entry->sizeChannels(); ++iChan)
    w += widthCalc(id, entry->channel(iChan), m);
  return w;
}

//--------------------------------------------------------------------------

// Calculate partial width of the particle without using interpolation.

double HadronWidths::widthCalc(int id, int prodA, int prodB, double m) const {

  // Find particle entry.
  pair<int, int> key = getKey(id, prodA, prodB);
  ParticleDataEntryPtr entry = particleDataPtr->findParticle(id);
  if (entry == nullptr)
    return 0.;

  // Search for the matching decay channel.
  for (int iChan = 0; iChan < entry->sizeChannels(); ++iChan) {
    DecayChannel& channel = entry->channel(iChan);
    if (channel.multiplicity() > 2)
      continue;
    if ( (channel.product(0) == key.first && channel.product(1) == key.second)
      || (channel.product(1) == key.first && channel.product(0) == key.second))
      return widthCalc(id, channel, m);
  }

  // Decay channel not found.
  infoPtr->errorMsg("Error in HadronWidths::widthCalc: "
    "decay channel not found",
    to_string(id) + " --> " + to_string(prodA) + " " + to_string(prodB));
  return 0.;
}

//--------------------------------------------------------------------------

// Calculate partial width of the particle without using interpolation.

double HadronWidths::widthCalc(int id, Pythia8::DecayChannel& channel,
  double m) const {

  // Get particle entry.
  ParticleDataEntryPtr entry = particleDataPtr->findParticle(id);
  if (entry == nullptr) {
    infoPtr->errorMsg("Error in HadronWidths::widthCalc: "
      "particle not found", to_string(id));
    return 0.;
  }

  // Store nominal mass and partial width.
  double m0 = entry->m0(), gamma0 = channel.bRatio() * entry->mWidth();

  // Only two-body decays can have mass-dependent width.
  if (channel.multiplicity() != 2)
    return channel.bRatio();
  auto prodA = particleDataPtr->findParticle(channel.product(0));
  auto prodB = particleDataPtr->findParticle(channel.product(1));

  if (m < prodA->mMin() + prodB->mMin())
    return 0.;

  // Get two-body angular momentum.
  int lType;
  if (channel.meMode() >= 3 && channel.meMode() <= 7)
    lType = 2 * (channel.meMode() - 3) + 1;
  else if (channel.meMode() == 2)
    lType = 3;
  else
    lType = 1;

  // Calculate phase space at the specified mass.
  double pM = psSize(m, prodA, prodB, lType);
  if (pM == 0.)
    return 0.;
  double pMS = psSize(m, prodA, prodB, lType - 1);
  if (pMS == 0.)
    return 0.;

  // Calculate phase space at on-shell mass.
  double pM0  = psSize(m0, prodA, prodB, lType);
  double pM0S = psSize(m0, prodA, prodB, lType - 1);
  if (pM0 <= 0 || pM0S <= 0) {
    infoPtr->errorMsg("Error in HadronWidths::widthCalc: "
      "on-shell decay is not possible",
      to_string(id) + " --> " + to_string(prodA->id())
       + " " + to_string(prodB->id()));
      return NAN;
  }

  // Return mass-dependent partial width.
  return gamma0 * (m0 / m) * (pM / pM0) * 1.2 / (1. + 0.2 * pMS / pM0S);
}

//--------------------------------------------------------------------------

// Regenerate parameterization for the specified particle.

bool HadronWidths::parameterize(int id, int precision) {

  // Get particle entry and validate input.
  ParticleDataEntryPtr entry = particleDataPtr->findParticle(id);

  if (entry == nullptr) {
    infoPtr->errorMsg("Error in HadronWidths::parameterize: "
      "particle does not exist", to_string(id));
    return false;
  }
  if (precision <= 1) {
    infoPtr->errorMsg("Error in HadronWidths::parameterize: "
      "precision must be at least 2");
    return false;
  }
  if (entry->mMin() >= entry->mMax()) {
    infoPtr->errorMsg("Error in HadronWidths::parameterize: "
      "particle has fixed mass", to_string(id));
    return false;
  }

  if (!entry->varWidth())
    infoPtr->errorMsg("Warning in HadronWidths::parameterize: "
      "particle does not have mass-dependent width", to_string(id));

  map<pair<int, int>, ResonanceDecayChannel> partialWidths;
  vector<double> totalWidthData(precision);

  double mMin = entry->mMin(), mMax = entry->mMax();
  double dm = (mMax - mMin) / (precision - 1);

  // Parameterize all channels.
  for (int iChan = 0; iChan < entry->sizeChannels(); ++iChan) {

    // Mass-dependent width is not defined for multibody channels.
    DecayChannel& channel = entry->channel(iChan);
    if (channel.multiplicity() != 2)
      continue;

    // Create key to put decay products in canonical order.
    pair<int, int> key = getKey(id, channel.product(0), channel.product(1));
    int prodA = key.first, prodB = key.second;

    // Calculate widths at regular mass intervals.
    vector<double> widthData(precision);
    for (int j = 0; j < precision; ++j) {
      double m = mMin + j * dm;
      widthData[j] = widthCalc(entry->id(), channel, m);
      totalWidthData[j] += widthData[j];
    }

    // Get two-body angular momentum.
    int lType;
    if (channel.meMode() >= 3 && channel.meMode() <= 7)
      lType = 2 * (channel.meMode() - 3) + 1;
    else if (channel.meMode() == 2)
      lType = 3;
    else
      lType = 1;

    // Add new ResonanceDecayChannel to map.
    partialWidths.emplace(make_pair(prodA, prodB), ResonanceDecayChannel {
      LinearInterpolator(mMin, mMax, widthData),
      prodA, prodB, lType,
      max(mMin, particleDataPtr->mMin(prodA) + particleDataPtr->mMin(prodB))
    });
  }

  // Create new or update existing HadronWidthEntry.
  HadronWidthEntry newEntry {
    LinearInterpolator(mMin, mMax, totalWidthData),
    partialWidths,
    true
  };
  auto iter = entries.find(id);
  if (iter == entries.end())
    entries.emplace(id, newEntry);
  else
    entries[id] = newEntry;

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Helper function to calculate CM momentum.

static double pCMS(double eCM, double mA, double mB) {
  if (eCM <= mA + mB) return 0;
  double sCM = eCM * eCM;
  return sqrt((sCM - pow2(mA + mB)) * (sCM - pow2(mA - mB))) / (2. * eCM);
}

//--------------------------------------------------------------------------

// Get total available phase space, integrating over mass-dependent widths.

double HadronWidths::psSize(double eCM, ParticleDataEntryPtr prodA,
  ParticleDataEntryPtr prodB, double lType) const {

  // Store some important values.
  int idA      = prodA->id(), idB = prodB->id();
  double m0A   = prodA->m0(), m0B = prodB->m0();
  double mMinA = prodA->mMin(), mMinB = prodB->mMin();
  double mMaxA = prodA->mMax(), mMaxB = prodB->mMax();
  bool varA = mMaxA > mMinA, varB = mMaxB > mMinB;

  if (eCM < mMinA + mMinB)
    return 0.;

  double result;
  bool success = true;

  // No resonances.
  if (!varA && !varB)
    return pow(pCMS(eCM, m0A, m0B), lType);

  // A is resonance.
  else if (varA && !varB) {
    if (eCM <= mMinA + m0B)
      return 0.;

    // Integrate mass of A.
    auto f = [=](double mA) {
      return pow(pCMS(eCM, mA, m0B), lType) * mDistr(idA, mA); };
    if (!integrateGauss(result, f, mMinA, min(mMaxA, eCM - m0B)))
      success = false;
  }

  // B is resonance.
  else if (!varA && varB) {
    if (eCM <= m0A + mMinB)
      return 0.;

    // Integrate mass of B.
    auto f = [=](double mB) {
      return pow(pCMS(eCM, m0A, mB), lType) * mDistr(idB, mB); };
    if (!integrateGauss(result, f, mMinB, min(mMaxB, eCM - m0A)))
      success = false;
  }

  // Both are resonances.
  else {
    if (eCM <= mMinA + mMinB)
      return 0.;

    // Define integrand of outer integral.
    auto I = [=, &success](double mA) {

      // Define integrand of inner integral.
      auto f = [=](double mB) {
        return pow(pCMS(eCM, mA, mB), lType)
              * mDistr(idA, mA) * mDistr(idB, mB); };
      double res;

      // Integrate mass of B.
      if (!integrateGauss(res, f, mMinB, min(mMaxB, eCM - mA)))
        success = false;

      return res;
    };

    // Integrate mass of A.
    if (!integrateGauss(result, I, mMinA, min(mMaxA, eCM - mMinB)))
      success = false;
  }

  // Return result if successful.
  if (success)
    return result;
  else {
    infoPtr->errorMsg("Error in HadronWidths::psSize: Unable to integrate");
    return NAN;
  }
}

//--------------------------------------------------------------------------

// Generate parameterization for particle and its decay products if needed.

bool HadronWidths::parameterizeRecursive(int id, int precision) {

  // End recursion if data has already been generated.
  if (hasData(id))
    return true;

  // Get particle entry.
  ParticleDataEntryPtr entry = particleDataPtr->findParticle(id);

  // Iterate over all two-body channels.
  for (int iChannel = 0; iChannel < entry->sizeChannels(); ++iChannel) {
    DecayChannel& channel = entry->channel(iChannel);
    if (channel.multiplicity() == 2) {
      auto prodA = particleDataPtr->findParticle(channel.product(0));
      auto prodB = particleDataPtr->findParticle(channel.product(1));

      // Recursive call to parameterize decay product widths if necessary.
      if (prodA->varWidth() && !hasData(prodA->id()))
        if (!parameterizeRecursive(prodA->id(), precision)) return false;
      if (prodB->varWidth() && !hasData(prodB->id()))
        if (!parameterizeRecursive(prodB->id(), precision)) return false;
    }
  }

  // Perform the actual parameterization of this particle.
  infoPtr->errorMsg("Info from HadronWidths::parameterizeAll: "
    "parameterizing", to_string(id), true);
  return parameterize(id, precision);
}

//--------------------------------------------------------------------------

// Regenerate parameterization for all particles.

void HadronWidths::parameterizeAll(int precision) {

  // Get all particles with varWidth from particle database.
  vector<ParticleDataEntryPtr> variableWidthEntries;
  for (auto& mapEntry : *particleDataPtr) {
    ParticleDataEntryPtr entry = mapEntry.second;
    if (entry->varWidth())
      variableWidthEntries.push_back(entry);
  }

  // Clear existing data and parameterize new data.
  entries.clear();

  for (ParticleDataEntryPtr entry : variableWidthEntries) {
    if (!parameterizeRecursive(entry->id(), precision)) {
      infoPtr->errorMsg("Abort from HadronWidths::parameterizeAll: "
        "parameterization failed");
      return;
    }
  }
}

//--------------------------------------------------------------------------

// Write all widths data to an xml file.

bool HadronWidths::save(ostream& stream) const {

  if (!stream.good())
    return false;

  stream << "\n";

  for (auto& mapEntry : entries) {
    int id = mapEntry.first;
    const HadronWidthEntry& entry = mapEntry.second;

    // Counter for number of entries on current line, maximum 8 per line.
    int c = 0;

    // Write total width.
    stream << "<width id=\"" << id << "\" "
           << "left=\"" << entry.width.left() << "\" "
           << "right=\"" << entry.width.right() << "\" "
           << "data=\" \n";
    for (double dataPoint : entry.width.data()) {
      stream << " " << dataPoint;
      if (++c >= 7) {
        c = 0;
        stream << " \n";
      }
    }
    stream << "\"/> \n \n";

    // Write partial widths.
    for (auto& channelEntry : entry.decayChannels) {
      const ResonanceDecayChannel& channel = channelEntry.second;
      stream << "<partialWidth id=\"" << id << "\" "
        << "products=\"" << channel.prodA << " " << channel.prodB << "\" "
        << "lType=\"" << channel.lType << "\" data=\" \n";
      c = 0;
      for (double dataPoint : channel.partialWidth.data()){
        stream << " " << dataPoint;
        if (++c >= 7) {
          c = 0;
          stream << " \n";
        }
      }
      stream << "\"/> \n \n";
    }

    stream << " \n \n";
  }

  // Done.
  return true;
}

//==========================================================================

}
