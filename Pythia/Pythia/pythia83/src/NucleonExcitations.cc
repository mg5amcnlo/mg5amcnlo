// NucleonExcitations.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for NucleonExcitations class.

#include "Pythia8/NucleonExcitations.h"

namespace Pythia8 {

//==========================================================================

// The NucleonExcitations class.

//--------------------------------------------------------------------------

static double pCMS(double eCM, double mA, double mB) {
  if (eCM <= mA + mB) return 0;
  double sCM = eCM * eCM;
  return sqrt((sCM - pow2(mA + mB)) * (sCM - pow2(mA - mB))) / (2. * eCM);
}

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

//--------------------------------------------------------------------------

// Read in excitation data from the specified file.

bool NucleonExcitations::init(string path) {

  ifstream stream(path);
  if (!stream.is_open()) {
    infoPtr->errorMsg( "Error in NucleonExcitations::init: "
        "unable to open file", path);
    return false;
  }

  return init(stream);
}

//--------------------------------------------------------------------------

// Read in excitation data from the specified stream.

bool NucleonExcitations::init(istream& stream) {

  // Lower bound, needed for total cross section parameterization.
  double eMin = INFINITY;

  // Read header info.
  string line;
  if (!getline(stream, line)) {
    infoPtr->errorMsg("Error in NucleonExcitations::init: "
      "unable to read file");
    return false;
  }

  string word1;
  istringstream(line) >> word1;

  if (word1 != "<header") {
    infoPtr->errorMsg("Error in NucleonExcitations::init: header missing");
    return false;
  }
  completeTag(stream, line);

  // Configuration to use when parameterizing total cross section.
  double highEnergyThreshold = doubleAttributeValue(line, "threshold");
  int sigmaTotalPrecision = intAttributeValue(line, "sigmaTotalPrecision");

  // Process each line sequentially.
  while (getline(stream, line)) {

    if (!(istringstream(line) >> word1))
      continue;

    if (word1 == "<excitationChannel") {
      completeTag(stream, line);

      // Read channel data.
      int maskA = intAttributeValue(line, "maskA");
      int maskB = intAttributeValue(line, "maskB");
      double left  = doubleAttributeValue(line, "left");
      double right = doubleAttributeValue(line, "right");
      double scaleFactor = doubleAttributeValue(line, "scaleFactor");

      istringstream dataStr(attributeValue(line, "data"));
      vector<double> data;
      double currentData;
      while (dataStr >> currentData)
        data.push_back(currentData);

      // Update eMin if needed.
      if (eMin > left)
        eMin = left;

      // Add channel to the list.
      excitationChannels.push_back(ExcitationChannel {
          LinearInterpolator(left, right, data), maskA, maskB, scaleFactor });
    }
  }

  // Pre-sum sigmas to create one parameterization for the total sigma.
  vector<double> sigmaTotPts(sigmaTotalPrecision);
  double dE = (highEnergyThreshold - eMin) / (sigmaTotalPrecision - 1);
  for (int i = 0; i < sigmaTotalPrecision; ++i) {
    double eCM = eMin + i * dE;
    double sigma = 0.;
    for (auto& channel : excitationChannels)
      sigma += channel.sigma(eCM);
    sigmaTotPts[i] = sigma;
  }
  sigmaTotal = LinearInterpolator(eMin, highEnergyThreshold, sigmaTotPts);

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Validate that the loaded data makes sense.

bool NucleonExcitations::check() {

  // Check that all excitations make sense.
  for (auto excitationChannel : excitationChannels) {
    // Check that ids actually correspond to particles.
    for (int mask : { excitationChannel.maskA, excitationChannel.maskB })
    for (int id : { mask + 2210, mask + 2110 })
    if (!particleDataPtr->isParticle(id)) {
      infoPtr->errorMsg("Error in HadronWidths::check: "
        "excitation is not a particle", std::to_string(id));
      return false;
    }
  }
  return true;
}

//--------------------------------------------------------------------------

// Pick an excitation and mass distribution for the specified particles.

bool NucleonExcitations::pickExcitation(int idA, int idB, double eCM,
  int& idCOut, double& mCOut, int& idDOut, double& mDOut) {

  // Excitations are available only for nucleons.
  if (!(abs(idA) == 2112 || abs(idA) == 2212)
   || !(abs(idB) == 2112 || abs(idB) == 2212)) {
    infoPtr->errorMsg("Error in NucleonExcitations:pickExcitation: "
      "excitations are only available for NN collisions");
    return false;
  }

  // If antiparticles, flip signs and flip back at the end.
  int signA = (idA > 0 ? 1 : -1), signB = (idB > 0 ? 1 : -1);
  idA *= signA;
  idB *= signB;

  // Pick an excitation channel.
  vector<double> sigmas(excitationChannels.size());
  for (int i = 0; i < int(sigmas.size()); ++i) {
    // Below threshold, use parameterization.
    if (eCM < excitationChannels[i].sigma.right())
      sigmas[i] = excitationChannels[i].sigma(eCM);
    // Above threshold, use approximation (ignoring incoming phase space).
    else {
      double mA = particleDataPtr->m0(2210 + excitationChannels[i].maskA);
      double mB = particleDataPtr->m0(2210 + excitationChannels[i].maskB);
      sigmas[i] = pCMS(eCM, mA, mB) * excitationChannels[i].scaleFactor;
    }
  }
  auto& channel = excitationChannels[rndmPtr->pick(sigmas)];

  // The two nucleons have equal chance of becoming excited.
  int maskA = channel.maskA, maskB = channel.maskB;
  if (rndmPtr->flat() > 0.5)
    swap(maskA, maskB);

  // Construct ids of resonances from masks plus incoming quark content.
  int idCtmp = maskA + (idA - idA % 10);
  int idDtmp = maskB + (idB - idB % 10);

  // Pick masses.
  double mCtmp, mDtmp;
  if (!hadronWidthsPtr->pickMasses(idCtmp, idDtmp, eCM, mCtmp, mDtmp)) {
    infoPtr->errorMsg("Error in NucleonExcitations::pickExcitation: "
      "failed picking masses",
      "(for " + to_string(idA) + " + " + to_string(idB) + " --> "
      + to_string(idCtmp) + " + " + to_string(idDtmp) + ")");
    return false;
  }

  // Set output values and return.
  idCOut = signA * idCtmp;
  idDOut = signB * idDtmp;
  mCOut = mCtmp;
  mDOut = mDtmp;
  return true;
}

//--------------------------------------------------------------------------

// Get total excitation cross sections for NN at the specified energy.

double NucleonExcitations::sigmaExTotal(double eCM) const {
  // Below threshold, use parameterization.
  if (eCM < sigmaTotal.right())
    return sigmaTotal(eCM);
  // Above threshold, sum approximated integrals.
  else {
    double sig = 0.;
    for (auto channel : excitationChannels) {
      double mA = particleDataPtr->m0(2210 + channel.maskA);
      double mB = particleDataPtr->m0(2210 + channel.maskB);
      sig += channel.scaleFactor * pCMS(eCM, mA, mB);
    }

    // Average over incoming phase space.
    return sig / pCMS(eCM, 0.938, 0.938) / pow2(eCM);
  }
}

//--------------------------------------------------------------------------

  // Get cross section for NN -> CD. Quark content in masks is ignored.

double NucleonExcitations::sigmaExPartial(double eCM,
  int maskC, int maskD) const {

  // Remove quark content from masks.
  maskC = maskC - 10 * ((maskC / 10) % 1000);
  maskD = maskD - 10 * ((maskD / 10) % 1000);

  // Ensure ordering is ND, NX* or DX*.
  if (maskD == 0002 || (maskD == 0004 && maskC > 0004))
    swap(maskC, maskD);

  // Find the corresponding channel.
  for (auto& channel : excitationChannels)
    if (channel.maskA == maskC && channel.maskB == maskD) {
      // At low energy, use interpolation.
      if (eCM < channel.sigma.right())
        return channel.sigma(eCM);

      // At high energy, use parameterization.
      double mA = particleDataPtr->m0(2210 + channel.maskA);
      double mB = particleDataPtr->m0(2210 + channel.maskB);
      return channel.scaleFactor / pow2(eCM)
           * pCMS(eCM, mA, mB) / pCMS(eCM, 0.938, 0.938);
    }

  // Cross section is zero if channel does not exist.
  return 0.;
}

//--------------------------------------------------------------------------

// Get masks (ids without quark content) for all implemented cross sections.

vector<pair<int, int>> NucleonExcitations::getChannels() const {
  vector<pair<int, int>> result;
  for (auto channel : excitationChannels)
    result.push_back(make_pair(channel.maskA, channel.maskB));
  return result;
}

//--------------------------------------------------------------------------

// Get all nucleon excitations from particle data.

vector<int> NucleonExcitations::getExcitationMasks() const {

  vector<int> results;
  for (auto& kvPair : *particleDataPtr) {
    int id = kvPair.first;
    int quarkContent = ((id / 10) % 1000);
    int mask = id - 10 * quarkContent;

    // Check quark content to make sure each mask is included only once.
    if ( ((mask == 0004) || (mask >= 10000 && mask < 1000000))
      && quarkContent == 221 )
      results.push_back(mask);
  }

  return results;
}

//--------------------------------------------------------------------------

// Calculate partial excitation cross section without using interpolation.

double NucleonExcitations::sigmaCalc(double eCM, int maskC, int maskD) const {

  // Convert masks to particle ids.
  int quarkContentC = (maskC / 10) % 1000, quarkContentD = (maskD / 10) % 1000;
  maskC -= 10 * quarkContentC;
  maskD -= 10 * quarkContentD;
  ParticleDataEntryPtr entryC = particleDataPtr->findParticle(2210 + maskC);
  ParticleDataEntryPtr entryD = particleDataPtr->findParticle(2210 + maskD);

  // No cross section below threshold.
  if (eCM < entryC->mMin() + entryD->mMin())
    return 0.;

  // Calculate matrix element, based on method by UrQMD.
  double matrixElement;
  if (maskC == 0002 && maskD == 0004) {
    constexpr double A = 40000, mD2 = pow2(1.232), GammaD2 = pow2(0.115);
    matrixElement = A * mD2 * GammaD2 /
      (pow2(eCM * eCM - mD2) + mD2 * GammaD2);
  }
  else if (maskC == 0004 && maskD == 0004)
    matrixElement = 2.8;
  else {
    double mD = particleDataPtr->m0(2210 + maskD);
    double mC, A;
    if (maskC == 0002) {
      mC = 0.938;
      if (particleDataPtr->isParticle(2220 + maskD))
        A = 12.0;
      else
        A = 6.3;
    }
    else {
      mC = 1.232;
      A = 3.5;
    }

    matrixElement = A / (pow2(mD - mC) * pow2(mD + mC));
  }

  // Return cross section.
  return entryC->spinType() * entryD->spinType() * matrixElement
       * psSize(eCM, *entryC, *entryD) / pCMS(eCM, 0.938, 0.938) / pow2(eCM);
}

//--------------------------------------------------------------------------

// Regenerate parameterization for all cross sections.

bool NucleonExcitations::parameterizeAll(int precision, double threshold) {

  if (precision <= 1){
    infoPtr->errorMsg("Error in NucleonExcitations::parameterizeAll: "
      "precision must be at least 2");
    return false;
  }

  double mN = particleDataPtr->m0(2212), mD = particleDataPtr->m0(2214);

  // Calculate high energy scale factor for nucleons and Delta(1232).
  double scaleFactorN = 2.;

  double scaleFactorD;
  bool valid = integrateGauss(scaleFactorD, [&](double m) {
    return hadronWidthsPtr->mDistr(2214, m);
  }, particleDataPtr->mMin(2214), particleDataPtr->mMax(2214));
  if (!valid) {
    infoPtr->errorMsg("Abort from NucleonExcitations::parameterizeAll: "
        "unable to integrate excitation mass distribution", "2214");
    return false;
  }
  scaleFactorD *= 4;

  // Create new excitation channels.
  excitationChannels.clear();
  for (auto maskEx : getExcitationMasks()) {

    int idEx = 2210 + maskEx;
    infoPtr->errorMsg("Info from NucleonExcitations::parameterizeAll: "
      "parameterizing", to_string(idEx), true);

    // Define helpful variables for the current excitation.
    ParticleDataEntryPtr entry = particleDataPtr->findParticle(idEx);
    double mEx = entry->m0(), mMinEx = entry->mMin();
    bool isDelta = particleDataPtr->isParticle(2220 + maskEx);

    // Calculate high energy scale factor.
    double scaleFactorEx;
    valid = integrateGauss(scaleFactorEx, [&](double m) {
      return hadronWidthsPtr->mDistr(idEx, m);
    }, entry->mMin(), entry->mMax());

    if (!valid) {
      infoPtr->errorMsg("Abort from NucleonExcitations::parameterizeAll: "
        "unable to integrate excitation mass distribution", to_string(idEx));
      return false;
    }
    scaleFactorEx *= entry->spinType();

    // Generate N + X cross sections.
    double eMin = mN + mMinEx;
    double de = (threshold - eMin) / (precision - 1);
    vector<double> dataPointsNX(precision);
    for (int ie = 0; ie < precision; ++ie) {
      double eNow = eMin + de * ie;
      dataPointsNX[ie] = sigmaCalc(eNow, 0002, maskEx);
    }

    double scaleN = (maskEx == 0004) ? 0.
      : scaleFactorN * scaleFactorEx * (isDelta ? 12.0 : 6.3)
        / (pow2(mN - mEx) * pow2(mN + mEx));
    excitationChannels.push_back(ExcitationChannel {
      LinearInterpolator(eMin, threshold, dataPointsNX),
      0002, maskEx, scaleN
    });

    // Generate Delta(1232) + X cross sections.
    eMin = mD + mMinEx;
    de = (threshold - eMin) / (precision - 1);
    vector<double> dataPointsDX(precision);
    for (int ie = 0; ie < precision; ++ie) {
      double eNow = eMin + de * ie;
      dataPointsDX[ie] = sigmaCalc(eNow, 0004, maskEx);
    }

    double scaleD = scaleFactorD * scaleFactorEx *
      (maskEx == 0004 ? 2.8 : 3.5 / (pow2(mD - mEx) * pow2(mD + mEx)));
    excitationChannels.push_back(ExcitationChannel {
      LinearInterpolator(eMin, threshold, dataPointsDX),
      0004, maskEx, scaleD
    });
  }

  // Reparameterize total cross section.
  vector<double> sigmaTotPts(precision);
  double eMin = mN + mD;
  double de = (threshold - eMin) / (precision - 1);
  for (int ie = 0; ie < precision; ++ie) {
    double eNow = eMin + de * ie;
    sigmaTotPts[ie] = 0;
    for (auto& channel : excitationChannels)
      sigmaTotPts[ie] += channel.sigma(eNow);
  }
  sigmaTotal = LinearInterpolator(eMin, threshold, sigmaTotPts);

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Get total available phase space, integrating over mass-dependent widths.

double NucleonExcitations::psSize(double eCM, ParticleDataEntry& prodA,
  ParticleDataEntry& prodB) const {

  // Store some important values.
  int idA      = prodA.id(), idB = prodB.id();
  double m0A   = prodA.m0(), m0B = prodB.m0();
  double mMinA = prodA.mMin(), mMinB = prodB.mMin();
  double mMaxA = prodA.mMax(), mMaxB = prodB.mMax();
  bool varA = mMaxA > mMinA, varB = mMaxB > mMinB;

  if (eCM < mMinA + mMinB)
    return 0.;

  double result;
  bool success = true;

  // No resonances.
  if (!varA && !varB)
    return pCMS(eCM, m0A, m0B);

  // A is resonance.
  else if (varA && !varB) {
    if (eCM <= mMinA + m0B)
      return 0.;

    // Integrate mass of A.
    auto f = [=](double mA) {
      return pCMS(eCM, mA, m0B) * hadronWidthsPtr->mDistr(idA, mA); };
    if (!integrateGauss(result, f, mMinA, min(mMaxA, eCM - m0B)))
      success = false;
  }

  // B is resonance.
  else if (!varA && varB) {
    if (eCM <= m0A + mMinB)
      return 0.;

    // Integrate mass of B.
    auto f = [=](double mB) {
      return pCMS(eCM, m0A, mB) * hadronWidthsPtr->mDistr(idB, mB); };
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
        return pCMS(eCM, mA, mB)
              * hadronWidthsPtr->mDistr(idA, mA)
              * hadronWidthsPtr->mDistr(idB, mB); };
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

// Write all cross section data to an xml file.

bool NucleonExcitations::save(ostream& stream) const {

  if (!stream.good())
    return false;

  // Write header
  stream << "<header "
         << "threshold=\"" << sigmaTotal.right() << "\" "
         << "sigmaTotalPrecision=\"" << sigmaTotal.data().size() << "\" /> "
         << endl << endl;

  // Write channels.
  for (auto& channel : excitationChannels) {
    stream << "<excitationChannel "
           << "maskA=\"" << channel.maskA << "\" "
           << "maskB=\"" << channel.maskB << "\" "
           << "left=\"" << channel.sigma.left() << "\" "
           << "right=\"" << channel.sigma.right() << "\" "
           << "scaleFactor=\"" << channel.scaleFactor << "\" "
           << "data=\" \n";

    for (double dataPoint : channel.sigma.data())
      stream << dataPoint << " ";
    stream << "\n /> \n \n";
  }

  // Done.
  return true;
}

//==========================================================================

}
