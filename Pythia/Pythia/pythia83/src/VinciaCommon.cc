// VinciaCommon.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the headers) for the Colour, Rambo, and
// VinciaCommon classes, and related auxiliary methods.

#include "Pythia8/VinciaCommon.h"
#include "Pythia8/MathTools.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// The VinciaColour class.

//--------------------------------------------------------------------------

// Initialize.

bool VinciaColour::init() {

  // Sanity check.
  if (!isInitPtr) return false;

  // Set verbosity level.
  verbose = settingsPtr->mode("Vincia:verbose");

  // Set parameters. CR disabled in this version.
  inheritMode       = settingsPtr->mode("Vincia:CRinheritMode");

  // Sett initialization.
  isInit = true;
  return isInit;

}

//--------------------------------------------------------------------------

// Map a list of particles with ordinary Les Houches colour tags into
// a new set of Les Houches colour tags with the last index running
// from 1-9.
//
// Note on coloured resonances. These should have their colours
//  defined by whatever system they were produced by, and should not
//  be recoloured when decaying. This means that only colour lines for
//  which both the colour AND the corresponding anticolour line are
//  found inside the same system should be redefined (assuming the
//  coloured resonance itself does not appear explicitly in its
//  corresponding decay partonSystem.

bool VinciaColour::colourise(int iSys, Event& event) {

  // Sanity checks.
  if (!isInit) {
    printOut("VinciaColour::colourise","ERROR! not initialised");
    return false;
  }
  else if (partonSystemsPtr->sizeAll(iSys) <= 1) return false;

  // Construct colour map and assign new tags.
  map<int, int> colourMap;
  int startTag = event.lastColTag();
  int nTries = 0;
  bool accept = false;

  // Do not recolour decaying resonances (as they will already have
  // been coloured when they were produced).
  int colRes(0), acolRes(0);
  if (partonSystemsPtr->hasInRes(iSys)) {
    int iRes = partonSystemsPtr->getInRes(iSys);
    colRes  = event[iRes].col();
    acolRes = event[iRes].acol();
  }

  // Reject assignments that would produce (subleading) singlet gluons.
  while (!accept && ++nTries < 10) {
    colourMap.clear();

    // Example: if startTag is 220-229, nextTagBase is 230.
    int nextTagBase = 10*int((startTag/10)+1);
    for (int i=0; i<partonSystemsPtr->sizeAll(iSys); ++i) {
      int i1 = partonSystemsPtr->getAll(iSys,i);
      if (i1 <= 0) continue;
      Particle* partonPtr = &event[i1];
      if (partonPtr->colType() == 0) continue;
      int col, acol;

      // Cross initial-state colours.
      if (i < partonSystemsPtr->sizeAll(iSys)
        - partonSystemsPtr->sizeOut(iSys)) {
        acol = partonPtr->col();
        col  = partonPtr->acol();
        if (col  == acolRes)  col = 0;
        if (acol ==  colRes) acol = 0;
      }
      else {
        col  = partonPtr->col();
        acol = partonPtr->acol();
        if (col == colRes)   col  = 0;
        if (acol == acolRes) acol = 0;
      }
      if (col == 0 && acol == 0) continue;
      int colIndx = colourMap[col];
      int acolIndx = colourMap[acol];
      if (col != 0) {

        // First time a tag is encountered, mark it negative -> one end found.
        if (colIndx == 0) {
          // Ensure gluons have different col and acol indices.
          while (colIndx == 0 || colIndx == acolIndx) {
            colIndx = nextTagBase + int(rndmPtr->flat()*9) + 1;
          }
          colourMap[col]  = -colIndx;
        }
        // Second time mark it positive -> both ends found
        else colourMap[col] = abs(colourMap[col]);
      }

      if (acol != 0) {
        // First time a tag is encountered, mark it negative -> one end found
        if (acolIndx == 0) {
          // Ensure gluons have different col and acol indices
          while (acolIndx == 0 || colIndx == acolIndx) {
            acolIndx = nextTagBase + int(rndmPtr->flat()*9) + 1;
          }
          colourMap[acol] = -acolIndx;
        }
        // Second time mark it positive -> both ends found
        else colourMap[acol] = abs(colourMap[acol]);
      }
      // Update nextTagBase
      nextTagBase += 10;
    }

    // Check if these assignments would produce any singlet gluons
    accept = true;
    for (int i=0; i<partonSystemsPtr->sizeAll(iSys); ++i) {
      int i1 = partonSystemsPtr->getAll(iSys,i);
      Particle* partonPtr = &event[i1];
      if (partonPtr->colType() != 2) continue;
      int colIndexNew  = colourMap[partonPtr->col()] % 10;
      int acolIndexNew = colourMap[partonPtr->acol()] % 10;
      if (colIndexNew == acolIndexNew) {
        accept=false;
        break;
      }
    }
  }

  // Check for failure to find acceptable conf.
  if (!accept) {
    if (verbose >= REPORT) event.list();
    printOut(__METHOD_NAME__,"Warning! failed to avoid singlet gluon(s).");
  }

  // Update event.
  for (int i = 0; i < partonSystemsPtr->sizeAll(iSys); ++i) {
    int ip = partonSystemsPtr->getAll(iSys,i);
    Particle* partonPtr = &event[ip];
    if (partonPtr->colType() == 0) continue;
    if ( colourMap[partonPtr->col()] > 0 )
      partonPtr->col(colourMap[partonPtr->col()]);
    if ( colourMap[partonPtr->acol()] > 0 )
      partonPtr->acol(colourMap[partonPtr->acol()]);

    // Update max used colour tag.
    int lastTag = event.lastColTag();
    int colMax  = max(abs(partonPtr->col()),abs(partonPtr->acol()));
    while (colMax > lastTag) lastTag = event.nextColTag();
  }

  // Return successful.
  return true;
}

//--------------------------------------------------------------------------

// Order a list of partons in colour sequence.

vector<int> VinciaColour::colourSort(vector<Particle*> partons) {

  // Output vector (starts empty).
  vector<int> iSorted;

  // Shorthand for final-state parton multiplicities.
  int nPartons=partons.size();
  if (nPartons <= 1) return iSorted;

  // Find string endpoints and colour types of particles.
  vector<int> iTrip, iAnti, iOct, iOtherIn, iOtherOut;

  // Definition of colType (classified by multiplet up to total charge
  // p+q = 4).
  //  +- 1 : triplet (e.g., Red)        [1,0] / [0,1] {quark, antidiquark}
  //     2 : octet (e.g., R-Gbar)       [1,1] {
  //           gluon, incoherent qqbar, partially coherent gluon-gluon
  //           eg R-(Bbar-B)-Gbar -> R-Gbar (no junction)
  //           or (R-B)-(Bbar-Gbar) -> Gbar-R (junction-antijunction)}
  //  +- 3 : sextet (e.g., 2 x Red)     [2,0] / [0,2] {2 incoherent quarks}
  //  +- 4 : fifteen (e.g., R-R-Gbar)   [2,1] / [1,2] {incoherent qg}
  //  +- 5 : decuplet (e.g., 3 x Red)   [3,0] / [0,3] {
  //           3 incoherent quarks / partially coherent gluon-gluon
  //           eg R-Gbar-R-Bbar -> R-R-R}
  //     6 : vigintiseptet (e.g., 2 x R-Gbar)   [2,2] {2 incoherent gluons}
  //  +- 7 : fifteen' (e.g., 4 x Red)   [4,0] / [0,4] {4 incoherent quarks}
  //  +- 8 : vigintiquartet (e.g., R-R-R-Gbar)  [3,1] / [1,3]

  map<int, int> iOfAcol;
  for (int i=partons.size()-1; i>=0; --i) {
    int sign    = (partons[i]->isFinal() ? 1 : -1);
    int colType = particleDataPtr->colType(partons[i]->id());

    // Store indices of anticolour partners.
    if (sign == 1 && ( colType == -1 || colType == 2))
      iOfAcol[partons[i]->acol()] = i;
    else if (sign == -1 && ( colType == 1 || colType == 2 ))
      iOfAcol[partons[i]->col()] = i;

    // Construct list of triplets (= starting points).
    if (colType * sign == 1) iTrip.push_back(i);

    // List of antitriplets.
    else if (colType * sign == -1) iAnti.push_back(i);
    // List of octets.
    else if (colType == 2) iOct.push_back(i);

    // Higher representations.
    else if (abs(colType) >= 3) {
      cout << "colourSort(): ERROR! handling of coloured particles in "
           << "representations higher than triplet or octet is not implemented"
           << endl;
    }

    // Colourless particles.
    else if (sign == -1) iOtherIn.push_back(i);
    else iOtherOut.push_back(i);
  }

  // Now sort particles.
  int  i1 = -1;
  bool beginNewChain = true;

  // Keep looping until we have sorted all particles.
  while (iSorted.size() < partons.size()) {

    // Start new piece (also add colourless particles at front and end).
    if (beginNewChain) {

      // Insert any incoming colourless particles at front of iSorted.
      if (iOtherIn.size() > 0) {
        iSorted.push_back(iOtherIn.back());
        iOtherIn.pop_back();

      // Triplet starting point (charge += 1).
      } else if (iTrip.size() > 0) {
        beginNewChain = false;
        iSorted.push_back(iTrip.back());
        iTrip.pop_back();

      // Octet starting point if no triplets/sextets available.
      } else if (iOct.size() > 0) {
        beginNewChain = false;
        iSorted.push_back(iOct.back());
        iOct.pop_back();

      } else if (iOtherOut.size() > 0) {
        iSorted.push_back(iOtherOut.back());
        iOtherOut.pop_back();
      }

      // Index of current starting parton.
      i1 = iSorted.back();

    // Step to next parton in this chain.
    } else {
      bool isFinal = partons[iSorted.back()]->isFinal();
      int col = (isFinal) ? partons[iSorted.back()]->col()
        : partons[iSorted.back()]->acol();
      int iNext = iOfAcol[col];

      // Sanity check.
      if (iNext < 0) {
        cout << "colourSort(): ERROR! cannot step to < 0" << endl;
        beginNewChain = true;

      // Catch close of gluon ring.
      } else if (iNext == i1) {
        beginNewChain = true;

      // Step to next parton; end if not gluon (antiquark or IS quark).
      } else {
        // Add to sorted list.
        iSorted.push_back(iNext);
        // If endpoint reached, begin new chain.
        if (particleDataPtr->colType(partons[iNext]->id()) != 2)
          beginNewChain = true;
        // Octet: continue chain and erase this octet from list.
        else {
          // Erase this endpoint from list.
          for (int i=0; i<(int)iOct.size(); ++i) {
            if (iOct[i] == iNext) {
              iOct.erase(iOct.begin()+i);
              break;
            }
          }
        }
      }
    } // End step to next parton.
  }

  // Normal return.
  return iSorted;

}

//--------------------------------------------------------------------------

// Make colour maps and construct list of parton pairs that form QCD dipoles.

void VinciaColour::makeColourMaps(const int iSysIn, const Event& event,
  map<int,int>& indexOfAcol, map<int,int>& indexOfCol,
  vector< pair<int,int> >& antLC, const bool findFF, const bool findIX) {

  // Loop over all parton systems.
  int iSysBeg = (iSysIn >= 0) ? iSysIn : 0;
  int iSysEnd = (iSysIn >= 0) ? iSysIn + 1: partonSystemsPtr->sizeSys();
  for (int iSys = iSysBeg; iSys < iSysEnd; ++iSys) {

    // Loop over a single parton system.
    int sizeSystem = partonSystemsPtr->sizeAll(iSys);
    for (int i = 0; i < sizeSystem; ++i) {
      int i1 = partonSystemsPtr->getAll( iSys, i);
      if ( i1 <= 0 ) continue;

      // Save to colour maps.
      int col  = event[i1].col();
      int acol = event[i1].acol();

      // Switch colours for initial partons.
      if (!event[i1].isFinal()) {
        col  = acol;
        acol = event[i1].col();
      }

      // Save colours (taking negative-index sextets into account).
      if (col > 0) indexOfCol[col] = i1;
      else if (col < 0) indexOfAcol[-col] = i1;
      if (acol > 0) indexOfAcol[acol] = i1;
      else if (acol < 0) indexOfCol[-acol] = i1;

      // Look for partner on colour side.
      if (col > 0 && indexOfAcol.count(col) == 1) {
        int i2 = indexOfAcol[col];
        if ( event[i1].isFinal() && event[i2].isFinal() ) {
          if (findFF) antLC.push_back( make_pair(i1,i2) );
        } else if (findIX) antLC.push_back( make_pair(i1,i2) );
      }

      // Look for partner on anticolour side.
      if (acol > 0 && indexOfCol.count(acol) == 1) {
        int i2 = indexOfCol[acol];
        // Coloured parton first -> i2, i1 instead of i1, i2)
        if (event[i1].isFinal() && event[i2].isFinal()) {
          if (findFF) antLC.push_back( make_pair(i2, i1) );
        } else if (findIX) antLC.push_back( make_pair(i2,i1) );
      }

      // Allow for sextets: negative acol -> extra positive col.
      if (acol < 0 && indexOfAcol.count(-acol) == 1) {
        int i2 = indexOfAcol[-acol];
        if (event[i1].isFinal() && event[i2].isFinal()) {
          if (findFF) antLC.push_back( make_pair(i1,i2) );
        } else if (findIX) antLC.push_back( make_pair(i1,i2) );
      }
      if (col < 0 && indexOfCol.count(-col) == 1) {
        int i2 = indexOfAcol[-acol];
        if (event[i1].isFinal() && event[i2].isFinal()) {
          if (findFF) antLC.push_back( make_pair(i1,i2) );
        } else if (findIX) antLC.push_back( make_pair(i1,i2) );
      }
    }
  }
  return;

}

//--------------------------------------------------------------------------

// Determine which of two antennae inherits the old colour tag after a
// branching. Default is that the largest invariant has the largest
// probability to inherit, with a few alternatives also implemented.

bool VinciaColour::inherit01(double s01, double s12) {
  // Initialization check.
  if (!isInit) {
    printOut("VinciaColour::inherit01",
      "ERROR! not initialised");
    if (isInitPtr && rndmPtr->flat() < 0.5) return false;
    else return true;
  }

  // Mode 0: Purely random.
  if (inheritMode == 0) {
    if (rndmPtr->flat() < 0.5) return true;
    else return false;
  }

  // Safety checks: small, or approximately equal s01, s12.
  double a12 = abs(s01);
  double a23 = abs(s12);

  // Inverted mode (smallest invariant inherits - should only be used
  // for extreme variation checks).
  if (inheritMode < 0) {
    a12 = abs(s12);
    a23 = abs(s01);
    inheritMode = abs(inheritMode);
  }

  // Winner-takes-all mode.
  if (inheritMode == 2) {
    if (a12 > a23) return true;
    else return false;
  }
  double p12 = 0.5;
  if ( max(a12,a23) > NANO ) {
    if ( a12 < NANO ) p12 = 0.;
    else if ( a23 < NANO ) p12 = 1.;
    else {
      double r = a23/a12;
      if (r < NANO) p12 = 1. - r;
      else if (r > 1./NANO) p12 = 1./r;
      else p12 = 1./(1. + r);
    }
  }
  if (rndmPtr->flat() < p12) return true;
  else return false;

}

//==========================================================================

// The Resolution class.

//--------------------------------------------------------------------------

// Initialize.

bool Resolution::init() {

  // Check that pointers initialized.
  if (!isInitPtr) {
    printOut("Resolution::init","Cannot initialize, pointers not set.");
    return false;
  }

  // Set members.
  verbose          = settingsPtr->mode("Vincia:verbose");
  nFlavZeroMassSav = settingsPtr->mode("Vincia:nFlavZeroMass");
  isInit           = true;
  return isInit;

}

//--------------------------------------------------------------------------

// Method to calculate evolution variable for a given clustering.

double Resolution::q2evol(VinciaClustering& clus) {
  // 2 -> 3 branchings.
  if (clus.is2to3()) {
    // Fetch invariants.
    double sAB = 0., sar = 0., srb = 0., sab = 0.;
    if (clus.invariants.size() >= 4) {
      sAB = clus.invariants.at(0);
      sar = clus.invariants.at(1);
      srb = clus.invariants.at(2);
      sab = clus.invariants.at(3);
    }
    else {
      if (verbose >= NORMAL)
        infoPtr->errorMsg(__METHOD_NAME__,
          "Invariant vectors aren't initialised.");
      return -1.;
    }
    // Fetch masses.
    double ma2 = 0., mr2 = 0., mb2 = 0.;
    if (clus.massesChildren.size() >= 3) {
      ma2 = pow2(clus.massesChildren.at(0));
      mr2 = pow2(clus.massesChildren.at(1));
      mb2 = pow2(clus.massesChildren.at(2));
    }
    double mA2 = 0., mB2 = 0.;
    if (clus.massesMothers.size() >= 2) {
      mA2 = pow2(clus.massesMothers.at(0));
      mB2 = pow2(clus.massesMothers.at(1));
    }

    // Final-Final configuration.
    if (clus.isFF()) {
      double mar2 = sar + ma2 + mr2;
      double mrb2 = srb + mr2 + mb2;
      clus.Q2evol = (mar2 - mA2) * (mrb2 - mB2) / sAB;
      return clus.Q2evol;
    }

    // Initial-Final/Resonance-Final configuration.
    if (clus.isIF() || clus.isRF()) {
      double mar2 = -sar + ma2 + mr2;
      double mrb2 = srb + mb2 + mr2;
      clus.Q2evol = (mA2 - mar2) * (mrb2 - mB2) / (sar + sab);
      return clus.Q2evol;
    }

    // Initial-Initial configuration.
    if (clus.isII()) {
      double mar2 = -sar + ma2 + mr2;
      double mrb2 = -srb + mb2 + mr2;
      clus.Q2evol = (mA2 - mar2) * (mB2 - mrb2) / sab;
      return clus.Q2evol;
    }
  }
  // Implement other branchings here.

  if (verbose >= NORMAL)
    infoPtr->errorMsg(__METHOD_NAME__,"evolution variable not implemented.");
  return -1.;
}

//--------------------------------------------------------------------------

// Get dimensionless evolution variable.

double Resolution::xTevol(VinciaClustering& clus) {
  // Calculate dimensionful evolution variable first.
  double q2 = q2evol(clus);
  if (q2 >= 0.) {
    // Get normalisation.
    double sNorm = -1.;
    if (clus.isFF()) {
      // Norm is sIK.
      sNorm = clus.invariants.at(0);
    }
    else if (clus.isRF() || clus.isIF()) {
      // Norm is saj+sak.
      sNorm = clus.invariants.at(1) + clus.invariants.at(3);
    }
    else if (clus.isII()) {
      // Norm is sab.
      sNorm = clus.invariants.at(3);
    }
    double xT = q2 / sNorm;
    if (xT >= 0. && xT <= 1.) return xT;
  }

  return -1.;
}

//--------------------------------------------------------------------------

// Top-level function to calculate sector resolution for a given clustering.

double Resolution::q2sector(VinciaClustering& clus) {
  // 2 -> 3 branchings.
  if (clus.is2to3()) {
    if (clus.isFF()) return q2sector2to3FF(clus);
    if (clus.isRF()) return q2sector2to3RF(clus);
    if (clus.isIF()) return q2sector2to3IF(clus);
    if (clus.isII()) return q2sector2to3II(clus);
  }
  // Implement other branchings here.

  if (verbose >= NORMAL)
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Sector resolution not implemented.");
  return -1.;
}

//--------------------------------------------------------------------------

// Top-level function to find sector with minimal resolution while resolving
// a Born configuration.

VinciaClustering Resolution::findSector(vector<Particle>& state,
  map<int, int> nFlavsBorn) {

  // Get all clusterings.
  vector<VinciaClustering> clusterings;
  clusterings = vinComPtr->findClusterings(state, nFlavsBorn);

  // Sanity check.
  if (clusterings.size() < 1) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Warning in Resolution::findSector():"
        " No sector found.");
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,"Born flavour list:");
      for (auto it(nFlavsBorn.begin()); it!=nFlavsBorn.end(); ++it) {
        if (it->second > 0)
          cout << "      " << it->first << ": " << it->second << endl;
      }
      vinComPtr->list(state);
    }
    return VinciaClustering();
  }

  // Return clustering with smallest resolution.
  return getMinSector(clusterings);
}

//--------------------------------------------------------------------------

// Top-level function to find sector with minimal resolution while retaining
// a minimal number of quark pairs and gluons.

VinciaClustering Resolution::findSector(vector<Particle>& state, int nqpMin,
  int ngMin) {

  // Get all clusterings.
  vector<VinciaClustering> clusterings;
  clusterings = vinComPtr->findClusterings(state, nqpMin, ngMin);

  // Return clustering with smallest resolution.
  return getMinSector(clusterings);
}

//--------------------------------------------------------------------------

// Check sector veto, given two clusterings.

bool Resolution::sectorVeto(const VinciaClustering& clusMin,
  const VinciaClustering& clus) {
  // Sanity check: always veto if NANs.
  if (std::isnan(clusMin.Q2res) || std::isnan(clus.Q2res))
    return true;

  // Check resolution scales.
  // Note: in principle, we could implement a random choice here,
  // when the two scales are two close.
  if (clus.Q2res <= clusMin.Q2res) return false;
  return true;
}

//--------------------------------------------------------------------------

// Private functions.

//--------------------------------------------------------------------------

// Sector resolution functions for final-final 2->3 branchings
// as defined in sector shower paper.

double Resolution::q2sector2to3FF(VinciaClustering& clus) {
  // Fetch invariants and masses.
  double sIK = clus.invariants[0];
  double sij = clus.invariants[1];
  double sjk = clus.invariants[2];
  double mj2 = pow2(clus.massesChildren[1]);

  // Gluon splitting.
  // Note: it is assumed that the splitting was I -> ij (as in GXsplitFF).
  if (clus.antFunType == GXsplitFF)
    clus.Q2res = (sij + 2.*mj2) * sqrt((sjk + mj2)/sIK);
  // Gluon emission.
  else clus.Q2res = sij*sjk/sIK;

  return clus.Q2res;
}

//--------------------------------------------------------------------------

// Sector resolution functions for resonance-final 2->3 branchings
// as defined in sector shower paper.

double Resolution::q2sector2to3RF(VinciaClustering& clus) {
  // Fetch invariants and masses.
  double saj = clus.invariants[1];
  double sjk = clus.invariants[2];
  double sak = clus.invariants[3];
  double mj2 = pow2(clus.massesChildren[1]);

  // Gluon splitting.
  // Note: it is assumed that the splitting was J -> jk (as in XGsplitRF).
  if (clus.antFunType == XGsplitRF)
    clus.Q2res = (sjk + 2.*mj2) * sqrt((saj - mj2)/(saj + sak));
  // Gluon emission.
  else clus.Q2res = saj*sjk/(saj + sak);

  return clus.Q2res;
}

//--------------------------------------------------------------------------

// Sector resolution functions for initial-final 2->3 branchings
// as defined in sector shower paper.

double Resolution::q2sector2to3IF(VinciaClustering& clus) {
  // Fetch invariants and masses.
  double saj = clus.invariants[1];
  double sjk = clus.invariants[2];
  double sak = clus.invariants[3];
  double mj2 = pow2(clus.massesChildren[1]);

  // Initial-state gluon splitting.
  // Note: it is assumed that the splitting was a -> Aj (as in QXsplitIF).
  if (clus.antFunType == QXsplitIF)
    clus.Q2res = saj * sqrt((sjk + mj2)/(saj + sak));
  // (Initial-state) Quark conversion.
  // Note: it is assumed that the conversion was a -> Aj (as in GXconvIF).
  else if (clus.antFunType == GXconvIF)
    clus.Q2res = (saj - 2.*mj2) * sqrt((sjk + mj2)/(saj + sak));
  // Final-state gluon splitting.
  // Note: it is assumed that the splitting was J -> jk (as in XGsplitIF).
  else if (clus.antFunType == XGsplitIF)
    clus.Q2res = (sjk + 2.*mj2) * sqrt((saj - mj2)/(saj + sak));
  // Gluon emission.
  else clus.Q2res = saj*sjk/(saj + sak);

  return clus.Q2res;
}

//--------------------------------------------------------------------------

// Sector resolution functions for initial-initial 2->3 branchings
// as defined in sector shower paper.

double Resolution::q2sector2to3II(VinciaClustering& clus) {
  // Fetch invariants and masses.
  double saj = clus.invariants[1];
  double sjb = clus.invariants[2];
  double sab = clus.invariants[3];
  double mj  = clus.massesChildren[1];
  double mj2 = (mj != 0.) ? 0. : pow2(mj);

  // (Initial-state) Gluon splitting.
  // Note: it is assumed that the splitting was a -> Aj (as in QXsplitII).
  if (clus.antFunType == QXsplitII)
    clus.Q2res = (saj - 2.*mj2) * sqrt((sjb - mj2)/sab);
  // (Initial-state) Quark conversion.
  // Note: it is assumed that the conversion was a -> Aj (as in GXconvII).
  else if (clus.antFunType == GXconvII)
    clus.Q2res = saj * sqrt((sjb - mj2)/sab);
  // Gluon emission.
  else clus.Q2res = saj*sjb/sab;

  return clus.Q2res;
}

//--------------------------------------------------------------------------

// Find sector with minimal Q2sector in list of clusterings.

VinciaClustering Resolution::getMinSector(
  vector<VinciaClustering>& clusterings) {

  // Set starting scale.
  double q2min = 1.e19;

  // Loop over all clusterings and save one with minimal resolution.
  VinciaClustering clusMin;
  for (int iClu(0); iClu<(int)clusterings.size(); ++iClu) {
    VinciaClustering* clus = &clusterings.at(iClu);
    q2sector(*clus);
    if (verbose >= DEBUG)
      printOut(__METHOD_NAME__,
        " Sector " + num2str(iClu,2) + ": q2res = "
        + num2str(clus->Q2res,6)
        + " (q2min = " + num2str(q2min,6) + ")");
    if (clus->Q2res < q2min) {
      clusMin = *clus;
      q2min = clusMin.Q2res;
    }
  }
  return clusMin;
}

//--------------------------------------------------------------------------

// Sector resolution function for 3->4 branchings (currently only used
// for gluon splitting, with m2qq as the measure).
//TODO: currently disabled.

// double Resolution::q2sector3to4(const Particle*, const Particle* ,
//   const Particle* j1, const Particle* j2) {
//   Vec4   pqq  = j1->p() + j2->p();
//   double m2qq = pqq.m2Calc();
//   return m2qq;
// }

//--------------------------------------------------------------------------

// Sector resolution function for 2->4 branchings (double emission).
// Assume j1 and j2 are colour connected, with a and b hard recoilers.
//TODO: currently disabled.

// double Resolution::q2sector2to4(const Particle* a, const Particle* b,
//   const Particle* j1, const Particle* j2) {
//   // Double gluon emissions.
//   if (j1->isGluon() && j2->isGluon()) {
//     return min(q2sector2to3(a,j2,j1,a->id(),j2->id()),
//       q2sector2to3(j1,b,j2,j1->id(),b->id()));
//   }
//   // Gluon Splittings.
//   else {
//     // Final-Final.
//     if (a->isFinal() && b->isFinal()) {
//       return -1.;
//     }
//     // Initial-Final with b in final state.
//     else if (b->isFinal()) {
//       return -1.;
//     }
//     // Initial-Final with a in final state.
//     else if (a->isFinal()) {
//       return -1.;
//     }
//     // Initial-Initial.
//     else {
//       return -1.;
//     }
//   }

//   return -1.;
// }

//--------------------------------------------------------------------------

// Sector resolution function for 3->5 branchings
// (emission + splitting).
//TODO: currently disabled.

// double Resolution::q2sector3to5(Particle* a, Particle* b,
//   Particle* j1, Particle* j2, Particle* j3) {

//   // j3 is gluon.
//   Particle* gluPtr;
//   Particle* qPtr;
//   Particle* qBarPtr;
//   if (j1->id() == 21) {
//     gluPtr  = j1;
//     qPtr    = (j2->id() > 0 ? j2 : j3);
//     qBarPtr = (j2->id() < 0 ? j2 : j3);
//   } else if (j2->id() == 21) {
//     gluPtr  = j2;
//     qPtr    = (j1->id() > 0 ? j1 : j3);
//     qBarPtr = (j1->id() < 0 ? j1 : j3);
//   } else if (j3->id() == 21) {
//     gluPtr  = j3;
//     qPtr    = (j2->id() > 0 ? j2 : j1);
//     qBarPtr = (j2->id() < 0 ? j2 : j1);
//   } else {
//     cout << " q2sector3to5: unable to identify branching type" << endl;
//     return 1.e19;
//   }
//   Vec4   pqq  = qPtr->p() + qBarPtr->p();
//   double m2qq = pqq.m2Calc();
//   Particle* colPtr = a;
//   if (a->col()   != gluPtr->acol()) colPtr  = j1;
//   if (j1->col()  != gluPtr->acol()) colPtr  = j2;
//   if (j2->col()  != gluPtr->acol()) colPtr  = j3;
//   if (j3->col()  != gluPtr->acol()) colPtr  = b;
//   Particle* acolPtr = b;
//   if (b->acol()  != gluPtr->col())  acolPtr = j3;
//   if (j3->acol() != gluPtr->col())  acolPtr = j2;
//   if (j2->acol() != gluPtr->col())  acolPtr = j1;
//   if (j1->acol() != gluPtr->col())  acolPtr = a;
//   double q2emit = q2sector2to3(colPtr,acolPtr,gluPtr,21,gluPtr->id());
//   return min(q2emit,m2qq);

// }

//==========================================================================

// The VinciaClustering struct.

//--------------------------------------------------------------------------

// Set information based on current state and indices of children.

void VinciaClustering::setChildren(Event& state, int child1In,
  int child2In, int child3In) {

  // Store indices of children.
  child1 = child1In;
  child2 = child2In;
  child3 = child3In;

  setInvariantsAndMasses(state);

}

void VinciaClustering::setChildren(vector<Particle>& state, int child1In,
  int child2In, int child3In) {

  // Store indices of children.
  child1 = child1In;
  child2 = child2In;
  child3 = child3In;

  setInvariantsAndMasses(state);
}

//--------------------------------------------------------------------------

bool VinciaClustering::initInvariantAndMassVecs() {
  // Save masses of children.
  double ma = massesChildren[0];
  double mj = massesChildren[1];
  double mb = massesChildren[2];

  // Calculate sAB and masses of mothers according to configuration.
  double mA = -1., mB = -1.;
  double sAB = -1.;
  if (isFSR) {
    // Final-final gluon splitting.
    if (antFunType == GXsplitFF) {
      mA = 0.;
      mB = mb;
      sAB = saj + sab + sjb + pow2(ma) + pow2(mj);
    }
    // Final-final gluon emission.
    else if (antFunType == QQemitFF || antFunType == QGemitFF ||
      antFunType == GQemitFF || antFunType == GGemitFF) {
      mA = ma;
      mB = mb;
      sAB = saj + sab + sjb;
    }
    // Resonance-final gluon splitting.
    else if (antFunType == XGsplitRF) {
      mA = ma;
      mB = 0.;
      sAB = saj + sab - sjb - pow2(mj) - pow2(mb);
    }
    // Resonance-final gluon emission.
    else if (antFunType == QQemitRF || antFunType == QGemitRF) {
      mA = ma;
      mB = mb;
      sAB = saj + sab - sjb;
    }
  }
  else {
    // IF initial-state gluon splitting.
    if (antFunType == QXsplitIF) {
      mA = mj;
      mB = mb;
      sAB = saj + sab - sjb - pow2(ma);
    }
    // IF quark conversion.
    else if (antFunType == GXconvIF) {
      mA = 0.;
      mB = mb;
      sAB = saj + sab - sjb - pow2(ma) - pow2(mj);
    }
    // IF final-state gluon splitting.
    else if (antFunType == XGsplitIF){
      mA = ma;
      mB = 0.;
      sAB = saj + sab - sjb - pow2(mj) - pow2(mb);
    }
    // IF gluon emission.
    else if (antFunType == QQemitIF || antFunType == QGemitIF ||
      antFunType == GQemitIF || antFunType == GGemitIF) {
      mA = ma;
      mB = mb;
      sAB = saj + sab - sjb;
    }
    // II initial-state gluon splitting.
    else if (antFunType == QXsplitII) {
      mA = mj;
      mB = mb;
      sAB = sab - saj - sjb + pow2(ma);
    }
    // II quark conversion.
    else if (antFunType == GXconvII) {
      mA = 0.;
      mB = mb;
      sAB = sab - saj - sjb + pow2(ma) + pow2(mj);
    }
    // II Gluon emission.
    else if (antFunType == QQemitII || antFunType == GQemitII ||
      antFunType == GGemitII) {
      mA = ma;
      mB = mb;
      sAB = sab - saj - sjb;
    }
  }

  // Check if masses and antenna invariant make sense.
  if (mA < 0. || mB < 0.) return false;
  // Check if we have phase space left for this emission.
  if (sAB < 0.) return false;

  // Save masses and invariants
  invariants.clear();
  invariants.push_back(sAB);
  invariants.push_back(saj);
  invariants.push_back(sjb);
  invariants.push_back(sab);
  massesMothers.clear();
  massesMothers.push_back(mA);
  massesMothers.push_back(mB);

  return true;
}

//--------------------------------------------------------------------------

void VinciaClustering::setInvariantsAndMasses(Event& state) {
  // Save masses.
  massesChildren.clear();
  massesChildren.push_back(max(0.,state[child1].m()));
  massesChildren.push_back(max(0.,state[child2].m()));
  massesChildren.push_back(max(0.,state[child3].m()));

  // Calculate invariants.
  saj = 2. * state[child1].p() * state[child2].p();
  sjb = 2. * state[child2].p() * state[child3].p();
  sab = 2. * state[child1].p() * state[child3].p();
}

//--------------------------------------------------------------------------

void VinciaClustering::setInvariantsAndMasses(vector<Particle>& state) {
  // Save masses.
  massesChildren.clear();
  massesChildren.push_back(max(0.,state[child1].m()));
  massesChildren.push_back(max(0.,state[child2].m()));
  massesChildren.push_back(max(0.,state[child3].m()));

  // Calculate invariants.
  saj = 2. * state[child1].p() * state[child2].p();
  sjb = 2. * state[child2].p() * state[child3].p();
  sab = 2. * state[child1].p() * state[child3].p();
}

//--------------------------------------------------------------------------

// Get Vincia name of current antenna.

string VinciaClustering::getAntName() const {
  if (isFSR) {
    if (antFunType == QQemitFF) return "QQEmitFF";
    if (antFunType == QGemitFF) return "QGEmitFF";
    if (antFunType == GQemitFF) return "GQEmitFF";
    if (antFunType == GGemitFF) return "GGEmitFF";
    if (antFunType == GXsplitFF) return "GXsplitFF";
    if (antFunType == QQemitRF) return "QQEmitRF";
    if (antFunType == QGemitRF) return "QGEmitRF";
    if (antFunType == XGsplitRF) return "XGsplitRF";
    return "noVinciaName";
  }
  else {
    if (antFunType == QQemitII) return "QQEmitII";
    if (antFunType == GQemitII) return "GQEmitII";
    if (antFunType == GGemitII) return "GGEmitII";
    if (antFunType == QXsplitII) return "QXsplitII";
    if (antFunType == GXconvII) return "GXconvII";
    if (antFunType == QQemitIF) return "QQEmitIF";
    if (antFunType == QGemitIF) return "QGEmitIF";
    if (antFunType == GQemitIF) return "GQEmitIF";
    if (antFunType == GGemitIF) return "GGEmitIF";
    if (antFunType == QXsplitIF) return "QXsplitIF";
    if (antFunType == GXconvIF) return "GXconvIF";
    if (antFunType == XGsplitIF) return "XGsplitIF";
    return "noVinciaName";
  }
}

//==========================================================================

// The VinciaCommon class.

//--------------------------------------------------------------------------

// Initialize the class.

bool VinciaCommon::init() {

  // Check initPtr.
  if (!isInitPtr) {
    printOut(__METHOD_NAME__,"Error! pointers not initialized");
    return false;
  }

  // Verbosity level and checks.
  verbose   = settingsPtr->mode("Vincia:verbose");
  epTolErr  = settingsPtr->parm("Check:epTolErr");
  epTolWarn = settingsPtr->parm("Check:epTolWarn");
  mTolErr   = settingsPtr->parm("Check:mTolErr");
  mTolWarn  = settingsPtr->parm("Check:mTolWarn");

  // Counters
  nUnkownPDG    = 0;
  nIncorrectCol = 0;
  nNAN          = 0;
  nVertex       = 0;
  nChargeCons   = 0;
  nMotDau       = 0;
  nUnmatchedMass.resize(2);
  nEPcons.resize(2);
  for (int i=0; i<2; i++) {
    nUnmatchedMass[i] = 0;
    nEPcons[i]        = 0;
  }

  // Quark masses
  mt                 = particleDataPtr->m0(6);
  if (mt < NANO) mt  = 171.0;
  mb                 = min(mt,particleDataPtr->m0(5));
  if (mb < NANO) mb  = min(mt,4.8);
  mc                 = min(mb,particleDataPtr->m0(4));
  if (mc < NANO) mc  = min(mb,1.5);
  ms                 = min(mc,particleDataPtr->m0(3));
  if (ms < NANO) ms  = min(mc,0.1);

  // Number of flavours to treat as massless in clustering and
  // kinematics maps.
  nFlavZeroMass = settingsPtr->mode("Vincia:nFlavZeroMass");

  // Default alphaS, with and without CMW.
  double alphaSvalue = settingsPtr->parmDefault("Vincia:alphaSvalue");
  int    alphaSorder = settingsPtr->modeDefault("Vincia:alphaSorder");
  int    alphaSnfmax = settingsPtr->modeDefault("Vincia:alphaSnfmax");
  bool   useCMW      = settingsPtr->flagDefault("Vincia:useCMW");
  alphaStrongDef.init(    alphaSvalue, alphaSorder, alphaSnfmax, false);
  alphaStrongDefCMW.init( alphaSvalue, alphaSorder, alphaSnfmax, true);

  // Strong coupling for use in merging.
  alphaSvalue  = settingsPtr->parm("Vincia:alphaSvalue");
  alphaSorder  = settingsPtr->mode("Vincia:alphaSorder");
  alphaSnfmax  = settingsPtr->mode("Vincia:alphaSnfmax");
  useCMW       = settingsPtr->flag("Vincia:useCMW");
  alphaS.init(alphaSvalue, alphaSorder, alphaSnfmax, useCMW);

  // User alphaS, with and without CMW.
  alphaSvalue  = settingsPtr->parm("Vincia:alphaSvalue");
  alphaSorder  = settingsPtr->mode("Vincia:alphaSorder");
  alphaSnfmax  = settingsPtr->mode("Vincia:alphaSnfmax");
  useCMW       = settingsPtr->flag("Vincia:useCMW");
  alphaStrong.init(    alphaSvalue, alphaSorder, alphaSnfmax, false);
  alphaStrongCMW.init( alphaSvalue, alphaSorder, alphaSnfmax, true);

  // Freeze and minimum scales.
  mu2freeze    = pow2(settingsPtr->parm("Vincia:alphaSmuFreeze"));
  alphaSmax    = settingsPtr->parm("Vincia:alphaSmax");

  // Find the overall minimum scale. Take into account the freezeout
  // scale, Lambda pole, and alphaSmax.
  double muMin = max(sqrt(mu2freeze),1.05*alphaS.Lambda3());
  double muMinASmax;
  if (alphaStrong.alphaS(mu2min) < alphaSmax) {
    muMinASmax = muMin;
  } else if (settingsPtr->mode("Vincia:alphaSorder") == 0) {
    muMinASmax = muMin;
  } else {
    muMinASmax = muMin;
    while (true) {
      if (alphaS.alphaS(pow2(muMinASmax)) < alphaSmax) break;
      muMinASmax += 0.001;
    }
  }
  mu2min = pow2( max(muMinASmax, muMin) );

  // EM coupling for use in merging. Dummy, as no EW clusterings.
  alphaEM.init(1, settingsPtr);

  // Return.
  isInit = true;
  return true;

}

//--------------------------------------------------------------------------

// Function to find the lowest meson mass for a parton pair treating
// gluons as down quarks. Used to determine hadronisation boundaries
// consistent with assumption of string length > 1 meson.

double VinciaCommon::mHadMin(const int id1in, const int id2in) {

  // Treat gluons as down quarks for purposes of minimum meson mass.
  int id1 = abs(id1in);
  if (id1 == 21 || id1 <= 2) id1 = 1;
  int id2 = abs(id2in);
  if (id2 == 21 || id2 <= 2) id2 = 1;

  // No hadronisation cut for ID codes >= 6.
  if (max(id1,id2) >= 6) return 0.;

  // ID of would-be pseudoscalar meson.
  int idMes = max(id1,id2)*100 + min(id1,id2)*10 + 1;

  // Special for ssbar, use eta rather than eta'.
  if (idMes == 331) idMes = 221;
  return particleDataPtr->m0(idMes);

}

//--------------------------------------------------------------------------

// Function to check the event after each branching, mostly copied
// from Pythia8.

bool VinciaCommon::showerChecks(Event& event, bool ISR) {

  // Only for verbose >= REPORT.
  if (verbose < REPORT) return true;

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // First check if this event has any beams.
  bool hasBeams = false;
  bool hasBeamRemnants = false;
  int  iBeg     = 0;
  for (int i = 0; i < event.size(); ++i) {
    if (event[i].statusAbs() == 12) {
      iBeg     = i+1;
      hasBeams = true;
    }
    if (abs(event[i].status()) == 63) {
      hasBeamRemnants = true;
      break;
    }
  }

  // Count incoming partons with negative momentum and charge.
  Vec4 pSum;
  double chargeSum = 0.0;
  // If beam remnants were added, define incoming as sum of beams.
  if (hasBeamRemnants) {
    pSum = -(event[1].p() + event[2].p());
    chargeSum = -(event[1].charge()+event[2].charge());
  }
  // If there are incoming beams (but no remnants yet), use incoming partons.
  else if (hasBeams) {
    for (int i = iBeg; i < event.size(); ++i) {
      if ( (event[i].mother1() == 1) || (event[i].mother1() == 2) ) {
        pSum      -= event[i].p();
        chargeSum -= event[i].charge();
      }
    }
  }
  // No beams. Incoming defined by partons with mother = 0.
  else {
    for (int i = 1; i < event.size(); ++i) {
      if (event[i].mother1() == 0) {
        pSum      -= event[i].p();
        chargeSum -= event[i].charge();
      }
      else break;
    }
  }

  double eLab = abs(pSum.e());

  // Loop over particles in the event.
  for (int i = iBeg; i < event.size(); ++i) {

    // Look for any unrecognized particle codes.
    int id = event[i].id();
    if (id == 0 || !particleDataPtr->isParticle(id)) {
      nUnkownPDG++;
      if (nUnkownPDG == 1) {
        cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
             << ": unknown particle code"
             << ", i = " << i << ", id = " << id << endl;
        return false;
      }
    }

    // Check that colour assignments are the expected ones.
    else {
      int colType = event[i].colType();
      int col     = event[i].col();
      int acol    = event[i].acol();
      if (    (colType ==  0 && (col  > 0 || acol  > 0))
        || (colType ==  1 && (col <= 0 || acol  > 0))
        || (colType == -1 && (col  > 0 || acol <= 0))
        || (colType ==  2 && (col <= 0 || acol <= 0)) ) {
        nIncorrectCol++;
        if (nIncorrectCol == 1) {
          cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
               << ": incorrect colours"
               << ", i = " << i << ", id = " << id << " cols = " << col
               << " " << acol << endl;
          return false;
        }
      }
    }

    // Look for particles with mismatched or not-a-number energy/momentum/mass.
    if (abs(event[i].px()) >= 0.0 && abs(event[i].py()) >= 0.0
        && abs(event[i].pz()) >= 0.0 && abs(event[i].e())  >= 0.0
        && abs(event[i].m())  >= 0.0 ) {
      double errMass  = abs(event[i].mCalc() - event[i].m()) /
        max( 1.0, event[i].e());

      if (errMass > mTolErr) {
        nUnmatchedMass[0]++;
        if (nUnmatchedMass[0] == 1) {
          cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
               << ": unmatched particle energy/momentum/mass"
               << ", i = " << i << ", id = " << id << endl;
          return false;
        }
      } else if (errMass > mTolWarn) {
        nUnmatchedMass[1]++;
        if (nUnmatchedMass[1] == 1) {
          cout << "WARNING in Vincia::ShowerChecks"
               << (ISR ? "(ISR)" : "(FSR)")
               << ": not quite matched particle energy/momentum/mass"
               << ", i = " << i << ", id = " << id << endl;
        }
      }
    } else {
      nNAN++;
      if (nNAN == 1) {
        cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
             << ": not-a-number energy/momentum/mass"
             << ", i = " << i << ", id = " << id << endl;
        return false;
      }
    }

    // Look for particles with not-a-number vertex/lifetime.
    if (abs(event[i].xProd()) >= 0.0 && abs(event[i].yProd()) >= 0.0
        && abs(event[i].zProd()) >= 0.0 && abs(event[i].tProd()) >= 0.0
        && abs(event[i].tau())   >= 0.0) {
    } else {
      nVertex++;
      if (nVertex == 1) {
        cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
             << ": not-a-number vertex/lifetime"
             << ", i = " << i << ", id = " << id << endl;
        return false;
      }
    }

    // Add final-state four-momentum and charge.
    if (event[i].isFinal()) {
      pSum      += event[i].p();
      chargeSum += event[i].charge();
    }
  } // End of particle loop.

  // Check energy-momentum/charge conservation.
  double epDev = abs( pSum.e()) + abs(pSum.px()) + abs(pSum.py())
    + abs(pSum.pz() );
  if (epDev > epTolErr * eLab) {
    nEPcons[0]++;
    if (nEPcons[0] == 1) {
      cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
           << ": energy-momentum not conserved" << endl;
      cout << " epDev = " << epDev << " Ein = " << eLab
           << " pSum = " << pSum << endl;
      return false;
    }
  } else if (epDev > epTolWarn * eLab) {
    nEPcons[1]++;
    if (nEPcons[1] == 1)
      cout << "WARNING in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
           << ": energy-momentum not quite conserved" << endl;
  }
  if (abs(chargeSum) > 0.1) {
    nChargeCons++;
    if (nChargeCons == 1) {
      cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
           << ": charge not conserved" << endl;
      return false;
    }
  }

  // Check that mother and daughter information match for each particle.
  vector<int> noMot, noDau;
  vector< pair<int,int> > noMotDau;

  // Loop through the event and check mother/daughter lists.
  for (int i = iBeg; i < event.size(); ++i) {
    int status = event[i].status();

    // Check that mother and daughter lists not empty where not expected to.
    vector<int> mList = event[i].motherList();
    vector<int> dList = event[i].daughterList();
    if (mList.size() == 0 && abs(status) != 11 && abs(status) != 12)
      noMot.push_back(i);
    if (dList.size() == 0 && status < 0 && status != -11)
      noDau.push_back(i);

    // Check that the particle appears in the daughters list of each mother.
    for (int j = 0; j < int(mList.size()); ++j) {
      if (event[mList[j]].daughter1() <= i
          && event[mList[j]].daughter2() >= i) continue;
      vector<int> dmList = event[mList[j]].daughterList();
      bool foundMatch = false;
      for (int k = 0; k < int(dmList.size()); ++k)
        if (dmList[k] == i) {
          foundMatch = true;
          break;
        }
      if (!hasBeams && mList.size() == 1 && mList[0] == 0) foundMatch = true;
      if (!foundMatch) {
        bool oldPair = false;
        for (int k = 0; k < int(noMotDau.size()); ++k)
          if (noMotDau[k].first == mList[j] && noMotDau[k].second == i) {
            oldPair = true;
            break;
          }
        if (!oldPair) noMotDau.push_back( make_pair( mList[j], i) );
      }
    }

    // Check that the particle appears in the mothers list of each daughter.
    for (int j = 0; j < int(dList.size()); ++j) {
      if (event[dList[j]].statusAbs() > 80
          && event[dList[j]].statusAbs() < 90
          && event[dList[j]].mother1() <= i
          && event[dList[j]].mother2() >= i ) continue;
      vector<int> mdList = event[dList[j]].motherList();
      bool foundMatch = false;
      for (int k = 0; k < int(mdList.size()); ++k)
        if (mdList[k] == i) {
          foundMatch = true;
          break;
        }
      if (!foundMatch) {
        bool oldPair = false;
        for (int k = 0; k < int(noMotDau.size()); ++k)
          if (noMotDau[k].first == i && noMotDau[k].second == dList[j]) {
            oldPair = true;
            break;
          }
        if (!oldPair) noMotDau.push_back( make_pair( i, dList[j]) );
      }
    } // End loop through the event.
  }

  // Warn if any errors were found.
  if (noMot.size() > 0 || noDau.size() > 0 || noMotDau.size() > 0) {
    nMotDau++;
    if (nMotDau == 1) {
      cout << "ERROR in Vincia::ShowerChecks" << (ISR ? "(ISR)" : "(FSR)")
           << ": mismatch in daughter and mother lists" << endl;
      // Print some more info.
      if (noMot.size() > 0) {
        cout << " missing mothers for particles ";
        for (int i = 0; i < int(noMot.size()); ++i) cout << noMot[i] << " ";
        cout << endl;
      }
      if (noDau.size() > 0) {
        cout << " missing daughters for particles ";
        for (int i = 0; i < int(noDau.size()); ++i) cout << noDau[i] << " ";
        cout << endl;
      }
      if (noMotDau.size() > 0) {
        cout << " inconsistent history for (mother,daughter) pairs ";
        for (int i = 0; i < int(noMotDau.size()); ++i)
          cout << "(" << noMotDau[i].first << ","
               << noMotDau[i].second << ") ";
        cout << endl;
      }
      return false;
    }
  }

  // Made it to here: no major problems.
  return true;

}

//--------------------------------------------------------------------------

// Get the shower starting scale.

double VinciaCommon::getShowerStartingScale(int iSys, const Event& event,
  double sbbSav) {

  // Depending on user choice shower starts at q2maxFudge *
  // factorization scale of phase space maximum.
  int    qMaxMatch    = settingsPtr->mode("Vincia:QmaxMatch");
  double q2maxFudge   = pow2(settingsPtr->parm("Vincia:QmaxFudge"));
  bool   hasFSpartons = false;
  int nOut = partonSystemsPtr->sizeOut(iSys);
  vector<int> iFS;
  for (int iOut = 0; iOut < nOut; ++iOut) {
    int i = partonSystemsPtr->getOut(iSys,iOut);
    int idAbs = event[i].idAbs();
    if ((idAbs >= 1 && idAbs <= 5) || idAbs == 21) hasFSpartons = true;
    iFS.push_back(i);
  }
  if (qMaxMatch == 1 || (qMaxMatch == 0 && hasFSpartons) ) {
    double Q2facSav = sbbSav;
    double Q2facFix  = settingsPtr->parm("SigmaProcess:factorFixScale");
    double Q2facMult = settingsPtr->parm("SigmaProcess:factorMultFac");

    // Ask Pythia about 2 -> 1 scale.
    if (nOut == 1) {
      int factorScale1 = settingsPtr->mode("SigmaProcess:factorScale1");
      Q2facSav = ( factorScale1 == 1 ? Q2facMult*event[iFS[0]].m2Calc() :
        Q2facFix );

    // Ask Pythia about 2 -> 2 scale.
    } else if (iFS.size() == 2) {
      int factorScale2 = settingsPtr->mode("SigmaProcess:factorScale2");
      double mT21 = event[iFS[0]].mT2(), mT22 = event[iFS[1]].mT2();
      double sHat = m2(event[iFS[0]],event[iFS[1]]);
      double tHat = m2(event[3],event[iFS[0]]);
      if (factorScale2 == 1) Q2facSav = min(mT21,mT22);
      else if (factorScale2 == 2) Q2facSav = sqrt(mT21*mT22);
      else if (factorScale2 == 3) Q2facSav = 0.5*(mT21+mT22);
      else if (factorScale2 == 4) Q2facSav = sHat;
      else if (factorScale2 == 5) Q2facSav = Q2facFix;
      else if (factorScale2 == 6) Q2facSav = abs(-tHat);
      if (factorScale2 != 5) Q2facSav *= Q2facMult;

    // Ask Pythia about 2 -> 3 scale.
    } else if (iFS.size() == 3) {
      int factorScale3 = settingsPtr->mode("SigmaProcess:factorScale3");
      double mT21 = event[iFS[0]].mT2(), mT22 = event[iFS[1]].mT2(),
        mT23 = event[iFS[2]].mT2();
      double mT2min1 = min(mT21,min(mT22,mT23));
      double mT2min2 = max(max(min(mT21,mT22),min(mT21,mT23)),min(mT22,mT23));
      double sHat    = m2(event[iFS[0]],event[iFS[1]],event[iFS[2]]);
      if (factorScale3 == 1) Q2facSav = mT2min1;
      else if (factorScale3 == 2) Q2facSav = sqrt(mT2min1*mT2min2);
      else if (factorScale3 == 3) Q2facSav = pow(mT21*mT22*mT23,1.0/3.0);
      else if (factorScale3 == 4) Q2facSav = (mT21+mT22+mT23)/3.0;
      else if (factorScale3 == 5) Q2facSav = sHat;
      else if (factorScale3 == 6) Q2facSav = Q2facFix;
      if (factorScale3 != 6) Q2facSav *= Q2facMult;

    // Unknown, leave as is, all emissions allowed now.
    } else {;}
    return q2maxFudge*Q2facSav;
  }
  return sbbSav;

}

//--------------------------------------------------------------------------

// Find all possible clusterings for given configuration while retaining
// a minimum number of quark pairs and gluons in the event.

vector<VinciaClustering> VinciaCommon::findClusterings(vector<Particle>& state,
  int nqpMin, int ) {

  // Initialise.
  vector<VinciaClustering> clusterings;

  // Dummy flavour map.
  map<int, int> nFlavsBorn;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavsBorn[21] = 0;
    nFlavsBorn[i] = 0;
  }

  // Find number of quark pairs in current state.
  int nqpNow = 0;
  for (auto& ptcl : state)
    if (ptcl.isQuark()) ++nqpNow;
  if (nqpNow % 2 != 0) return clusterings;
  nqpNow /= 2;
  if (nqpNow < nqpMin) return clusterings;

  // Find all tentative clusterings.
  clusterings = findClusterings(state, nFlavsBorn);

  // Erase those reducing the number of quark pairs below the minimum.
  // (Only when the current state has the minimum number of quark pairs).
  if (nqpNow == nqpMin) {
    auto itClus = clusterings.begin();
    while (itClus != clusterings.end()) {
      VinciaClustering clus = *itClus;
      if (clus.isFSR) {
        if (clus.antFunType == GXsplitFF || clus.antFunType == XGsplitRF) {
          clusterings.erase(itClus);
        }
        else ++itClus;
      }
      else {
        // Note: no QXsplit, as this leaves the total number of
        // quark pairs invariant.
        if (clus.antFunType == GXconvIF || clus.antFunType == GXconvII
          || clus.antFunType == XGsplitIF)
          clusterings.erase(itClus);
        else ++itClus;
      }
    }
  }

  return clusterings;
}

//--------------------------------------------------------------------------

bool VinciaCommon::isValidClustering(const VinciaClustering& clus,
  const Event& event, int verboseIn) {
  // Fetch children.
  const Particle* a = &event[clus.child1];
  const Particle* j = &event[clus.child2];
  const Particle* b = &event[clus.child3];

  // Do not consider emissions into the initial state.
  if (!j->isFinal()) return false;

  // Fetch colour connection.
  bool ajColCon = colourConnected(*a,*j);
  bool jbColCon = colourConnected(*j,*b);
  bool abColCon = colourConnected(*a,*b);

  // Quark-antiquark clusterings.
  if (j->isQuark()) {
    // We can have multiple antennae contributing to this state.
    // Only return false if nothing is found.
    bool foundValid = false;
    // (Initial-state) quark conversion on side a.
    if (!a->isFinal() && a->isQuark()) {
      // Quark a needs to have the same flavour as j and must not
      // be colour-connected to it to not create singlet gluons.
      if (!ajColCon && a->id() == j->id()) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid quark conversion clustering on side a.");
        foundValid = true;
      }
    }
    // Initial-state gluon splitting on side a.
    else if (!a->isFinal() && a->isGluon()) {
      // Gluon a has to be colour-connected with both j and b.
      if (ajColCon && abColCon) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid initial-state gluon splitting clustering on side a.");
        foundValid = true;
      }
    }
    // Final-state gluon splitting on side a.
    else if (a->isFinal() && a->isQuark()) {
      // Quark a needs to have the antiflavour of j and must not
      // be colour-connected to it to not create singlet gluons.
      if (!ajColCon && a->id() == -j->id()) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid final-state gluon splitting clustering on side a.");
        foundValid = true;
      }
    }

    // (Initial-state) quark conversion on side b.
    if (!b->isFinal() && b->isQuark()) {
      // Quark b needs to have the same flavour as j and must not
      // be colour-connected to it to not create singlet gluons.
      if (!jbColCon && b->id() == j->id()) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid quark conversion clustering on side b.");
        foundValid = true;
      }
    }
    // Initial-state gluon splitting on side b.
    else if (!b->isFinal() && b->isGluon()) {
      // Gluon a has to be colour-connected with both j and b.
      if (jbColCon && abColCon) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid initial-state gluon splitting clustering on side b.");
        foundValid = true;
      }
    }
    // Final-state gluon splitting on side b.
    else if (b->isFinal() && b->isQuark()) {
      // Quark b needs to have the antiflavour of j and must not
      // be colour-connected to it to not create singlet gluons.
      if (!jbColCon && b->id() == -j->id()) {
        if (verboseIn >= DEBUG)
          printOut(__METHOD_NAME__,
            "Found valid final-state gluon splitting clustering on side b.");
        foundValid = true;
      }
    }

    // If nothing was found, return false.
    if (!foundValid) return false;
  }
  // For gluon emissions, child2 should be colour-connected with 1 and 3.
  else {
    if (!ajColCon || !jbColCon)
      return false;
    if (verboseIn >= DEBUG)
      printOut(__METHOD_NAME__,
        "Found valid gluon emission clustering.");
  }

  return true;
}

//--------------------------------------------------------------------------

bool VinciaCommon::clus3to2(const VinciaClustering& clus,
  const vector<Particle>& state, vector<Particle>& pClustered) {

  pClustered.clear();

  // Save indices of children in event.
  int ia = clus.child1;
  int ij = clus.child2;
  int ib = clus.child3;

  //TODO polarisations.
  int polA = 9;
  int polB = 9;

  // Find clustered colours.
  pair<int, int> colsA;
  pair<int, int> colsB;
  if (!getCols3to2(&state[ia], &state[ij], &state[ib],
      clus, colsA, colsB)) {
    if (verbose >= NORMAL) {
      string msg = ": Failed to cluster colours.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }
  int colA  = colsA.first;
  int acolA = colsA.second;
  int colB  = colsB.first;
  int acolB = colsB.second;
  // Check if the colour assignment created singlet particles.
  if ((colA == 0 && acolA == 0) || colA == acolA) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << ": created colour-singlet parent A with colours"
         << " (" << colA << ", " << acolA << ")";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__, ss.str());
    }
    return false;
  }
  if ((colB == 0 && acolB == 0) || colB == acolB) {
    if (verbose >= NORMAL) {
      stringstream ss;
      ss << ": created colour-singlet parent B with colours"
         << " (" << colB << ", " << acolB << ")";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__, ss.str());
    }
    return false;
  }

  // Find clustered momenta.
  vector<Vec4> momNow;
  vector<Vec4> momClus;
  for (int iPtcl(0); iPtcl<(int)state.size(); ++iPtcl)
    momNow.push_back(state.at(iPtcl).p());
  if (!getMomenta3to2(momNow, momClus, clus)) {
    if (verbose >= NORMAL) {
      string msg = ": Failed to cluster momenta.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }

  // Initialise clustered particles (momenta are set in loop below).
  Particle pAnew = state.at(ia);
  pAnew.id(clus.idMoth1);
  pAnew.cols(colA,acolA);
  pAnew.pol(polA);
  pAnew.setPDEPtr(particleDataPtr->findParticle(clus.idMoth1));
  pAnew.m(clus.massesMothers.at(0));
  Particle pBnew = state.at(ib);
  pBnew.id(clus.idMoth2);
  pBnew.cols(colB,acolB);
  pBnew.pol(polB);
  pBnew.m(clus.massesMothers.at(1));
  pBnew.setPDEPtr(particleDataPtr->findParticle(clus.idMoth2));

  // Set list of clustered particles with new momenta.
  int iOffset = 0;
  for (int iPtcl(0); iPtcl<(int)momNow.size(); ++iPtcl) {
    if (iPtcl == ij) {
      iOffset = 1;
      continue;
    }
    else if (iPtcl == ia) {
      pClustered.push_back(pAnew);
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
    else if (iPtcl == ib) {
      pClustered.push_back(pBnew);
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
    else {
      pClustered.push_back(state.at(iPtcl));
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
  }

  // Done.
  return true;
}

bool VinciaCommon::clus3to2(const VinciaClustering& clus, const Event& event,
  vector<Particle>& pClustered) {

  pClustered.clear();

  // Save indices of children in event.
  int iaEvt = clus.child1;
  int ijEvt = clus.child2;
  int ibEvt = clus.child3;

  //TODO polarisations.
  int polA = 9;
  int polB = 9;

  // Find clustered colours.
  pair<int, int> colsA;
  pair<int, int> colsB;
  if (!getCols3to2(&event[iaEvt], &event[ijEvt], &event[ibEvt],
      clus, colsA, colsB)) {
    if (verbose >= NORMAL) {
      string msg = ": Failed to cluster colours.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }
  int colA  = colsA.first;
  int acolA = colsA.second;
  int colB  = colsB.first;
  int acolB = colsB.second;
  // Check if the colour assignment created singlet particles.
  if ((colA == 0 && acolA == 0) || colA == acolA) {
    if (verbose >= NORMAL) {
      string msg = ": created colour-singlet parent A.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }
  if ((colB == 0 && acolB == 0) || colB == acolB) {
    if (verbose >= NORMAL) {
      string msg = ": created colour-singlet parent B.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }

  // Find clustered momenta.
  vector<Vec4> momNow;
  vector<Vec4> momClus;
  int iEvtOff = 3;
  for (int iPtcl(iEvtOff); iPtcl<event.size(); ++iPtcl)
    momNow.push_back(event[iPtcl].p());
  if (!getMomenta3to2(momNow, momClus, clus, iEvtOff)) {
    if (verbose >= NORMAL) {
      string msg = ": Failed to cluster momenta.";
      infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
    }
    return false;
  }

  // Initialise clustered particles (momenta are set in loop below).
  Particle pAnew = event[iaEvt];
  pAnew.id(clus.idMoth1);
  pAnew.cols(colA,acolA);
  pAnew.pol(polA);
  pAnew.m(clus.massesMothers.at(0));
  pAnew.setPDEPtr(particleDataPtr->findParticle(clus.idMoth1));
  Particle pBnew = event[ibEvt];
  pBnew.id(clus.idMoth2);
  pBnew.cols(colB,acolB);
  pBnew.pol(polB);
  pBnew.m(clus.massesMothers.at(1));
  pBnew.setPDEPtr(particleDataPtr->findParticle(clus.idMoth2));

  // Set list of clustered particles with new momenta.
  int iOffset = 0;
  for (int iPtcl(0); iPtcl<(int)momNow.size(); ++iPtcl) {
    if (iPtcl == ijEvt-iEvtOff) {
      iOffset = 1;
      continue;
    }
    else if (iPtcl == iaEvt-iEvtOff) {
      pClustered.push_back(pAnew);
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
    else if (iPtcl == ibEvt-iEvtOff) {
      pClustered.push_back(pBnew);
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
    else {
      pClustered.push_back(event[iPtcl+3]);
      pClustered.back().p(momClus.at(iPtcl-iOffset));
    }
  }

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// Get clustered colours in 3->2 clustering.

bool VinciaCommon::getCols3to2(const Particle* a, const Particle* j,
  const Particle* b, const VinciaClustering& clus,
  pair<int,int>& colsA, pair<int,int>& colsB) {

  // Is the emission an antiquark?
  bool jIsAntiQ = j->isQuark() ? (j->id() < 0) : false;

  // Colours and colour chains of clustered particles (mothers).
  int  colA = 0;
  int acolA = 0;
  int  colB = 0;
  int acolB = 0;
  if (clus.isFSR) {
    // Final-final gluon splitting.
    if (clus.antFunType == GXsplitFF) {
      colA = jIsAntiQ ? a->col() : j->col();
      acolA = jIsAntiQ ? j->acol() : a->acol();
      colB = b->col();
      acolB = b->acol();
    }
    // Resonance-final gluon splitting.
    else if (clus.antFunType == XGsplitRF) {
      colA = a->col();
      acolA = a->acol();
      colB = jIsAntiQ ? b->col() : j->col();
      acolB = jIsAntiQ ? j->acol() : b->acol();
    }
    // Gluon emission.
    else {
      // Choose to annihilate colour pair of child2 and child3.
      colA = a->col();
      acolA = a->acol();
      // Note: by construction, b and j are always in the final state.
      if (b->col() == j->acol()) {
        colB  = j->col();
        acolB = b->acol();
      }
      else if (b->acol() == j->col()) {
        colB  = b->col();
        acolB = j->acol();
      }
    }
  }
  else {
    // Initial-state gluon splitting.
    if (clus.antFunType == QXsplitII || clus.antFunType == QXsplitIF) {
      colA = jIsAntiQ ? a->col() : 0;
      acolA = jIsAntiQ ? 0 : a->acol();
      colB = b->col();
      acolB = b->acol();
    }
    // Quark conversion.
    else if (clus.antFunType == GXconvII || clus.antFunType == GXconvIF) {
      // Find out whether children have been swapped to not have "emissions
      // into the initial state", cf. setClusterlist in VinciaHistory.
      if (j->id() == a->id()
        && !a->isFinal() ) { // && !ajColCon) {
        colA = jIsAntiQ ? j->acol() : a->col();
        acolA = jIsAntiQ ? a->acol() : j->col();
        colB = b->col();
        acolB = b->acol();
      }
      else if (j->id() == b->id()
        && !b->isFinal() ) { // && jbColCon) {
        colB = jIsAntiQ ? j->acol() : b->col();
        acolB = jIsAntiQ ? b->acol() : j->col();
        colA = a->col();
        acolA = a->acol();
      }
      else {
        if (verbose >= REPORT) {
          string msg = ": Colour of parents couldn't be assigned";
          msg += " in quark conversion clustering.";
          infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
        }
        return false;
      }
    }
    // Final-state gluon splitting.
    else if (clus.antFunType == XGsplitIF) {
      colA = a->col();
      acolA = a->acol();
      colB = jIsAntiQ ? b->col() : j->col();
      acolB = jIsAntiQ ? j->acol() : b->acol();
    }
    // Gluon emission.
    else {
      // Choose to annihilate colour pair of child1 and child2.
      colB  = b->col();
      acolB = b->acol();
      // Child1 is definitely initial and child2 final.
      if (a->col() == j->col()) {
        colA  = j->acol();
        acolA = a->acol();
      }
      else if (a->acol() == j->acol()) {
        colA  = a->col();
        acolA = j->col();
      }
      else {
        if (verbose >= REPORT) {
          string msg = ": Colour of parents couldn't be assigned";
          msg += " in gluon emission clustering.";
          infoPtr->errorMsg("Error in "+__METHOD_NAME__,msg);
        }
        return false;
      }
    }
  }

  colsA = make_pair(colA, acolA);
  colsB = make_pair(colB, acolB);

  return true;
}

//--------------------------------------------------------------------------

// Get clustered momenta in 3->2 clustering.
// Note: iOffset shifts the indices stored in the VinciaClustering object.
//       Is needed when these point to an event record.

bool VinciaCommon::getMomenta3to2(vector<Vec4>& momNow,
  vector<Vec4>& momClus, const VinciaClustering& clus, int iOffset) {

  momClus.clear();

  // Fetch indices.
  int ia = clus.child1-iOffset;
  int ij = clus.child2-iOffset;
  int ib = clus.child3-iOffset;

  // Fetch masses (for clarity).
  double mj = clus.massesChildren.at(1);
  double mk = clus.massesChildren.at(2);
  double mA = clus.massesMothers.at(0);
  double mB = clus.massesMothers.at(1);

  // Cluster momenta according to antenna function.
  if (clus.isFSR) {
    // Use antenna index to identify resonance-final branchings.
    if (clus.antFunType >= QQemitRF) {
      if (!map3to2RF(momClus, momNow, ia, ij, ib, mB))
        return false;
    }
    // Otherwise final-final branching.
    else {
      if (!map3to2FF(momClus, momNow, clus.kMapType, ia, ij, ib, mA, mB))
        return false;
    }
  }
  else {
    // Use antenna index to identify initial-final branchings.
    if (clus.antFunType >= QQemitIF) {
      if (!map3to2IF(momClus, momNow, ia, ij, ib, mj, mk, mB))
        return false;
    }
    // Otherwise initial-initial branching.
    else {
      if (!map3to2II(momClus, momNow, true, ia, ij, ib, mj))
        return false;
    }
  }

  return true;
}

//--------------------------------------------------------------------------

// Find all possible clusterings for given configuration (optionally while
// retaining a Born configuration).

vector<VinciaClustering> VinciaCommon::findClusterings(vector<Particle>& state,
  map<int, int> nFlavsBorn) {

  // Check if we have sufficient information about the flavours in Born.
  if (nFlavsBorn.size() < 12) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__, "Will not resolve Born.");
    }
    for (int i(-6); i<=6; ++i) {
      if (i == 0) nFlavsBorn[21] = 0;
      nFlavsBorn[i] = 0;
    }
  }

  // Vector of all clusterings.
  vector<VinciaClustering> allClusterings;

  // Initialise information about state.
  map<int,int> col2ind;
  map<int,int> acol2ind;
  map<int, vector<int>> flav2inds;
  vector<int> hels;
  map<int, int> nFlavs;
  for (int i(-6); i<=6; ++i) {
    if (i == 0) nFlavs[21] = 0;
    nFlavs[i] = 0;
  }
  for (int iPtcl(0); iPtcl<(int)state.size(); ++iPtcl) {
    Particle thisPtcl = state[iPtcl];
    if (!thisPtcl.isFinal()) {
      // Initial-state colour treated as anticolour.
      if (thisPtcl.col()!=0)
        acol2ind[thisPtcl.col()] = iPtcl;
      if (thisPtcl.acol()!=0)
        col2ind[thisPtcl.acol()] = iPtcl;
      // Helicity flipped for initial state.
      hels.push_back(-thisPtcl.pol());
      // Initial-state quark treated as antiquark (and vice versa).
      if (thisPtcl.isQuark()) {
        flav2inds[-thisPtcl.id()].push_back(iPtcl);
        nFlavs[-thisPtcl.id()]++;
      } else if (thisPtcl.isGluon()) nFlavs[21]++;
    } else {
      if (thisPtcl.col()!=0)
        col2ind[thisPtcl.col()] = iPtcl;
      if (thisPtcl.acol()!=0)
        acol2ind[thisPtcl.acol()] = iPtcl;
      hels.push_back(thisPtcl.pol());
      if (thisPtcl.isQuark()) {
        flav2inds[thisPtcl.id()].push_back(iPtcl);
        nFlavs[thisPtcl.id()]++;
      } else if (thisPtcl.isGluon()) nFlavs[21]++;
    }
  }

  // Loop through all 3 -> 2 clusterings.
  for (int ij(0); ij<(int)state.size(); ++ij) {
    // Gluon emission.
    if (state[ij].isGluon() && state[ij].isFinal()) {
      // Check if we are allowed to cluster this gluon.
      if (nFlavs[21] <= nFlavsBorn[21]) continue;

      VinciaClustering thisClus;

      // Get colour-connected partners.
      int ia = col2ind[state[ij].acol()];
      int ib = acol2ind[state[ij].col()];
      thisClus.setChildren(state, ia, ij, ib);
      thisClus.setMothers(state[ia].id(), state[ib].id());

      // Get antenna function information.
      bool aIsGluon = state[ia].isGluon();
      bool bIsGluon = state[ib].isGluon();
      bool aIsFinal = state[ia].isFinal();
      bool bIsFinal = state[ib].isFinal();
      bool isFSR;
      enum AntFunType antFunType = NoFun;
      // Final-final.
      if (aIsFinal && bIsFinal) {
        isFSR = true;
        if (aIsGluon && bIsGluon) antFunType = GGemitFF;
        else if (aIsGluon && !bIsGluon) antFunType = GQemitFF;
        else if (!aIsGluon && bIsGluon) antFunType = QGemitFF;
        else antFunType = QQemitFF;
      }
      // Resonance-final.
      else if (state[ia].isResonance() && bIsFinal) {
        isFSR = true;
        if (bIsGluon) antFunType = QGemitRF;
        else antFunType = QQemitRF;
      }
      // Final-resonance: swap ia and ib.
      else if (aIsFinal && state[ib].isResonance()) {
        thisClus.swap13();
        isFSR = true;
        if (bIsGluon) antFunType = QGemitRF;
        else antFunType = QQemitRF;
      }
      // Initial-final.
      else if (!aIsFinal && bIsFinal) {
        isFSR = false;
        if (aIsGluon && bIsGluon) antFunType = GGemitIF;
        else if (aIsGluon && !bIsGluon) antFunType = GQemitIF;
        else if (!aIsGluon && bIsGluon) antFunType = QGemitIF;
        else antFunType = QQemitIF;
      }
      // Final-initial: swap ia and ib.
      else if (aIsFinal && !bIsFinal) {
        thisClus.swap13();
        swap(aIsGluon,bIsGluon);
        isFSR = false;
        if (aIsGluon && bIsGluon) antFunType = GGemitIF;
        else if (aIsGluon && !bIsGluon) antFunType = GQemitIF;
        else if (!aIsGluon && bIsGluon) antFunType = QGemitIF;
        else antFunType = QQemitIF;
      }
      // Initial-initial.
      else {
        isFSR = false;
        if (aIsGluon && bIsGluon) antFunType = GGemitII;
        else if (aIsGluon && !bIsGluon) antFunType = GQemitII;
        else if (!aIsGluon && bIsGluon) {
          thisClus.swap13();
          antFunType = GQemitII;
        }
        else antFunType = QQemitII;
      }

      // Save clustering to list of all clusterings.
      thisClus.setAntenna(isFSR, antFunType);
      if (!thisClus.initInvariantAndMassVecs()) {
        stringstream ss;
        ss << ": Couldn't initialise invariants and masses for "
           << thisClus.getAntName() << ". Skip clustering.";
        if (verbose >= DEBUG)
          printOut(__METHOD_NAME__, ss.str());
      }
      else {
        allClusterings.push_back(thisClus);
      }
    }

    // For quarks, distinguish different cases.
    else if (state[ij].isQuark()) {
      // If initial-state quark: only conversion possible.
      if (!state[ij].isFinal()) {
        // Find colour-connected partner of j.
        int ib = state[ij].id() > 0 ? col2ind[state[ij].col()]
          : acol2ind[state[ij].acol()];

        // Loop over all same-flavour (anti)quarks.
        vector<int> aList = flav2inds[state[ij].id()];
        for (int ifa = 0; ifa < (int)aList.size(); ++ifa) {
          // Get flavour partner.
          int ia = aList[ifa];

          // Check that we do not try to cluster the colour partner.
          if (ia == ib) continue;

          // Check if we are allowed to cluster this flavour.
          if (nFlavs[state[ia].id()] <= nFlavsBorn[state[ia].id()]) continue;

          // Get antenna information.
          bool isFSR = false;
          enum AntFunType antFunType =
            state[ib].isFinal() ? GXconvIF : GXconvII;

          // Append clustering to list.
          VinciaClustering thisClus;
          thisClus.setChildren(state,ij,ia,ib);
          thisClus.setMothers(21,state[ib].id());
          thisClus.setAntenna(isFSR, antFunType);
          if (!thisClus.initInvariantAndMassVecs()) {
            stringstream ss;
            ss << ": Couldn't initialise invariants and masses for "
               << thisClus.getAntName() << ". Skip clustering.";
            if (verbose >= DEBUG)
              printOut(__METHOD_NAME__, ss.str());
          }
          else {
            allClusterings.push_back(thisClus);
          }
        }

        // We do not consider any "other" emission into the initial state.
        continue;
      }

      // From here on, j is guaranteed to be in the final state.

      // Find colour-connected partner of j.
      int ib = ((state[ij].id() > 0) ? acol2ind[state[ij].col()]
        : col2ind[state[ij].acol()]);

      // Gluon splitting in the initial state.
      // Note: this does not reduce the total number of quarks!
      if (state[ib].isGluon() && !state[ib].isFinal()) {
        // The gluon now takes the role of a.
        int ia = ib;
        // The recoiler is the other parton colour-connected to the gluon.
        int iRec = (state[ij].id() > 0) ? acol2ind[state[ia].acol()]
          : col2ind[state[ia].col()];

        // Get antenna information.
        bool isFSR = false;
        enum AntFunType antFunType =
          state[iRec].isFinal() ? QXsplitIF : QXsplitII;

        // Append clustering to list.
        VinciaClustering thisClus;
        thisClus.setChildren(state,ia,ij,iRec);
        thisClus.setMothers(-state[ij].id(),state[iRec].id());
        thisClus.setAntenna(isFSR,antFunType);
        if (!thisClus.initInvariantAndMassVecs()) {
          stringstream ss;
          ss << ": Couldn't initialise invariants and masses for "
             << thisClus.getAntName() << ". Skip clustering.";
          if (verbose >= DEBUG)
            printOut(__METHOD_NAME__,ss.str());
        }
        else {
          allClusterings.push_back(thisClus);
        }
      }

      // From here on, we would reduce the number of quarks, so
      // check if we are allowed to cluster this flavour.
      if (nFlavs[state[ij].id()] <= nFlavsBorn[state[ij].id()]) continue;

      // Loop over all same-flavour (anti)quarks.
      vector<int> aList;
      aList = flav2inds[-state[ij].id()];
      for (int ifa = 0; ifa < (int)aList.size(); ++ifa) {
        // Get flavour partner.
        int ia = aList[ifa];

        // Check that we do not try to cluster the colour partner.
        if (ia == ib) continue;

        // Get antenna information.
        bool isFSR;
        enum AntFunType antFunType = NoFun;
        int  idA, idB;
        // Gluon splitting in the final state.
        if (state[ia].isFinal()) {
          // Final-final gluon splitting.
          if (state[ib].isFinal()) {
            isFSR = true;
            antFunType  = GXsplitFF;
            idA   = 21;
            idB   = state[ib].id();
          }
          // Final-resonance gluon splitting: swap ia and ib.
          else if (state[ib].isResonance()) {
            swap(ia,ib);
            isFSR = true;
            antFunType  = XGsplitRF;
            idA   = state[ia].id();
            idB   = 21;
          }
          // Final-initial gluon splitting: swap ia and ib.
          else {
            swap(ia,ib);
            isFSR = false;
            antFunType  = XGsplitIF;
            idA   = state[ia].id();
            idB   = 21;
          }
        }
        // Quark conversion (in the initial state).
        else {
          isFSR = false;
          idA   = 21;
          idB   = state[ib].id();
          // Initial-final quark conversion.
          if (state[ib].isFinal()) antFunType  = GXconvIF;
          // Initial-initial quark conversion.
          else antFunType = GXconvII;
        }

        // Append clustering to list.
        VinciaClustering thisClus;
        thisClus.setChildren(state,ia,ij,ib);
        thisClus.setMothers(idA,idB);
        thisClus.setAntenna(isFSR, antFunType);
        if (!thisClus.initInvariantAndMassVecs()) {
          stringstream ss;
          ss << ": Couldn't initialise invariants and masses for "
             << thisClus.getAntName() << ". Skip clustering.";
          if (verbose >= DEBUG)
            printOut(__METHOD_NAME__, ss.str());
        }
        else {
          allClusterings.push_back(thisClus);
        }
      }
    }
  }
  // Other clusterings (n -> m, QED, EW, ...) to be implemented here.

  // Summary.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Found " + std::to_string(allClusterings.size())
      + " sectors.");
    list(allClusterings, "", false);
    list(state, "Input State");
  }

  return allClusterings;
}

//--------------------------------------------------------------------------

// FF clustering map(s) for massless partons. Inputs are as follows:
//   kMapType = map number ( 1 : ARIADNE, 2 : PS, 3 : Kosower)
//   pIn      = Vec4 list (arbitrary length, but greater than 3)
//   a,r,b    = indices of 3 particles in pIn to be clustered
// Outputs are as follows:
//   pClu     = pIn but with the a and b momenta replaced by the clustered
//              aHat and bHat and with r erased.
// For example:
//   pIn(...,a,...,r-1,r,r+1,...,b,...) ->
//   pOut(...,aHat,...,r-1,r+1,...,bHat,...)
// with {a,r,b} -> {aHat,bHat} using kinematics map kMapType.

bool VinciaCommon::map3to2FFmassless(vector<Vec4>& pClu, vector<Vec4> pIn,
  int kMapType, int a, int r, int b) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Intialize and sanity check.
  pClu=pIn;
  if (max(max(a,r),b) > int(pIn.size()) || min(min(a,r),b) < 0) {
    if (verbose >= REPORT)
      printOut(__METHOD_NAME__,
        "Error! Unable to cluster (a,r,b) = "+
        num2str(a)+num2str(r)+num2str(b)+" p.size ="
        +num2str(int(pIn.size())));
    return false;
  }

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "called with ");
    cout << "pi = " << pIn[a];
    cout << "pj = " << pIn[r];
    cout << "pk = " << pIn[b];
  }

  // Compute total invariant mass squared.
  Vec4 pSum    = pIn[a] + pIn[r] + pIn[b];
  double m2Ant = pSum.m2Calc();
  if (m2Ant < 1e-20) {
    printOut(__METHOD_NAME__,
      "Massless or spacelike system. Cannot find rest frame");
    return false;
  }

  // ARIADNE and PS maps (recoded from old F77 maps for v.1.2.01)
  // (maps -1 and -2 are special: force A or B to take all recoil).
  if (kMapType == 1 || kMapType == 2 || kMapType == -1 || kMapType == -2) {

    // Make copies of PA, PB, and compute sum of LAB momenta and CM energy.
    Vec4 paDum   = pIn[a];
    Vec4 pbDum   = pIn[b];
    double eCM = sqrt(m2Ant);
    paDum.bstback(pSum);
    pbDum.bstback(pSum);

    // Rotate so a goes into upper half of (x,z) plane.
    double phiA = paDum.phi();
    paDum.rot(0.,-phiA);
    pbDum.rot(0.,-phiA);

    // Rotate so a goes onto z axis.
    double theta = paDum.theta();
    pbDum.rot(-theta,0.);

    // Rotate so (r,b) go into (x,z) plane.
    double phiB = pbDum.phi();

    // Compute psi(a,ahat) from A, B energies and theta(AB).
    double thetaAB = pbDum.theta();
    double psi = 0.0;

    // ARIADNE angle (smoothly varying antenna recoil).
    if (kMapType == 1) {
      psi = pbDum.e()*pbDum.e()/(paDum.e()*paDum.e() + pbDum.e()*pbDum.e())
        * (M_PI - thetaAB);

    // PS angle (direction a fixed if sar > srb, and vice versa). Use
    // org momenta, since new ones may not all have been rotated.
    } else if (kMapType == 2) {
      Vec4 pAR = pIn[a] + pIn[r];
      Vec4 pRB = pIn[r] + pIn[b];
      double sar = pAR.m2Calc();
      double srb = pRB.m2Calc();
      if (sar > srb) psi = 0.0;
      else psi = (M_PI - thetaAB);

    // Force A to be the emitter. B recoils longitudinally.
    } else if (kMapType == -1) {
      psi = M_PI - thetaAB;

    // Force B to be the emitter. A recoils longitudinally.
    } else if (kMapType == -2) {
      psi = 0.0;
    }
    // Now we know everything:
    // CM -> LAB : -PSI, PHIB, THETA, PHIA, BOOST

    // Set up initial massless AHAT, BHAT with AHAT along z axis.
    pClu[a] = Vec4(0.,0.,eCM/2.,eCM/2.);
    pClu[b] = Vec4(0.,0.,-eCM/2.,eCM/2.);

    // 1) Take into account antenna recoil, and rotate back in phiB.
    pClu[a].rot(-psi,phiB);
    pClu[b].rot(-psi,phiB);

    // 2) Rotate back in theta and phiA.
    pClu[a].rot(theta,phiA);
    pClu[b].rot(theta,phiA);

    // 3) Boost back to LAB.
    pClu[a].bst(pSum);
    pClu[b].bst(pSum);

  // kMapType = 3, 4 (and catchall for undefined kMapType
  // values). Implementation of the massless Kosower antenna map(s).
  } else {

    // Compute invariants.
    double s01  = 2*pIn[a]*pIn[r];
    double s12  = 2*pIn[r]*pIn[b];
    double s02  = 2*pIn[a]*pIn[b];

    // Check whether the arguments need to be reversed for kMapType == 4.
    if (kMapType == 4 && ( ! (s01 < s12) ) ) {
      if (verbose >= DEBUG) {
        printOut(__METHOD_NAME__,
          "choose parton i as the recoiler");
      }
      // Call the function with reverse arguments, then return.
      return map3to2FFmassless(pClu, pIn, kMapType, b, r, a);
    }
    double sAnt = s01 + s12 + s02;

    // Compute coefficients.
    //  kMapType  = 3: GGG choice
    //  kMapType >= 4: r=1
    double rMap = kMapType == 3 ? s12/(s01 + s12) : 1;
    double rho  = sqrt(1.0+(4*rMap*(1.0-rMap)*s01*s12)/sAnt/s02);
    double x    = 0.5/(s01+s02)*((1+rho)*(s01+s02)+(1+rho-2*rMap)*s12);
    double z    = 0.5/(s12+s02)*((1-rho)*sAnt-2*rMap*s01);

    // Compute reclustered momenta.
    pClu[a]     =     x*pIn[a] +     rMap*pIn[r] +     z*pIn[b];
    pClu[b]     = (1-x)*pIn[a] + (1-rMap)*pIn[r] + (1-z)*pIn[b];
  }

  // A dimensionless quantitiy to compare with precision target.
  // Note: allow bigger difference for events from Les Houches files.
  double PREC = infoPtr->isLHA() ? DECI : NANO;
  if (pClu[a].m2Calc()/m2Ant >= PREC || pClu[b].m2Calc()/m2Ant >= PREC) {
    if (verbose >= REPORT)
      printOut(__METHOD_NAME__,
        "on-shell check failed. m2I/sIK ="
        + num2str(pClu[a].m2Calc()/m2Ant)+" m2K/m2Ant ="
        + num2str(pClu[b].m2Calc()/m2Ant)+" m2Ant = "+num2str(m2Ant));
    return false;
  }

  // Erase the clustered momentum and return.
  pClu.erase(pClu.begin()+r);
  return true;

}

//--------------------------------------------------------------------------

// Implementations of FF clustering maps for massive partons. See
// VinciaCommon::map3to2FFmassless for details but with the additional
// input of masses mI and mK for the first and second parent
// particles.

bool VinciaCommon::map3to2FFmassive(vector<Vec4>& pClu, vector<Vec4> pIn,
  int kMapType, double mI, double mK, int a, int r, int b) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // If both parent masses are negligible and all the momenta are
  // measure off-shellness normalised to average energy of the partons
  // to be clustered, avoids small denominator for collinear massless
  // p_a, p_r, p_b.
  double eAv = 1.0/3.0*( pIn[a].e() + pIn[r].e() + pIn[b].e() );
  if (mI/eAv < NANO && mK/eAv < NANO && pIn[a].mCalc()/eAv < NANO
    && pIn[r].mCalc()/eAv < NANO && pIn[b].mCalc()/eAv < NANO ) {
    return map3to2FFmassless(pClu, pIn, kMapType, a, r, b);
  }

  // Intialize and sanity check.
  pClu = pIn;
  if (max(max(a,r),b) > int(pIn.size()) || min(min(a,r),b) < 0) return false;

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, "called with ");
    cout << "p0 = " << pIn[a];
    cout << "p1 = " << pIn[r];
    cout << "p2 = " << pIn[b];
  }

  // ARIADNE map not defined for massive partons; use Kosower map instead.
  if (kMapType == 1) kMapType = 3;

  // Longitudinal map; use Kosower map with r = 1.
  if (kMapType == 2) kMapType = 4;

  // Forced-longitudinal maps.
  if (kMapType < 0) {
    printOut(__METHOD_NAME__, "longitudinal clustering maps "
      "not yet implemented for massive partons.");
    return false;
  }

  // Compute invariants.
  double m0   = pIn[a].mCalc();
  double m1   = pIn[r].mCalc();
  double m2   = pIn[b].mCalc();
  double s01  = 2*pIn[a]*pIn[r];
  double s12  = 2*pIn[r]*pIn[b];
  double s02  = 2*pIn[a]*pIn[b];

  // Check whether the arguments need to be reversed for mapType == 4.
  if (kMapType == 4 && (! (s01 < s12) ) ) {
    // Call the function with reverse arguments, then return.
    return map3to2FFmassive(pClu, pIn, kMapType, b, r, a, mK, mI);
  }
  Vec4   pAnt    = pIn[a] + pIn[r] + pIn[b];
  double m2Ant   = pAnt.m2Calc();
  double mAnt    = sqrt(m2Ant);

  // Rewrite the determination in terms of dimensionless variables.
  // Note normalisation choice here is mAnt, rather than sAnt.
  double mu0 = m0/mAnt;
  double mu1 = m1/mAnt;
  double mu2 = m2/mAnt;
  double y01 = s01/m2Ant;
  double y12 = s12/m2Ant;
  double y02 = s02/m2Ant;
  double y01min = 2*mu0*mu1;
  double y12min = 2*mu1*mu2;
  double y02min = 2*mu0*mu2;
  double muI = mI/mAnt;
  double muK = mK/mAnt;
  double yIK = 1. - pow2(muI) - pow2(muK);
  double yIKmin = 2*muI*muK;
  double sig2 = 1 + pow2(muI) - pow2(muK);
  double gdetdimless = gramDet(y01,y12,y02,mu0,mu1,mu2);
  double gdetdimless01 = (y02*y12-2.*pow2(mu2)*y01)/4.;
  double gdetdimless12 = (y02*y01-2.*pow2(mu0)*y12)/4.;
  double rMap = 1.;
  if ( kMapType == 3) {
    rMap =
      (
        sig2
        + sqrt( pow2(yIK) - pow2(yIKmin) )
        *( y12 - y12min - ( y01 - y01min ) )
        /( y12 - y12min + ( y01 - y01min ) )
       )/2.;

  // Fallback: map with massless r -> 1.
  } else  {
    double mu01squa = pow2(mu0) + pow2(mu1) + y01;
    double lambda = 1 + pow2(mu01squa) + pow4(mu2) - 2*mu01squa - 2*pow2(mu2)
      - 2*mu01squa*pow2(mu2);
    rMap = (sig2 + sqrt(pow2(yIK) - pow2(yIKmin))
            *(1 - pow2(mu0) - pow2(mu1) + pow2(mu2) - y01)/sqrt(lambda))/2.;
  }

  // Compute reclustered momenta.
  double bigsqr = sqrt(
    16.*( rMap*(1.-rMap) - (1.-rMap)*pow2(muI) - rMap*pow2(muK) )*gdetdimless
    + ( pow2(y02) - pow2(y02min) )*( pow2(yIK) - pow2(yIKmin) ));
  double x = (
    sig2*( pow2(y02) - pow2(y02min) + 4.*gdetdimless01)
    + 8.*rMap*( gdetdimless - gdetdimless01 )
    + bigsqr*( 1. - pow2(mu0) - pow2(mu1) + pow2(mu2) - y01)
              )/(2.*( 4.*gdetdimless + pow2(y02) - pow2(y02min) ));
  double z = (
    sig2*( pow2(y02) - pow2(y02min) + 4.*gdetdimless12)
    + 8.*rMap*( gdetdimless - gdetdimless12 )
    - bigsqr*( 1. + pow2(mu0) - pow2(mu1) - pow2(mu2) - y12)
              )/(2.*( 4.*gdetdimless + pow2(y02) - pow2(y02min)));
  pClu[a] =     x*pIn[a] +     rMap*pIn[r] +     z*pIn[b];
  pClu[b] = (1-x)*pIn[a] + (1-rMap)*pIn[r] + (1-z)*pIn[b];

  // Check if on-shell.
  double offshellnessI = abs(pClu[a].m2Calc() - pow2(mI))/m2Ant;
  double offshellnessK = abs(pClu[b].m2Calc() - pow2(mK))/m2Ant;
  if (offshellnessI > NANO || offshellnessK > NANO) {
    if (verbose >= REPORT) {
      printOut(__METHOD_NAME__,"on-shell check failed");
    }
    return false;
  }

  // Erase the clustered parton and return.
  pClu.erase(pClu.begin()+r);
  return true;

}

//--------------------------------------------------------------------------

// Inverse kinematic map for the resonance-final antenna.

bool VinciaCommon::map3to2RF(vector<Vec4>& pClu, vector<Vec4>& pIn,
  int a, int r, int b, double mK){

  // 1) Extract momenta.
  if (pIn.size()<3) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Not enough input momenta.");
    return false;
  }

  // Momenta to cluster.
  Vec4 pA = pIn.at(a);
  Vec4 pj = pIn.at(r);
  Vec4 pk = pIn.at(b);

  // Get pX'.
  Vec4 pXBeforeClus = pA - pj - pk;

  // 2) Momenta are completely pre-determined by the masses.
  // Take mK as input - may change if e.g. clustering a splitting.
  double mA  = pA.mCalc();
  double mX  = pXBeforeClus.mCalc();
  double pXz = sqrt(kallenFunction( mA*mA , mX*mX , mK*mK ))/(2.*mA);
  double EX  = sqrt(pXz*pXz + mX*mX);
  double EK  = sqrt(pXz*pXz + mK*mK);

  // 3) Boost to Top Centre of Mass frame.
  Vec4 pXBeforeInCoM = pXBeforeClus;
  pXBeforeInCoM.bstback(pA);
  // pX along z.
  Vec4 pX(0. ,0., pXz, EX);
  Vec4 pK(0. ,0. ,-pXz, EK);
  // 4) Fetch the orientation of X' and rotate X
  double phiX = pXBeforeInCoM.phi();
  double thetaX = pXBeforeInCoM.theta();
  pX.rot(thetaX,phiX);
  pK.rot(thetaX,phiX);

  // 5) Boost back to lab frame.
  pX.bst(pA);
  pK.bst(pA);

  Vec4 diff = pA - pK - pX;
  if( abs(diff.e())  / abs(pA.e()) > MILLI ||
    abs(diff.px()) / abs(pA.e()) > MILLI ||
    abs(diff.py()) / abs(pA.e()) > MILLI ||
    abs(diff.pz()) / abs(pA.e()) > MILLI) {
    string msg=": Failed momentum conservation test.";
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+msg);
    return false;
  }

  // 6) Boost recoilers and save clustered momenta.
  Vec4 recSumAfter(0., 0., 0., 0.);
  pClu.clear();
  for (int ip(0); ip<(int)pIn.size(); ++ip) {
    if (ip == r) continue;
    if (ip == a) pClu.push_back(pA);
    else if (ip == b) pClu.push_back(pK);
    else {
      Vec4 pRecNow=pIn.at(ip);
      pRecNow.bst(pX-pXBeforeClus);
      pClu.push_back(pRecNow);
      recSumAfter+=pRecNow;
    }
  }

  // Check sum.
  recSumAfter-=pX;
  if( abs(recSumAfter.e())  / abs(pX.e()) > MILLI ||
    abs(recSumAfter.px()) / abs(pX.e()) > MILLI ||
    abs(recSumAfter.py()) / abs(pX.e()) > MILLI ||
    abs(recSumAfter.pz()) / abs(pX.e()) > MILLI) {
    if(verbose >= NORMAL){
      string msg="Recoilers failed momentum conservation. Violation:";
      printOut(__METHOD_NAME__,msg);
      cout << "  " << num2str(abs(recSumAfter.e() / abs(pX.e())), 6) << endl;
      cout << "  " << num2str(abs(recSumAfter.px() / abs(pX.px())), 6) << endl;
      cout << "  " << num2str(abs(recSumAfter.py() / abs(pX.py())), 6) << endl;
      cout << "  " << num2str(abs(recSumAfter.pz() / abs(pX.pz())), 6) << endl;
    }
    return false;
  }

  return true;
}

//--------------------------------------------------------------------------

// Implementations of IF clustering maps for massive partons.
// NOTE: particle A and a are assumed massless (no initial-state masses).

bool VinciaCommon::map3to2IF(vector<Vec4>& pClu, vector<Vec4>& pIn,
  int a, int r, int b, double mj, double mk, double mK) {

  // Initialise and sanity check.
  pClu = pIn;
  if (max(max(a,r),b) > int(pIn.size()) || min(min(a,r),b) < 0) return false;

  // Save momenta for clustering.
  Vec4 pa = pIn[a];
  Vec4 pr = pIn[r];
  Vec4 pb = pIn[b];

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, " called with ");
    cout << "  pa = " << pa;
    cout << "  pr = " << pr;
    cout << "  pb = " << pb;
    cout << "  mK = " << mK << endl;
  }

  // Calculate invariants.
  double saj = 2. * pa * pr;
  double sjk = 2. * pr * pb;
  double sak = 2. * pa * pb;
  double m2K = (mK <= NANO ? 0. : pow2(mK));
  double m2j = (mj <= NANO ? 0. : pow2(mj));
  double m2k = (mk <= NANO ? 0. : pow2(mk));
  double sAK = saj + sak - sjk + m2K - m2j - m2k;

  Vec4 pA = pa;
  pA.rescale4(sAK/(saj + sak));
  Vec4 pK = pA - pa + pr + pb;

  pClu[a] = pA;
  pClu[b] = pK;
  pClu.erase(pClu.begin()+r);

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, " after clustering: ");
    cout << "  pA = " << pA;
    cout << "  pK = " << pK;
  }

  Vec4 pTot(0.,0.,0.,0.);
  for (const auto& p : pIn) pTot += p;
  for (const auto& p : pClu) pTot -= p;
  double m2tot = pTot.m2Calc();
  if (m2tot >= MILLI) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Momentum not conserved (m2tot = "+num2str(m2tot)+")");
    return false;
  }

  return true;

}

//--------------------------------------------------------------------------

// Implementations of II clustering maps for massive partons.
// NOTE: particle A, a, B, and b are assumed massless.

bool VinciaCommon::map3to2II(vector<Vec4>& pClu, vector<Vec4>& pIn,
  bool doBoost, int a, int r, int b, double mj) {

  // Initialisation and sanity check.
  pClu = pIn;
  if (max(max(a,r),b) > int(pIn.size()) || min(min(a,r),b) < 0) return false;

  // Save momenta for clustering.
  Vec4 pa = pIn[a];
  Vec4 pr = pIn[r];
  Vec4 pb = pIn[b];

  // Verbose output.
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__, " called with ");
    cout << "\tpa = " << pa;
    cout << "\tpr = " << pr;
    cout << "\tpb = " << pb;
    cout << "\tand " << pIn.size() - 3 << " recoilers." << endl;
  }

  // Calculate invariants.
  double saj = 2. * pa * pr;
  double sjb = 2. * pr * pb;
  double sab = 2. * pa * pb;
  double m2j = (mj <= NANO ? 0. : pow2(mj));
  double sAB = sab - saj - sjb + m2j;

  // Scale factors and momenta.
  double rescaleFacA = 1./sqrt(sab/sAB * (sab - saj)/(sab - sjb));
  double rescaleFacB = 1./sqrt(sab/sAB * (sab - sjb)/(sab - saj));
  Vec4 pA = pa;
  pA.rescale4(rescaleFacA);
  Vec4 pB = pb;
  pB.rescale4(rescaleFacB);
  Vec4 pInSum = pa + pb - pr;
  Vec4 pCluSum = pA + pB;
  // Clustered momenta and recoilers.
  pClu[a] = pA;
  pClu[b] = pB;
  // Perform boost - if doBoost, we boost back to the lab frame.
  // Adjust recoiling momenta.
  if (doBoost) {
    Vec4 pSum(0., 0., 0., 0.);
    for (int i(0); i<(int)pClu.size(); ++i) {
      if (i != a && i != r && i != b) {
        pClu[i].bstback(pInSum);
        pClu[i].bst(pCluSum);
        pSum += pClu[i];
      }
    }
  }
  // Otherwise stay in the current frame. Adjust clustered momenta.
  else {
    for (int i=0; i<(int)pClu.size(); ++i) {
      if (i == a || i == b) {
        pClu[i].bstback(pCluSum);
        pClu[i].bst(pInSum);
      }
    }
  }
  pClu.erase(pClu.begin()+r);

  return true;

}

//--------------------------------------------------------------------------

// 2->3 branching kinematics: output arguments first, then input
// arguments.  The output is p[i,j,k] whil the inputs are p[I,K],
// kMapType, inviariants[sIK,sij,sjk], phi, and
// masses[mi,mj,mk]. Note, sab defined as 2*pa.pb.

bool VinciaCommon::map2to3FFmassive(vector<Vec4>& pThree,
  const vector<Vec4>& pTwo, int kMapType, const vector<double>& invariants,
  double phi, vector<double> masses) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Check for ultrarelativistic particles.
  // Gluon/Photon splitting assumed to happen on legs 01 with 2 as recoiler.
  if (masses.size() >= 3) {
    if (masses[0] > 0. && masses[0]/pTwo[0].e() < MICRO) masses[0] = 0.;
    if (masses[1] > 0. && masses[1]/pTwo[0].e() < MICRO) masses[1] = 0.;
    if (masses[2] > 0. && masses[2]/pTwo[1].e() < MICRO) masses[2] = 0.;
  }

  // Hand off to massless map if all masses < 1 keV.
  if (masses.size() < 3 ||
    (masses[0] <= MICRO && masses[1] <= MICRO && masses[2] <= MICRO))
    return map2to3FFmassless(pThree,pTwo,kMapType,invariants,phi);

  // Antenna invariant mass and sIK = 2*pI.pK.
  double m2Ant = m2(pTwo[0],pTwo[1]);
  double mAnt  = sqrt(m2Ant);
  double sAnt  = invariants[0];
  // Recompute sAnt if needed.
  if (sAnt < 0.0 || std::isnan(sAnt)) {
    double m2I = max(0.0,m2(pTwo[0]));
    double m2K = max(0.0,m2(pTwo[1]));
    sAnt = m2Ant - m2I - m2K;
  }
  if (sAnt <= 0.0) return false;

  // Masses and invariants
  double mass0 = max(0.,masses[0]);
  double mass1 = max(0.,masses[1]);
  double mass2 = max(0.,masses[2]);
  // Check for totally closed phase space. Should normally have
  // happened before generation of invariants but put last-resort
  // check here since not caught by Gram determinant.
  if (mAnt < mass0 + mass1 + mass2) {
    if (verbose >= REPORT) {
      stringstream ssDau; ssDau<<(mass0+mass1+mass2);
      stringstream ssAnt; ssAnt<<mAnt;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": no on-shell solution",
        "(antenna mass = " + ssAnt.str() + " GeV < sum of daughter masses = "
        + ssDau.str() + "GeV)");
    }
    return false;
  }
  double s01 = max(0.,invariants[1]);
  double s12 = max(0.,invariants[2]);
  double s02 = m2Ant - s01 - s12 - pow2(mass0) - pow2(mass1) - pow2(mass2);
  if (s02 <= 0.) return false;

  // Check whether we are inside massive phase space.
  double gDet = gramDet(s01, s12, s02, mass0, mass1, mass2);
  if (gDet <= 0.) {
    if (verbose >= DEBUG)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": Failed massive phase space check.");
    return false;
  }

  // Verbose output.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "mAnt =" << num2str(mAnt)
       << "   sqrts(IK,ij,jk,ik) ="
       << num2str(sqrt(sAnt)) << " " << num2str(sqrt(s01)) << " "
       << num2str(sqrt(s12)) << " " << num2str(sqrt(s02))
       << "   m(i,j,k) =" << num2str(mass0) << " "
       << num2str(mass1) << " " << num2str(mass2) << " D = " << gDet;
    printOut(__METHOD_NAME__,ss.str());
    RotBstMatrix M;
    Vec4 p1cm = pTwo[0];
    Vec4 p2cm = pTwo[1];
    M.toCMframe(p1cm,p2cm);
    p1cm.rotbst(M);
    p2cm.rotbst(M);
    Vec4 tot = p1cm+p2cm;
    printOut(__METHOD_NAME__,"Pre-branching momenta in Antenna CM:");
    cout << "  p1cm = " << p1cm << "  p2cm = " << p2cm
         << "  total = " << tot;
  }

  // Set up kinematics in rest frame.
  double E0 = (pow2(mass0) + s01/2 + s02/2)/mAnt;
  double E1 = (pow2(mass1) + s12/2 + s01/2)/mAnt;
  double E2 = (pow2(mass2) + s02/2 + s12/2)/mAnt;

  // Make sure energies > masses (should normally be ensured by
  // combination of open phase space and positive Gram determinant).
  if (E0 < mass0 || E1 < mass1 || E2 < mass2) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": daughter energy value(s) < mass(es).");
    return false;
  }
  double ap0 = sqrt( pow2(E0) - pow2(mass0) );
  double ap1 = sqrt( pow2(E1) - pow2(mass1) );
  double ap2 = sqrt( pow2(E2) - pow2(mass2) );
  double cos01 = (E0*E1 - s01/2)/(ap0*ap1);
  double cos02 = (E0*E2 - s02/2)/(ap0*ap2);

  // Protection: num. precision loss for small (ultracollinear) invariants.
  if ( 1-abs(cos01) < 1e-15 ) cos01 = cos01 > 0 ? 1. : -1.;
  if ( 1-abs(cos02) < 1e-15 ) cos02 = cos02 > 0 ? 1. : -1.;

  // Use positive square root for sine.
  double sin01 = (abs(cos01) < 1) ? sqrt(abs(1.0 - pow2(cos01))) : 0.0;
  double sin02 = (abs(cos02) < 1) ? sqrt(abs(1.0 - pow2(cos02))) : 0.0;

  // Set momenta in CMz frame (frame with 1 oriented along positive z
  // axis and event in (x,z) plane).
  Vec4 p1(0.0,0.0,ap0,E0);
  Vec4 p2(-ap1*sin01,0.0,ap1*cos01,E1);
  Vec4 p3(ap2*sin02,0.0,ap2*cos02,E2);

  // Verbose output.
  if (verbose >= DEBUG) {
    Vec4 tot = p1+p2+p3;
    printOut(__METHOD_NAME__,
      "Post-branching momenta in CM* (def: 1 along z):");
    cout  << "  k1* =  " << p1 << "  k2* =  " << p2 << "  k3* =  " << p3
          << "  total = " << tot << endl;
  }

  // Choose global rotation around axis perpendicular to event plane.
  double psi;

  // kMapType = -2(-1): force A(B) to be purely longitudinal recoiler.
  if (kMapType == -2) psi = 0.0;
  else if (kMapType == -1) psi = M_PI - acos(cos02);
  // Else general antenna-like recoils.
  else {
    double m2I = max(0.0,m2(pTwo[0]));
    double m2K = max(0.0,m2(pTwo[1]));
    double sig2 = m2Ant + m2I - m2K;
    double sAntMin = 2*sqrt(m2I*m2K);
    double s01min = max(0.0,2*mass0*mass1);
    double s12min = max(0.0,2*mass1*mass2);
    double s02min = max(0.0,2*mass0*mass2);

    // The r and R parameters in arXiv:1108.6172.
    double rAntMap = ( sig2 + sqrt( pow2(sAnt) - pow2(sAntMin) )
      * ( s12-s12min - (s01-s01min) )
      / ( s01-s01min + s12-s12min ) ) / (2*m2Ant);
    double bigRantMap2 = 16*gDet * ( m2Ant*rAntMap * (1.-rAntMap)
      - (1.-rAntMap)*m2I - rAntMap*m2K )
      + ( pow2(s02) - pow2(s02min) )
      * ( pow2(sAnt) - pow2(sAntMin) );
    if (bigRantMap2 < 0.) {
      stringstream ss;
      ss<<"On line "<<__LINE__;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": kinematics map is broken."+ss.str());
      return false;
    }
    double bigRantMap = sqrt( bigRantMap2 );
    double p1dotpI = (sig2*(pow2(s02) - pow2(s02min))*
      (m2Ant + pow2(mass0) - pow2(mass1) - pow2(mass2) - s12)
      +8*rAntMap*(m2Ant + pow2(mass0) - pow2(mass1) - pow2(mass2) - s12)*gDet
      -bigRantMap*(pow2(s02) - pow2(s02min) + s01*s02-2*s12*pow2(mass0)))
      /(4*(4*gDet + m2Ant*(pow2(s02) - pow2(s02min))));

    // Norm of the three-momentum and the energy of the first parent
    // particle (could also be obtained by explicitly boosting
    // incoming particles to CM).
    double apInum2 = pow2(m2Ant) + pow2(m2I) + pow2(m2K) - 2*m2Ant*m2I
      - 2*m2Ant*m2K - 2*m2I*m2K;
    if (apInum2 < 0.) {
      stringstream ss;
      ss<<"On line "<<__LINE__;
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": kinematics map is broken.",ss.str());
      return false;
    }
    double apI = sqrt(apInum2)/(2*mAnt);
    double EI = sqrt( pow2(apI) + m2I );

    // Protect against rounding errors before taking acos.
    double cospsi = ((E0*EI) - p1dotpI)/(ap0*apI);
    if (cospsi >= 1.0) {
      psi = 0.;
    } else if (cospsi <= -1.0) {
      psi = M_PI;
    } else if(std::isnan(cospsi)){
      psi= 0.;
      infoPtr->errorMsg("Error in "+__METHOD_NAME__+": cos(psi) = nan.");
      if (verbose >= REPORT) {
        cout << " pIn0 = " << pTwo[0];
        cout << " pIn1 = " << pTwo[1];
        cout << " mAnt = " << mAnt << " sAnt = " << sAnt << " m2I = " << m2I
             << " m2K = " << m2K
             << " sum1 = " << pow2(sAnt) + pow2(m2I) + pow2(m2K)
             << " sum2 = " << 2*sAnt*m2I + 2*sAnt*m2K + 2*m2I*m2K << endl
             << "  ap0 = " << ap0 << " apI = " << apI << " E0 = " << E0
             << " m0 = " << mass0 << " m1 = " << mass1  << " m2 = "
             << mass2 << " s01 = " << s01 << " s02 = " << s02 << endl;
      }
      return false;
    }
    else{
      psi = acos( cospsi );
    }
  }

  // Perform global rotations.
  p1.rot(psi,phi);
  p2.rot(psi,phi);
  p3.rot(psi,phi);

  // Verbose output.
  if (verbose >= DEBUG) {
    Vec4 tot = p1+p2+p3;
    printOut(__METHOD_NAME__, "phi = " + num2str(phi,6)
             + " cospsi = " + num2str(cos(psi),6) + " psi = " + num2str(psi,6)
             + " mapType = " + num2str(kMapType));
    printOut(__METHOD_NAME__,"final post-branching momenta in Antenna CM:");
    cout << "  k1cm = " << p1 << "  k2cm = " << p2 << "  k3cm = " << p3
         << "  total = " << tot << endl;
  }

  // Check CM frame energy and momentum conservation
  // Dimensionless, normalised by ECM, so that it does not fail for small
  // differences of very large absolute momentum values (FCC, etc).
  // Note masses can still be quite different from mass[i].
  Vec4 pSumCM = (p1+p2+p3);
  pSumCM /= mAnt;
  if (abs(pSumCM.e() - 1) > MICRO || abs(pSumCM.px()) > MICRO
    || abs(pSumCM.py()) > MICRO || abs(pSumCM.pz()) > MICRO ){
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+
      ": Failed momentum conservation test. Aborting.");
    if (verbose >= REPORT) {
      cout << " pDiffCM/eCM = " << scientific << num2str(pSumCM.px()) << " "
           << num2str(pSumCM.py()) << " " << num2str(pSumCM.pz()) <<" "
           << num2str(pSumCM.e()-1.) << "    (" << num2str(pSumCM.mCalc()-1.)
           << ")" << endl;
    }
    infoPtr->setAbortPartonLevel(true);
    return false;
  }
  if (isnan(pSumCM)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": (E,p) = nan.");
    return false;
  }

  // Rotate and boost to lab frame.
  RotBstMatrix M;
  M.fromCMframe(pTwo[0],pTwo[1]);
  if (verbose >= DEBUG) {
    Vec4 tot = pTwo[0]+pTwo[1];
    printOut(__METHOD_NAME__,"boosting to LAB frame defined by:");
    cout << "  p1 =   " << pTwo[0] << "  p2 =   " << pTwo[1]
         << "  total = " << tot << endl;
  }
  p1.rotbst(M);
  p2.rotbst(M);
  p3.rotbst(M);
  if (verbose >= DEBUG) {
    Vec4 tot = p1+p2+p3;
    printOut(__METHOD_NAME__,"final post-branching momenta in LAB frame:");
    cout << "  k1 =   " << p1 << "  k2 =   " << p2 << "  k3 =   " << p3
         << "  total = " << tot << endl;
  }

  // Save momenta.
  pThree.resize(0);
  pThree.push_back(p1);
  pThree.push_back(p2);
  pThree.push_back(p3);

  // Return.
  return true;

}

//--------------------------------------------------------------------------

// 2->3 branching kinematics, massless with output arguments first,
// then input arguments. The output is p3, while the inputs are
// kMapType, invariants(sIK, s01, s12), and phi.

bool VinciaCommon::map2to3FFmassless(vector<Vec4>& pThree,
  const vector<Vec4>& pTwo, int kMapType, const vector<double>& invariants,
  double phi) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Antenna invariant mass.
  double m2Ant = m2(pTwo[0],pTwo[1]);
  double mAnt  = sqrt(m2Ant);

  // Compute invariants (massless relation).
  double s01 = invariants[1];
  double s12 = invariants[2];
  double s02 = m2Ant - s01 - s12;

  // Can check alternative hadronization vetos here.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "m  = " << num2str(mAnt)
         << "   m12 =" << num2str(sqrt(s01))
         << "   m23 =" << num2str(sqrt(s12))
         << "   m13 =" << num2str(sqrt(s02));
    printOut(__METHOD_NAME__,ss.str());
    RotBstMatrix M;
    Vec4 p1cm = pTwo[0];
    Vec4 p2cm = pTwo[1];
    M.toCMframe(p1cm,p2cm);
    p1cm.rotbst(M);
    p2cm.rotbst(M);
    Vec4 tot = p1cm+p2cm;
    printOut(__METHOD_NAME__,"Starting dipole in CM");
    cout << " p1cm = " << p1cm << " p2cm = " << p2cm
         << " total = " << tot;
  }

  // Set up kinematics in rest frame.
  double E0 = 1/mAnt*(s01/2 + s02/2);
  double E1 = 1/mAnt*(s01/2 + s12/2);
  double E2 = 1/mAnt*(s02/2 + s12/2);
  double ap0 = E0;
  double ap1 = E1;
  double ap2 = E2;
  double cos01 = (E0*E1 - s01/2)/(ap0*ap1);
  double cos02 = (E0*E2 - s02/2)/(ap0*ap2);

  // Protection: num. precision loss for small (ultracollinear) invariants.
  if ( 1-abs(cos01) < 1e-15 ) cos01 = cos01 > 0 ? 1. : -1.;
  if ( 1-abs(cos02) < 1e-15 ) cos02 = cos02 > 0 ? 1. : -1.;
  double sin01 = (abs(cos01) < 1) ? sqrt(abs(1.0 - pow2(cos01))) : 0.0;
  double sin02 = (abs(cos02) < 1) ? sqrt(abs(1.0 - pow2(cos02))) : 0.0;

  // Set momenta in CMz frame (with 1 oriented along positive z axis
  // and event in (x,z) plane).
  Vec4 p1(0.0,0.0,ap0,E0);
  Vec4 p2(-ap1*sin01,0.0,ap1*cos01,E1);
  Vec4 p3(ap2*sin02,0.0,ap2*cos02,E2);

  // Verbose output.
  if (verbose >= DEBUG) {
    Vec4 tot = p1+p2+p3;
    printOut(__METHOD_NAME__,"Configuration in CM* (def: 1 along z)");
    cout << " k1* =  " << p1 << " k2* =  " << p2 << " k3* =  " << p3
         << " total = " << tot;
  }

  // Choose global rotation around axis perpendicular to event plane.
  double psi;

  // Force A to be longitudinal recoiler.
  if (kMapType == -2) {
    psi = 0.0;

  // Force B to be longitudinal recoiler.
  } else if (kMapType == -1) {
    psi = M_PI - acos(max(-1.,min(1.,cos02)));

  // ARIADNE map.
  } else if (kMapType == 1) {
    psi = E2*E2/(E0*E0+E2*E2)*(M_PI-acos(cos02));

  // psi PYTHIA-like. "Recoiler" remains along z-axis.
  } else if (kMapType == 2) {
    psi = 0.;
    if (s01 < s12 || (s01 == s12 && rndmPtr->flat() > 0.5) )
      psi = M_PI-acos(cos02);

  // Kosower's map. Similar to ARIADNE.
  } else {
    double rMap(1);
    if (kMapType == 3) rMap = s12/(s01+s12);
    double rho=sqrt(1.0+4.0*rMap*(1.0-rMap)*s01*s12/s02/m2Ant);
    double s00=-( (1.0-rho)*m2Ant*s02 + 2.0*rMap*s01*s12 ) / 2.0 /
      (m2Ant - s01);
    psi=acos(1.0+2.0*s00/(m2Ant-s12));
  }

  // Perform global rotations.
  p1.rot(psi,phi);
  p2.rot(psi,phi);
  p3.rot(psi,phi);

  // Verbose output.
  if (verbose >= DEBUG) {
    Vec4 tot = p1+p2+p3;
    printOut(__METHOD_NAME__, "phi = " + num2str(phi,6) + "psi = " +
             num2str(psi,6));
    printOut(__METHOD_NAME__, "Final momenta in CM:");
    cout << " k1cm = " << p1 << " k2cm = " << p2 << " k3cm = " << p3
         << " total = " << tot;
  }

  // Rotate and boost to lab frame.
  RotBstMatrix M;
  M.fromCMframe(pTwo[0],pTwo[1]);
  Vec4 total = pTwo[0] + pTwo[1];
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Boosting to LAB frame defined by");
    cout << " p1 =   " << pTwo[0] << " p2 =   " << pTwo[1]
         << " total = " << total;
  }
  p1.rotbst(M);
  p2.rotbst(M);
  p3.rotbst(M);
  if (verbose >= DEBUG) {
    Vec4 tot = p1 + p2 + p3 ;
    printOut(__METHOD_NAME__,"Final momenta in LAB");
    cout <<" k1 =   "<<p1<<" k2 =   "<<p2<<" k3 =   "<<p3
         <<" total = "<<tot;
  }

  // Save momenta.
  pThree.resize(0);
  pThree.push_back(p1);
  pThree.push_back(p2);
  pThree.push_back(p3);

  // Check momentum conservation.
  Vec4 diff = total - (p1+p2+p3);
  if (abs(diff.e())  / abs(total.e()) > MILLI ||
     abs(diff.px()) / abs(total.e()) > MILLI ||
     abs(diff.py()) / abs(total.e()) > MILLI ||
     abs(diff.pz()) / abs(total.e()) > MILLI) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": (E,p) not conserved.","Aborting.");
    cout << setprecision(10) << " difference = " << total.px() << " "
         << total.py() << " " << total.pz() << " " << total.e() << endl;
    infoPtr->setAbortPartonLevel(true);
    return false;
  }

  // Return.
  return true;

}

//--------------------------------------------------------------------------

// Implementations of RF clustering maps for massive partons.

bool VinciaCommon::map2to3RF(vector<Vec4>& pThree, vector<Vec4> pTwo,
  vector<double> invariants,double phi,
  vector<double> masses) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Get momenta and boost to lab frame.
  if (pTwo.size() != 2) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Wrong number of momenta provided.");
    return false;
  }

  // Momentum of recoiler(s), final state parton, and (modified) resonance.
  Vec4 pAKBefore = pTwo.at(0);
  Vec4 pKBefore  = pTwo.at(1);
  Vec4 pABefore  = pKBefore + pAKBefore;
  Vec4 pACoM     = pABefore;

  // Boost to CoM frame of (modified) resonance.
  pKBefore.bstback(pABefore);
  pAKBefore.bstback(pABefore);
  pACoM.bstback(pABefore);

  // Get the polar and phi angle in CoM frame of K.
  double thetaK = pKBefore.theta();
  double phiK = pKBefore.phi();


  // Construct recoiled momenta in (modified) resonance CoM
  // frame. Recover masses and unscaled invariants.
  double saj = invariants[1];
  double sjk = invariants[2];
  double sak = invariants[3];
  double mA  = masses[0];
  double mj  = masses[1];
  double mk  = masses[2];
  double mAK = masses[3];
  double invDiff = mA*mA + mj*mj + mk*mk -saj-sak+sjk;
  invDiff -= mAK*mAK;

  // Get energies.
  double EjAfter = saj/(2.0*mA);
  double EkAfter = sak/(2.0*mA);
  if (EkAfter < mk)  return false;
  else if (EjAfter < mj) return false;
  else if (invDiff > MILLI) return false;

  // Get cosTheta.
  double cosTheta = costheta(EjAfter,EkAfter, mj,mk, sjk);
  if (abs(cosTheta) > 1.0) return false;
  double sinTheta = sqrt(1.0 - cosTheta*cosTheta);
  double pk = sqrt(EkAfter*EkAfter-mk*mk);
  double pj = sqrt(EjAfter*EjAfter-mj*mj);

  // Construct three momenta, assuming k was along z.
  Vec4 pkAfter(0.,0.,pk, EkAfter);
  Vec4 pjAfter(pj*sinTheta,0.,pj*cosTheta, EjAfter);
  Vec4 pajkAfter = pACoM - pkAfter - pjAfter;

  // Give some transverse recoil to k.
  double thetaEff = -(M_PI-pajkAfter.theta());
  pkAfter.rot(thetaEff,0.);
  pjAfter.rot(thetaEff,0.);
  pajkAfter.rot(thetaEff,0.);

  // Rotate by arbitrary phi.
  pkAfter.rot(0.,phi);
  pjAfter.rot(0.,phi);
  pajkAfter.rot(0.,phi);

  // Rotate to recover original orientation of frame.
  pkAfter.rot(thetaK,phiK);
  pjAfter.rot(thetaK,phiK);
  pajkAfter.rot(thetaK,phiK);

  // Boost to lab frame.
  pkAfter.bst(pABefore);
  pjAfter.bst(pABefore);
  pajkAfter.bst(pABefore);
  pThree.clear();
  pThree.push_back(pajkAfter);
  pThree.push_back(pjAfter);
  pThree.push_back(pkAfter);

  // Return.
  return true;

}

//--------------------------------------------------------------------------

// 1->2 decay map for (already offshell) resonance decay.

bool VinciaCommon::map1to2RF(vector<Vec4>& pNew, Vec4 pRes, double m1,
  double m2, double theta, double phi) {

  pNew.clear();

  // Fetch resonance mass.
  double m2R = pRes.m2Calc();

  // Square input masses.
  double m21 = m1*m1;
  double m22 = m2*m2;

  // Set up kinematics in the CoM frame.
  double p2z = kallenFunction(m2R,m21,m22)/(4.*m2R);

  // No solution if kallenFunction negative.
  if (p2z < 0.) return false;

  double E1 = sqrt(m21 + p2z);
  double E2 = sqrt(m22 + p2z);
  double pz = sqrt(p2z);

  Vec4 p1(0, 0, pz, E1);
  Vec4 p2(0, 0, -pz, E2);

  // Rotate.
  p1.rot(theta,phi);
  p2.rot(theta,phi);

  // Boost to lab frame.
  p1.bst(pRes);
  p2.bst(pRes);

  // Check.
  if (verbose >= DEBUG) {
    Vec4 total = pRes - p1 - p2;
    printOut(__METHOD_NAME__,"Checking momentum in lab frame:");
    cout<<" pRes = "<< pRes.e()<<" "<< pRes.px()
        <<" "<< pRes.py()<<" "<< pRes.pz()<<endl;
    cout<<" p1 = "<< p1.e()<<" "<< p1.px()
        <<" "<< p1.py()<<" "<< p1.pz()<<endl;
    cout<<" p2 = "<< p2.e()<<" "<< p2.px()
        <<" "<< p2.py()<<" "<< p2.pz()<<endl;
    cout<<" total = "<< total.e()<<" "<< total.px()
        <<" "<< total.py()<<" "<< total.pz()<<endl;
  }

  pNew.push_back(p1);
  pNew.push_back(p2);

  return true;

}

//--------------------------------------------------------------------------

// Implementations of resonance kineatic maps for massive partons. Inputs
// are as follows:
//   pBefore    = momenta of resonance and  all downstream recoilers
//                before emission.
//   posF       = position in pBefore of the momentum of the F end of the
//                antenna.
//   invariants = yaj and yjk scaled invariants.
//   phi        = azimuthal angle of gluon emission.
//   mui        = masses of a, j, k.
// The output is as follows:
//  pAfter = momenta of resonance, emission and all downstream recoilers
//           after emission.
//           [0]   = pa - will be unchanged
//           [1]   = pj
//           [2]   = pk
//           [i>3] = recoilers

bool VinciaCommon::map2toNRF(vector<Vec4>& pAfter, vector<Vec4> pBefore,
  unsigned int posR, unsigned int posF, vector<double> invariants,double phi,
  vector<double> masses) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  pAfter.clear();

  // Momentum of "R", "F" end of antenna, and sum of downstream recoilers.
  Vec4 pR = pBefore.at(posR);
  Vec4 pF = pBefore.at(posF);
  Vec4 pSum(0.,0.,0.,0.);
  vector<Vec4> pRec;
  for(unsigned int imom = 0; imom < pBefore.size(); imom++) {
    if (imom==posF || imom==posR) {
      continue;
    } else {
      pSum+= pBefore.at(imom);
      pRec.push_back(pBefore.at(imom));
    }
  }
  vector<Vec4> pTwo;
  vector<Vec4> pThree;
  pTwo.push_back(pSum);
  pTwo.push_back(pF);

  // Recoil AK system.
  if (!map2to3RF(pThree,pTwo,invariants,phi,masses)) {
    return false;
  } else if (pThree.size() != 3) {
    return false;
  }

  // Add pa, pj, and k. Check mass.
  pAfter.push_back(pR);
  pAfter.push_back(pThree.at(1));
  pAfter.push_back(pThree.at(2));
  Vec4 pSumAfter = pThree.at(0);
  if (abs(pSumAfter.mCalc() - pSum.mCalc()) > MILLI) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Failed to conserve mass of system.");
    return false;
  }

  // If only a single recoiler, it just takes the remaining momentum.
  if (pRec.size() == 1) {
    pAfter.push_back(pSumAfter);

  // Boost the downstream recoilers appropriately
  } else {
    for(unsigned int imom = 0; imom < pRec.size(); imom++) {
      double mRecBef = pRec.at(imom).mCalc();
      pRec.at(imom).bstback(pSum,pSum.mCalc());
      pRec.at(imom).bst(pSumAfter,pSum.mCalc());
      double mRecAfter = pRec.at(imom).mCalc();

      // Check mass.
      if (abs(mRecAfter- mRecBef) > MILLI) {
        infoPtr->errorMsg("Error in "+__METHOD_NAME__
          +": Failed to conserve mass of recoilers.");
        return false;
      }
      pAfter.push_back(pRec.at(imom));
    }
  }

  // Done.
  return true;
}

//--------------------------------------------------------------------------

// 2 -> 3 kinematics map for initial-initial antennae, for general mj.

bool VinciaCommon::map2to3IImassive(vector<Vec4>& pNew, vector<Vec4>& pRec,
  vector<Vec4>& pOld, double sAB, double saj, double sjb, double sab,
  double phi, double mj2) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Do massive mapping.
  pNew.clear();
  pNew.resize(3);

  // Force incoming momenta on shell (massless) with mass squared = sAB.
  pOld[0].py(0.);
  pOld[0].px(0.);
  pOld[1].py(0.);
  pOld[1].px(0.);
  double sCM = m2(pOld[0] + pOld[1]);
  double fac = sqrt(sAB/sCM);
  double e0 = pOld[0].e();
  double e1 = pOld[1].e();
  if (abs(1. - fac) > NANO) {
    if (verbose >= REPORT && abs(1.-fac) > 1.01)
      printOut(__METHOD_NAME__, "Warning: scaling AB so m2(AB) = sAB");
    e0 *= fac;
    e1 *= fac;
  }
  int sign = pOld[0].pz() > 0 ? 1 : -1;
  pOld[0].e(e0);
  pOld[0].pz(sign * e0);
  pOld[1].e(e1);
  pOld[1].pz(-sign * e1);

  // Initialise new momenta.
  pNew[0] = pOld[0];
  pNew[2] = pOld[1];

  // Check if we're inside massive phase space.
  double G = saj*sjb*sab - mj2*sab*sab;
  if (G < 0. || sab < 0.) return false;
  if ((sab <= sjb) || (sab <= saj)) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__
      +": Incompatible invariants.");
    return false;
  }

  // Incoming momenta.
  double rescaleFacA = sqrt(sab/sAB * (sab - saj)/(sab - sjb));
  double rescaleFacB = sqrt(sab/sAB * (sab - sjb)/(sab - saj));
  pNew[0].rescale4(rescaleFacA);
  pNew[2].rescale4(rescaleFacB);

  // Emission.
  double preFacA = sjb*sqrt((sab - saj)/(sab - sjb)/sab/sAB);
  double preFacB = saj*sqrt((sab - sjb)/(sab - saj)/sab/sAB);
  double preFacT = sqrt(saj*sjb/sab - mj2);
  Vec4 pTrans(cos(phi), sin(phi), 0.0, 0.0);
  pNew[1] = preFacA*pOld[0] + preFacB*pOld[1] + preFacT*pTrans;

  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Invariants are");
    cout << scientific << "    sAB = " << sAB << " saj = " << saj
         << " sjb = " << sjb << " sab = " << sab << endl
         << " Given momenta are" << endl;
    for (int i = 0; i < 2; i++) cout << "    " << pOld[i];
    cout << " New momenta are" << endl;
    for (int i = 0; i < 3; i++) cout << "    " << pNew[i];
  }

  // Check the invariants, allow 0.1% difference.
  double sajNew = 2*pNew[0]*pNew[1];
  double sjbNew = 2*pNew[1]*pNew[2];
  double sabNew = 2*pNew[0]*pNew[2];
  if (abs(sabNew - sab)/sab > MILLI) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
        "invariant(s)","yab ("+num2str(abs(sabNew - sab)/sab)+")");
    if (verbose >= REPORT) {
      cout << scientific << " sab (" << sab << ") fracdiff = ydiff = "
           << abs(sabNew-sab)/sab << endl << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
    }
  } else if (abs(sajNew - saj)/sab > MILLI) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
        "invariant(s)"," yaj ("+num2str(abs(sajNew - saj)/sab)+")");
    if (verbose >= REPORT) {
      cout << scientific << " saj (" << saj << ") fracdiff = "
           << abs(sajNew-saj)/saj << " ydiff = "
           << abs(sajNew-saj)/sab << endl << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
    }
  } else if (abs(sjbNew - sjb)/sab > MILLI) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
        "invariant(s)","yjb ("+num2str(abs(sjbNew - sjb)/sab)+")");
    if (verbose >= REPORT) {
      cout << scientific << " sjb (" << sjb << ") fracdiff = "
           << abs(sjbNew-sjb)/sjb << " ydiff = " << abs(sjbNew-sjb)/sab
           << endl << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
    }
  }

  // Change the final state recoiler. The recoiler is currently sum
  // of initial guys => E,0,0,pz. Boost in center-of-mass frame AB
  // E,0,0,0.
  Vec4 pSum = pOld[0] + pOld[1];
  Vec4 pRecSumBefore(0.,0.,0.,0.);
  Vec4 pRecSumAfter(0.,0.,0.,0.);
  for (int i=0; i<(int)pRec.size(); i++) {
    pRecSumBefore+=pRec[i];
    pRec[i].bstback(pSum);
  }

  // Now boost from E,0,0,0 to E',px',py',pz'.
  Vec4 pPrime = pNew[0] + pNew[2] - pNew[1];
  for (int i=0; i<(int)pRec.size(); i++) {
    pRec[i].bst(pPrime, pPrime.mCalc());
    pRecSumAfter+=pRec[i];
  }
  if (verbose >= DEBUG) {
    Vec4 total= pOld[0]+pOld[1];
    cout << " Total In before" << total
         << " Total Out before" << pRecSumBefore;
    total = pNew[0] + pNew[2] - pNew[1];
    cout << " Total In After" << total
         << " Total Out After" << pRecSumAfter
         << " Total diff After" << total-pRecSumAfter;
  }
  return true;

}

//--------------------------------------------------------------------------

// 2 -> 3 kinematics map for initial-initial antennae, for mj= 0.

bool VinciaCommon::map2to3IImassless(vector<Vec4>& pNew, vector<Vec4>& pRec,
  vector<Vec4>& pOld, double sAB, double saj, double sjb, double sab,
  double phi) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  pNew.clear();
  pNew.resize(3);

  // Force incoming momenta on shell (massless) with mass squared = sAB.
  pOld[0].py(0.);
  pOld[0].px(0.);
  pOld[1].py(0.);
  pOld[1].px(0.);
  double sCM = m2(pOld[0] + pOld[1]);
  double fac = sqrt(sAB/sCM);
  double e0 = pOld[0].e();
  double e1 = pOld[1].e();
  if (abs(1. - fac) > NANO) {
    if (verbose >= REPORT && abs(1. - fac) > 1.01)
      printOut(__METHOD_NAME__,
        "Warning: scaling AB so m2(AB) = sAB");
    e0 *= fac;
    e1 *= fac;
  }
  int sign = pOld[0].pz() > 0 ? 1 : -1;
  pOld[0].e(e0);
  pOld[0].pz(sign * e0);
  pOld[1].e(e1);
  pOld[1].pz(-sign * e1);

  // Initialise new momenta.
  pNew[0] = pOld[0];
  pNew[2] = pOld[1];

  // Incoming momenta.
  double rescaleFacA = sqrt(sab/(sAB+saj) * (1. + sjb/sAB));
  double rescaleFacB = sqrt(sab/(sAB+sjb) * (1. + saj/sAB));
  pNew[0].rescale4(rescaleFacA);
  pNew[2].rescale4(rescaleFacB);

  // Emission.
  double preFacA = sjb*sqrt((sAB+sjb)/(sAB+saj)/sab/sAB);
  double preFacB = saj*sqrt((sAB+saj)/(sAB+sjb)/sab/sAB);
  double preFacT = sqrt(saj*sjb/sab);
  Vec4 pTrans(cos(phi), sin(phi), 0.0, 0.0);
  pNew[1] = preFacA*pOld[0] + preFacB*pOld[1] + preFacT*pTrans;

  // Debugging info.
  if (verbose >= DEBUG) {
    stringstream ss;
    ss << "Invariants are: "
       << scientific << "    sAB = " << sAB << " saj = " << saj
       << " sjb = " << sjb << " sab = " << sab;
    printOut(__METHOD_NAME__, ss.str());
    printOut(__METHOD_NAME__, "Given momenta are");
    for (int i = 0; i < 2; i++) cout << "    " << pOld[i];
    printOut(__METHOD_NAME__, "New momenta are");
    for (int i = 0; i < 3; i++) cout << "    " << pNew[i];
  }

  // Check the invariants.
  double sajNew = 2*pNew[0]*pNew[1];
  double sjbNew = 2*pNew[1]*pNew[2];
  double sabNew = 2*pNew[0]*pNew[2];
  if (abs(sabNew - sab)/sab > MILLI) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
      "invariant(s)","yab ("+num2str(abs(sabNew - sab)/sab)+")");
    cout << scientific << " sab (" << sab << ") fracdiff = ydiff = "
         << abs(sabNew - sab)/sab << endl
         << " Old momenta are" << endl;
    for (int i=0; i<2; i++) cout << "    " << pOld[i];
    cout << " New momenta are" << endl;
    for (int i=0; i<3; i++) cout << "    " << pNew[i];
  } else if (abs(sajNew - saj)/sab > MILLI) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
      "invariant(s)","yaj ("+num2str(abs(sajNew - saj)/sab)+")");
    cout << scientific << " saj (" << saj << ") fracdiff = "
         << abs(sajNew-saj)/saj << " ydiff = "<< abs(sajNew - saj)/sab
         << endl << " Old momenta are" << endl;
    for (int i=0; i<2; i++) cout << "    " << pOld[i];
    cout << " New momenta are" << endl;
    for (int i=0; i<3; i++) cout << "    " << pNew[i];
  } else if ( abs(sjbNew-sjb)/sab > MILLI) {
    infoPtr->errorMsg("Warning in "+__METHOD_NAME__+": inconsistent "
      "invariant(s)","yjb ("+num2str(abs(sjbNew - sjb)/sab)+")");
    cout << scientific << " sjb (" << sjb << ") fracdiff = "
         << abs(sjbNew-sjb)/sjb << " ydiff = "<< abs(sjbNew - sjb)/sab
         << endl << " Old momenta are" << endl;
    for (int i=0; i<2; i++) cout << "    " << pOld[i];
    cout << " New momenta are" << endl;
    for (int i=0; i<3; i++) cout << "    " << pNew[i];
  }

  // Change the final state recoiler. The recoiler is currently sum
  // of initial guys => E,0,0,pz. Boost in center-of-mass frame AB
  // E,0,0,0.
  Vec4 pSum = pOld[0] + pOld[1];
  for (int i=0; i<(int)pRec.size(); i++) pRec[i].bstback(pSum);

  // Now boost from E,0,0,0 to E',px',py',pz' and return.
  Vec4 pPrime = pNew[0] + pNew[2] - pNew[1];
  for (int i=0; i<(int)pRec.size(); i++) pRec[i].bst(pPrime);
  return true;

}

//--------------------------------------------------------------------

// 2->3 kinematics map for local recoils, for general mj,mk. Assumes
// partons from proton explicitly massless

bool VinciaCommon::map2to3IFlocal(vector<Vec4>& pNew, vector<Vec4>& pOld,
  double sAK, double saj, double sjk, double sak, double phi,
  double mK2, double mj2, double mk2) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  pNew.clear();
  pNew.resize(3);
  if (verbose >= DEBUG) {
    printOut(__METHOD_NAME__,"Invariants are");
    cout << "    sAK = " << sAK << " saj = " << saj
         << " sjk = " << sjk << " sak = " << sak << endl
         << "    mK = " << sqrt(mK2) << " mj = " << sqrt(mj2)
         << " mk = " << sqrt(mk2) << endl
         << " Given momenta are" << endl;
    for (int i=0; i<2; i++) cout << "    " << pOld[i];
  }

  // Check invariants.
  double inv1Norm = (saj + sak)/(sAK + sjk);
  double inv2Norm = 1.0  + (mj2 + mk2 - mK2)/(sAK + sjk);
  double diff = abs(inv1Norm-inv2Norm);
  if (diff > MILLI) {
    if (verbose >= NORMAL) infoPtr->errorMsg("Warning in "+__METHOD_NAME__
      +": Inconsistent invariant(s)");
    if (verbose >= REPORT) {
      cout <<" yaj + yak = " << inv1Norm
           << " 1 + muj2 + muk2 - muK2 = "<< inv2Norm
           << " Diff = " << diff << endl;
    }
  }

  // Check if we're inside massive phase space.
  double G = saj*sjk*sak - mj2*sak*sak - mk2*saj*saj;
  if (G < 0. || sak < 0.) return false;

  // Set up some variables for boosting pT.
  Vec4 pSum = pOld[0] + pOld[1];
  Vec4 pOldBst = pOld[0];
  pOldBst.bstback(pSum);
  double thetaRot = pOldBst.theta();
  double phiRot = pOldBst.phi();
  Vec4 pTrans(cos(phi), sin(phi), 0.0, 0.0);

  // Rotate and boost.
  pTrans.rot(thetaRot, phiRot);
  pTrans.bst(pSum);

  // Check if pT was properly boosted, allow 0.1% difference.
  if (pTrans*pOld[0] > pOld[0][0]*1e-3 || pTrans*pOld[1] > pOld[1][0]*1e-3) {
    if (verbose >= NORMAL) {
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": The transverse momentum is not transverse after boosting");
    }
    return false;
  }
  double sig = sak + saj;
  double cj = (sig*(sak + mj2 - mk2) + mK2*(sak - saj) - sAK*sak)/(sAK*sig);
  double ck = (sig*(saj - mj2 + mk2) + mK2*(saj - sak) - sAK*saj)/(sAK*sig);
  double dj = saj/sig;
  double dk = sak/sig;

  // Construct post-branching initial-state momentum (explicitly massless).
  double pzAnew = pOld[0].pz() * sig/sAK;
  pNew[0] = Vec4(0., 0., pzAnew, abs(pzAnew));
  // Construct post-branching final-state momenta.
  pNew[1] = cj*pOld[0] + dj*pOld[1] + (sqrt(G)/sig)*pTrans;
  pNew[2] = ck*pOld[0] + dk*pOld[1] - (sqrt(G)/sig)*pTrans;

  // Check the invariants, allow 0.1% difference (due to boost).
  double sakNew = pNew[0]*pNew[2]*2;
  double sajNew = pNew[0]*pNew[1]*2;
  double sjkNew = pNew[1]*pNew[2]*2;
  if (abs(sakNew - sak)/sak > MILLI) {
    if (verbose >= NORMAL) {
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": Inconsistent invariant(s)","sak");
    }
    if (verbose >= REPORT) {
      cout << scientific << " sak (" << sak << ") diff = "
           << abs(sakNew-sak)/sak << endl
           << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
      cout <<"Masses:    mK2 = " << mK2 << " mj2 = " << mj2
           << " mk2 = " << mk2 << endl;
    }
  }
  if (abs(sajNew - saj)/saj > MILLI) {
    if (verbose >= NORMAL) {
      infoPtr->errorMsg("Warning in "+__METHOD_NAME__
        +": Inconsistent invariant(s)","saj");
    }
    if (verbose >= REPORT) {
      cout << scientific << " saj (" << saj << ") diff = ";
      cout << abs(sajNew-saj)/saj << endl;
      cout << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
      cout <<"Masses:    mK2 = " << mK2 << " mj2 = " << mj2
           << " mk2 = " << mk2 << endl;
    }
  }
  if (abs(sjkNew-sjk)/sjk > MILLI) {
    if (verbose >= NORMAL){
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": Inconsistent invariant(s)","sjk");
    }
    if (verbose >= REPORT) {
      cout << scientific << " sjk (" << sjk << ") diff = ";
      cout << abs(sjkNew-sjk)/sjk << endl;
      cout << " Old momenta are" << endl;
      for (int i=0; i<2; i++) cout << "    " << pOld[i];
      cout << " New momenta are" << endl;
      for (int i=0; i<3; i++) cout << "    " << pNew[i];
      cout <<"Masses:    mK2 = " << mK2 << " mj2 = " << mj2
           << " mk2 = " << mk2 << endl;
    }
  }
  return true;
}

//--------------------------------------------------------------------------

// 2->3 kinematics map for global recoils, for general mj,mk.  Assumes
// partons from proton explicitly massless.

bool VinciaCommon::map2to3IFglobal(vector<Vec4>& pNew,
  vector<Vec4>& pRec, vector<Vec4>& pOld, Vec4 &pB,
  double sAK, double saj, double sjk, double sak, double phi,
  double mK2, double mj2, double mk2) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  pNew.clear();
  pNew.resize(3);

  // Set up some variables for boosting pT.
  Vec4 pSum = pOld[0] + pOld[1];
  Vec4 pAcm = pOld[0];
  pAcm.bstback(pSum);
  double thetaRot = pAcm.theta();
  double phiRot = pAcm.phi();
  Vec4 pTrans(cos(phi), sin(phi), 0.0, 0.0);

  // Rotate and boost.
  pTrans.rot(thetaRot, phiRot);
  pTrans.bst(pSum);

  // Check if pT was properly boosted, allow 0.1% difference.
  if (pTrans*pOld[0] > pOld[0].e()*1e-3 || pTrans*pOld[1] > pOld[1].e()*1e-3) {
    if (verbose >= DEBUG) {
      printOut(__METHOD_NAME__,
        "The transverse momentum is not transverse after boosting");
    }
    return false;
  }

  // Set up starting (massless) solution. Saved in case of restart.
  double sig = sAK - saj;
  vector<double> vStart; vStart.resize(5);
  vStart[0] = sak/sig;
  vStart[1] = (saj*sjk)/(sAK*sig);
  vStart[2] = sjk/sig;
  vStart[3] = (saj*sak)/(sAK*sig);
  vStart[4] = sqrt(sjk*saj*sak)/sig;

  // Initialise from massless solution.
  vector<double> v(vStart);

  // Root finding with Newton-Raphson in 5D.
  int nCount = 0;
  int nFails = 0;

  // Current Newton-Raphson error
  double eps = numeric_limits<double>::max();

  do {
    nCount++;

    // Construct function.
    vector<double> f(5, 0);
    f[0] = v[0]*v[1]*sAK + pow2(v[1])*mK2 - v[4];
    f[1] = v[2]*v[3]*sAK + pow2(v[3])*mK2 - v[4] - mj2;
    f[2] = (v[0] - v[2] - 1)*(v[1] - v[3] + 1)*sAK
      + pow2(v[1] - v[3] + 1)*mK2 - mk2;
    f[3] = (v[0]*v[3] + v[1]*v[2])*sAK + 2*v[1]*v[3]*mK2 - 2*v[4] - saj;
    f[4] = (v[2]*(v[1] - v[3] + 1) + v[3]*(v[0] - v[2] - 1))*sAK
      + 2*v[3]*(v[1] - v[3] + 1)*mK2 - sjk;

    // Construct Jacobian.
    vector<vector<double> > A(5, vector<double>(5, 0));
    A[0][0] = v[1]*sAK;
    A[0][1] = v[0]*sAK + 2*v[1]*mK2;
    A[0][2] = 0;
    A[0][3] = 0;
    A[0][4] = -2*v[4];
    A[1][0] = 0;
    A[1][1] = 0;
    A[1][2] = v[3]*sAK;
    A[1][3] = v[2]*sAK + 2*v[3]*mK2;
    A[1][4] = -2*v[4];
    A[2][0] = (v[1] - v[3] + 1)*sAK;
    A[2][1] = (v[0] - v[2] - 1)*sAK + 2*(v[1] - v[3] + 1)*mK2;
    A[2][2] = -(v[1] - v[3] + 1)*sAK;
    A[2][3] = -( (v[0] - v[2] - 1)*sAK + 2*(v[1] - v[3] + 1)*mK2 );
    A[2][4] = 0;
    A[3][0] = v[3]*sAK;
    A[3][1] = v[2]*sAK + 2*v[3]*mK2;
    A[3][2] = v[1]*sAK;
    A[3][3] = v[0]*sAK + 2*v[1]*mK2;
    A[3][4] = -4*v[2];
    A[4][0] = v[3]*sAK;
    A[4][1] = v[2]*sAK + 2*v[3]*mK2;
    A[4][2] = (v[1] - 2*v[3] + 1)*sAK;
    A[4][3] = (v[0] - 2*v[2] - 1)*sAK + 2*(v[1] - 2*v[3] + 1)*mK2;
    A[4][4] = 0;

    // Invert Jacobian and append identity.
    int n = 5;
    for (int i = 0; i < n; i++) {
      A[i].resize(2*n);
      A[i][n+i] = 1;
    }

    for (int i = 0; i < n; i++) {
      // Find column max.
      double eleMax = abs(A[i][i]);
      int rowMax = i;
      for (int k=i+1; k<n; k++) {
        if (abs(A[k][i]) > eleMax) {
          eleMax = A[k][i];
          rowMax = k;
        }
      }

      // Swap maximum row with current row.
      A[rowMax].swap(A[i]);

      // Reduce current column.
      for (int k = i+1; k < n; k++) {
        double c = -A[k][i]/A[i][i];
        for (int j = i; j < 2*n; j++) {
          if (i == j) {
            A[k][j] = 0;
          } else {
            A[k][j] += c * A[i][j];
          }
        }
      }
    }

    // Solve equation Ax = b for upper triangular matrix A.
    for (int i = n-1; i >= 0; i--) {
      for (int k = n; k < 2*n; k++) {
        A[i][k] /= A[i][i];
      }
      for (int j = i-1; j >= 0; j--) {
        for (int k = n; k < 2*n; k++) {
          A[j][k] -= A[i][k] * A[j][i];
        }
      }
    }

    // Remove remaining identity.
    for (int i = 0; i < n; i++) {
      A[i].erase(A[i].begin(), A[i].begin()+n);
    }

    // Find next iteration.
    vector<double> vNew(5, 0);
    for (int i = 0; i < n; i++) {
      vNew[i] = v[i];
      for (int j=0; j<n; j++) {
        vNew[i] -= A[i][j]*f[j];
      }
    }

    // Perform sanity checks to decide if we should reset.
    bool resetToRandom = false;

    // Check for nans.
    for (int i=0; i<n; i++) {
      if (isnan(vNew[i])) {resetToRandom = true;}
    }

    // vNew[4] is a sqrt - should not be negative.
    if (vNew[4] < 0) {resetToRandom = true;}

    // Check for all negative solution.
    if (vNew[0] < 0 && vNew[1] < 0 && vNew[2] < 0 &&
        vNew[3] < 0) {
      resetToRandom = true;
    }

    if (nCount == 100) {resetToRandom = true;}

    // Do reset.
    if (resetToRandom) {
      nCount = 0;
      nFails++;

      // Start again from massless values.
      v = vStart;

      // Randomly vary variables.
      for (int i=0; i<n; i++) {
        v[i] *= (2*rndmPtr->flat() - 1);
      }
    }
    else {
      // Compute current error.
      eps = 0;
      for (int i=0; i<n; i++) {
        if (abs(vNew[i] - v[i])/abs(v[i]) > eps) {
          eps = abs(vNew[i] - v[i])/abs(v[i]);
        }
      }

      // Set to new values.
      v = vNew;
    }
  }
  while (eps > MICRO && nFails < 10);

  // Did we fail solving?
  if (nFails == 10) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+": failed to converge.");
    return false;
  }

  // Construct post-branching momenta.
  pNew[0] = v[0]*pOld[0] + v[1]*pOld[1] - v[4]*pTrans;
  pNew[1] = v[2]*pOld[0] + v[3]*pOld[1] - v[4]*pTrans;
  pNew[2] = (v[0] - v[2] - 1)*pOld[0] + (v[1] - v[3] + 1)*pOld[1];

  // Check if these momenta are on-shell
  bool failedOnShell = false;

  if (pNew[0].m2Calc() > MILLI) {
    failedOnShell = true;
  }

  double mj2New = pNew[1].m2Calc();
  if (mj2 == 0.) {
    if (abs(mj2New) > MILLI) {
      failedOnShell = true;
    }
  }
  else {
    if (abs(mj2New - mj2)/mj2 > 1E-4) {
      failedOnShell = true;
    }
  }

  double mk2New = pNew[2].m2Calc();
  if (mk2 == 0.) {
    if (abs(mk2New) > MILLI) {
      failedOnShell = true;
    }
  }
  else {
    if (abs(mk2New - mk2)/mk2 > 1E-4) {
      failedOnShell = true;
    }
  }

  if (failedOnShell) {
    infoPtr->errorMsg("Error in "+__METHOD_NAME__+
      ": failed on-shell check.");
    return false;
  }

  // Set up the boost.
  Vec4 pa = pNew[0];
  Vec4 pA = pOld[0];
  double qaB = pa*pB;
  double qAB = pA*pB;
  double qAa = pA*pa;

  // Perform boost.
  for (int i=0; i<3; i++) {
    Vec4 p = pNew[i];
    pNew[i] += pB*((pa*p)/qaB) - pa*((pB*p)/qaB) + pA*((pB*p)/qAB)
      - pB*((pA*p)/qAB) + pB*(qAa*(pB*p)/(qAB*qaB));

    // Force the initial state to be on the beam axis.
    if (i==0) {
      double ea = pNew[i].e();
      double sign = (pNew[0].pz() > 0) ? 1 : -1;
      pNew[0] = Vec4(0, 0, sign*ea, ea);
    }
  }

  // Perform boost on the rest of the system and return.
  for (int i=0; i<(int)pRec.size(); i++) {
    Vec4 p = pRec[i];
    pRec[i] += pB*((pa*p)/qaB) - pa*((pB*p)/qaB) + pA*((pB*p)/qAB)
      - pB*((pA*p)/qAB) + pB*(qAa*(pB*p)/(qAB*qaB));
  }

  double sajNew = 2*pNew[0]*pNew[1];
  double sjkNew = 2*pNew[1]*pNew[2];
  double sakNew = 2*pNew[0]*pNew[2];

  double checkAJ = abs(sajNew - saj)/saj;
  double checkJK = abs(sjkNew - sjk)/sjk;
  double checkAK = abs(sakNew - sak)/sak;

  if ( checkAJ > MILLI || checkJK > MILLI || checkAK > MILLI) {
    if (verbose >= NORMAL)
      infoPtr->errorMsg("Error in "+__METHOD_NAME__
        +": inconsistent invariant(s)");
    if (verbose >= REPORT) {
      cout<<setprecision(3)
          <<" Input:  sAK = "<<sAK<<"  saj = "<<saj<<"  sjk = "<<sjk
          <<"  sak = "<<sak<<"  mK2 = "<<mK2<<"  mj2 = "<<mj2
          <<"  mk2 = "<<mk2<<endl;
      cout<< "   pA = "<<pOld[0];
      cout<< "   pK = "<<pOld[1];
      cout<<"=>"<<endl;
      cout<< "   pa = "<<pNew[0];
      cout<< "   pj = "<<pNew[1];
      cout<< "   pk = "<<pNew[2];
      cout<<"       delta(aj,jk,ak) = "<<checkAJ<<" "<<checkJK
          <<" "<<checkAK<<"\n\n";
    }
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------

// Check if 2-particle system is on-shell and rescale if not.

bool VinciaCommon::onShellCM(Vec4& p1, Vec4& p2, double m1, double m2,
  double tol) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  double s1     = pow2(m1);
  double s2     = pow2(m2);
  double s01    = Vec4(p1+p2).m2Calc();
  double s1Calc = p1.m2Calc();
  double s2Calc = p2.m2Calc();
  if (abs(s1Calc-s1)/s01 > tol || abs(s2Calc-s2)/s01 > tol) {
    if (verbose >= REPORT)
      printOut(__METHOD_NAME__,"forcing particles on mass shell");
    RotBstMatrix M;
    M.fromCMframe(p1,p2);

    // Define massive on-shell momenta.
    double E0 = (s01 + s1 - s2)/(2*sqrt(s01));
    double E1 = (s01 - s1 + s2)/(2*sqrt(s01));
    double pz = pow2(E0)-s1;
    Vec4 p1new = Vec4(0.0,0.0,-pz,E0);
    Vec4 p2new = Vec4(0.0,0.0,pz,E1);
    p1new.rotbst(M);
    p2new.rotbst(M);
    double s1Test = p1new.m2Calc();
    double s2Test = p2new.m2Calc();
    if (verbose >= REPORT) {
      cout << " p1   : " << p1 << " p1new: " << p1new
           << " p2   : " << p1 << " p2new: " << p1new;
    }

    // If this got them closer to mass shell, replace momenta.
    if (abs(s1Test-s1)/s01 <= abs(s1Calc-s1)/s01
      && abs(s2Test-s2)/s01 <= abs(s2Calc-s2)/s01) {
      p1 = p1new;
      p2 = p2new;
    }
    return false;
  }
  else return true;

}

//--------------------------------------------------------------------------

// Map partons partonSystems[iSys] to equivalent massless ones.
// Return true if succeeded. Note, a method using only Particles or
// Vec4 as input could in principle be split off from this, if needed,
// but has not been required so far.

bool VinciaCommon::mapToMassless(int iSys, Event& event, bool makeNewCopies) {

  if (verbose >= DEBUG) printOut(__METHOD_NAME__, "begin", dashLen);

  // Start by making new copies, if requested to do so.
  if (makeNewCopies) {
    int iOld, iNew;

    // Copy incoming partons, interpret the copying operation as the
    // two incoming partons recoiling off each other. Assign status
    // code -42, incoming copy of recoiler (as mother).
    if (partonSystemsPtr->hasInAB(iSys)) {
      iOld = partonSystemsPtr->getInA(iSys);
      iNew = event.copy(iOld,-42);
      partonSystemsPtr->replace(iSys, iOld, iNew);
      iOld = partonSystemsPtr->getInB(iSys);
      iNew = event.copy(iOld,-42);
      partonSystemsPtr->replace(iSys, iOld, iNew);
    }
    // Note, a decaying resonance is not copied to preserve structure
    // of production and decay. Copy outgoing partons (use status code
    // 52).
    for (int i=0; i<partonSystemsPtr->sizeOut(iSys); ++i) {
      iOld = partonSystemsPtr->getOut(iSys,i);
      iNew = event.copy(iOld, 52);
    } // End loop to make new copies.
  } // End if new copies requested.

  // Initial-state partons, always assumed massless in VINCIA.
  if (partonSystemsPtr->hasInAB(iSys) ) {
    int iA = partonSystemsPtr->getInA(iSys);
    int iB = partonSystemsPtr->getInB(iSys);
    if (event[iA].m() != 0.0 || event[iB].m() != 0.0) {

      // Below we assume iA is the one with pz > 0; swap if opposite case.
      if (event[iA].pz() < 0 || event[iB].pz() > 0) {
        iA = partonSystemsPtr->getInB(iSys);
        iB = partonSystemsPtr->getInA(iSys);
      }

      // Transverse components assumed zero: check.
      if (event[iA].pT() > 1.e-6 || event[iB].pT() > 1.e-6) {
        stringstream ss;
        ss << ": incoming partons have non-vanishing transverse momenta"
           << " for system iSys = " << iSys << ". Giving up.";
        infoPtr->errorMsg("Error in "+__METHOD_NAME__+ss.str());
        return false;
      }

      // Verbose output.
      if (verbose >= DEBUG) {
        stringstream ss;
        ss << ": forcing initial"
          "-state partons to be massless for system "<<iSys;
        infoPtr->errorMsg("Warning in "+__METHOD_NAME__+ss.str());
      }

      // Define explicitly massless momenta (same as in Pythia::PartonLevel).
      double pPos = event[iA].pPos() + event[iB].pPos();
      double pNeg = event[iA].pNeg() + event[iB].pNeg();
      event[iA].pz( 0.5 * pPos);
      event[iA].e ( 0.5 * pPos);
      event[iA].m ( 0.);
      event[iB].pz(-0.5 * pNeg);
      event[iB].e ( 0.5 * pNeg);
      event[iB].m ( 0.);
    }
  } // End make initial-state partons massless.

  // Final-state partons.
  if (partonSystemsPtr->sizeOut(iSys) >= 2) {
    vector<Vec4> momenta;
    vector<double> massOrg;
    bool makeMassless = false;
    Vec4 pSysOrg;
    for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
      momenta.push_back(event[partonSystemsPtr->getOut(iSys,i)].p());
      massOrg.push_back(event[partonSystemsPtr->getOut(iSys,i)].m());
      if (massOrg[i] > 0. && event[partonSystemsPtr->getOut(iSys,i)].idAbs()
        <= nFlavZeroMass) makeMassless = true;
      pSysOrg += momenta[i];
    }
    // Return if nothing needs to be done.
    if (!makeMassless) return true;

    // Create copy of original momenta (in case of failure).
    vector<Vec4> momentaOrg = momenta;

    // Boost to CM if original system not at rest.
    double sCM = m2(pSysOrg);
    bool isInCM = ( pow2(pSysOrg.pAbs())/sCM < 1e-10 );
    if (!isInCM)
      for (int i=0; i<(int)momenta.size(); ++i) momenta[i].bstback(pSysOrg);

    // Define vector for computing CM energy of modified system.
    Vec4 pSysNew;

    // Identify particles to be made massless (by ID code) and rescale
    // their momenta along direction of motion.
    for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
      int ip = partonSystemsPtr->getOut(iSys,i);
      if (event[ip].idAbs() <= nFlavZeroMass && event[ip].m() != 0.) {
        double facInv = momenta[i].pAbs()/momenta[i].e();
        // Sanity check.
        if (facInv <= 0.) {
          if (verbose >= NORMAL)
            printOut(__METHOD_NAME__,
              "Remap failed. Particle is spacelike or at rest.");
          // Restore masses in case any were already changed.
          for (int j=0; j < partonSystemsPtr->sizeOut(iSys); ++j)
            event[partonSystemsPtr->getOut(iSys,j)].m(massOrg[j]);
          // Failed.
          return false;
        }
        momenta[i].rescale3(1./facInv);
        event[ip].m(0.);
        // Check new 4-vector.
        double mNew = momenta[i].mCalc();
        if (pow2(mNew/momenta[i].e()) > NANO) {
          printOut(__METHOD_NAME__,"Warning: rounding problem.");
          if (verbose >= DEBUG) {
            cout<<scientific << "(p,e) = "<<momenta[i].pAbs() << "  "
                << momenta[i].e() << " facInv = " << facInv
                << " 1/facInv = " << 1./facInv << " mNew = " << mNew << endl;
          }
        } // End check new 4-vector.
      } // End if massless flavour with mass > 0.

      // Add to new CM momentum.
      pSysNew += momenta[i];
    } // End loop over FS particles.

    // New system generally has smaller invariant mass and some
    // motion. Determine if additional scalings or boosts are needed.
    Vec4 delta = pSysOrg - pSysNew;
    if (delta.e()/sqrt(sCM) < NANO && delta.pAbs()/sqrt(sCM) < NANO) {
      // Update event record (masses already updated above).
      for (int i = 0; i < (int)momenta.size(); ++i)
        event[partonSystemsPtr->getOut(iSys,i)].p(momenta[i]);
      if (verbose >= DEBUG)
        printOut(__METHOD_NAME__,"No further rescalings needed.");
      return true;
    }

    // If the new system has a different CM energy, rescale all
    // energies and momenta to restore the same CM energy as before.
    double sCMnew   = m2(pSysNew);
    double scaleFac = sqrt(sCM/sCMnew);
    if (verbose >= DEBUG && pow2(scaleFac-1.0) > NANO)
      printOut(__METHOD_NAME__,"Rescaling 4-vectors to restore eCM");
    Vec4 pSysNewB;
    for (int i = 0; i < (int)momenta.size(); ++i) {
      momenta[i].rescale4(scaleFac);
      pSysNewB += momenta[i];
    }
    double sCMnewB = m2(pSysNewB);
    if (verbose >= DEBUG) {
      cout << "old CM energy = " << sqrt(sCM) << " intermediate CM energy = "
           << sqrt(sCMnew) << " new CM energy = " << sqrt(sCMnewB) << endl;
      printOut(__METHOD_NAME__,"Boosting back to CM frame");
    }
    // Then boost to CM frame (preserves CM energy).
    for (int i=0; i<(int)momenta.size(); ++i) {
      // Boost to new CM frame
      momenta[i].bstback(pSysNewB);
      // If required, also boost back to frame of input system
      if (!isInCM) momenta[i].bst(pSysOrg);

      // Update event record (masses already updated above).
      event[partonSystemsPtr->getOut(iSys,i)].p(momenta[i]);

    } // End do boosts.

    // Verbose output: final configuration.
    if (verbose >= DEBUG) {
      cout << "Final configuration:" << endl;
      for (int i = 0; i < (int)momenta.size(); ++i)
        cout << "  " << i << " " << momenta[i];
    }

  } // End make final-state momenta massless.

  // We made it.
  return true;

}

//--------------------------------------------------------------------------

// Make list of particles as vector<Particle>.

vector<Particle> VinciaCommon::makeParticleList(const int iSys,
  const Event& event, const vector<Particle> &pNew, const vector<int> &iOld) {

  // Put incoming ones (initial-state partons or decaying resonance) first.
  vector<Particle> state;
  // reserve size for state
  state.reserve(3 + pNew.size() + partonSystemsPtr->sizeOut(iSys));
  if (partonSystemsPtr->hasInAB(iSys)) {
    int iA = partonSystemsPtr->getInA(iSys);
    int iB = partonSystemsPtr->getInB(iSys);
    for (int j = 0; j < (int)iOld.size(); ++j) {
      // Exclude any partons in old state that should be replaced.
      if (iOld[j] == iA) iA = -1;
      if (iOld[j] == iB) iB = -1;
      if (iA == -1 && iB == -1) break;
    }
    if (iA >= 0) state.push_back(event[iA]);
    if (iB >= 0) state.push_back(event[iB]);
  } else if (partonSystemsPtr->hasInRes(iSys)) {
    int iRes = partonSystemsPtr->getInRes(iSys);
    for (int j = 0; j < (int)iOld.size(); ++j) {
      // Exclude any partons in old state that should be replaced.
      if (iOld[j] == iRes){
        iRes = -1;
        break;
      }
    }
    if (iRes >= 0) state.push_back(event[iRes]);
  } else {
    // If neither hasInAB() nor hasInRes(), assume hadron decay. Find mother.
    int iMot = event[partonSystemsPtr->getOut(iSys,0)].mother1();
    while (iMot > 0 && !event[iMot].mayDecay())
      iMot = event[iMot].mother1();
    if (iMot > 0) state.push_back(iMot);
  }
  // Add any post-branching incoming particles.
  for (int j = 0; j < (int)pNew.size(); ++j) {
    if (!pNew[j].isFinal()) state.push_back(pNew[j]);
  }

  // Note: state size can be zero at this point if this is a forced shower
  // off a user-defined final state without any specified decaying resonance
  // or incoming beams; see e.g., VinciaFSR::shower().

  // Then put outgoing ones.
  for (int i = 0; i < partonSystemsPtr->sizeOut(iSys); ++i) {
    int i1 = partonSystemsPtr->getOut(iSys, i);
    // Do not add any that are marked as branched.
    for (int j = 0; j < (int)iOld.size(); ++j) {
      if (iOld[j] == i1){
        i1 = -1;
        break;
      }
    }
    if (i1 >= 0) state.push_back(event[i1]);
  }
  // Add any post-branching outgoing partons.
  for (int j=0; j<(int)pNew.size(); ++j)
    if (pNew[j].isFinal()) state.push_back(pNew[j]);

  // Return the state.
  return state;

}

//--------------------------------------------------------------------------

// Based on current state, find all antennae
// for this branching and swap children if needed.

vector<VinciaClustering> VinciaCommon::findAntennae(Event& state,
  int i1, int i2, int i3) {
  // Initialise.
  vector<VinciaClustering> clusterings;
  VinciaClustering clus;
  clus.setChildren(state, i1, i2, i3);

  // Final-final branching.
  if (state[clus.child1].isFinal() && state[clus.child3].isFinal()) {
    clus.isFSR = true;

    // Gluon emission.
    if (state[clus.child2].isGluon()) {
      if (state[clus.child1].isGluon()) {
        if (state[clus.child3].isGluon()) clus.antFunType = GGemitFF;
        else clus.antFunType = GQemitFF;
      }
      else {
        if (state[clus.child3].isGluon()) clus.antFunType = QGemitFF;
        else clus.antFunType = QQemitFF;
      }

      // No flavour change for gluon emissions.
      clus.setMothers(state[clus.child1].id(),state[clus.child3].id());

      // Save.
      clusterings.push_back(clus);
    }

    // Gluon splitting.
    else {
      // Check colour connection to find out who is splitting.
      bool colCon12 = colourConnected(state[clus.child1],state[clus.child2]);
      bool colCon23 = colourConnected(state[clus.child2],state[clus.child3]);
      if (colCon12 && !colCon23) {
        clus.swap13();
        std::swap(colCon12, colCon23);
      }
      // Check flavours.
      if (state[clus.child1].id() == -state[clus.child2].id()) {
        // Check colour connection.
        if (!colCon12 && colCon23) {
          // Children 1 and 2 are clustered to a gluon.
          clus.antFunType = GXsplitFF;
          clus.setMothers(21, state[clus.child3].id());

          // Save.
          clusterings.push_back(clus);
        }
      }
    }
  }

  // Initial-initial branching.
  else if (!state[clus.child1].isFinal() && !state[clus.child3].isFinal()) {
    clus.isFSR = false;

    // Gluon emission.
    if (state[clus.child2].isGluon()) {
      if (state[clus.child1].isGluon()) {
        if (state[clus.child3].isGluon()) clus.antFunType = GGemitII;
        else clus.antFunType = GQemitII;
      }
      else {
        if (state[clus.child3].isGluon()) clus.antFunType = GQemitII;
        else clus.antFunType = QQemitII;
      }

      // No flavour change for gluon emissions.
      clus.setMothers(state[clus.child1].id(),state[clus.child3].id());

      // Save.
      clusterings.push_back(clus);
    }

    // For splittings, we can have more than one antenna.
    else {
      // Quark conversion of clus.child1.
      if (state[clus.child2].id() == state[clus.child1].id()) {
        // Check that the colour connection is sensible, otherwise skip.
        bool colCon12 = colourConnected(state[clus.child1],state[clus.child2]);
        bool colCon23 = colourConnected(state[clus.child2],state[clus.child3]);
        bool colCon13 = colourConnected(state[clus.child1],state[clus.child3]);
        if (!colCon12 && (colCon23 || colCon13)) {
          // Children 1 and 2 are clustered to a gluon.
          clus.setMothers(21,state[clus.child3].id());
          clus.antFunType = GXconvII;

          // Save.
          clusterings.push_back(clus);
        }
      }
      // Quark conversion of clus.child3.
      if (state[clus.child2].id() == state[clus.child3].id()) {
        // Swap children.
        clus.swap13();

        // Check that the colour connection is sensible, otherwise skip.
        bool colCon12 = colourConnected(state[clus.child1],state[clus.child2]);
        bool colCon23 = colourConnected(state[clus.child2],state[clus.child3]);
        bool colCon13 = colourConnected(state[clus.child1],state[clus.child3]);
        if (!colCon12 && (colCon23 || colCon13)) {
          // Now children 1 and 2 are clustered to a gluon.
          clus.setMothers(21,state[clus.child3].id());
          clus.antFunType = GXconvII;

          // Save.
          clusterings.push_back(clus);
        }
      }

      // Gluon splitting of clus.child1.
      if (state[clus.child1].isGluon()) {
        // Check that the colour connection is sensible, otherwise skip.
        if (colourConnected(state[clus.child1], state[clus.child2])
          && colourConnected(state[clus.child1], state[clus.child3])) {
          // Children 1 and 2 are clustered to quark of anti-flavour of 2.
          // Child 3 does not change flavour.
          clus.setMothers(-state[clus.child2].id(), state[clus.child3].id());
          clus.antFunType = QXsplitII;

          // Save.
          clusterings.push_back(clus);
        }
      }
      // Gluon splitting of clus.child3.
      if (state[clus.child3].isGluon()) {
        // Swap children.
        clus.swap13();

        // Check that the colour connection is sensible, otherwise skip.
        if (colourConnected(state[clus.child1], state[clus.child2])
          && colourConnected(state[clus.child1], state[clus.child3])) {
          // Children 1 and 2 are clustered to quark of anti-flavour of 2.
          // Child 3 does not change flavour.
          clus.setMothers(-state[clus.child2].id(), state[clus.child3].id());
          clus.antFunType = QXsplitII;

          // Save.
          clusterings.push_back(clus);
        }
      }
    }
  }

  // Resonance-final branching.
  else if ((state[clus.child1].isResonance() && !state[clus.child1].isFinal())
    || (state[clus.child3].isResonance() && !state[clus.child3].isFinal())) {
    clus.isFSR = true;

    // Always assume clus.child1 is resonance.
    if (!state[clus.child1].isResonance()) clus.swap13();

    // Resonance always stays the same.
    int idA = state[clus.child1].id();

    // Gluon emission.
    if (state[clus.child2].isGluon()) {
      if (state[clus.child3].isGluon()) clus.antFunType = QGemitRF;
      else clus.antFunType = QQemitRF;

      // No flavour change for gluon emissions.
      clus.setMothers(idA, state[clus.child3].id());

      // Save.
      clusterings.push_back(clus);
    }
    else {
      clus.antFunType = XGsplitRF;

      // Explicitly check colour connection.
      if (!colourConnected(state[clus.child2],state[clus.child3])
        && colourConnected(state[clus.child2],state[clus.child1])) {
        // Children 2 and 3 get clustered to a gluon.
        clus.setMothers(idA, 21);

        // Save.
        clusterings.push_back(clus);
      }
    }
  }

  // Initial-final branching.
  else {
    clus.isFSR = false;

    // Always assume child1 is in the initial state.
    if (state[clus.child1].isFinal()) clus.swap13();

    // Gluon emission.
    if (state[clus.child2].isGluon()) {
      if (state[clus.child1].isGluon()) {
        if (state[clus.child3].isGluon()) clus.antFunType = GGemitIF;
        else clus.antFunType = GQemitIF;
      }
      else {
        if (state[clus.child3].isGluon()) clus.antFunType = QGemitIF;
        else clus.antFunType = QQemitIF;
      }

      // No flavour change for gluon emissions.
      clus.setMothers(state[clus.child1].id(),state[clus.child3].id());

      // Save.
      clusterings.push_back(clus);
    }
    // For gluon splittings, we have more than one antenna.
    else {
      // Gluon splitting in the final state.
      if (state[clus.child2].id() == -state[clus.child3].id()
        && !colourConnected(state[clus.child2], state[clus.child3])) {
        // Children 2 and 3 are clustered to a gluon.
        clus.setMothers(state[clus.child1].id(), 21);
        clus.antFunType = XGsplitIF;

        // Save.
        clusterings.push_back(clus);
      }
      // Gluon splitting in the initial state.
      if (state[clus.child1].isGluon()) {
        // Check that the colour connection is sensible, skip otherwise.
        if (colourConnected(state[clus.child1],state[clus.child2])
          && colourConnected(state[clus.child1],state[clus.child3])) {
          // Children 1 and 2 are clustered to quark of anti-flavour of 2.
          // Child 3 does not change flavour.
          clus.setMothers(-state[clus.child2].id(), state[clus.child3].id());
          clus.antFunType = QXsplitIF;

          // Save.
          clusterings.push_back(clus);
        }
      }
      // Quark conversion (in the initial state).
      if (state[clus.child2].id() == state[clus.child1].id()) {
        // Check that the colour connection is sensible, skip otherwise.
        bool colCon12 = colourConnected(state[clus.child1],state[clus.child2]);
        bool colCon23 = colourConnected(state[clus.child2],state[clus.child3]);
        bool colCon13 = colourConnected(state[clus.child1],state[clus.child3]);
        if (!colCon12 && (colCon23 || colCon13)) {
          // Children 1 and 2 are clustered to a gluon.
          clus.setMothers(21, state[clus.child3].id());
          clus.antFunType = GXconvIF;

          // Save.
          clusterings.push_back(clus);
        }
      }
    }
  }

  return clusterings;
}

//--------------------------------------------------------------------------

// Check whether two particles are colour-connected.

bool VinciaCommon::colourConnected(const Particle& ptcl1,
  const Particle& ptcl2) {
  int  col1 = ptcl1.isFinal() ? ptcl1.col() : ptcl1.acol();
  int acol1 = ptcl1.isFinal() ? ptcl1.acol() : ptcl1.col();
  int  col2 = ptcl2.isFinal() ? ptcl2.col() : ptcl2.acol();
  int acol2 = ptcl2.isFinal() ? ptcl2.acol() : ptcl2.col();

  if ((col1 != 0 && col1 == acol2) || (acol1 != 0 && acol1 == col2))
    return true;
  return false;
}

//--------------------------------------------------------------------------

// Print a list of particles.

void VinciaCommon::list(const vector<Particle>& state, string title,
  bool footer) {

  if (title == "") title = " ------------------------";
  else {
    title = "- " + title + "  ";
    int nDashes = 25 - title.size();
    if (nDashes > 0)
      for (int i(0); i<nDashes; ++i) title += "-";
  }
  cout << " --------  Particle List " << title << "----------";
  cout << "----------------------";
  cout << endl << endl;
  cout << "   ind          id      colours"
       << setw(14) << "px" << setw(10) << "py" << setw(10) << "pz"
       << setw(10) << "e" << setw(11) << "m" << endl;
  for (int i(0); i<(int)state.size(); ++i)
    cout << " " << num2str(i,5) << " " << num2str(state[i].id(),9)
         << "    "
         << num2str(state[i].col(),4) << " " << num2str(state[i].acol(),4)
         << "    " << state[i].p();
  cout << endl;
  if (footer) {
    cout << " -----------------------------------------------------------";
    cout << "-------------------";
    cout << endl;
  }
}

//--------------------------------------------------------------------------

// Print a list of clusterings.

void VinciaCommon::list(const vector<VinciaClustering>& clusterings,
  string title, bool footer) {

  // Get breakdown of clusterings.
  int nTot = 0;
  int nFF  = 0;
  int nRF  = 0;
  int nIF  = 0;
  int nII  = 0;
  for (const auto& c : clusterings) {
    if (c.isFF()) ++nFF;
    else if (c.isRF()) ++nRF;
    else if (c.isIF()) ++nIF;
    else if (c.isII()) ++nII;
    ++nTot;
  }

  if (title == "") title = " ------------------------";
  else {
    title = "- " + title + "  ";
    int nDashes = 25 - title.size();
    if (nDashes > 0)
      for (int i(0); i<nDashes; ++i) title += "-";
  }
  cout << " --------  Clusterings Summary " << title << "----";
  cout << "-------------------";
  cout << endl << endl;
  cout << "  Found " << nTot << " clustering"
       << (nTot != 1 ? "s." : ".") << endl;
  cout << "    -> FF clusterings: " << setw(2) << nFF << endl;
  cout << "    -> RF clusterings: " << setw(2) << nRF << endl;
  cout << "    -> IF clusterings: " << setw(2) << nIF << endl;
  cout << "    -> II clusterings: " << setw(2) << nII << endl;
  cout << endl;
  cout << "  Clusterings:" << endl;
  for (int i(0); i<nTot; ++i) {
    VinciaClustering c = clusterings.at(i);
    cout << "    Sector " << i << ": " << num2str(c.child1,3) << " "
         << num2str(c.child2,3) << " " << num2str(c.child3,3)
         << " (" << c.getAntName() << ")" << endl;
  }
  cout << endl;
  if (footer) {
    cout << " -----------------------------------------------------------";
    cout << "-------------------";
    cout << endl;
  }
}

//==========================================================================

// VINCIA Auxiliary helper functions.

//--------------------------------------------------------------------------

// External auxiliaries, extra four-products.


//--------------------------------------------------------------------------

// External auxiliaries, string manipulation.

string num2str(int i, int width) {
  ostringstream tmp;
  if (width <= 1) tmp << i;
  else if (abs(i) < pow(10.0, width - 1) || ( i > 0 && i < pow(10.0, width)))
    tmp << fixed << setw(width) << i;
  else {
    string ab = "k";
    double r = i;
    if      (abs(i) < 1e5)       {r/=1e3;}
    else if (abs(i) < 1e8)  {r/=1e6;  ab = "M";}
    else if (abs(i) < 1e11) {r/=1e9;  ab = "G";}
    else if (abs(i) < 1e14) {r/=1e12; ab = "T";}
    tmp << fixed << setw(width - 1)
        << (r > 10 ? setprecision(width-4) : setprecision(width-3)) << r << ab;
  }
  return tmp.str();
}

string num2str(double r, int width) {
  ostringstream tmp;
  if (width <= 0) tmp << r;
  else if (r == 0.0 || (abs(r) > 0.1 && abs(r) < pow(10., max(width-3,1)))
           || width <= 8) tmp << fixed << setw(max(width,3))
                              << setprecision(min(3, max(1, width - 2))) << r;
  else tmp << scientific << setprecision(max(2, width - 7))
           << setw(max(9, width)) << r;
  return tmp.str();
}

string bool2str(bool b, int width) {
  string tmp = b ? "on" : "off";
  int nPad = width - tmp.length();
  for (int i = 1; i <= nPad; ++i) tmp = " " + tmp;
  return tmp;
}

//--------------------------------------------------------------------------

// Print "(place) message" with option for padding to len with padChar.

void printOut(string place, string message, int len, char padChar) {
  cout.setf(ios::internal);
  cout << " (" << (place + ") ") << message;
  // Are we asked to pad until a certain length?
  if (len > 0) {
    int nPad = len - place.length() - 5 - message.length();
    string padString(max(0,nPad),padChar);
    cout<<" "<<padString;
  }
  cout << "\n";
}

//==========================================================================

} // end namespace Pythia8
