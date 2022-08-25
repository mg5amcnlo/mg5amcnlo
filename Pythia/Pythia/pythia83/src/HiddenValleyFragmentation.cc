// HiddenValleyFragmentation.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the
// HiddenValleyFragmentation class and its helper classes.

#include "Pythia8/HiddenValleyFragmentation.h"

namespace Pythia8 {

//==========================================================================

// The HVStringFlav class is used to select HV-quark and HV-hadron flavours.

//--------------------------------------------------------------------------

// Initialize data members of the flavour generation.

void HVStringFlav::init() {

  // Read in data from Settings.
  separateFlav    = flag("HiddenValley:separateFlav");
  nFlav           = mode("HiddenValley:nFlav");
  probFlav        = settingsPtr->pvec("HiddenValley:probFlav");
  probDiquark     = parm("HiddenValley:probDiquark");
  probVector      = parm("HiddenValley:probVector");
  probKeepEta1    = parm("HiddenValley:probKeepEta1");

  // Sum of flavour suppression factors. Auxiliaries for eta1 suppression.
  sumProbFlav     = 0.;
  for (int i = 0; i < nFlav; ++i) sumProbFlav += probFlav[i];
  probKeepLast    = (1. - probVector) * probKeepEta1 + probVector;
  probVecEta1     = probVector / probKeepLast;

  // Some settings not allowed for this scenario. (Plus several more?)
  thermalModel    = false;
  useWidthPre     = false;
  closePacking    = false;
  mT2suppression  = false;

  // Overwrite some HV-hadron names when simplified displayed flavours.
  if (!separateFlav) {
    particleDataPtr->name( 4900111, "pivDiag");
    particleDataPtr->names(4900211, "pivUp", "pivDn");
    particleDataPtr->name( 4900113, "rhovDiag");
    particleDataPtr->names(4900213, "rhovUp", "rhovDn");
    particleDataPtr->names(4901114, "Deltav", "Deltavbar");

    // Also overwrite hv-quark masses to agree in internal fragmentation.
    double mqv1   = particleDataPtr->m0(4900101);
    for (int i = 2; i < 9; ++i) particleDataPtr->m0(4900100 + i, mqv1);
  }

  // Switch off Zv decays to qv's above the nFlav number.
  ParticleDataEntryPtr zvPtr = particleDataPtr->particleDataEntryPtr(4900023);
  for (int i = 0; i < zvPtr->sizeChannels(); ++i) {
    int idDec = abs(zvPtr->channel(i).product(1));
    if (idDec > 4900100 + nFlav && idDec < 4900109)
      zvPtr->channel(i).onMode(0);
  }

}

//--------------------------------------------------------------------------

// Pick a new HV-flavour given an incoming one.

FlavContainer HVStringFlav::pick(FlavContainer& flavOld, double, double,
  bool) {

  // Initial values for old and new flavour.
  bool isNotDiquark = ( (abs(flavOld.id)/1000)%10 == 0 );
  FlavContainer flavNew;
  flavNew.rank = flavOld.rank + 1;

  // HV-diquark selection, leading to HV-baryon production.
  if (isNotDiquark && rndmPtr->flat() < probDiquark) {
    flavNew.id = (flavOld.id > 0) ? 4901103 : -4901103;
    return flavNew;
  }

  // Pick new HV-flavour according to suppression factors; keep track of sign.
  // Prepare for suppression of eta_1 pseudoscalar, the highest-numbered state.
  bool isLastDiag = false;
  do {
    double rndmFlav = sumProbFlav * rndmPtr->flat();
    int index = -1;
    do rndmFlav -= probFlav[++index];
    while (rndmFlav > 0. && index < nFlav - 1);
    flavNew.id = 4900101 + index;
    if ( isNotDiquark && flavOld.id > 0) flavNew.id = -flavNew.id;
    if (!isNotDiquark && flavOld.id < 0) flavNew.id = -flavNew.id;
    isLastDiag = (flavOld.id + flavNew.id == 0
      && abs(flavOld.id) == 4900100 + nFlav);
  } while (isLastDiag && rndmPtr->flat() > probKeepLast);

  // Done.
  return flavNew;

}

//--------------------------------------------------------------------------

// Combine two HV-flavours to produce an HV-hadron.

int HVStringFlav::combine(FlavContainer& flav1, FlavContainer& flav2) {

  // Check whether incoming diquarks. No solution for diquark-antidiquark.
  bool isDiquark1 = ( (abs(flav1.id)/1000)%10 != 0 );
  bool isDiquark2 = ( (abs(flav2.id)/1000)%10 != 0 );
  if (isDiquark1 && isDiquark2) return 0;

  // HV-baryon from diquark plus quark.
  if (isDiquark1 || isDiquark2) {
    int id1Abs = abs(flav1.id) - 4900000;
    int id2Abs = abs(flav2.id) - 4900000;
    if (isDiquark1) swap(id1Abs, id2Abs);
    int idBaryon = 4900004 +  1000 * (id1Abs % 10) + (id2Abs / 10);
    return (flav1.id > 0) ? idBaryon : -idBaryon;
  }

  // Positive and negative flavour. Note that with kinetic mixing
  // the Fv are really intended to represent qv, so remap.
  int idMeson = 0;
  int idPos =  max( flav1.id, flav2.id) - 4900000;
  int idNeg = -min( flav1.id, flav2.id) - 4900000;
  if (idPos < 20) idPos = 101;
  if (idNeg < 20) idNeg = 101;

  // Pick HV-meson code: full spectrum of codes or simplified option.
  if (separateFlav) {
    if (idNeg == idPos)     idMeson =   4889001 + 110 * idPos;
    else if (idPos > idNeg) idMeson =   4889001 + 100 * idPos + 10 * idNeg;
    else                    idMeson = -(4889001 + 100 * idNeg + 10 * idPos);
  } else {
    if (idNeg == idPos)     idMeson =  4900111;
    else if (idPos > idNeg) idMeson =  4900211;
    else                    idMeson = -4900211;
  }

  // Pick spin either 0 or 1. Include suppression of eta_1 pseudoscalar.
  double probVecNow = (idNeg == idPos && idPos == 100 + nFlav)
                    ? probVecEta1 : probVector;
  if (rndmPtr->flat() < probVecNow) idMeson += ((idMeson > 0) ? 2 : -2);

  // Done.
  return idMeson;

}

//==========================================================================

// The HVStringPT class is used to select pT in HV fragmentation.

//--------------------------------------------------------------------------

// Initialize data members of the string pT selection.

void HVStringPT::init() {

  // Parameter of the pT width. No enhancement, since this is finetuning.
  double sigmamqv  = parm("HiddenValley:sigmamqv");
  double sigma     = sigmamqv * particleDataPtr->m0( 4900101);
  sigmaQ           = sigma / sqrt(2.);
  enhancedFraction = 0.;
  enhancedWidth    = 0.;

  // Parameter for pT suppression in MiniStringFragmentation.
  sigma2Had        = 2. * pow2( max( particleDataPtr->m0( 4900111), sigma) );
  thermalModel     = false;
  useWidthPre      = false;
  closePacking     = false;

}

//==========================================================================

// The HVStringZ class is used to select z in HV fragmentation.

//--------------------------------------------------------------------------

// Initialize data members of the string z selection.

void HVStringZ::init() {

  // Paramaters of Lund/Bowler symmetric fragmentation function.
  aLund    = parm("HiddenValley:aLund");
  bmqv2    = parm("HiddenValley:bmqv2");
  rFactqv  = parm("HiddenValley:rFactqv");

  // Use qv mass to set scale of bEff = b * m^2;
  mqv2     = pow2( particleDataPtr->m0( 4900101) );
  bLund    = bmqv2 / mqv2;

  // Mass of qv meson used to set stop scale for fragmentation iteration.
  mhvMeson = particleDataPtr->m0( 4900111);

}

//--------------------------------------------------------------------------

// Generate the fraction z that the next hadron will take using Lund/Bowler.

double HVStringZ::zFrag( int , int , double mT2) {

  // Shape parameters of Lund symmetric fragmentation function.
  double bShape = bLund * mT2;
  double cShape = 1. + rFactqv * bmqv2;
  return zLund( aLund, bShape, cShape);

}

//==========================================================================

// The HiddenValleyFragmentation class.

//--------------------------------------------------------------------------

// Initialize and save pointers.

bool HiddenValleyFragmentation::init() {

  // Check whether Hidden Valley fragmentation switched on, and SU(N).
  doHVfrag = flag("HiddenValley:fragment");
  if (mode("HiddenValley:Ngauge") < 2) doHVfrag = false;
  if (!doHVfrag) return false;

  // Several qv copies may be assumed, with separated or identical handling.
  separateFlav = flag("HiddenValley:separateFlav");
  nFlav        = mode("HiddenValley:nFlav");

  // Hidden Valley meson mass used to choose hadronization mode.
  mhvMeson = particleDataPtr->m0(4900111);

  // Minimal mass by initial flavour when separated handling.
  if (separateFlav) for (int i = 1; i <= nFlav; ++i) {
    mhvMin[i] = particleDataPtr->m0(4900001 + 110 * i);
    for (int j = 1; j < i; ++j) mhvMin[i] = min( mhvMin[i],
      particleDataPtr->m0(4900001 + 100 * i + 10 * j) );
    for (int j = i + 1; j <= nFlav; ++j) mhvMin[i] = min( mhvMin[i],
      particleDataPtr->m0(4900001 + 100 * j + 10 * i) );
    mhvMeson = min( mhvMeson, mhvMin[i]);
  }

  // Initialize the hvEvent instance of an event record.
  hvEvent.init( "(Hidden Valley fragmentation)", particleDataPtr);

  // Create HVStringFlav instance for HV-flavour selection.
  hvFlavSel.init();

  // Create HVStringPT instance for pT selection in HV fragmentation.
  hvPTSel.init();

  // Create HVStringZ instance for z selection in HV fragmentation.
  hvZSel.init();

  // Initialize auxiliary administrative class.
  hvColConfig.init(infoPtr, &hvFlavSel);

  // Initialize HV-string and HV-ministring fragmentation.
  hvStringFrag.init(&hvFlavSel, &hvPTSel, &hvZSel);
  hvMinistringFrag.init(&hvFlavSel, &hvPTSel, &hvZSel);

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Perform the fragmentation.

bool HiddenValleyFragmentation::fragment(Event& event) {

  // Reset containers for next event.
  hvEvent.reset();
  hvColConfig.clear();
  ihvParton.resize(0);

  // Extract HV-particles from event to hvEvent. Done if none found.
  if (!extractHVevent(event)) return true;

  // Assign HV-colours to hidden partons.
  if (!assignHVevent()) return false;

  // Store found string system. Analyze its properties.
  if (!hvColConfig.insert(ihvParton, hvEvent)) return false;

  // Collect sequentially all partons in the HV subsystem.
  // Copy also if already in order, or else history tracing may fail.
  hvColConfig.collect(0, hvEvent, false);

  // Mass used to decide how to fragment system.
  mSys = hvColConfig[0].mass;

  // Minimal meson masses given endpoints.
  double mMinEnd1 = mhvMeson;
  double mMinEnd2 = mhvMeson;
  if (separateFlav) {
    idEnd1 = hvEvent[hvColConfig[0].iParton.front()].idAbs() - 4900100;
    idEnd2 = hvEvent[hvColConfig[0].iParton.back()].idAbs() - 4900100;
    mMinEnd1 = mhvMin[idEnd1];
    mMinEnd2 = mhvMin[idEnd2];
  }

  // HV-string fragmentation when enough mass to produce >= 3 HV-mesons.
  if (mSys > mMinEnd1 + mMinEnd2 + 1.5 * mhvMeson) {
    if (!hvStringFrag.fragment( 0, hvColConfig, hvEvent)) return false;

  // HV-ministring fragmentation when enough mass to produce 2 HV-mesons.
  } else if (mSys > mMinEnd1 + mMinEnd2 + 0.1 * mhvMeson) {
    if (!hvMinistringFrag.fragment( 0, hvColConfig, hvEvent, true))
    return false;

  // If only enough mass for one HV-meson assume HV-glueballs emitted.
  } else if (!collapseToMeson()) return false;

  // Insert HV particles from hvEvent to event.
  insertHVevent(event);

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Extract HV-particles from event to hvEvent.

bool HiddenValleyFragmentation::extractHVevent(Event& event) {

  // Copy Hidden-Valley particles to special event record.
  for (int i = 0; i < event.size(); ++i) {
    int idAbs = event[i].idAbs();
    bool isHV = (idAbs > 4900000 && idAbs < 4900007)
             || (idAbs > 4900010 && idAbs < 4900017)
             || idAbs == 4900021
             || (idAbs > 4900100 && idAbs < 4900109);
    if (isHV) {
      int iHV = hvEvent.append( event[i]);
      // Convert HV-gluons into normal ones so as to use normal machinery.
      if (event[i].id() ==  4900021) hvEvent[iHV].id(21);
      // Second mother points back to position in complete event;
      // otherwise construct the HV history inside hvEvent.
      hvEvent[iHV].mothers( 0, i);
      hvEvent[iHV].daughters( 0, 0);
      hvEvent[iHV].cols( 0, 0);
      int iMother = event[i].mother1();
      for (int iHVM = 1; iHVM < hvEvent.size(); ++iHVM)
      if (hvEvent[iHVM].mother2() == iMother) {
        hvEvent[iHV].mother1( iHVM);
        if (hvEvent[iHVM].daughter1() == 0) hvEvent[iHVM].daughter1(iHV);
        else                                hvEvent[iHVM].daughter2(iHV);
      }
    }
  }

  // Fail if no HV particles found. (But that is OK as well.)
  hvOldSize = hvEvent.size();
  return (hvOldSize > 1);

}

//--------------------------------------------------------------------------

// Assign HV-colours to the partons in the extracted HV-event.

bool HiddenValleyFragmentation::assignHVevent() {

  // Initial colour - anticolour parton pair.
  hvEvent.initColTag();
  int colBeg = hvEvent.nextColTag();
  for (int iHV = 1; iHV < hvOldSize; ++iHV)
  if (hvEvent[iHV].mother1() == 0) {
    if (hvEvent[iHV].id() > 0) hvEvent[iHV].cols( colBeg, 0);
    else                       hvEvent[iHV].cols( 0, colBeg);
  }

  // Trace colours through full history.
  for (int iRad = 1; iRad < hvOldSize; ++iRad) {
    int iRadBef   = hvEvent[iRad].mother1();
    int statusRad = hvEvent[iRad].statusAbs();
    int iEmt      = iRad + 1;

    // Skip initial partons that already had colours assigned.
    if (hvEvent[iRad].col() != 0 || hvEvent[iRad].acol() != 0) {
      continue;

    // Copy of particle by resonance decay, recoil from normal sector,
    // beam-remnant recoil or pre-hadronization ordering.
    } else if (statusRad == 23 || statusRad == 52
      || (statusRad == 51 && (iEmt == hvOldSize
      || hvEvent[iEmt].statusAbs() != 51
      || hvEvent[iEmt].mother1() != iRadBef))
      || (statusRad > 60 && statusRad < 80) ) {
      hvEvent[iRad].cols( hvEvent[iRadBef].col(), hvEvent[iRadBef].acol());

    // Dipole branching (iRadBef, iRecBef) -> (iRad, iEmt) + (iEmt, iRec).
    } else if (iEmt < hvOldSize && statusRad == 51
      && hvEvent[iEmt].statusAbs() == 51
      && hvEvent[iEmt].mother1() == iRadBef) {
      int iRec    = iRad + 2;
      int iRecBef = (iRec < hvOldSize && hvEvent[iRec].statusAbs() == 52)
        ? hvEvent[iRec].mother1() : 0;

      // Find whether colour or anticolour end of dipole is radiating.
      int colBef  = hvEvent[iRadBef].col();
      int acolBef = hvEvent[iRadBef].acol();
      int colType = 0;
      if (acolBef == 0) colType = 1;
      else if (colBef == 0) colType = -1;
      else if (iRecBef > 0 && hvEvent[iRecBef].acol() == colBef) colType = 1;
      else if (iRecBef > 0 && hvEvent[iRecBef].col() == acolBef) colType = -1;
      // Recoil in normal sector: trace up to qv/Qv or qvbar/QvBar.
      else {
        int iUp = hvEvent[iRadBef].mother1();
        while (iUp > 0 && hvEvent[iUp].id() == hvEvent[iRad].id())
          iUp = hvEvent[iUp].mother1();
        if (iUp > 0) colType = (hvEvent[iUp].id() > 0) ? 1 : -1;
      }

      // Set new colours of radiator and emitted. Error checks.
      int colNew  = hvEvent.nextColTag();
      if (colType == 1) {
        hvEvent[iRad].cols( colNew, acolBef);
        hvEvent[iEmt].cols( colBef, colNew);
      } else if (colType == -1) {
        hvEvent[iRad].cols( colBef, colNew);
        hvEvent[iEmt].cols( colNew, acolBef);
      } else {
        infoPtr->errorMsg("Error in HiddenValleyFragmentation::extractHVevent:"
        " failed to assign dipole colours");
        return false;
      }
      ++iRad;
    } else {
      infoPtr->errorMsg("Error in HiddenValleyFragmentation::extractHVevent:"
      " failed to assign other colours");
      return false;
    }
  }

  // Pick up the colour end.
  int colNow = 0;
  for (int iHV = 1; iHV < hvOldSize; ++iHV)
  if (hvEvent[iHV].isFinal() && hvEvent[iHV].acol() == 0) {
    ihvParton.push_back( iHV);
    colNow = hvEvent[iHV].col();
  }

  // Trace colour by colour until reached anticolour end.
  while (colNow > 0) {
    for (int iHV = 1; iHV < hvOldSize; ++iHV)
    if (hvEvent[iHV].isFinal() && hvEvent[iHV].acol() == colNow) {
      ihvParton.push_back( iHV);
      colNow = hvEvent[iHV].col();
      break;
    }
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Collapse of light system to one HV-meson, by the emission of HV-glueballs.

bool HiddenValleyFragmentation::collapseToMeson() {

  // Lightest mass, given flavour content.
  int idhvLight   = 4900111;
  double mhvLight = mhvMeson;
  if (separateFlav) {
    idhvLight = 4900001 + 100 * max(idEnd1, idEnd2) + 10 * min(idEnd1, idEnd2);
    mhvLight  = particleDataPtr->m0(idhvLight);
  }

  // If too low mass then cannot do anything. Should not happen.
  if (mSys < 1.001 * mhvLight) {
    infoPtr->errorMsg("Error in HiddenValleyFragmentation::collapseToMeson:"
      " too low mass to do anything");
    return false;
  }

  // Choose mass of collective HV-glueball states flat between limits.
  double mhvGlue = (0.001 + 0.998 * rndmPtr->flat()) * (mSys - mhvLight);

  // Find momentum in rest frame, with isotropic "decay" angles.
  double pAbs = 0.5 * sqrtpos( pow2(mSys*mSys - mhvLight*mhvLight
    - mhvGlue*mhvGlue) - pow2(2. * mhvLight * mhvGlue) ) / mSys;
  double pz   = (2 * rndmPtr->flat() - 1.) * pAbs;
  double pT   = sqrtpos( pAbs*pAbs - pz*pz);
  double phi  = 2. * M_PI * rndmPtr->flat();
  double px   = pT * cos(phi);
  double py   = pT * sin(phi);

  // Construct four-vectors and boost them to event frame.
  Vec4 phvMeson( px, py, pz, sqrt(mhvLight*mhvLight + pAbs*pAbs) );
  Vec4 phvGlue( -px, -py, -pz, sqrt(mhvGlue*mhvGlue + pAbs*pAbs) );
  phvMeson.bst( hvColConfig[0].pSum );
  phvGlue.bst(  hvColConfig[0].pSum );

  // Add produced particles to the event record.
  vector<int> iParton = hvColConfig[0].iParton;
  int iFirst = hvEvent.append( idhvLight, 82,  iParton.front(),
    iParton.back(), 0, 0, 0, 0, phvMeson, mhvLight);
  int iLast  = hvEvent.append( 4900991, 82,  iParton.front(),
    iParton.back(), 0, 0, 0, 0, phvGlue, mhvGlue);

  // Mark original partons as hadronized and set their daughter range.
  for (int i = 0; i < int(iParton.size()); ++i) {
    hvEvent[ iParton[i] ].statusNeg();
    hvEvent[ iParton[i] ].daughters(iFirst, iLast);
  }

  // Done.
  return true;

}

//--------------------------------------------------------------------------

// Insert HV-particles from hvEvent to event.

bool HiddenValleyFragmentation::insertHVevent(Event& event) {

  // Offset for mother/daughter indices.
  hvNewSize = hvEvent.size();
  int nOffset = event.size() - hvOldSize;

  // Copy back HV-particles.
  int iNew, iMot1, iMot2, iDau1, iDau2;
  for (int iHV = hvOldSize; iHV < hvNewSize; ++iHV) {
    iNew = event.append( hvEvent[iHV]);

    // Restore HV-gluon codes. Do not keep HV-colours, to avoid confusion.
    if (hvEvent[iHV].id() == 21) event[iNew].id(4900021);
    event[iNew].cols( 0, 0);

    // Begin history construction.
    iMot1 = hvEvent[iHV].mother1();
    iMot2 = hvEvent[iHV].mother2();
    iDau1 = hvEvent[iHV].daughter1();
    iDau2 = hvEvent[iHV].daughter2();
    // Special mother for partons copied from event, else simple offset.
    // Also set daughters of mothers in original record.
    if (iMot1 > 0 && iMot1 < hvOldSize) {
      iMot1 = hvEvent[iMot1].mother2();
      event[iMot1].statusNeg();
      event[iMot1].daughter1(iNew);
    } else if (iMot1 > 0) iMot1 += nOffset;
    if (iMot2 > 0 && iMot2 < hvOldSize) {
      iMot2 = hvEvent[iMot2].mother2();
      event[iMot2].statusNeg();
      if (event[iMot2].daughter1() == 0) event[iMot2].daughter1(iNew);
      else                               event[iMot2].daughter2(iNew);
    } else if (iMot2 > 0) iMot2 += nOffset;
    if (iDau1 > 0) iDau1 += nOffset;
    if (iDau2 > 0) iDau2 += nOffset;
    event[iNew].mothers( iMot1, iMot2);
    event[iNew].daughters( iDau1, iDau2);
  }

  // Done.
  return true;

}

//==========================================================================

} // end namespace Pythia8
