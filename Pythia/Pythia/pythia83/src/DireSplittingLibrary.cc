// DireSplitingLibrary.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the Dire
// splitting library.

#include "Pythia8/DireSplittingLibrary.h"

namespace Pythia8 {

//==========================================================================

// class SplittingLibrary

//--------------------------------------------------------------------------

// Clean up
void DireSplittingLibrary::clear() {
  for ( std::unordered_map<string,DireSplitting*>::iterator
    it = splittings.begin();
    it != splittings.end(); ++it ) if (it->second) delete it->second;
  splittings.clear();
}

//--------------------------------------------------------------------------

// Initialisation.
void DireSplittingLibrary::init( Info* infoPtrIn, BeamParticle* beamA,
  BeamParticle* beamB, DireInfo* direInfo, DireHooks* hooks) {

  // Store infrastructure pointers.
  infoPtr      = infoPtrIn;
  settingsPtr     = infoPtr->settingsPtr;
  particleDataPtr = infoPtr->particleDataPtr;
  rndmPtr         = infoPtr->rndmPtr;
  beamAPtr        = beamA;
  beamBPtr        = beamB;
  coupSMPtr       = infoPtr->coupSMPtr;
  direInfoPtr     = direInfo;

  if (!hooksPtr) hooksPtr        = hooks;
  if (hooksPtr)  hasExternalHook = true;

  // Initialise splitting names.
  clear();
  initISR();
  initFSR();

  fsrQCD_1_to_1_and_21            = shash("Dire_fsr_qcd_1->1&21");
  fsrQCD_1_to_21_and_1            = shash("Dire_fsr_qcd_1->21&1");
  fsrQCD_21_to_21_and_21a         = shash("Dire_fsr_qcd_21->21&21a");
  fsrQCD_21_to_21_and_21b         = shash("Dire_fsr_qcd_21->21&21b");
  fsrQCD_21_to_1_and_1a           = shash("Dire_fsr_qcd_21->1&1a");
  fsrQCD_21_to_1_and_1b           = shash("Dire_fsr_qcd_21->1&1b");
  fsrQCD_1_to_2_and_1_and_2       = shash("Dire_fsr_qcd_1->2&1&2");
  fsrQCD_1_to_1_and_1_and_1       = shash("Dire_fsr_qcd_1->1&1&1");
  fsrQCD_1_to_1_and_21_notPartial = shash("Dire_fsr_qcd_1->1&21_notPartial");
  fsrQCD_21_to_21_and_21_notPartial =
    shash("Dire_fsr_qcd_21->21&21_notPartial");
  fsrQCD_21_to_1_and_1_notPartial = shash("Dire_fsr_qcd_21->1&1_notPartial");
  fsrQCD_1_to_1_and_21_and_21     = shash("Dire_fsr_qcd_1->1&21&21");
  fsrQCD_1_to_1_and_1_and_1a      = shash("Dire_fsr_qcd_1->1&d&dbar");
  fsrQCD_1_to_1_and_1_and_1b      = shash("Dire_fsr_qcd_1->1&dbar&d");
  fsrQCD_1_to_1_and_2_and_2a      = shash("Dire_fsr_qcd_1->1&u&ubar");
  fsrQCD_1_to_1_and_2_and_2b      = shash("Dire_fsr_qcd_1->1&ubar&u");
  fsrQCD_1_to_1_and_3_and_3a      = shash("Dire_fsr_qcd_1->1&s&sbar");
  fsrQCD_1_to_1_and_3_and_3b      = shash("Dire_fsr_qcd_1->1&sbar&s");
  fsrQCD_1_to_1_and_4_and_4a      = shash("Dire_fsr_qcd_1->1&c&cbar");
  fsrQCD_1_to_1_and_4_and_4b      = shash("Dire_fsr_qcd_1->1&cbar&c");
  fsrQCD_1_to_1_and_5_and_5a      = shash("Dire_fsr_qcd_1->1&b&bbar");
  fsrQCD_1_to_1_and_5_and_5b      = shash("Dire_fsr_qcd_1->1&bbar&b");
  fsrQCD_21_to_21_and_21_and_21   = shash("Dire_fsr_qcd_21->21&21&21");
  fsrQCD_21_to_21_and_1_and_1a    = shash("Dire_fsr_qcd_21->21&d&dbar");
  fsrQCD_21_to_21_and_1_and_1b    = shash("Dire_fsr_qcd_21->21&dbar&d");
  fsrQCD_21_to_21_and_2_and_2a    = shash("Dire_fsr_qcd_21->21&u&ubar");
  fsrQCD_21_to_21_and_2_and_2b    = shash("Dire_fsr_qcd_21->21&ubar&u");
  fsrQCD_21_to_21_and_3_and_3a    = shash("Dire_fsr_qcd_21->21&s&sbar");
  fsrQCD_21_to_21_and_3_and_3b    = shash("Dire_fsr_qcd_21->21&sbar&s");
  fsrQCD_21_to_21_and_4_and_4a    = shash("Dire_fsr_qcd_21->21&c&cbar");
  fsrQCD_21_to_21_and_4_and_4b    = shash("Dire_fsr_qcd_21->21&cbar&c");
  fsrQCD_21_to_21_and_5_and_5a    = shash("Dire_fsr_qcd_21->21&b&bbar");
  fsrQCD_21_to_21_and_5_and_5b    = shash("Dire_fsr_qcd_21->21&bbar&b");

  fsrQED_1_to_1_and_22            = shash("Dire_fsr_qed_1->1&22");
  fsrQED_1_to_22_and_1            = shash("Dire_fsr_qed_1->22&1");
  fsrQED_11_to_11_and_22          = shash("Dire_fsr_qed_11->11&22");
  fsrQED_11_to_22_and_11          = shash("Dire_fsr_qed_11->22&11");
  fsrQED_22_to_1_and_1a           = shash("Dire_fsr_qed_22->1&1a");
  fsrQED_22_to_1_and_1b           = shash("Dire_fsr_qed_22->1&1b");
  fsrQED_22_to_2_and_2a           = shash("Dire_fsr_qed_22->2&2a");
  fsrQED_22_to_2_and_2b           = shash("Dire_fsr_qed_22->2&2b");
  fsrQED_22_to_3_and_3a           = shash("Dire_fsr_qed_22->3&3a");
  fsrQED_22_to_3_and_3b           = shash("Dire_fsr_qed_22->3&3b");
  fsrQED_22_to_4_and_4a           = shash("Dire_fsr_qed_22->4&4a");
  fsrQED_22_to_4_and_4b           = shash("Dire_fsr_qed_22->4&4b");
  fsrQED_22_to_5_and_5a           = shash("Dire_fsr_qed_22->5&5a");
  fsrQED_22_to_5_and_5b           = shash("Dire_fsr_qed_22->5&5b");
  fsrQED_22_to_11_and_11a         = shash("Dire_fsr_qed_22->11&11a");
  fsrQED_22_to_11_and_11b         = shash("Dire_fsr_qed_22->11&11b");
  fsrQED_22_to_13_and_13a         = shash("Dire_fsr_qed_22->13&13a");
  fsrQED_22_to_13_and_13b         = shash("Dire_fsr_qed_22->13&13b");
  fsrQED_22_to_15_and_15a         = shash("Dire_fsr_qed_22->15&15a");
  fsrQED_22_to_15_and_15b         = shash("Dire_fsr_qed_22->15&15b");
  fsrQED_1_to_1_and_22_notPartial = shash("Dire_fsr_qed_1->1&22_notPartial");
  fsrQED_11_to_11_and_22_notPartial =
    shash("Dire_fsr_qed_11->11&22_notPartial");

  fsrEWK_1_to_1_and_23            = shash("Dire_fsr_ew_1->1&23");
  fsrEWK_1_to_23_and_1            = shash("Dire_fsr_ew_1->23&1");
  fsrEWK_23_to_1_and_1a           = shash("Dire_fsr_ew_23->1&1a");
  fsrEWK_23_to_1_and_1b           = shash("Dire_fsr_ew_23->1&1b");
  fsrEWK_24_to_1_and_1a           = shash("Dire_fsr_ew_24->1&1a");
  fsrEWK_24_to_1_and_1b           = shash("Dire_fsr_ew_24->1&1b");
  fsrEWK_25_to_24_and_24          = shash("Dire_fsr_ew_25->24&24");
  fsrEWK_25_to_22_and_22          = shash("Dire_fsr_ew_25->22&22");
  fsrEWK_25_to_21_and_21          = shash("Dire_fsr_ew_25->21&21");
  fsrEWK_24_to_24_and_22          = shash("Dire_fsr_ew_24->24&22");

  isrQCD_1_to_1_and_21            = shash("Dire_isr_qcd_1->1&21");
  isrQCD_21_to_1_and_1            = shash("Dire_isr_qcd_21->1&1");
  isrQCD_21_to_21_and_21a         = shash("Dire_isr_qcd_21->21&21a");
  isrQCD_21_to_21_and_21b         = shash("Dire_isr_qcd_21->21&21b");
  isrQCD_1_to_21_and_1            = shash("Dire_isr_qcd_1->21&1");
  isrQCD_1_to_2_and_1_and_2       = shash("Dire_isr_qcd_1->2&1&2");
  isrQCD_1_to_1_and_1_and_1       = shash("Dire_isr_qcd_1->1&1&1");

  isrQED_1_to_1_and_22            = shash("Dire_isr_qed_1->1&22");
  isrQED_11_to_11_and_22          = shash("Dire_isr_qed_11->11&22");
  isrQED_1_to_22_and_1            = shash("Dire_isr_qed_1->22&1");
  isrQED_11_to_22_and_11          = shash("Dire_isr_qed_11->22&11");
  isrQED_22_to_1_and_1            = shash("Dire_isr_qed_22->1&1");
  isrQED_22_to_11_and_11          = shash("Dire_isr_qed_22->11&11");

  isrEWK_1_to_1_and_23            = shash("Dire_isr_ew_1->1&23");

  fsrU1N_1_to_1_and_22            = shash("Dire_fsr_u1new_1->1&22");
  fsrU1N_1_to_22_and_1            = shash("Dire_fsr_u1new_1->22&1");
  fsrU1N_11_to_11_and_22          = shash("Dire_fsr_u1new_11->11&22");
  fsrU1N_11_to_22_and_11          = shash("Dire_fsr_u1new_11->22&11");
  fsrU1N_22_to_1_and_1a           = shash("Dire_fsr_u1new_22->1&1a");
  fsrU1N_22_to_1_and_1b           = shash("Dire_fsr_u1new_22->1&1b");
  fsrU1N_22_to_2_and_2a           = shash("Dire_fsr_u1new_22->2&2a");
  fsrU1N_22_to_2_and_2b           = shash("Dire_fsr_u1new_22->2&2b");
  fsrU1N_22_to_3_and_3a           = shash("Dire_fsr_u1new_22->3&3a");
  fsrU1N_22_to_3_and_3b           = shash("Dire_fsr_u1new_22->3&3b");
  fsrU1N_22_to_4_and_4a           = shash("Dire_fsr_u1new_22->4&4a");
  fsrU1N_22_to_4_and_4b           = shash("Dire_fsr_u1new_22->4&4b");
  fsrU1N_22_to_5_and_5a           = shash("Dire_fsr_u1new_22->5&5a");
  fsrU1N_22_to_5_and_5b           = shash("Dire_fsr_u1new_22->5&5b");
  fsrU1N_22_to_11_and_11a         = shash("Dire_fsr_u1new_22->11&11a");
  fsrU1N_22_to_11_and_11b         = shash("Dire_fsr_u1new_22->11&11b");
  fsrU1N_22_to_13_and_13a         = shash("Dire_fsr_u1new_22->13&13a");
  fsrU1N_22_to_13_and_13b         = shash("Dire_fsr_u1new_22->13&13b");
  fsrU1N_22_to_15_and_15a         = shash("Dire_fsr_u1new_22->15&15a");
  fsrU1N_22_to_15_and_15b         = shash("Dire_fsr_u1new_22->15&15b");
  fsrU1N_22_to_211_and_211a       = shash("Dire_fsr_u1new_22->211&211a");
  fsrU1N_22_to_211_and_211b       = shash("Dire_fsr_u1new_22->211&211b");

  isrU1N_1_to_1_and_22            = shash("Dire_isr_u1new_1->1&22");
  isrU1N_1_to_22_and_1            = shash("Dire_isr_u1new_1->22&1");
  isrU1N_22_to_1_and_1            = shash("Dire_isr_u1new_22->1&1");
  isrU1N_11_to_11_and_22          = shash("Dire_isr_u1new_11->11&22");
  isrU1N_11_to_22_and_11          = shash("Dire_isr_u1new_11->22&11");
  isrU1N_22_to_11_and_11          = shash("Dire_isr_u1new_22->11&11");

  // Done.
}

//--------------------------------------------------------------------------

void DireSplittingLibrary::initFSR() {

  // Add corrections to the LO splitting kernel.
  // order  = 0 -> A1, B1
  // order  = 1 -> A1, A2, B1
  // order  = 2 -> A1, A2, A3, B1
  // order  = 3 -> A1, A2, A3, B1, B2
  int order = settingsPtr->mode("DireTimes:kernelOrder");
  string name = "";

  // QCD splittings.
  if (settingsPtr->flag("TimeShower:QCDshower")) {
    // Q -> Q G, soft part + collinear
    name = "Dire_fsr_qcd_1->1&21";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2QG( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // Q -> G Q, soft part + collinear
    // At leading order, this can be trivially combined with Q->QG because of
    // symmetry under z --> 1 -z .
    // Since this is no longer possible at NLO, we keep the kernels separately.
    name = "Dire_fsr_qcd_1->21&1";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2GQ( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // G -> G G, soft part + collinear
    name = "Dire_fsr_qcd_21->21&21a";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2GG1( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // G -> G G, soft part + collinear
    name = "Dire_fsr_qcd_21->21&21b";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2GG2( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // G -> Q Q (regular DGLAP kernel)
    name = "Dire_fsr_qcd_21->1&1a";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2QQ1( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // G -> Q Q (regular DGLAP kernel)
    name = "Dire_fsr_qcd_21->1&1b";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2QQ2( name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // Q -> Q' Q Q'bar (pure NLO kernel)
    name = "Dire_fsr_qcd_1->2&1&2";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2qQqbarDist( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Qbar Q Q (pure NLO kernel)
    name = "Dire_fsr_qcd_1->1&1&1";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2QbarQQId( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    // Full DGLAP kernels for shower w/o color-connected recoiler.
    name = "Dire_fsr_qcd_1->1&21_notPartial";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2QG_notPartial( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qcd_21->21&21_notPartial";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2GG_notPartial( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qcd_21->1&1_notPartial";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2QQ_notPartial( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q G G
    name = "Dire_fsr_qcd_1->1&21&21";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2QGG( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q d dbar
    name = "Dire_fsr_qcd_1->1&d&dbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar( 1, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q dbar d
    name = "Dire_fsr_qcd_1->1&dbar&d";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar(-1, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q u ubar
    name = "Dire_fsr_qcd_1->1&u&ubar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar( 2, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q ubar u
    name = "Dire_fsr_qcd_1->1&ubar&u";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar(-2, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q s sbar
    name = "Dire_fsr_qcd_1->1&s&sbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar( 3, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q sbar s
    name = "Dire_fsr_qcd_1->1&sbar&s";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar(-3, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q c cbar
    name = "Dire_fsr_qcd_1->1&c&cbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar( 4, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q cbar c
    name = "Dire_fsr_qcd_1->1&cbar&c";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar(-4, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q b bbar
    name = "Dire_fsr_qcd_1->1&b&bbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar( 5, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q bbar b
    name = "Dire_fsr_qcd_1->1&bbar&b";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_Q2Qqqbar(-5, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G G G
    name = "Dire_fsr_qcd_21->21&21&21";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2GGG( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G d dbar
    name = "Dire_fsr_qcd_21->21&d&dbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar( 1, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G dbar d
    name = "Dire_fsr_qcd_21->21&dbar&d";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar(-1, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G u ubar
    name = "Dire_fsr_qcd_21->21&u&ubar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar( 2, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G ubar u
    name = "Dire_fsr_qcd_21->21&ubar&u";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar(-2, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G s sbar
    name = "Dire_fsr_qcd_21->21&s&sbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar( 3, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G sbar s
    name = "Dire_fsr_qcd_21->21&sbar&s";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar(-3, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G c cbar
    name = "Dire_fsr_qcd_21->21&c&cbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar( 4, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G cbar c
    name = "Dire_fsr_qcd_21->21&cbar&c";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar(-4, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G b bbar
    name = "Dire_fsr_qcd_21->21&b&bbar";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar( 5, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G bbar b
    name = "Dire_fsr_qcd_21->21&bbar&b";
    splittings.insert( make_pair( name, new Dire_fsr_qcd_G2Gqqbar(-5, name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // QED splittings
  if (settingsPtr->flag("TimeShower:QEDshowerByQ")) {
    name = "Dire_fsr_qed_1->1&22";
    splittings.insert( make_pair( name, new Dire_fsr_qed_Q2QA( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_1->22&1";
    splittings.insert( make_pair( name, new Dire_fsr_qed_Q2AQ( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->1&1a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 1, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->1&1b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-1, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->2&2a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 2, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->2&2b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-2, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->3&3a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 3, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->3&3b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-3, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->4&4a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 4, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->4&4b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-4, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->5&5a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 5, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->5&5b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-5, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // Complete DGLAG kernel for photon emission off charge, as e.g. needed
    // after charged hadron decay.
    name = "Dire_fsr_qed_1->1&22_notPartial";
    splittings.insert( make_pair( name, new Dire_fsr_qed_Q2QA_notPartial( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
  }

  if (settingsPtr->flag("TimeShower:QEDshowerByL")) {
    name = "Dire_fsr_qed_11->11&22";
    splittings.insert( make_pair( name, new Dire_fsr_qed_L2LA( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_11->22&11";
    splittings.insert( make_pair( name, new Dire_fsr_qed_L2AL( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->11&11a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 11, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->11&11b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-11, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->13&13a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 13, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->13&13b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-13, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->15&15a";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF( 15, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    name = "Dire_fsr_qed_22->15&15b";
    splittings.insert( make_pair( name, new Dire_fsr_qed_A2FF(-15, name, order,
      settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
      infoPtr, direInfoPtr)) );
    // Complete DGLAG kernel for photon emission off charge, as e.g. needed
    // after charged hadron decay.
    name = "Dire_fsr_qed_11->11&22_notPartial";
    splittings.insert( make_pair( name, new Dire_fsr_qed_L2LA_notPartial( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // H -> A A
  name = "Dire_fsr_ew_25->22&22";
  splittings.insert( make_pair( name, new Dire_fsr_ew_H2AA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr, direInfoPtr)) );

  name = "Dire_fsr_ew_25->21&21";
  splittings.insert( make_pair( name, new Dire_fsr_ew_H2GG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr, direInfoPtr)) );

  name = "Dire_fsr_ew_24->24&22";
  splittings.insert( make_pair( name, new Dire_fsr_ew_W2WA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr, direInfoPtr)) );

  // New U(1) splittings
  if (settingsPtr->flag("TimeShower:U1newShowerByQ")) {
    name = "Dire_fsr_u1new_22->211&211a";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2SS(211, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_22->211&211b";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2SS(-211, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
  }

  if (settingsPtr->flag("TimeShower:U1newShowerByL")) {
    name = "Dire_fsr_u1new_11->11&22";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_L2LA( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_11->22&11";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_L2AL( name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_22->11&11a";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2FF( 11, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_22->11&11b";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2FF(-11, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_22->13&13a";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2FF( 13, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_fsr_u1new_22->13&13b";
    splittings.insert( make_pair( name, new Dire_fsr_u1new_A2FF(-13, name,
      order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
      coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // Read more kernels.
  if (hasExternalHook && hooksPtr->canLoadFSRKernels())
    hooksPtr->doLoadFSRKernels(splittings);

  // Done.
}

//--------------------------------------------------------------------------

void DireSplittingLibrary::initISR() {

  // Add corrections to the LO splitting kernel.
  // order  = 0 -> A1, B1
  // order  = 1 -> A1, A2, B1
  // order  = 2 -> A1, A2, A3, B1
  // order  = 3 -> A1, A2, A3, B1, B2
  int order = settingsPtr->mode("DireSpace:kernelOrder");
  string name = "";

  // QCD splittings.
  if (settingsPtr->flag("SpaceShower:QCDshower")) {
    // Q -> Q G, soft and collinear part.
    name = "Dire_isr_qcd_1->1&21";
    splittings.insert( make_pair( name, new Dire_isr_qcd_Q2QG( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> Q Q~ (regular DGLAP kernel)
    name = "Dire_isr_qcd_21->1&1";
    splittings.insert( make_pair( name, new Dire_isr_qcd_G2QQ( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // G -> G G, soft part.
    name = "Dire_isr_qcd_21->21&21a";
    splittings.insert( make_pair( name, new Dire_isr_qcd_G2GG1( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    name = "Dire_isr_qcd_21->21&21b";
    splittings.insert( make_pair( name, new Dire_isr_qcd_G2GG2( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> G Q (regular DGLAP kernel)
    name = "Dire_isr_qcd_1->21&1";
    splittings.insert( make_pair( name, new Dire_isr_qcd_Q2GQ( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Q' Q Q'bar (pure NLO kernel)
    name = "Dire_isr_qcd_1->2&1&2";
    splittings.insert( make_pair( name, new Dire_isr_qcd_Q2qQqbarDist( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
    // Q -> Qbar Q Q (pure NLO kernel)
    name = "Dire_isr_qcd_1->1&1&1";
    splittings.insert( make_pair( name, new Dire_isr_qcd_Q2QbarQQId( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // QED splittings
  if (settingsPtr->flag("SpaceShower:QEDshowerByQ")) {
    // Q -> Q A, soft and collinear part.
    name = "Dire_isr_qed_1->1&22";
    splittings.insert( make_pair( name, new Dire_isr_qed_Q2QA( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
  }

  if (settingsPtr->flag("SpaceShower:QEDshowerByL")) {
    // L -> L A, soft and collinear part.
    name = "Dire_isr_qed_11->11&22";
    splittings.insert( make_pair( name, new Dire_isr_qed_L2LA( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // New U(1) splittings
  if (settingsPtr->flag("SpaceShower:U1newShowerByL")) {
    name = "Dire_isr_u1new_11->11&22";
    splittings.insert( make_pair( name, new Dire_isr_u1new_L2LA( name,
    order, settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr,
    coupSMPtr, infoPtr, direInfoPtr)) );
  }

  // Read more kernels.
  if (hasExternalHook && hooksPtr->canLoadISRKernels())
    hooksPtr->doLoadISRKernels(splittings);

  // Done.
}

//--------------------------------------------------------------------------

// Overload index operator to access element of splitting vector.

DireSplitting* DireSplittingLibrary::operator[](string id) {
  if (splittings.find(id) != splittings.end()) return splittings[id];
  return NULL;
}

const DireSplitting* DireSplittingLibrary::operator[](string id) const {
  if (splittings.find(id) != splittings.end()) return splittings.at(id);
  return NULL;
}

//--------------------------------------------------------------------------

// Generate identifier of radiator before the splitting

vector<int> DireSplittingLibrary::getSplittingRadBefID(const Event& event,
  int rad, int emt) { return getSplittingRadBefID_new(event, rad, emt); }

//--------------------------------------------------------------------------

vector<int> DireSplittingLibrary::getSplittingRadBefID_new(const Event& event,
  int rad, int emt) {

  vector<int>radBefIDs;
  for ( std::unordered_map<string,DireSplitting*>::iterator it =
          splittings.begin();
    it != splittings.end(); ++it ) {
    int idNow = it->second->radBefID(event[rad].id(), event[emt].id());
    if (idNow != 0) radBefIDs.push_back(idNow);
  }

  return radBefIDs;

}

//--------------------------------------------------------------------------

// Generate name for a splitting

vector<string> DireSplittingLibrary::getSplittingName_new(const Event& event,
  int rad, int emt) {

  vector<string> names;
  for ( std::unordered_map<string,DireSplitting*>::iterator it =
          splittings.begin();
    it != splittings.end(); ++it ) {

    int type = event[rad].isFinal() ? 1 :-1;
    if (type < 0 && it->first.find("isr") == string::npos) continue;
    if (type > 0 && it->first.find("fsr") == string::npos) continue;

    // Find radiator before emission.
    int idNow = it->second->radBefID(event[rad].id(), event[emt].id());

    // Now check that after emission, we would find same flavors.
    vector <int> radAndEmt;
    if (idNow != 0) radAndEmt = it->second->radAndEmt(idNow,0);

    bool valid = false;
    if (radAndEmt.size() == 2) {
      if (radAndEmt[1] == event[emt].id())
        valid = true;
      if (event[emt].isQuark() && event[emt].colType() > 0
        && radAndEmt[1] == 1)
        valid = true;
      if (event[emt].isQuark() && event[emt].colType() < 0
        && radAndEmt[1] == 1)
        valid = true;
    }

    // Found valid splitting name.
    if (valid && idNow != 0) names.push_back(it->first);
  }

  return names;

}

//--------------------------------------------------------------------------

// Generate name for a splitting

vector<string> DireSplittingLibrary::getSplittingName(const Event& event,
  int rad, int emt) { return getSplittingName_new(event, rad, emt); }

//--------------------------------------------------------------------------

// Return the total number of emissions for a particular splitting

int DireSplittingLibrary::nEmissions( string name ) {

  std::unordered_map<string, DireSplitting*>::iterator it =
    splittings.find(name);
  if (it != splittings.end() && abs(it->second->kinMap()) == 2) return 2;

  // Flavour-changing 1->3 splitting for FSR implemented.
  if ( name.find("Dire_fsr_qcd_1->2&1&2") != string::npos ) return 2;

  // Flavour-preserving 1->3 splitting for FSR implemented.
  if ( name.find("Dire_fsr_qcd_1->1&1&1") != string::npos ) return 2;

  // Flavour-changing 1->3 splitting for FSR implemented.
  if ( name.find("Dire_isr_qcd_1->2&1&2") != string::npos ) return 2;

  // Flavour-preserving 1->3 splitting for FSR implemented.
  if ( name.find("Dire_isr_qcd_1->1&1&1") != string::npos ) return 2;

  // Default is one emission.
  return 1;

}

//==========================================================================

} // end namespace Pythia8
