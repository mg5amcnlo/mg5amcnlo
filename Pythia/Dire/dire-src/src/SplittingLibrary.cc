
#include "Dire/SplittingLibrary.h"

namespace Pythia8 {
 
//==========================================================================

// class SplittingLibrary

//--------------------------------------------------------------------------

// Clean up
void SplittingLibrary::clear() {
  for ( map<string,Splitting*>::iterator it = splittings.begin();
    it != splittings.end(); ++it ) if (it->second) delete it->second;
  splittings.clear();
}

//--------------------------------------------------------------------------

// Initialisation.
void SplittingLibrary::init(Settings* settings, ParticleData* particleData,
  Rndm* rndm, BeamParticle* beamA, BeamParticle* beamB, CoupSM* coupSM,
  Info* info, Hooks* hooks) {

  // Store infrastructure pointers.
  settingsPtr     = settings;
  particleDataPtr = particleData;
  rndmPtr         = rndm;
  beamAPtr        = beamA;
  beamBPtr        = beamB;
  coupSMPtr       = coupSM;
  infoPtr         = info;

  if (!hooksPtr) hooksPtr        = hooks;
  if (hooksPtr)  hasExternalHook = true;

  // Initialise splitting names.
  clear();
  initISR();
  initFSR();

  fsrQCD_1_to_1_and_21            = shash("fsr_qcd_1->1&21_CS");
  fsrQCD_1_to_21_and_1            = shash("fsr_qcd_1->21&1_CS");
  fsrQCD_21_to_21_and_21a         = shash("fsr_qcd_21->21&21a_CS");
  fsrQCD_21_to_21_and_21b         = shash("fsr_qcd_21->21&21b_CS");
  fsrQCD_21_to_1_and_1a           = shash("fsr_qcd_21->1&1a_CS");
  fsrQCD_21_to_1_and_1b           = shash("fsr_qcd_21->1&1b_CS");
  fsrQCD_1_to_2_and_1_and_2       = shash("fsr_qcd_1->2&1&2_CS");
  fsrQCD_1_to_1_and_1_and_1       = shash("fsr_qcd_1->1&1&1_CS");
  fsrQCD_1_to_1_and_21_notPartial = shash("fsr_qcd_1->1&21_notPartial");
  fsrQCD_1_to_1_and_21_and_21     = shash("fsr_qcd_1->1&21&21_CS");
  fsrQCD_21_to_21_and_21_and_21   = shash("fsr_qcd_21->21&21&21_CS");
  fsrQCD_21_to_1_and_1_and_21     = shash("fsr_qcd_21->1&1&21_CS");
  fsrQCD_21_to_2_and_2_and_21     = shash("fsr_qcd_21->2&2&21_CS");
  fsrQED_1_to_1_and_22            = shash("fsr_qed_1->1&22_CS");
  fsrQED_1_to_22_and_1            = shash("fsr_qed_1->22&1_CS");
  fsrQED_11_to_11_and_22          = shash("fsr_qed_11->11&22_CS");
  fsrQED_11_to_22_and_11          = shash("fsr_qed_11->22&11_CS");
  fsrQED_22_to_1_and_1a           = shash("fsr_qed_22->1&1a_CS");
  fsrQED_22_to_1_and_1b           = shash("fsr_qed_22->1&1b_CS");
  fsrQED_22_to_2_and_2a           = shash("fsr_qed_22->2&2a_CS");
  fsrQED_22_to_2_and_2b           = shash("fsr_qed_22->2&2b_CS");
  fsrQED_22_to_3_and_3a           = shash("fsr_qed_22->3&3a_CS");
  fsrQED_22_to_3_and_3b           = shash("fsr_qed_22->3&3b_CS");
  fsrQED_22_to_4_and_4a           = shash("fsr_qed_22->4&4a_CS");
  fsrQED_22_to_4_and_4b           = shash("fsr_qed_22->4&4b_CS");
  fsrQED_22_to_5_and_5a           = shash("fsr_qed_22->5&5a_CS");
  fsrQED_22_to_5_and_5b           = shash("fsr_qed_22->5&5b_CS");
  fsrQED_22_to_11_and_11a         = shash("fsr_qed_22->11&11a_CS");
  fsrQED_22_to_11_and_11b         = shash("fsr_qed_22->11&11b_CS");
  fsrQED_22_to_13_and_13a         = shash("fsr_qed_22->13&13a_CS");
  fsrQED_22_to_13_and_13b         = shash("fsr_qed_22->13&13b_CS");
  fsrQED_22_to_15_and_15a         = shash("fsr_qed_22->15&15a_CS");
  fsrQED_22_to_15_and_15b         = shash("fsr_qed_22->15&15b_CS");
  fsrEWK_1_to_1_and_23            = shash("fsr_ew_1->1&23_CS");
  fsrEWK_1_to_23_and_1            = shash("fsr_ew_1->23&1_CS");
  fsrEWK_23_to_1_and_1a           = shash("fsr_ew_23->1&1a_CS");
  fsrEWK_23_to_1_and_1b           = shash("fsr_ew_23->1&1b_CS");
  fsrEWK_24_to_1_and_1a           = shash("fsr_ew_24->1&1a_CS");
  fsrEWK_24_to_1_and_1b           = shash("fsr_ew_24->1&1b_CS");
  fsrEWK_25_to_24_and_24          = shash("fsr_ew_25->24&24_CS");
  isrQCD_1_to_1_and_21            = shash("isr_qcd_1->1&21_CS");
  isrQCD_21_to_1_and_1            = shash("isr_qcd_21->1&1_CS");
  isrQCD_21_to_21_and_21a         = shash("isr_qcd_21->21&21a_CS");
  isrQCD_21_to_21_and_21b         = shash("isr_qcd_21->21&21b_CS");
  isrQCD_1_to_21_and_1            = shash("isr_qcd_1->21&1_CS");
  isrQCD_1_to_2_and_1_and_2       = shash("isr_qcd_1->2&1&2_CS");
  isrQCD_1_to_1_and_1_and_1       = shash("isr_qcd_1->1&1&1_CS");
  isrQED_1_to_1_and_22            = shash("isr_qed_1->1&22_CS");
  isrQED_11_to_11_and_22          = shash("isr_qed_11->11&22_CS");
  isrQED_1_to_22_and_1            = shash("isr_qed_1->22&1_CS");
  isrQED_11_to_22_and_11          = shash("isr_qed_11->22&11_CS");
  isrQED_22_to_1_and_1            = shash("isr_qed_22->1&1_CS");
  isrQED_22_to_11_and_11          = shash("isr_qed_22->11&11_CS");
  isrEWK_1_to_1_and_23            = shash("isr_ew_1->1&23_CS");

  // Done.
}

//--------------------------------------------------------------------------

void SplittingLibrary::initFSR() {

  /********************* FSR ******************************************/

  // Add corrections to the LO splitting kernel.
  // order  = 0 -> A1, B1
  // order  = 1 -> A1, A2, B1
  // order  = 2 -> A1, A2, A3, B1
  // order  = 3 -> A1, A2, A3, B1, B2
  int order = settingsPtr->mode("DireTimes:kernelOrder");
  string name = "";

  // Q -> Q G, soft part + collinear
  name = "fsr_qcd_1->1&21_CS";
  splittings.insert( make_pair( name, new fsr_qcd_Q2QG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> G Q, soft part + collinear
  // At leading order, this can be trivially combined with Q->QG because of
  // symmetry under z --> 1 -z .
  // Since this is no longer possible at NLO, we keep the kernels separately.
  name = "fsr_qcd_1->21&1_CS";
  splittings.insert( make_pair( name, new fsr_qcd_Q2GQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> G G, soft part + collinear
  name = "fsr_qcd_21->21&21a_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2GG1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> G G, soft part + collinear
  name = "fsr_qcd_21->21&21b_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2GG2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> Q Q (regular DGLAP kernel)
  name = "fsr_qcd_21->1&1a_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2QQ1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> Q Q (regular DGLAP kernel)
  name = "fsr_qcd_21->1&1b_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2QQ2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Q' Q Q'bar (pure NLO kernel)
  name = "fsr_qcd_1->2&1&2_CS";
  splittings.insert( make_pair( name, new fsr_qcd_Q2qQqbarDist( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Qbar Q Q (pure NLO kernel)
  name = "fsr_qcd_1->1&1&1_CS";
  splittings.insert( make_pair( name, new fsr_qcd_Q2QbarQQId( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  name = "fsr_qcd_1->1&21_notPartial";
  splittings.insert( make_pair( name, new fsr_qcd_Q2QG_notPartial( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  /*// New 1->3 kernels.
  // Q -> Q G G
  name = "fsr_qcd_1->1&21&21_CS";
  splittings.insert( make_pair( name, new fsr_qcd_Q2QGG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  // G -> G G G
  name = "fsr_qcd_21->21&21&21_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2GGG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  // G -> Q Q~ G, one kernel for each flavor.
  name = "fsr_qcd_21->1&1&21_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2DDG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  name = "fsr_qcd_21->2&2&21_CS";
  splittings.insert( make_pair( name, new fsr_qcd_G2UUG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );*/

  // QED splittings

  // Q -> Q A, soft part + collinear
  name = "fsr_qed_1->1&22_CS";
  splittings.insert( make_pair( name, new fsr_qed_Q2QA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> A Q, soft part + collinear
  name = "fsr_qed_1->22&1_CS";
  splittings.insert( make_pair( name, new fsr_qed_Q2AQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Q A, soft part + collinear
  name = "fsr_qed_11->11&22_CS";
  splittings.insert( make_pair( name, new fsr_qed_L2LA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> A Q, soft part + collinear
  name = "fsr_qed_11->22&11_CS";
  splittings.insert( make_pair( name, new fsr_qed_L2AL( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // A -> Q Q~ (regular DGLAP kernel)
  name = "fsr_qed_22->1&1a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 1, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->1&1b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-1, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->2&2a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 2, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->2&2b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-2, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->3&3a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 3, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->3&3b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-3, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->4&4a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 4, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->4&4b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-4, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->5&5a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 5, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->5&5b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-5, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  // A -> L L~ (regular DGLAP kernel)
  name = "fsr_qed_22->11&11a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 11, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->11&11b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-11, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->13&13a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 13, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->13&13b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-13, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->15&15a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF( 15, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );
  name = "fsr_qed_22->15&15b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2FF(-15, name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr,
    infoPtr)) );

  /*// EW splittings

  // Q -> Q Z
  name = "fsr_ew_1->1&23_CS";
  splittings.insert( make_pair( name, new fsr_ew_Q2QZ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Z Q
  name = "fsr_ew_1->23&1_CS";
  splittings.insert( make_pair( name, new fsr_ew_Q2ZQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Z -> Q Q~
  name = "fsr_ew_23->1&1a_CS";
  splittings.insert( make_pair( name, new fsr_ew_Z2QQ1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Z -> Q~ Q
  name = "fsr_ew_23->1&1b_CS";
  splittings.insert( make_pair( name, new fsr_ew_Z2QQ2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // W -> Q Q~
  name = "fsr_ew_24->1&1a_CS";
  splittings.insert( make_pair( name, new fsr_ew_W2QQ1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // W -> Q~ Q
  name = "fsr_ew_24->1&1b_CS";
  splittings.insert( make_pair( name, new fsr_ew_W2QQ2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // H -> W W
  name = "fsr_ew_25->24&24_CS";
  splittings.insert( make_pair( name, new fsr_ew_H2WW( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  */

  // Read more kernels.
  if (hasExternalHook && hooksPtr->canLoadFSRKernels())
    hooksPtr->doLoadFSRKernels(splittings);

  // Done.
}

//--------------------------------------------------------------------------

void SplittingLibrary::initISR() {

  /********************* ISR off Drell-Yan ****************************/

  // Add corrections to the LO splitting kernel.
  // order  = 0 -> A1, B1
  // order  = 1 -> A1, A2, B1
  // order  = 2 -> A1, A2, A3, B1
  // order  = 3 -> A1, A2, A3, B1, B2
  int order = settingsPtr->mode("DireSpace:kernelOrder");
  string name = "";

  // Q -> Q G, soft and collinear part.
  name = "isr_qcd_1->1&21_CS";
  splittings.insert( make_pair( name, new isr_qcd_Q2QG( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> Q Q~ (regular DGLAP kernel)
  name = "isr_qcd_21->1&1_CS";
  splittings.insert( make_pair( name, new isr_qcd_G2QQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // G -> G G, soft part.
  name = "isr_qcd_21->21&21a_CS";
  splittings.insert( make_pair( name, new isr_qcd_G2GG1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  name = "isr_qcd_21->21&21b_CS";
  splittings.insert( make_pair( name, new isr_qcd_G2GG2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> G Q (regular DGLAP kernel)
  name = "isr_qcd_1->21&1_CS";
  splittings.insert( make_pair( name, new isr_qcd_Q2GQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Q' Q Q'bar (pure NLO kernel)
  name = "isr_qcd_1->2&1&2_CS";
  splittings.insert( make_pair( name, new isr_qcd_Q2qQqbarDist( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // Q -> Qbar Q Q (pure NLO kernel)
  name = "isr_qcd_1->1&1&1_CS";
  splittings.insert( make_pair( name, new isr_qcd_Q2QbarQQId( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // QED splittings

  // Q -> Q A, soft and collinear part.
  name = "isr_qed_1->1&22_CS";
  splittings.insert( make_pair( name, new isr_qed_Q2QA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // L -> L A, soft and collinear part.
  name = "isr_qed_11->11&22_CS";
  splittings.insert( make_pair( name, new isr_qed_L2LA( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  /*// Q -> A Q (regular DGLAP kernel)
  name = "isr_qed_1->22&1_CS";
  splittings.insert( make_pair( name, new isr_qed_Q2AQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // L -> A L (regular DGLAP kernel)
  name = "isr_qed_11->22&11_CS";
  splittings.insert( make_pair( name, new isr_qed_L2AL( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // A -> Q Q~ (regular DGLAP kernel)
  name = "isr_qed_22->1&1_CS";
  splittings.insert( make_pair( name, new isr_qed_A2QQ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // A -> L L~ (regular DGLAP kernel)
  name = "isr_qed_22->11&11_CS";
  splittings.insert( make_pair( name, new isr_qed_A2LL( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  */

  /*// EW splittings

  // Q -> Q Z
  name = "isr_ew_1->1&23_CS";
  splittings.insert( make_pair( name, new isr_ew_Q2QZ( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  */

  // Read more kernels.
  if (hasExternalHook && hooksPtr->canLoadISRKernels())
    hooksPtr->doLoadISRKernels(splittings);

  // Done.
}

//--------------------------------------------------------------------------

// Overload index operator to access element of splitting vector.

Splitting* SplittingLibrary::operator[](string id) {
  if (splittings.find(id) != splittings.end()) return splittings[id];
  return NULL;
}

const Splitting* SplittingLibrary::operator[](string id) const {
  if (splittings.find(id) != splittings.end()) return splittings.at(id);
  return NULL;
}

//--------------------------------------------------------------------------

// Generate identifier of radiator before the splitting

vector<int> SplittingLibrary::getSplittingRadBefID(const Event& event,
  int rad, int emt) { return getSplittingRadBefID_new(event, rad, emt); }

//--------------------------------------------------------------------------

vector<int> SplittingLibrary::getSplittingRadBefID_new(const Event& event,
  int rad, int emt) {

  vector<int>radBefIDs;
  for ( map<string,Splitting*>::iterator it = splittings.begin();
    it != splittings.end(); ++it ) {
    int idNow = it->second->radBefID(event[rad].id(), event[emt].id());
    if (idNow != 0) radBefIDs.push_back(idNow);
  }

  return radBefIDs;

}

//--------------------------------------------------------------------------

// Generate name for a splitting

vector<string> SplittingLibrary::getSplittingName_new(const Event& event, int rad,
  int emt) {

  vector<string> names;
  for ( map<string,Splitting*>::iterator it = splittings.begin();
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

vector<string> SplittingLibrary::getSplittingName(const Event& event, int rad,
  int emt) { return getSplittingName_new(event, rad, emt); }

//--------------------------------------------------------------------------

// Return the total number of emissions for a particular splitting 

int SplittingLibrary::nEmissions( string name ) {

  map<string, Splitting*>::iterator it = splittings.find(name);
  if (it != splittings.end() && abs(it->second->kinMap()) == 2) return 2;

  // Flavour-changing 1->3 splitting for FSR implemented.
  if ( name.find("fsr_qcd_1->2&1&2_CS") != string::npos ) return 2;

  // Flavour-preserving 1->3 splitting for FSR implemented.
  if ( name.find("fsr_qcd_1->1&1&1_CS") != string::npos ) return 2;

  // Flavour-changing 1->3 splitting for FSR implemented.
  if ( name.find("isr_qcd_1->2&1&2_CS") != string::npos ) return 2;

  // Flavour-preserving 1->3 splitting for FSR implemented.
  if ( name.find("isr_qcd_1->1&1&1_CS") != string::npos ) return 2;

  // Default is one emission.
  return 1; 

}

//==========================================================================

} // end namespace Pythia8
