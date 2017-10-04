
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

  /*// A -> Q Q~ (regular DGLAP kernel)
  name = "fsr_qed_22->1&1a_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2QQ1( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );

  // A -> Q~ Q (regular DGLAP kernel)
  name = "fsr_qed_22->1&1b_CS";
  splittings.insert( make_pair( name, new fsr_qed_A2QQ2( name, order,
    settingsPtr, particleDataPtr, rndmPtr, beamAPtr, beamBPtr, coupSMPtr, infoPtr)) );
  */

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
  int rad, int emt) {

return getSplittingRadBefID_new(event, rad, emt);

  // Get ID of radiator before branching.
  int type = event[rad].isFinal() ? 1 :-1;
  int emtID  = event[emt].id();
  int radID  = event[rad].id();
  int emtCOL = event[emt].col();
  int radCOL = event[rad].col();
  int emtACL = event[emt].acol();
  int radACL = event[rad].acol();

  int emtColType = event[emt].colType();
  int radColType = event[rad].colType();

  bool colConnected = ((type == 1) && ( (emtCOL !=0 && (emtCOL ==radACL))
                                     || (emtACL !=0 && (emtACL ==radCOL)) ))
                    ||((type ==-1) && ( (emtCOL !=0 && (emtCOL ==radCOL))
                                     || (emtACL !=0 && (emtACL ==radACL)) ));

  // SQCD bookkeeping
  int radSign = (radID < 0) ? -1 : 1;
  int offsetL = 1000000;
  int offsetR = 2000000;
  int emtSign = (emtID < 0) ? -1 : 1;
  // Get PDG numbering offsets.
  int emtOffset = 0;
  if ( abs(emtID) > offsetL && abs(emtID) < offsetL+10 )
    emtOffset = offsetL;
  if ( abs(emtID) > offsetR && abs(emtID) < offsetR+10 )
    emtOffset = offsetR;
  int radOffset = 0;
  if ( abs(radID) > offsetL && abs(radID) < offsetL+10 )
    radOffset = offsetL;
  if ( abs(radID) > offsetR && abs(radID) < offsetR+10 )
    radOffset = offsetR;

  // Vector of possible flavours of radiator before the emission. Can have more
  // than one entry if multiple splittings yield the same final state
  // (e.g. Z --> ff~ and gamma --> ff~)
  vector<int>radBefIDs;

  // QCD splittings
  // Gluon radiation
  if ( emtID == 21 ) {
    radBefIDs.push_back(radID);
  // Final state gluon emission from quark, (1-z) term
  } else if ( type == 1 && abs(emtID) < 10 && radID == 21 ) {
    radBefIDs.push_back(emtID);
  // Final state gluon splitting
  } else if ( type == 1 && emtID == -radID && !colConnected ) {
    radBefIDs.push_back(21);
  // Initial state s-channel gluon splitting
  } else if ( type ==-1 && radID == 21 ) {
    radBefIDs.push_back(-emtID);
  // Initial state t-channel gluon splitting
  } else if ( type ==-1 && !colConnected
  && emtID != 21 && radID != 21 && abs(emtID) < 10 && abs(radID) < 10) {
    radBefIDs.push_back(21);

  // Gluino radiation
  } else if ( emtID == 1000021 ) {
    // Gluino radiation combined with quark yields squark.
    if (abs(radID) < 10 ) {
      int offset = offsetL;
      // Check if righthanded squark present. If so, make the reclustered
      // squark match. Works for squark pair production + gluino emission.
      for (int i=0; i < int(event.size()); ++i)
        if ( event[i].isFinal()
          && event[i].idAbs() < offsetR+10 && event[i].idAbs() > offsetR)
          offset = offsetR;
      radBefIDs.push_back(radSign*(abs(radID)+offset));
    // Gluino radiation combined with squark yields quark.
    } else if (abs(radID) > offsetL && abs(radID) < offsetL+10 ) {
      radBefIDs.push_back(radSign*(abs(radID)-offsetL));
    } else if (abs(radID) > offsetR && abs(radID) < offsetR+10 ) {
      radBefIDs.push_back(radSign*(abs(radID)-offsetR));
    // Gluino radiation off gluon yields gluino.
    } else if (radID == 21 ) {
      radBefIDs.push_back(emtID);
    }

  // Final state gluino splitting
  } else if ( type == 1 && !colConnected
    && emtColType != 0 && radColType != 0) {
    // Emitted squark, radiating quark.
    if ( emtOffset > 0 && radOffset == 0
      && emtSign*(abs(emtID) - emtOffset) == -radID ) {
      radBefIDs.push_back(1000021);
    // Emitted quark, radiating squark.
    } else if ( emtOffset == 0 && radOffset > 0
      && emtID == -radSign*(abs(radID) - radOffset) ) {
      radBefIDs.push_back(1000021);
    }
  // Initial state s-channel gluino splitting
  } else if ( type ==-1 && radID == 1000021 ) {
    // Quark entering underlying hard process.
    if ( emtOffset > 0 ) {
      radBefIDs.push_back(-emtSign*(abs(emtID) - emtOffset));
    // Squark entering underlying hard process.
    } else {
      radBefIDs.push_back(-emtSign*(abs(emtID) + emtOffset));
    }
  // Initial state t-channel gluino splitting.
  } else if ( type ==-1
    && ( (abs(emtID) > offsetL && abs(emtID) < offsetL+10)
      || (abs(emtID) > offsetR && abs(emtID) < offsetR+10))
    && ( (abs(radID) > offsetL && abs(radID) < offsetL+10)
      || (abs(radID) > offsetR && abs(radID) < offsetR+10))
    && emtSign*(abs(emtID)+emtOffset) == radSign*(abs(radID) - radOffset)
    && !colConnected ) {
    radBefIDs.push_back(1000021);
  // Photon / Z / Higgs radiation
  } else if ( emtID == 22 || emtID == 23 || emtID == 25) {
    radBefIDs.push_back(radID);
  // Final state Photon / Z / Higgs emission, (1-z) term
  } else if ( type == 1 && abs(emtID) < 10
    && (radID == 22 || radID == 23 || radID == 25) ) {
    radBefIDs.push_back(emtID);
  // Final state Photon / Z / Higgs splitting
  } else if ( type == 1 && emtID == -radID && colConnected){
    radBefIDs.push_back(22);
    radBefIDs.push_back(23);
    radBefIDs.push_back(25);
  // Initial state s-channel photon / Z / Higgs splitting
  } else if ( type ==-1 && (radID == 22 || radID == 23 || radID == 25) ){
    radBefIDs.push_back(-emtID);
  // Initial state t-channel photon / Z splitting to quarks.
  } else if ( type ==-1 && particleDataPtr->isQuark(emtID) && radID == emtID
    && colConnected){
    radBefIDs.push_back(22);
    radBefIDs.push_back(23);
    radBefIDs.push_back(25);
  // Initial state t-channel photon / Z splitting to charged leptons.
  } else if ( type ==-1 && particleDataPtr->isLepton(emtID)
    && particleDataPtr->chargeType(emtID) != 0 && radID == emtID){
    radBefIDs.push_back(22);
    radBefIDs.push_back(23);
    radBefIDs.push_back(25);
  // Initial state t-channel Z splitting to neutrinos.
  } else if ( type ==-1 && particleDataPtr->isLepton(emtID)
    && particleDataPtr->chargeType(emtID) == 0 && radID == emtID){
    radBefIDs.push_back(23);
  // W+ radiation
  } else if ( emtID == 24 && radID < 0 ) {
    radBefIDs.push_back(radID + 1);
  } else if ( emtID == 24 && radID > 0 ) {
    radBefIDs.push_back(radID + 1);
  // W- radiation
  } else if ( emtID ==-24 && radID < 0 ) {
    radBefIDs.push_back(radID - 1);
  } else if ( emtID ==-24 && radID > 0 ) {
    radBefIDs.push_back(radID - 1);
  // Final state W splitting
  } else if ( type == 1 && colConnected
    && (emtSign*(abs(emtID)+1) == -radID || emtID == -radSign*(abs(radID)+1)) ){
    int chg = event[emt].charge() + event[rad].charge();
    if (chg > 0) radBefIDs.push_back( 24);
    else         radBefIDs.push_back(-24);
  }


  // Done.
  return radBefIDs;

}

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
    if (valid && idNow != 0) { names.push_back(it->first);}
  }

  return names;

}

//--------------------------------------------------------------------------

// Generate name for a splitting

vector<string> SplittingLibrary::getSplittingName(const Event& event, int rad,
  int emt) {

  return getSplittingName_new(event, rad, emt);

  // Get ID of radiator before branching.
  int type               = event[rad].isFinal() ? 1 :-1;
  vector<int> radBefIDs  = getSplittingRadBefID(event, rad, emt);

  // Construct all possible names.
  vector<string> names;

  for (int i = 0; i < int(radBefIDs.size()); ++i) {
    int radBefID = radBefIDs[i];

    if (radBefID == 0) continue;

    // Construct name.
    string name = "";
    if (type ==  1) name +="fsr_";
    else name +="isr_";

    if      (event[emt].id()    == 22 || radBefID      == 22) name +="qed_";
    else if (event[emt].id()    == 23 || radBefID      == 23) name +="ew_";
    else if (event[emt].idAbs() == 24 || abs(radBefID) == 24) name +="ew_";
    else if (event[emt].id()    == 25 || radBefID      == 25) name +="ew_";
    else name +="qcd_";

    int radAftID = event[rad].id();
    if (type == -1) swap(radBefID,radAftID);
    int emtAftID = event[emt].id();

    int rbID = (abs(radBefID) < 10) ? 1 : abs(radBefID); 
    stringstream ss; ss << rbID;
    name += ss.str(); name +="->";

    int rID = (abs(radAftID) < 10) ? 1 : abs(radAftID); 
    ss.str(""); ss << rID;
    name += ss.str(); name +="&";

    int eID = (abs(emtAftID) < 10) ? 1 : abs(emtAftID); 
    ss.str(""); ss << eID;
    name += ss.str();

    // Distinguish FSR x->qq~ and x->q~q
    if (name.find("fsr") !=string::npos && name.find("->1&1") !=string::npos){
      if (event[emt].id() > 0) name += "a";
      if (event[emt].id() < 0) name += "b";
    }

    // Two colour structures for g->gg.
    if ( name.compare("isr_qcd_21->21&21") == 0
      || name.compare("fsr_qcd_21->21&21") == 0) {
      string nameA=name + "a_CS";
      // Check if the name exists.
      bool foundName = (splittings.find(nameA) != splittings.end());
      if (foundName) names.push_back(nameA);
      nameA=name + "b_CS";
      // Check if the name exists.
      foundName = (splittings.find(nameA) != splittings.end());
      if (foundName) names.push_back(nameA);
    } else {
      name += "_CS";
      // Check if the name exists.
      bool foundName = (splittings.find(name) != splittings.end());
      // Done for this name.
      if (foundName) names.push_back(name);
    }

  }

  // Done.
  return names;
}

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
