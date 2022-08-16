// LHAFortran.h is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for Fortran Les Houches Accord user process information.
// LHAup_outFortran: derived class with the HEPRUP and HEPEUP Fortran info.
// You are expected to create a derived class that sup_outplies the fillHepRup_out
// and fillHepEup_out methods (and returns true when successful).

#ifndef Pythia8_LHAFortran_aMCatNLO_H
#define Pythia8_LHAFortran_aMCatNLO_H

#include "Pythia8/Pythia.h"

namespace Pythia8 {

//==========================================================================

// Give access to the HEPRUP and HEPEUP Fortran commonblocks.

#ifdef _WIN32
  #define heprup_out_ HEPRUP_OUT
  #define hepeup_out_ HEPEUP_OUT
  #define heprup_in_ HEPRUP_IN
  #define hepeup_in_ HEPEUP_IN
#endif

extern "C" {

  extern struct {
    int idbmup_in[2];
    double ebmup_in[2];
    int pdfgup_in[2], pdfsup_in[2], idwtup_in, nprup_in;
    double xsecup_in[100], xerrup_in[100], xmaxup_in[100];
    int lprup_in[100];
  } heprup_in_;

  extern struct {
    int nup_in, idprup_in;
    double xwgtup_in, scalup_in, aqedup_in, aqcdup_in;
    int idup_in[500], istup_in[500], mothup_in[500][2], icolup_in[500][2];
    double pup_in[500][5], vtimup_in[500],spinup_in[500], scales_in[500][2];
    int ifks_in, jfks_in;
  } hepeup_in_;

}

//==========================================================================

extern "C" {

  extern struct {
    int idbmup_out[2];
    double ebmup_out[2];
    int pdfgup_out[2], pdfsup_out[2], idwtup_out, nprup_out;
    double xsecup_out[100], xerrup_out[100], xmaxup_out[100];
    int lprup_out[100];
  } heprup_out_;

  extern struct {
    int nup_out, idprup_out;
    double xwgtup_out, scalup_out, aqedup_out, aqcdup_out;
    int idup_out[500], istup_out[500], mothup_out[500][2], icolup_out[500][2];
    double pup_out[500][5], vtimup_out[500],spinup_out[500],
           scales_out[500][2];
    int ifks_out, jfks_out;
  } hepeup_out_;

}

//==========================================================================

extern "C" {

  extern struct {
    int is_pythia_active;
	char pythia_cmd_file[500];
  } pythia_control_;

}

//==========================================================================

// A derived class with initialization information from the HEPRUP
// Fortran commonblock and event information from the HEPEUP one.

class LHAupFortran_aMCatNLO : public LHAup {

public:

  // Constructor.
  LHAupFortran_aMCatNLO() {}

  // Routine for doing the job of setting initialization info.
  bool setInit() {
    // Call the routine that does the job.
    if (!fillHepRup()) return false;
    // Store beam and strategy info.
    setBeamA(heprup_out_.idbmup_out[0], heprup_out_.ebmup_out[0], heprup_out_.pdfgup_out[0],
      heprup_out_.pdfsup_out[0]);
    setBeamB(heprup_out_.idbmup_out[1], heprup_out_.ebmup_out[1], heprup_out_.pdfgup_out[1],
      heprup_out_.pdfsup_out[1]);
    setStrategy(heprup_out_.idwtup_out);
    // Store process info. Protect against vanishing cross section.
    for (int ip = 0; ip < heprup_out_.nprup_out; ++ip) {
      double xsec = max( 1e-10, heprup_out_.xsecup_out[ip]);
      addProcess( heprup_out_.lprup_out[ip], xsec, heprup_out_.xerrup_out[ip],
        heprup_out_.xmaxup_out[ip] );
    }
    // Store the beam energies to calculate x values later.
    eBeamA = heprup_out_.ebmup_out[0];
    eBeamB = heprup_out_.ebmup_out[1];
    // Done.
    return true;
  }

  // Routine for doing the job of setting info on next event.
  bool setEvent(int idProcIn = 0) {
    // In some strategies the type of the next event has been set.
    hepeup_out_.idprup_out = idProcIn;
    // Call the routine that does the job.
    if (!fillHepEup()) return false;
    // Store process info.
    setProcess(hepeup_out_.idprup_out, hepeup_out_.xwgtup_out, hepeup_out_.scalup_out,
      hepeup_out_.aqedup_out, hepeup_out_.aqcdup_out);
    // Store particle info.
    for (int ip = 0; ip < hepeup_out_.nup_out; ++ip) {
      double scale(0.0);
      if (settingsPtr->flag("Beams:setProductionScalesFromLHEF")) {
        if ( hepeup_out_.icolup_out[ip][0] != 0)
          scale = hepeup_out_.scales_out[ip][0];
        if ( hepeup_out_.icolup_out[ip][1] != 0)
          scale = hepeup_out_.scales_out[ip][1];
        if ( hepeup_out_.icolup_out[ip][0] != 0
          && hepeup_out_.icolup_out[ip][1] != 0)
          scale = max(hepeup_out_.scales_out[ip][0],
                      hepeup_out_.scales_out[ip][1]);
      }
      addParticle(hepeup_out_.idup_out[ip],
      hepeup_out_.istup_out[ip], hepeup_out_.mothup_out[ip][0], hepeup_out_.mothup_out[ip][1],
      hepeup_out_.icolup_out[ip][0], hepeup_out_.icolup_out[ip][1], hepeup_out_.pup_out[ip][0],
      hepeup_out_.pup_out[ip][1], hepeup_out_.pup_out[ip][2], hepeup_out_.pup_out[ip][3],
      hepeup_out_.pup_out[ip][4], hepeup_out_.vtimup_out[ip], hepeup_out_.spinup_out[ip],
      scale);
    }

    // Store x values (here E = pup_out[ip][3]), but note incomplete info.
    setPdf( hepeup_out_.idup_out[0], hepeup_out_.idup_out[1], hepeup_out_.pup_out[0][3]/eBeamA,
      hepeup_out_.pup_out[1][3]/eBeamB, 0., 0., 0., false);

    // Read in scales.
    scalesNow.clear();
    scalesNow.muf   = hepeup_out_.scalup_out;
    scalesNow.mur   = hepeup_out_.scalup_out;
    scalesNow.mups  = hepeup_out_.scalup_out;
    int offset = 3;
    //string tag = (forUnderlying) ? "mups_underlying" : "mups_event";
    string tag = "mups_underlying";
    for (int i = 0; i < hepeup_out_.nup_out; ++i) {
      stringstream name;
      name << tag << "_col_" << i+offset;
      if (hepeup_out_.icolup_out[i][0] != 0) {
        stringstream name;
        if (hepeup_out_.istup_out[i] < 0)
             name << tag << "_inc_col_" << hepeup_out_.icolup_out[i][0];
        else name << tag << "_out_col_" << hepeup_out_.icolup_out[i][0];
        double scale = settingsPtr->flag("Beams:setProductionScalesFromLHEF")
                     ? hepeup_out_.scales_out[i][0] : hepeup_out_.scalup_out;
        scalesNow.attributes.insert(make_pair(name.str(), scale));
      }
      if (hepeup_out_.icolup_out[i][1] != 0) {
        stringstream name;
        if (hepeup_out_.istup_out[i] < 0)
             name << tag << "_inc_acol_" << hepeup_out_.icolup_out[i][1];
        else name << tag << "_out_acol_" << hepeup_out_.icolup_out[i][1];
        double scale = settingsPtr->flag("Beams:setProductionScalesFromLHEF")
                     ? hepeup_out_.scales_out[i][1] : hepeup_out_.scalup_out;
        scalesNow.attributes.insert(make_pair(name.str(), scale));
      }
    }

    infoPtr->scales = &scalesNow;

    evAttributes.clear();

    stringstream ss;
    ss << hepeup_out_.ifks_out;
    evAttributes.insert(make_pair("ifks",ss.str()));
    ss.str("");
   ss << hepeup_out_.jfks_out;
    evAttributes.insert(make_pair("jfks",ss.str()));
    infoPtr->eventAttributes = &evAttributes;

    // Done.
    return true;
  }

  Settings* settingsPtr;
  void setPointers(Settings* settings) {settingsPtr = settings;}

protected:

  // User-written routine that does the intialization and fills heprup_out.
  virtual bool fillHepRup() {return false;}

  // User-written routine that does the event generation and fills hepeup_out.
  virtual bool fillHepEup() {return false;}

private:

  // Save beam energies to calculate x values.
  double eBeamA, eBeamB;
  LHAscales scalesNow;
  map<string,string> evAttributes;

};

//==========================================================================

class MyLHAupFortran : public LHAupFortran_aMCatNLO {
  public:

  MyLHAupFortran(){
    initialised = false;
  }

  MyLHAupFortran(Settings* settings){
    settingsPtr=settings;
    initialised = false;
  }

  //the common blocks should be alredy filled at the fortran level
  //so simply return true
  bool fillHepRup(){
    initialised = true;
    return true;
  }
  bool fillHepEup(){
    return true;
  }
  
  bool is_initialised(){
    return initialised;
  }

  private:
  bool initialised;
};

//==========================================================================

// LHA3FromPythia8 class.

class LHA3FromPythia8 : public LHEF3FromPythia8 {

public:

  // Constructor.
  LHA3FromPythia8(Event* eventPtrIn, Settings* settingsPtrIn,
    const Info* infoPtrIn, ParticleData* particleDataPtrIn, int pDigitsIn = 15,
    bool = false) : LHEF3FromPythia8(eventPtrIn, infoPtrIn,
    pDigitsIn, false) {}

  // Routine for reading, setting and printing the initialisation info.
  bool setInit();

  // Routine for reading, setting and printing the next event.
  void setProcessPtr(const Event* evPtr) { processPtr = evPtr; }
  bool setEvent(int = 0);

private:

  // Pointer to event that should be printed.
  const Event* processPtr;

};

typedef shared_ptr<LHA3FromPythia8> LHA3FromPythia8Ptr;

//--------------------------------------------------------------------------

// Routine for reading, setting and printing the initialisation info.

bool LHA3FromPythia8::setInit() {

  heprup.clear();

  // PDG id's of beam particles. (first/second is in +/-z direction).
  heprup.IDBMUP = make_pair(infoPtr->idA(), infoPtr->idB());

  // Energy of beam particles given in GeV.
  heprup.EBMUP = make_pair(infoPtr->eA(),infoPtr->eB());

  // The author group for the PDF used for the beams according to the
  // PDFLib specification.
  heprup.PDFGUP = make_pair(0,0);

  // The id number the PDF used for the beams according to the
  // PDFLib specification.
  heprup.PDFSUP = make_pair(0,0);

  // Master switch indicating how the ME generator envisages the
  // events weights should be interpreted according to the Les Houches
  // accord.
  heprup.IDWTUP = -4;

  // The number of different subprocesses in this file.
  heprup.NPRUP = 1;

  // The cross sections for the different subprocesses in pb.
  vector<double> XSECUP;
  for ( int i=0; i < heprup.NPRUP; ++i)
    XSECUP.push_back(CONVERTMB2PB * infoPtr->sigmaGen());
  heprup.XSECUP = XSECUP;

  // The statistical error in the cross sections for the different
  // subprocesses in pb.
  vector<double> XERRUP;
  for ( int i=0; i < heprup.NPRUP; ++i)
    XERRUP.push_back(CONVERTMB2PB * infoPtr->sigmaErr());
  heprup.XERRUP = XERRUP;

  // The maximum event weights (in HEPEUP::XWGTUP) for different
  vector<double> XMAXUP;
  for ( int i=0; i < heprup.NPRUP; ++i) XMAXUP.push_back(0.0);
  heprup.XMAXUP = XMAXUP;

  // The subprocess code for the different subprocesses.
  vector<int> LPRUP;
  for ( int i=0; i < heprup.NPRUP; ++i) LPRUP.push_back(9999+i);
  heprup.LPRUP = LPRUP;

  // Now write to common block
  heprup_in_.idbmup_in[0] = heprup.IDBMUP.first;
  heprup_in_.idbmup_in[1] = heprup.IDBMUP.second;

  // Energy of beam particles given in GeV.
  heprup_in_.ebmup_in[0] = heprup.EBMUP.first;
  heprup_in_.ebmup_in[0] = heprup.EBMUP.second;

  // The author group for the PDF used for the beams according to the
  // PDFLib specification.
  heprup_in_.pdfgup_in[0] = 0;
  heprup_in_.pdfgup_in[1] = 0;

  // The id number the PDF used for the beams according to the
  // PDFLib specification.
  heprup_in_.pdfsup_in[0] = 0;
  heprup_in_.pdfsup_in[1] = 0;

  // Master switch indicating how the ME generator envisages the
  // events weights should be interpreted according to the Les Houches
  // accord.
  heprup_in_.idwtup_in = heprup.IDWTUP;

  // The number of different subprocesses in this file.
  heprup_in_.nprup_in = heprup.NPRUP;

  // The cross sections for the different subprocesses in pb.
  for ( int i=0; i < heprup.XSECUP.size(); ++i)
    heprup_in_.xsecup_in[i] = heprup.XSECUP[i];

  // The statistical error in the cross sections for the different
  // subprocesses in pb.
  for ( int i=0; i < heprup.XERRUP.size(); ++i)
    heprup_in_.xerrup_in[i] = heprup.XERRUP[i];

  // The maximum event weights (in HEPEUP::XWGTUP) for different
  for ( int i=0; i < heprup.XMAXUP.size(); ++i)
    heprup_in_.xmaxup_in[i] = heprup.XMAXUP[i];

  // The subprocess code for the different subprocesses.
  for ( int i=0; i < heprup.LPRUP.size(); ++i)
    heprup_in_.lprup_in[i] = heprup.LPRUP[i];

  // Done
  return true;
}

//--------------------------------------------------------------------------

// Routine for reading, setting and printing the next event.

bool LHA3FromPythia8::setEvent(int) {

  Event event = *eventPtr;

  // Begin filling Les Houches blocks.
  hepeup.clear();
  hepeup.resize(0);

  // The number of particle entries in the current event.
  hepeup.NUP = 2;
  for ( int i = 0; i < int(event.size()); ++i) {
    if ( event[i].status() == -22) ++hepeup.NUP;
    if ( event[i].isFinal()) ++hepeup.NUP;
  }

  // The subprocess code for this event (as given in LPRUP).
  hepeup.IDPRUP = 9999;

  // The weight for this event.
  hepeup.XWGTUP = infoPtr->weight();

  // The PDF weights for the two incoming partons. Note that this
  // variable is not present in the current LesHouches accord
  // (<A HREF="http://arxiv.org/abs/hep-ph/0109068">hep-ph/0109068</A>),
  // hopefully it will be present in a future accord.
  hepeup.XPDWUP = make_pair(0,0);

  // The scale in GeV used in the calculation of the PDF's in this
  // event.
  hepeup.SCALUP = eventPtr->scale();

  // The value of the QED coupling used in this event.
  hepeup.AQEDUP = infoPtr->alphaEM();

  // The value of the QCD coupling used in this event.
  hepeup.AQCDUP = infoPtr->alphaS();

  // Find incoming particles.
  int in1, in2;
  in1 = in2 = 0;
  for ( int i = 0; i < int( event.size()); ++i) {
    if ( event[i].mother1() == 1 && in1 == 0) in1 = i;
    if ( event[i].mother1() == 2 && in2 == 0) in2 = i;
  }

  // Find resonances in hard process.
  vector<int> hardResonances;
  for ( int i = 0; i < int(event.size()); ++i) {
    //if ( event[i].status() != -22) continue;
    if ( event[i].mother1() != 3) continue;
    if ( event[i].mother2() != 4) continue;
    if ( !(event[i].mayDecay() && event[i].isResonance()) ) continue;
    hardResonances.push_back(i);
  }

  // Find resonances and decay products after showering.
  vector<int> evolvedResonances;
  vector<pair<int,int> > evolvedDecayProducts;
  for ( int j = 0; j < int(hardResonances.size()); ++j) {
    for ( int i = int(event.size())-1; i > 0; --i) {
      if ( i == hardResonances[j]
        || (event[i].mother1() == event[i].mother2()
         && event[i].isAncestor(hardResonances[j])) ) {
        evolvedResonances.push_back(i);
        evolvedDecayProducts.push_back(
          make_pair(event[i].daughter1(), event[i].daughter2()) );
        break;
      }
    }
  }

  // Event for bookkeeping of resonances.
  Event now  = Event();
  now.init("(dummy event)", particleDataPtr);
  now.reset();

  // The PDG id's for the particle entries in this event.
  // The status codes for the particle entries in this event.
  // Indices for the first and last mother for the particle entries in
  // this event.
  // The colour-line indices (first(second) is (anti)colour) for the
  // particle entries in this event.
  // Lab frame momentum (Px, Py, Pz, E and M in GeV) for the particle
  // entries in this event.
  // Invariant lifetime (c*tau, distance from production to decay in
  // mm) for the particle entries in this event.
  // Spin info for the particle entries in this event given as the
  // cosine of the angle between the spin vector of a particle and the
  // 3-momentum of the decaying particle, specified in the lab frame.
  vector<int> iSkip;
  hepeup.IDUP.push_back(event[in1].id());
  iSkip.push_back(in1);
  hepeup.IDUP.push_back(event[in2].id());
  iSkip.push_back(in2);
  hepeup.ISTUP.push_back(-1);
  hepeup.ISTUP.push_back(-1);
  hepeup.MOTHUP.push_back(make_pair(0,0));
  hepeup.MOTHUP.push_back(make_pair(0,0));
  hepeup.ICOLUP.push_back(make_pair(event[in1].col(),event[in1].acol()));
  hepeup.ICOLUP.push_back(make_pair(event[in2].col(),event[in2].acol()));
  vector <double> p;
  p.push_back(0.0);
  p.push_back(0.0);
  p.push_back(event[in1].pz());
  p.push_back(event[in1].e());
  p.push_back(event[in1].m());
  hepeup.PUP.push_back(p);
  p.resize(0);
  p.push_back(0.0);
  p.push_back(0.0);
  p.push_back(event[in2].pz());
  p.push_back(event[in2].e());
  p.push_back(event[in2].m());
  hepeup.PUP.push_back(p);
  p.resize(0);
  hepeup.VTIMUP.push_back(event[in1].tau());
  hepeup.VTIMUP.push_back(event[in2].tau());
  hepeup.SPINUP.push_back(event[in1].pol());
  hepeup.SPINUP.push_back(event[in2].pol());

  now.append(event[in1]);
  now.append(event[in2]);

  // Attach resonances
  for ( int j = 0; j < int(evolvedResonances.size()); ++j) {
    int i = evolvedResonances[j];
    if (find(iSkip.begin(), iSkip.end(), i) != iSkip.end()) continue;
    hepeup.IDUP.push_back(event[i].id());
    iSkip.push_back(i);
    hepeup.ISTUP.push_back(2);
    hepeup.MOTHUP.push_back(make_pair(1,2));
    hepeup.ICOLUP.push_back(make_pair(event[i].col(),event[i].acol()));
    p.push_back(event[i].px());
    p.push_back(event[i].py());
    p.push_back(event[i].pz());
    p.push_back(event[i].e());
    p.push_back(event[i].m());
    hepeup.PUP.push_back(p);
    p.resize(0);
    hepeup.VTIMUP.push_back(event[i].tau());
    hepeup.SPINUP.push_back(event[i].pol());
    now.append(event[i]);
    now.back().statusPos();
  }

  // Loop through event and attach remaining decays
  int iDec = 0;
  do {

    if ( now[iDec].isFinal() && now[iDec].canDecay()
      && now[iDec].mayDecay() && now[iDec].isResonance() ) {

      int iD1 = now[iDec].daughter1();
      int iD2 = now[iDec].daughter2();

      // Done if no daughters exist.
      if ( iD1 == 0 || iD2 == 0 ) continue;

     // Attach daughters.
     for ( int k = iD1; k <= iD2; ++k ) {
       Particle partNow = event[k];
       if (find(iSkip.begin(), iSkip.end(), k) != iSkip.end()) continue;
       hepeup.IDUP.push_back(partNow.id());
       iSkip.push_back(k);
       hepeup.MOTHUP.push_back(make_pair(iDec,iDec));
       hepeup.ICOLUP.push_back(make_pair(partNow.col(),partNow.acol()));
       p.push_back(partNow.px());
       p.push_back(partNow.py());
       p.push_back(partNow.pz());
       p.push_back(partNow.e());
       p.push_back(partNow.m());
       hepeup.PUP.push_back(p);
       p.resize(0);
       hepeup.VTIMUP.push_back(partNow.tau());
       hepeup.SPINUP.push_back(partNow.pol());
       now.append(partNow);
       if ( partNow.canDecay() && partNow.mayDecay() && partNow.isResonance()){
         now.back().statusPos();
         hepeup.ISTUP.push_back(2);
       } else
         hepeup.ISTUP.push_back(1);
     }

     // End of loop over all entries.
    }
  } while (++iDec < now.size());

  // Attach final state particles
  for ( int i = 0; i < int(event.size()); ++i) {
    if (!event[i].isFinal()) continue;
    // Skip resonance decay products.
    bool skip = false;
    for ( int j = 0; j < int(evolvedDecayProducts.size()); ++j) {
      skip = ( i >= evolvedDecayProducts[j].first
            && i <= evolvedDecayProducts[j].second);
    }
    if (skip) continue;
    if (find(iSkip.begin(), iSkip.end(), i) != iSkip.end()) continue;

    hepeup.IDUP.push_back(event[i].id());
    iSkip.push_back(i);
    hepeup.ISTUP.push_back(1);
    hepeup.MOTHUP.push_back(make_pair(1,2));
    hepeup.ICOLUP.push_back(make_pair(event[i].col(),event[i].acol()));
    p.push_back(event[i].px());
    p.push_back(event[i].py());
    p.push_back(event[i].pz());
    p.push_back(event[i].e());
    p.push_back(event[i].m());
    hepeup.PUP.push_back(p);
    p.resize(0);
    hepeup.VTIMUP.push_back(event[i].tau());
    hepeup.SPINUP.push_back(event[i].pol());
    now.append(event[i]);
  }

  // Write to common block.
  hepeup_in_.nup_in = hepeup.NUP;
  hepeup_in_.idprup_in = hepeup.IDPRUP;

  // The weight for this event.
  hepeup_in_.xwgtup_in = hepeup.XWGTUP;

  // The scale in GeV used in the calculation of the PDF's in this
  // event.
  hepeup_in_.scalup_in = hepeup.SCALUP;

  // The value of the QED coupling used in this event.
  hepeup_in_.aqedup_in = hepeup.AQEDUP;

  // The value of the QCD coupling used in this event.
  hepeup_in_.aqcdup_in = hepeup.AQCDUP;

  for (int i=0; i < hepeup.IDUP.size(); ++i)
    hepeup_in_.idup_in[i] = hepeup.IDUP[i];
  for (int i=0; i < hepeup.ISTUP.size(); ++i)
    hepeup_in_.istup_in[i] = hepeup.ISTUP[i];
  for (int i=0; i < hepeup.MOTHUP.size(); ++i) {
    hepeup_in_.mothup_in[i][0] = hepeup.MOTHUP[i].first;
    hepeup_in_.mothup_in[i][1] = hepeup.MOTHUP[i].second;
  }
  for (int i=0; i < hepeup.ICOLUP.size(); ++i) {
    hepeup_in_.icolup_in[i][0] = hepeup.ICOLUP[i].first;
    hepeup_in_.icolup_in[i][1] = hepeup.ICOLUP[i].second;
  }
  for (int i=0; i < hepeup.PUP.size(); ++i) {
    hepeup_in_.pup_in[i][0] = hepeup.PUP[i][0];
    hepeup_in_.pup_in[i][1] = hepeup.PUP[i][1];
    hepeup_in_.pup_in[i][2] = hepeup.PUP[i][2];
    hepeup_in_.pup_in[i][3] = hepeup.PUP[i][3];
    hepeup_in_.pup_in[i][4] = hepeup.PUP[i][4];
  }
  for (int i=0; i < hepeup.VTIMUP.size(); ++i)
    hepeup_in_.vtimup_in[i] = hepeup.VTIMUP[i];
  for (int i=0; i < hepeup.SPINUP.size(); ++i)
    hepeup_in_.spinup_in[i] = hepeup.SPINUP[i];

  return true;

}

//==========================================================================

class PrintFirstEmission : public UserHooks {

public:

  PrintFirstEmission(LHA3FromPythia8Ptr lhawriterPtrIn)
    : lhawriterPtr(lhawriterPtrIn) {
    doRemoveDecayProducts=true;
    inputEvent.init("(hard process-modified)", particleDataPtr);
    resonances.resize(0);
  }

  bool canVetoISREmission() { return true; }
  bool canVetoFSREmission() { return true; }

  bool doVetoISREmission(int, const Event& event, int iSys) {
    nISR++;
    if (nISR + nFSR > 1) return false;
    // Reattach resonance decay products and write event.
    Event outEvent = makeHardEvent(iSys, event, true);
    reattachResonanceDecays(outEvent);
    lhawriterPtr->setEventPtr(&outEvent);
    lhawriterPtr->setEvent();
    // Done.
    return false;
  }

  bool doVetoFSREmission(int, const Event& event, int iSys) {
    nISR++;
    if (nISR + nFSR > 1) return false;
    // Reattach resonance decay products and write event.
    Event outEvent = makeHardEvent(iSys, event, true);
    reattachResonanceDecays(outEvent);
    lhawriterPtr->setEventPtr(&outEvent);
    lhawriterPtr->setEvent();
    // Done.
    return false;
  }

  bool canVetoProcessLevel() { return true; }
  bool doVetoProcessLevel(Event& process) {
    // Initailize and store resonance decay products.
    nISR = nFSR = 0;
    lhawriterPtr->setProcessPtr(&process);
    bareEvent(process,true);
    return false;
  }

  LHA3FromPythia8Ptr lhawriterPtr;

  Event inputEvent;
  vector< pair<int,int> > resonances;
  bool doRemoveDecayProducts;

  int nISR, nFSR;

  //--------------------------------------------------------------------------
  Event makeHardEvent( int iSys, const Event& state, bool isProcess) {

    bool hasSystems = !isProcess && partonSystemsPtr->sizeSys() > 0;
    int sizeSys     = (hasSystems) ? partonSystemsPtr->sizeSys() : 1;

    Event event = Event();
    event.clear();
    event.init( "(hard process-modified)", particleDataPtr );

    int in1 = 0;
    for ( int i = state.size()-1; i > 0; --i)
      if ( state[i].mother1() == 1 && state[i].mother2() == 0
        && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
        {in1 = i; break;}
    if (in1 == 0) in1 = partonSystemsPtr->getInA(iSys);

    int in2 = 0;
    for ( int i = state.size()-1; i > 0; --i)
      if ( state[i].mother1() == 2 && state[i].mother2() == 0
        && (!hasSystems || partonSystemsPtr->getSystemOf(i,true) == iSys))
        {in2 = i; break;}
    if (in2 == 0) in2 = partonSystemsPtr->getInB(iSys);

    // Try to find incoming particle in other systems, i.e. if the current
    // system arose from a resonance decay.
    bool resonantIncoming = false;
    if ( in1 == 0 && in2 == 0 ) {
      int iParentInOther = 0;
      int nSys = partonSystemsPtr->sizeAll(iSys);
      for (int iInSys = 0; iInSys < nSys; ++iInSys){
        int iNow = partonSystemsPtr->getAll(iSys,iInSys);
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == iSys) continue;
          int nOtherSys = partonSystemsPtr->sizeAll(iOtherSys);
          for (int iInOtherSys = 0; iInOtherSys < nOtherSys; ++iInOtherSys){
            int iOtherNow = partonSystemsPtr->getAll(iOtherSys,iInOtherSys);
            if (state[iNow].isAncestor(iOtherNow)) {
              iParentInOther = iOtherNow;
            }
          }
        }
      }
      in1 = iParentInOther;
      if (iParentInOther) resonantIncoming = true;
    } 

    event.append(state[0]);
    event.append(state[1]);
    event[1].daughters(3,0);
    event.append(state[2]);
    event[2].daughters(4,0);

    // Attach the first incoming particle.
    event.append(state[in1]);
    event[3].mothers(1,0);
    if (resonantIncoming) event[3].status(-22);
    else event[3].status(-21);

    // Attach the second incoming particle.
    event.append(state[in2]);
    event[4].mothers(2,0);
    event[4].status(-21);

    for ( int i = 0; i < state.size(); ++i) {
      // Careful when builing the sub-events: A particle that is currently
      // intermediate in one system could be the pirogenitor of another
      // system, i.e. when resonance decays are present. In this case, the
      // intermediate particle in the current system should be final. 
      bool isFin   = state[i].isFinal();
      bool isInSys = (partonSystemsPtr->getSystemOf(i) == iSys);

      bool isParentOfOther = false;
      if (!isFin && isInSys) {
        for (int iOtherSys = 0; iOtherSys < sizeSys; ++iOtherSys){
          if (iOtherSys == iSys) continue;
          double nSys = partonSystemsPtr->sizeAll(iOtherSys);
          for (int iInSys = 0; iInSys < nSys; ++iInSys){
            int iNow = partonSystemsPtr->getAll(iOtherSys,iInSys);
            if (state[iNow].isAncestor(i)) {isParentOfOther=true; break;}
          }
        }
      }

      if ( (isFin || isParentOfOther) && (!hasSystems || isInSys) ) {
      //if ( state[i].isFinal() 
      //  && (!hasSystems || partonSystemsPtr->getSystemOf(i) == iSys)) {
        int iN = event.append(state[i]);
        event[iN].daughters(0,0);
        event[iN].mothers(3,4);
        event[iN].status(23);
      }
    }

    // Set daughters of initial particles.
    event[3].daughters(5,event.size()-1);
    event[4].daughters(5,event.size()-1);
    return event;

  }


  //--------------------------------------------------------------------------
  // Return event stripped off decay products.
  Event bareEvent(const Event& inputEventIn,
    bool storeInputEvent ) {

    // Find and detach decay products.
    Event newProcess = Event();
    newProcess.init("(hard process-modified)", particleDataPtr);

    // If desired, store input event.
    if ( storeInputEvent ) {
      resonances.resize(0);
      inputEvent.clear();
      inputEvent = inputEventIn;
    }

    // Now remove decay products.
    if ( doRemoveDecayProducts ) {

      // Add the beam and initial partons to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if ( inputEventIn[i].mother1() > 4
          || inputEventIn[i].statusAbs() == 22
          || inputEventIn[i].statusAbs() == 23)
          break;
        newProcess.append(inputEventIn[i]);
      }

      // Add the intermediate particles to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if (inputEventIn[i].mother1() > 4) break;
        if ( inputEventIn[i].status() == -22) {
          int j = newProcess.append(inputEventIn[i]);
          newProcess[j].statusPos();
          if ( storeInputEvent ) resonances.push_back( make_pair(j, i) );
          newProcess[j].daughters(0, 0);
        }
      }

      // Add remaining outgoing particles to the event record.
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if (inputEventIn[i].mother1() > 4) break;
        if ( inputEventIn[i].statusAbs() != 11
          && inputEventIn[i].statusAbs() != 12
          && inputEventIn[i].statusAbs() != 21
          && inputEventIn[i].statusAbs() != 22)
          newProcess.append(inputEventIn[i]);
      }

      // Update event colour tag to maximum in whole process.
      int maxColTag = 0;
      for (int i = 0; i < inputEventIn.size(); ++ i) {
        if ( inputEventIn[i].col() > maxColTag )
          maxColTag = inputEventIn[i].col();
        if ( inputEventIn[i].acol() > maxColTag )
          maxColTag = inputEventIn[i].acol();
      }
      newProcess.initColTag(maxColTag);

      // Copy junctions from process to newProcess.
      for (int i = 0; i < inputEventIn.sizeJunction(); ++i)
        newProcess.appendJunction( inputEventIn.getJunction(i));

      newProcess.saveSize();
      newProcess.saveJunctionSize();

    } else {
      newProcess = inputEventIn;
    }

    // Remember scale
    newProcess.scale( inputEventIn.scale() );

    // Done
    return newProcess;

  }

  //--------------------------------------------------------------------------
  // Write event with decay products attached to argument. Only possible if an
  // input event with decay producs had been stored before.
  bool reattachResonanceDecays(Event& process ) {

    // Now reattach the decay products.
    if ( doRemoveDecayProducts && inputEvent.size() > 0 ) {

    int sizeBef = process.size();
    // Vector of resonances for which the decay products were already attached.
    vector<int> iAftChecked;
    // Reset daughters and status of intermediate particles.
    for ( int i = 0; i < int(resonances.size()); ++i ) {
      for (int j = 0; j < sizeBef; ++j ) {
        if ( j != resonances[i].first ) continue;

        int iOldDaughter1 = inputEvent[resonances[i].second].daughter1();
        int iOldDaughter2 = inputEvent[resonances[i].second].daughter2();

        // Get momenta in case of reclustering.
        int iHardMother      = resonances[i].second;
        Particle& hardMother = inputEvent[iHardMother];
        // Find current mother copy (after clustering).
        int iAftMother       = 0;
        for ( int k = 0; k < process.size(); ++k )
          if ( process[k].id() == inputEvent[resonances[i].second].id() ) {
            // Only attempt if decays of this resonance were not attached.
            bool checked = false;
            for ( int l = 0; l < int(iAftChecked.size()); ++l )
              if ( k == iAftChecked[l] )
                checked = true;
            if ( !checked ) {
              iAftChecked.push_back(k);
              iAftMother = k;
              break;
            }
          }

        Particle& aftMother  = process[iAftMother];

        // Resonance can have been moved by clustering,
        // so prepare to update colour and momentum information for system.
        int colBef  = hardMother.col();
        int acolBef = hardMother.acol();
        int colAft  = aftMother.col();
        int acolAft = aftMother.acol();
        RotBstMatrix M;
        M.bst( hardMother.p(), aftMother.p());

        // Attach resonance decay products.
        int iNewDaughter1 = 0;
        int iNewDaughter2 = 0;
        for ( int k = iOldDaughter1; k <= iOldDaughter2; ++k ) {
          if ( k == iOldDaughter1 )
            iNewDaughter1 = process.append(inputEvent[k] );
          else
            iNewDaughter2 = process.append(inputEvent[k] );
          process.back().statusPos();
          Particle& now = process.back();
          // Update colour and momentum information.
          if (now.col()  != 0 && now.col()  == colBef ) now.col(colAft);
          if (now.acol() != 0 && now.acol() == acolBef) now.acol(acolAft);
          now.rotbst( M);
          // Update vertex information.
          if (now.hasVertex()) now.vProd( aftMother.vDec() );
          // Update mothers.
          now.mothers(iAftMother,0);
        }

        process[iAftMother].daughters( iNewDaughter1, iNewDaughter2 );
        process[iAftMother].statusNeg();

        // Loop through event and attach remaining decays
        int iDec = 0;
        do {
          if ( process[iDec].isFinal() && process[iDec].canDecay()
            && process[iDec].mayDecay() && process[iDec].isResonance() ) {

            int iD1 = process[iDec].daughter1();
            int iD2 = process[iDec].daughter2();

            // Done if no daughters exist.
            if ( iD1 == 0 || iD2 == 0 ) continue;

            // Attach daughters.
            int iNewDaughter12 = 0;
            int iNewDaughter22 = 0;
            for ( int k = iD1; k <= iD2; ++k ) {
              if ( k == iD1 )
                iNewDaughter12 = process.append(inputEvent[k] );
              else
                iNewDaughter22 = process.append(inputEvent[k] );
              process.back().statusPos();
              Particle& now = process.back();
              // Update colour and momentum information.
              if (now.col() != 0 && now.col() == colBef ) now.col(colAft);
              if (now.acol()!= 0 && now.acol()== acolBef) now.acol(acolAft);
              now.rotbst( M);
              // Update vertex information.
              if (now.hasVertex()) now.vProd( process[iDec].vDec() );
              // Update mothers.
              now.mothers(iDec,0);
            }

            // Modify mother status and daughters.
            process[iDec].status(-22);
            process[iDec].daughters(iNewDaughter12, iNewDaughter22);

          // End of loop over all entries.
          }
        } while (++iDec < process.size());
      } // End loop over process entries.
    } // End loop over resonances.

    // Update event colour tag to maximum in whole process.
    int maxColTag = 0;
    for (int i = 0; i < process.size(); ++ i) {
      if (process[i].col() > maxColTag) maxColTag = process[i].col();
      if (process[i].acol() > maxColTag) maxColTag = process[i].acol();
    }
    process.initColTag(maxColTag);

    }

    // Done.
    return (doRemoveDecayProducts) ? inputEvent.size() > 0 : true;

  }

};

} // end namespace Pythia8

#endif // Pythia8_LHAFortran_aMCatNLO_H
