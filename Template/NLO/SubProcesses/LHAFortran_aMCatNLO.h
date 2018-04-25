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
    double pup_out[500][5], vtimup_out[500],spinup_out[500], scales_out[500][2];
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
    for (int i = 0; i < hepeup_out_.nup_out; ++i) {
      stringstream name;
      name << "mups_col_" << i+offset;
      scalesNow.attributes.insert(make_pair(name.str(),hepeup_out_.scales_out[i][0]));
      name.str("");
      name << "mups_acol_" << i+offset;
      scalesNow.attributes.insert(make_pair(name.str(),hepeup_out_.scales_out[i][1]));
    }

    infoPtr->scales = &scalesNow;

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

};

//==========================================================================

// LHA3FromPythia8 class.

class LHA3FromPythia8 : public LHAup {

public:

  // Constructor.
  LHA3FromPythia8(Event* eventPtrIn, Settings* settingsPtrIn,
    Info* infoPtrIn, ParticleData* particleDataPtrIn, int pDigitsIn = 15) :
    eventPtr(eventPtrIn),settingsPtr(settingsPtrIn), infoPtr(infoPtrIn),
    particleDataPtr(particleDataPtrIn), pDigits(pDigitsIn) {}

  // Routine for reading, setting and printing the initialisation info.
  bool setInit();

  // Routine for reading, setting and printing the next event.
  void setEventPtr(const Event* evPtr) { eventPtr = evPtr; }
  void setProcessPtr(const Event* evPtr) { processPtr = evPtr; }
  bool setEvent(int = 0);

private:

  // Pointer to event that should be printed.
  const Event* eventPtr;
  const Event* processPtr;

  // Pointer to settings and info objects.
  Settings* settingsPtr;
  Info* infoPtr;
  ParticleData* particleDataPtr;

  // Number of digits to set width of double write out
  int  pDigits;

  // Some internal init and event block objects for convenience.
  HEPRUP heprup;
  HEPEUP hepeup;

};

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
  hepeup.IDUP.push_back(event[in1].id());
  hepeup.IDUP.push_back(event[in2].id());
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
    hepeup.IDUP.push_back(event[i].id());
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
  vector<int> iSkip;
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
       hepeup.IDUP.push_back(partNow.id());
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

       iSkip.push_back(k);
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
    for ( int j = 0; j < int(iSkip.size()); ++j) {
      skip = ( i == iSkip[j] );
    }
    if (skip) continue;

    hepeup.IDUP.push_back(event[i].id());
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

} // end namespace Pythia8

#endif // Pythia8_LHAFortran_aMCatNLO_H
