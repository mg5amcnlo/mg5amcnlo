// VinciaISR.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains header information for the VinciaISR class for
// QCD initial-state antenna showers (II and IF), and auxiliary classes.

#ifndef Pythia8_VinciaISR_H
#define Pythia8_VinciaISR_H

#include "Pythia8/SpaceShower.h"
#include "Pythia8/VinciaAntennaFunctions.h"
#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaDiagnostics.h"
#include "Pythia8/VinciaQED.h"
#include "Pythia8/VinciaEW.h"
#include "Pythia8/VinciaWeights.h"

namespace Pythia8 {

// Forward declarations.
class VinciaFSR;

//==========================================================================

// Base class for initial-state trial generators. Note, base class is
// coded for a soft-eikonal trial function.

class TrialGeneratorISR {

public:

  // Constructor.
  TrialGeneratorISR() : isInit(false) {;}
  virtual ~TrialGeneratorISR() {;}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn);

  // Name of trial generator.
  virtual string name() {return "TrialGeneratorISR";}

  // Initialize.
  virtual void init(double mcIn, double mbIn);

  // Trial antenna function. Convention for what is coded here:
  //   when using x*PDF ratio <= const : aTrial
  //   when using PDF ratio   <= const : aTrial * sab / sAB
  // Base class implements soft eikonal with PDF ratio as overestimate.
  virtual double aTrial(double saj, double sjb, double sAB);

  // Evolution scale.
  virtual double getQ2(double saj, double sjb, double sAB) {
    return (saj*sjb/(saj + sjb + sAB));}
  virtual double getQ2max(double sAB, double, double) {
    return (0.25*pow2(shhSav - sAB)/shhSav);}

  // Generate new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0);

  // Generate new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio, double eA, double eB,
    double headroomFac = 1.0, double enhanceFac = 1.0);

  // Generate new Q value, with running of the PDFs towards the mass
  // threshold.
  virtual double genQ2thres(double q2old, double sAB, double zMin,
    double zMax, double colFac, double alphaSvalue, double PDFratio, int idA,
    int idB, double eA, double eB, bool useMpdf, double headroomFac = 1.0,
    double enhanceFac = 1.0);

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax);

  // The zeta integral.
  virtual double getIz(double zMin, double zMax);

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAB, double eA, double eBeamUsed);
  virtual double getZmax(double Qt2, double sAB, double eA, double eBeamUsed);

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB);
  virtual double getSj2(double Qt2, double zeta, double sAB);

  // Compute trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB,
    double Qt2A, double Qt2B);

  // Return last trial PDF ratio.
  virtual double getTrialPDFratio() {return trialPDFratioSav;}

  // Check initialization.
  bool checkInit();

  // Return the last trial flavor.
  int trialFlav() {return trialFlavSav;}

 protected:

  // Pointers.
  Info*     infoPtr{};
  Rndm*     rndmPtr{};
  Settings* settingsPtr{};

  // Use m or pT evolution for collinear singularities.
  bool useMevolSav;

  // s for hadron hadron.
  double shhSav;

  // For conversion trial generators.
  int trialFlavSav;
  int nGtoQISRSav;

  // Masses.
  double mbSav;
  double mcSav;

  // Doing a sector shower?
  bool sectorShower;

  // Saved trial PDF ratio and trial tolerance.
  double trialPDFratioSav;
  double TINYPDFtrial;

 private:

  // Status.
  bool isInit;

  // Verbosity level.
  int verbose;

};

//==========================================================================

// Soft-eikonal trial function for II (clone of base class but with
// name change).

class TrialIISoft : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIISoft";}

};

//==========================================================================

// A collinear trial function for initial-initial.

class TrialIIGCollA : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIIGCollA";}

  // Trial antenna function for g->gg (collinear to beam A).
  // Used with x*PDF <= const, so no extra x-factor.
  virtual double aTrial(double saj, double sjb, double sAB) override;

  // Evolution scale
  virtual double getQ2(double saj, double sjb, double sAB) override {
    return (saj*sjb/(saj + sjb + sAB));}
  virtual double getQ2max(double sAB, double, double) override {
    return (0.25*pow2(shhSav - sAB)/shhSav);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale
  virtual double getZmin(double Qt2, double sAB, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAB, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override;
  virtual double getSj2(double Qt2, double zeta, double sAB) override;

};

//==========================================================================

// B collinear trial function for initial-initial.

class TrialIIGCollB : public TrialIIGCollA {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIIGCollB";}

  // Trial antenna function.
  virtual double aTrial(double saj, double sjb, double sAB) override {
    // Note: arguments reversed intentionally!
    return TrialIIGCollA::aTrial(sjb, saj, sAB);}

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override {
    return TrialIIGCollA::getSj2(Qt2, zeta, sAB);}
  virtual double getSj2(double Qt2, double zeta, double sAB) override {
    return TrialIIGCollA::getS1j(Qt2, zeta, sAB);}

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB,
    double Qt2A, double Qt2B) override {
    // Note: arguments reversed intentionally!
    return TrialIIGCollA::trialPDFratio(
      beamBPtr, beamAPtr, iSys, idB, idA, eB, eA, Qt2B, Qt2A);}

};

//==========================================================================

// A splitting trial function for initial-initial, q -> gqbar.

class TrialIISplitA : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIISplitA";}

  // Trial antenna function. This trial used with PDF ratio <= const,
  // so has sab/sAB prefactor.
  virtual double aTrial(double saj, double sjb, double sAB) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjb, double sAB) override {
    return ((useMevolSav) ? saj : saj*sjb/(saj + sjb + sAB));}
  virtual double getQ2max(double sAB, double, double) override {
    return ((useMevolSav) ? shhSav-sAB : 0.25*pow2(shhSav-sAB)/shhSav);}

  // Generate new Q value, with first-order running alphaS. Same
  // expression for QT and QA; Iz is different.
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate new Q value, with running of the PDFs towards the mass
  // threshold.
  virtual double genQ2thres(double q2old, double sAB,
    double zMin, double zMax, double colFac, double alphaSvalue,
    double PDFratio, int idA, int idB, double eA, double eB, bool useMpdf,
    double headroomFac = 1.0, double enhanceFac = 1.0) override;

  // Generate a new zeta value in [zMin,zMax]. For QT 1/(1+z3)
  // distribution; for QA 1/z4 distribution.
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral (with alpha = 0). For QT integral dz4 1/(1+z3),
  // for QA integral dz4 1/z4.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAB, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAB, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override;
  virtual double getSj2(double Qt2, double zeta, double sAB) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// B splitting trial function for initial-initial, q -> gqbar.

class TrialIISplitB : public TrialIISplitA {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIISplitB";}

  // Trial antenna function.
  virtual double aTrial(double saj, double sjb, double sAB) override {
    // Note: arguments reversed intentionally!
    return TrialIISplitA::aTrial(sjb, saj, sAB);}

  // Evolution scale.
  virtual double getQ2(double saj, double sjb, double sAB) override {
    // Note: arguments reversed intentionally!
    return TrialIISplitA::getQ2(sjb, saj, sAB);}

  // Generate new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override {return TrialIISplitA::genQ2run(q2old, sAB, zMin, zMax, colFac,
      PDFratio, b0, kR, Lambda, eB, eA, headroomFac, enhanceFac);}

  // Generate new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override {return TrialIISplitA::genQ2(q2old, sAB, zMin, zMax, colFac,
      alphaSvalue, PDFratio, eB, eA, headroomFac, enhanceFac);}

  // Generate new Q value, with running of the PDFs towards the mass
  // threshold.
  virtual double genQ2thres(double q2old, double sAB,
    double zMin, double zMax, double colFac, double alphaSvalue,
    double PDFratio, int idA, int idB, double eA, double eB, bool useMpdf,
    double headroomFac = 1.0, double enhanceFac = 1.0) override {
    return TrialIISplitA::genQ2thres(q2old, sAB, zMin, zMax, colFac,
      alphaSvalue, PDFratio, idB, idA, eB, eA, useMpdf, headroomFac,
      enhanceFac);}

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override {
    return TrialIISplitA::getSj2(Qt2, zeta, sAB);}
  virtual double getSj2(double Qt2, double zeta, double sAB) override {
    return TrialIISplitA::getS1j(Qt2, zeta, sAB);}

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB, double Qt2A, double Qt2B)
    override {
    // Note: arguments reversed intentionally!
    return TrialIISplitA::trialPDFratio(beamBPtr, beamAPtr, iSys,
      idB, idA, eB, eA, Qt2B, Qt2A);}

};

//==========================================================================

// A conversion trial function for initial-initial, g -> qqbar.

class TrialIIConvA : public TrialGeneratorISR {

public:

  // Name of trial generator
  virtual string name() override {return "TrialIIConvA";}

  // Trial antenna function. Used with x*PDF ratio <= const, so no
  // extra prefactor.
  virtual double aTrial(double saj, double sjb, double sAB) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjb, double sAB) override {
    return ((useMevolSav) ? saj : (saj*sjb/(saj + sjb + sAB)));}
  virtual double getQ2max(double sAB, double, double) override {
    return ((useMevolSav) ? (shhSav - sAB) : 0.25*pow2(shhSav - sAB)/shhSav);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS. Same
  // expression for QT and QA; Iz is different.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAB, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAB, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override;
  virtual double getSj2(double Qt2, double zeta, double sAB) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// B conversion trial function for initial-initial, g -> qqbar.

class TrialIIConvB : public TrialIIConvA {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIIConvB";}

  // Trial antenna function.
  virtual double aTrial(double saj, double sjb, double sAB) override {
    // Note: arguments reversed intentionally!
    return TrialIIConvA::aTrial(sjb, saj, sAB);}

  // Evolution scale.
  virtual double getQ2(double saj, double sjb, double sAB) override {
    // Note: arguments reversed intentionally!
    return TrialIIConvA::getQ2(sjb, saj, sAB);}

  // Generate a new Q value, with first-order running alphaS
  virtual double genQ2run(double q2old, double sAB, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override {return TrialIIConvA::genQ2run(q2old, sAB, zMin, zMax, colFac,
      PDFratio, b0, kR, Lambda, eB, eA, headroomFac, enhanceFac);}

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAB, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eB, double headroomFac = 1.0, double enhanceFac = 1.0)
    override {return TrialIIConvA::genQ2(q2old, sAB, zMin, zMax, colFac,
      alphaSvalue, PDFratio, eB, eA, headroomFac, enhanceFac);}

  // Inverse transforms: to obtain saj and sjb from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAB) override {
    return TrialIIConvA::getSj2(Qt2, zeta, sAB);}
  virtual double getSj2(double Qt2, double zeta, double sAB) override {
    return TrialIIConvA::getS1j(Qt2, zeta, sAB);}

  // Trial PDF ratio
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idB, double eA, double eB,
    double Qt2A, double Qt2B) override {
    // Note: arguments reversed intentionally!
    return TrialIIConvA::trialPDFratio(
      beamBPtr, beamAPtr, iSys, idB, idA, eB, eA, Qt2B, Qt2A);}

};

//==========================================================================

// Soft-eikonal trial function for initial-final.

class TrialIFSoft : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFSoft";}

  // Trial antenna function. This trial generator uses x*PDF <= const
  // as overestimate => no x-factor.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return (saj*sjk/(sAK+sjk));}
  virtual double getQ2max(double sAK, double eA, double eBeamUsed) override {
    double eAmax = ((sqrt(shhSav)/2.0) - (eBeamUsed-eA));
    return (sAK*(eAmax-eA)/eA);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral: dzeta/zeta/(zeta-1).
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// Specialised soft-eikonal trial function for initial-final when
// initial-state parton is a valence quark.

class TrialVFSoft : public TrialIFSoft {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialVFSoft";}

  // Trial antenna function. This trial generator uses PDF <= const as
  // overestimate => x-factor.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral: dzeta/(zeta-1).
  virtual double getIz(double zMin, double zMax) override;

};

//==========================================================================

// A gluon collinear trial function for initial-final.

class TrialIFGCollA : public TrialGeneratorISR {

 public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFGCollA";}

  // Trial antenna function. This trial generator uses x*PDF ratio <=
  // const as overestimate so the trial function has no extra
  // x-factor.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return (saj*sjk/(sAK+sjk));}
  virtual double getQ2max(double sAK, double eA, double eBeamUsed) override {
    double eAmax = ( (sqrt(shhSav)/2.0) - (eBeamUsed-eA) );
    return (sAK*(eAmax-eA)/eA);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio (= just a simple headroom factor).
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// K gluon collinear trial function for initial-final sector shower.

class TrialIFGCollK : public TrialGeneratorISR {

 public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFGCollK";}

  // Trial antenna function.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return (saj*sjk/(sAK+sjk));
  }
  virtual double getQ2max(double sAK, double eA, double eAused) override {
    double eAmax = ( (sqrt(shhSav)/2.0) - (eAused-eA) );
    return (sAK*(eAmax-eA)/eA);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac=1.0, double enhanceFac=1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac=1.0, double enhanceFac=1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eAused)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eAused)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;
};

//==========================================================================

// A splitting trial function for initial-final, q -> gqbar.

class TrialIFSplitA : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFSplitA";}

  // Trial antenna function. This trial generator uses the xf
  // overestimate so no extra x-factor.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return ((useMevolSav) ? saj : saj*sjk/(sAK + sjk));}
  virtual double getQ2max(double sAK, double eA, double eBeamUsed) override {
    double xA    = eA/(sqrt(shhSav)/2.0);
    double eAmax = ((sqrt(shhSav)/2.0) - (eBeamUsed - eA));
    return ((useMevolSav) ? sAK/xA : sAK*(eAmax - eA)/eA);}

  // Generate new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate new Q value, with running of the PDFs towards the mass
  // threshold.
  virtual double genQ2thres(double q2old, double sAK,
    double zMin, double zMax, double colFac, double alphaSvalue,
    double PDFratio, int idA, int idK, double eA, double eK, bool useMpdf,
    double headroomFac = 1.0, double enhanceFac = 1.0) override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// K splitting trial function for initial-final, g -> qqbar.

class TrialIFSplitK : public TrialGeneratorISR {

 public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFSplitK";}

  // Trial antenna function. This trial uses the xf overestimate so no
  // extra x factor.
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return ((useMevolSav) ? sjk : saj*sjk/(sAK + sjk));}
  virtual double getQ2max(double sAK, double eA, double eBeamUsed) override {
    double xA    = eA/(sqrt(shhSav)/2.0);
    double eAmax = ((sqrt(shhSav)/2.0) - (eBeamUsed - eA));
    return ((useMevolSav) ? sAK*(1.0 - xA)/xA : sAK*(eAmax - eA)/eA);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// A conversion trial function for initial-final, g -> qqbar.

class TrialIFConvA : public TrialGeneratorISR {

public:

  // Name of trial generator.
  virtual string name() override {return "TrialIFConvA";}

  // Trial antenna function. This trial currently uses the xf
  // overestimate so no extra x-factor (but could be changed to use
  // the f overestimate if many violations, and/or for valence
  // flavours).
  virtual double aTrial(double saj, double sjk, double sAK) override;

  // Evolution scale.
  virtual double getQ2(double saj, double sjk, double sAK) override {
    return ((useMevolSav) ? saj : saj*sjk/(sAK + sjk));}
  virtual double getQ2max(double sAK, double eA, double eBeamUsed) override {
    double xA    = eA/(sqrt(shhSav)/2.0);
    double eAmax = ((sqrt(shhSav)/2.0) - (eBeamUsed - eA));
    return ((useMevolSav) ? sAK/xA : sAK*(eAmax-eA)/eA);}

  // Generate a new Q value, with first-order running alphaS.
  virtual double genQ2run(double q2old, double sAK, double zMin, double zMax,
    double colFac, double PDFratio, double b0, double kR, double Lambda,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new Q value, with constant trial alphaS.
  virtual double genQ2(double q2old, double sAK, double zMin, double zMax,
    double colFac, double alphaSvalue, double PDFratio,
    double eA, double eK, double headroomFac = 1.0, double enhanceFac = 1.0)
    override;

  // Generate a new zeta value in [zMin,zMax].
  virtual double genZ(double zMin, double zMax) override;

  // The zeta integral.
  virtual double getIz(double zMin, double zMax) override;

  // The zeta boundaries, for a given value of the evolution scale.
  virtual double getZmin(double Qt2, double sAK, double eA, double eBeamUsed)
    override;
  virtual double getZmax(double Qt2, double sAK, double eA, double eBeamUsed)
    override;

  // Inverse transforms to obtain saj and sjk from Qt2 and zeta.
  virtual double getS1j(double Qt2, double zeta, double sAK) override;
  virtual double getSj2(double Qt2, double zeta, double sAK) override;

  // Trial PDF ratio.
  virtual double trialPDFratio(BeamParticle* beamAPtr, BeamParticle* beamBPtr,
    int iSys, int idA, int idK, double eA, double eK,
    double Qt2A, double Qt2B) override;

};

//==========================================================================

// The BranchElementalISR class, container for 2 -> 3 trial branchings.

// Input: i1In     carries colour (or incoming anticolour)
//        i2In     carries anticolour (or incoming colour)
//        colIn    colour tag
//        isVal1In (true if i1In is incoming and is a valence parton on its
//                  side/system)
//        isVal2In (true if i2In is incoming and is a valence parton on its
//                  side/system)
// Internal storage for II:
//        i1sav: whichever of i1In, i2In has positive pz
//        i2sav: whichever of i1In, i2In has negative pz
//        is1A : always true (i1sav always has positive pz)
//        isII : true
// Internal storage for IF:
//        i1sav: whichever of i1In, i2In is the initial-state leg
//        i2sav: whichever of i1In, i2In is the final-state leg
//        is1A : true if i1sav has positive pz, false if it has negative pz
//        isII : false

class BranchElementalISR {

public:

  // Constructors.
  BranchElementalISR() = default;
  BranchElementalISR(int iSysIn, Event& event, int iOld1In,
    int iOld2In, int colIn, bool isVal1In, bool isVal2In) {
    reset(iSysIn, event, iOld1In, iOld2In, colIn, isVal1In, isVal2In);}

  // Main method to initialize/reset a BranchElemental.
  void reset(int iSysIn, Event& event, int i1In, int i2In, int colIn,
    bool isVal1In, bool isVal2In);

  // Antenna mass (negative if spacelike virtual).
  double mAnt()  const {return mAntSav;}
  double m2Ant() const {return m2AntSav;}
  double sAnt()  const {return sAntSav;}
  // Dot products of daughters.
  double s12() const {return 2*new1.p()*new2.p();}
  double s23() const {return 2*new2.p()*new3.p();}
  double s13() const {return 2*new1.p()*new3.p();}

  // This is an II or an IF type.
  bool isII() const {return isIIsav;}
  // Is 1 a side A (p+) guy (need to know for pdfs in IF).
  bool is1A() const {return is1Asav;}
  // Valence.
  bool isVal1()  const {return isVal1sav;}
  bool isVal2()  const {return isVal2sav;}
  int colType1() const {return colType1sav;}
  int colType2() const {return colType2sav;}
  int col() const {return colSav;}
  int geti1() {return i1sav;}
  int geti2() {return i2sav;}
  int getId1() {return id1sav;}
  int getId2() {return id2sav;}
  int getSystem() {return system;}

  // Function to reset all trial generators for this branch elemental.
  void clearTrialGenerators();

  // Add a trial generator to this BranchElemental.
  void addTrialGenerator(enum AntFunType antFunTypeIn, bool swapIn,
    TrialGeneratorISR* trialGenPtrIn);

  // Add to and get rescue levels.
  void addRescue(int iTrial) {nShouldRescue[iTrial]++;}
  int getNshouldRescue(int iTrial) {return nShouldRescue[iTrial];}
  void resetRescue() {
    for (int i=0; i<(int)nShouldRescue.size(); i++) nShouldRescue[i] = 0;}

  // Function to return number of trial generators for this antenna.
  int nTrialGenerators() const {return trialGenPtrsSav.size();}

  // Save a generated trial branching.
  void saveTrial(int iTrial, double qOld, double qTrial, double zMin=0.,
    double zMax=0., double colFac=0.,double alphaEff=0., double pdfRatio=0.,
    int trialFlav=0, double extraMpdf=0., double headroom = 1.0,
    double enhanceFac = 1.0);

  // Add the physical pdf ratio.
  void addPDF(int iTrial,double pdfRatio) {physPDFratioSav[iTrial] = pdfRatio;}

  // Generate invariants for saved branching.
  bool genTrialInvariants(double& s1j, double& sj2,
    double eBeamUsed, int iTrial = -1);

  // Get trial function index of winner.
  int getTrialIndex() const;

  // Check if a saved trial exists for a particular trialGenerator.
  bool hasTrial(int iTrial) const {
    if (iTrial < int(hasSavedTrial.size())) return hasSavedTrial[iTrial];
    else return false;}

  // Get whether physical antenna is swapped.
  bool getIsSwapped(int iTrial = -1) const {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return isSwappedSav[iTrial];}

  // Get physical antenna function index of winner.
  enum AntFunType antFunTypePhys(int iTrial = -1) const {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return antFunTypePhysSav[iTrial];}

  // Get scale for a specific saved trial.
  double getTrialScale(int iTrial) const {
    if (iTrial < int(scaleSav.size())) return scaleSav[iTrial];
    else return -1.0;}

  // Get scale of winner.
  double getTrialScale() const;

  // Get colour factor.
  double getColFac(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return colFacSav[iTrial];}

  // Get headroom factor.
  double getHeadroomFac(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return headroomSav[iTrial];}

  // Get headroom factor.
  double getEnhanceFac(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return enhanceFacSav[iTrial];}

  // Get alpha S.
  double getAlphaTrial(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return alphaSav[iTrial];}

  // Get pdf ratio.
  double getPDFratioTrial(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return trialPDFratioSav[iTrial];}

  // For gluon conversions, get ID of quark flavour to convert to.
  int getTrialFlav(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return trialFlavSav[iTrial];}

  // Get pdf ratio.
  double getPDFratioPhys(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return physPDFratioSav[iTrial];}

  // Get the extra factor when getting rid of the heavy quarks.
  double getExtraMassPDFfactor(int iTrial = -1) {
    if (iTrial <= -1) iTrial = getTrialIndex();
    return extraMassPDFfactorSav[iTrial];}

  // Flag to set if a saved trial should be ignored and a new one generated.
  // Default value -1 : force all trials to renew.
  void renewTrial(int iTrial = -1) {
    if (iTrial >= 0) hasSavedTrial[iTrial] = false;
    else for (iTrial = 0; iTrial < int(hasSavedTrial.size()); ++iTrial)
           hasSavedTrial[iTrial] = false;}

  // List function.
  void list(bool header = false, bool footer = false) const;

  // Data storage members.
  int i1sav{}, i2sav{}, id1sav{}, id2sav{}, colType1sav{}, colType2sav{},
  h1sav{}, h2sav{};
  double e1sav{}, e2sav{};
  bool isVal1sav{}, isVal2sav{}, isIIsav{}, is1Asav{};
  Particle new1{}, new2{}, new3{};
  // Colour, not obvious, since for e.g. gg -> H we have two II antennae.
  int colSav{};
  // System and counter for vetos.
  int system{0}, nVeto{}, nHull{}, nHadr{};
  // We have to force a splitting (heavy quarks).
  bool forceSplittingSav{};

  // Trial Generators and properties of saved trials.
  vector<TrialGeneratorISR*> trialGenPtrsSav{};
  vector<double> zMinSav{}, zMaxSav{}, colFacSav{}, alphaSav{};
  vector<double> physPDFratioSav{}, trialPDFratioSav{};
  vector<double> extraMassPDFfactorSav{};
  vector<double> scaleSav{}, scaleOldSav{}, headroomSav{}, enhanceFacSav{};
  vector<bool> hasSavedTrial{}, isSwappedSav{};
  vector<enum AntFunType> antFunTypePhysSav{};
  vector<int> nShouldRescue{}, trialFlavSav{};
  // Note: isSwapped = true for II means physical antenna function is
  // coded for side A but trial generator is for side B.  For IF, is1A
  // = true for 1 being on side A, false for 1 being on side B.

 private:

  // Saved antenna invariant mass value.
  double m2AntSav{}, mAntSav{}, sAntSav{};

};

//==========================================================================

// The VinciaISR class.
// Main shower class for initial-state (II and IF) antenna showers
// Inherits from SpaceShower in Pythia 8 so can be used as alternative to
// SpaceShower.
// Methods that must replace ones in SpaceShower are marked with override.

class VinciaISR : public SpaceShower {

  // Allow VinciaFSR to access private information.
  friend class VinciaFSR;
  friend class VinciaHistory;

public:

  // Constructor.
  VinciaISR() : isInit(false) {;}

  // Destructor.
  virtual ~VinciaISR() {;}

  // Initialize shower. Possibility to force re-initialization by hand.
  void init(BeamParticle* beamAPtrIn, BeamParticle* beamBPtrIn) override;

  // Force reset at beginning of each event.
  void onBeginEvent() override { isPrepared = false; }

  // Possible limitation of first emission.
  bool limitPTmax(Event& event, double Q2Fac = 0., double Q2Ren = 0.) override;

  // Prepare system for evolution; identify ME.
  void prepare(int iSys, Event& event, bool limitPTmaxIn = false) override;

  // Update dipole list after each ISR emission.
  void update( int iSys, Event& event, bool hasWeakRad = false) override;

  // Select next pT in downwards evolution.
  double pTnext(Event& event, double pTbegAll, double pTendAll,
    int nRadIn = -1, bool doTrialIn = false) override;

  // Perform a branching (as defined by current "winner"). Returns
  // true of the branching was accepted and false if rejected.
  bool branch(Event& event) override;

  // Print a list of II and IF dipole-antennae.
  void list() const override;

  // Initialize data members for calculation of uncertainty bands. So
  // far purely dummy in Vincia.
  bool initUncertainties() override {return false;}

  // Flag for failure in branch(...) that will force a retry of parton
  // level. Returns false in base class, and rescatterFail in
  // simpleTimeShower. Not implemented in Vincia yet since we do not
  // do rescattering.
  bool doRestart() const override {return false;}

  // Flag for last ISR branching being gamma -> qqbar. Not implemented
  // in Vincia's QED evolution yet.
  bool wasGamma2qqbar() override {return false;}

  // Tell whether ISR has done a weak emission. Not implemented in Vincia yet.
  bool getHasWeaklyRadiated() override {return false;}

  // Tell which system was the last processed one.
  int system() const override {return iSysWin;}

  // Potential enhancement factor of pTmax scale for hardest emission.
  // Used if limitPTmax = true.
  double enhancePTmax() const override {return pTmaxFudge;}

  // Initialise pointers to Vincia objects.
  void initVinciaPtrs(VinciaColour* colourPtrIn,
    shared_ptr<VinciaFSR> fsrPtrIn, MECs* mecsPtrIn,
    Resolution* resolutionPtrIn, VinciaCommon* vinComPtrIn,
    VinciaWeights* vinWeightsPtrIn);

  // Set pointer to object to use for diagnostics and profiling.
  void setDiagnosticsPtr(shared_ptr<VinciaDiagnostics> diagnosticsPtrIn) {
    diagnosticsPtr = diagnosticsPtrIn;
  }

  // Set EW shower module.
  void setEWShowerPtr(VinciaModulePtr ewShowerPtrIn) {
    ewShowerPtr = ewShowerPtrIn;
  }

  // Set QED shower module for hard process and resonance decays.
  void setQEDShowerHardPtr(VinciaModulePtr qedShowerPtrIn) {
    qedShowerHardPtr = qedShowerPtrIn;
  }

  // Set QED shower module for MPI and hadronisation.
  void setQEDShowerSoftPtr(VinciaModulePtr qedShowerPtrIn) {
    qedShowerSoftPtr = qedShowerPtrIn;
  }

  // Clear all containers.
  void clearContainers();

  // Initialize pointers to antenna sets.
  void initAntPtr(AntennaSetISR* antSetIn) {antSetPtr = antSetIn;}

  // Function to tell if VinciaISR::prepare() has treated this system.
  bool prepared(int iSys) {
    if (hasPrepared.find(iSys) != hasPrepared.end()) return hasPrepared[iSys];
    else return false;}

  // Wrapper function to return a specific antenna inside antenna set
  AntennaFunctionIX* getAntFunPtr(enum AntFunType antFunType) {
    return antSetPtr->getAntFunPtr(antFunType);}

  // Evolution windows, phase space region boundaries.
  int getRegion(double q) {
    for (int i=1; i<(int)regMinScalesSav.size(); i++)
      if (q < regMinScalesSav[i]) return i-1;
    return (int)regMinScalesSav.size() - 1;}

  // Evolution window, phase space region boundaries.
  double getQmin(int iRegion) {
    iRegion = max(0,iRegion);
    iRegion = min(iRegion,(int)regMinScalesSav.size()-1);
    return regMinScalesSav[iRegion];}

  // Number of active flavors.
  int getNf(int iRegion) {
    if (iRegion <= 1) return 3;
    else if (iRegion <= 2) return 4;
    else if (iRegion <= 4) return 5;
    else return 6;}

  // Lambda value
  double getLambda(int nFin) {
    if (nFin <= 3) return alphaSptr->Lambda3();
    else if (nFin <= 4) return alphaSptr->Lambda4();
    else if (nFin <= 5) return alphaSptr->Lambda5();
    else return alphaSptr->Lambda6();}

  // Add trial functions to a BranchElemental.
  void resetTrialGenerators(BranchElementalISR* trial);

  // Method to check if a gluon splitting in the initial state (to get
  // rid of heavy quarks) is still possible after the current
  // branching.
  bool checkHeavyQuarkPhaseSpace(vector<Particle> parts, int iSyst);

  // Method to check if heavy quark left after passing the evolution window.
  bool heavyQuarkLeft(double qTrial);

  // Function to ask if a system is hard.
  bool isSysHard(int iSys) {
    if (!isInit) return false;
    if ((int)isHardSys.size() <= iSys) return false;
    return isHardSys[iSys];}

  // Return a vector of the masses.
  vector<double> getMasses() {return vector<double> {mt, mtb, mb, mc, ms};}

  // Get number of systems.
  int getNsys() {return nBranchISR.size();}
  // Get number of branchings in a system (return -1 if no such system).
  int getNbranch(int iSys = -1) {
    int n = 0;
    if (iSys < 0) for (int i = 0; i < (int)nBranchISR.size(); ++i)
                    n += nBranchISR[iSys];
    else if (iSys < (int)nBranchISR.size()) n = nBranchISR[iSys];
    else n = -1;
    return n;}

  // Communicate information about trial showers for merging.
  void setIsTrialShower(bool isTrialIn){ isTrialShower = isTrialIn; }
  void setIsTrialShowerRes(bool isTrialIn){ isTrialShowerRes = isTrialIn; }

  // Save the flavour content of system in Born state
  // (needed for sector shower).
  void saveBornState(Event& born, int iSys);
  // Save the flavour content of Born for trial shower.
  void saveBornForTrialShower(Event& born);

  // Set verbosity level.
  void setVerbose(int verboseIn) {verbose = verboseIn;}

  // Check the antennae.
  bool checkAntennae(const Event& event);

  // Pointer to global AlphaStrong instance.
  AlphaStrong* alphaSptr;

private:

  // Set starting scale of shower (power vs wimpy) for system iSys.
  void setStartScale(int iSys, Event& event);

  // Function to return headroom factor.
  double getHeadroomFac(int iSys, enum AntFunType antFunTypePhysIn,
    double qMinNow);

  // Generate trial branching kinematics and check physical phase space
  bool generateKinematics(Event& event, BranchElementalISR* trialPtr,
    vector<Vec4>& pRec) {
    return ( trialPtr->isII()
      ? generateKinematicsII(event, trialPtr, pRec)
      : generateKinematicsIF(event, trialPtr, pRec) ); }

  // Generate kinematics (II) and set flavours and masses.
  bool generateKinematicsII(Event& event, BranchElementalISR* trialPtr,
    vector<Vec4>& pRec);

  // Generate kinematics (IF) and set flavours and masses.
  bool generateKinematicsIF(Event& event, BranchElementalISR* trialPtr,
    vector<Vec4>& pRec);

  // Main trial accept function.
  bool acceptTrial(const Event& event, BranchElementalISR* winnerPtr);

  // Method to assign colour flow.
  bool assignColourFlow(Event& event, BranchElementalISR* trialPtr);

  // Initialised.
  bool isInit;
  bool isPrepared;

  // Possibility to allow user veto of emission step.
  bool hasUserHooks, canVetoEmission;

  // Beams, saved as positive and negative pz respectively.
  int beamFrameType;
  double eBeamA, eBeamB, eCMBeamsSav, m2BeamsSav;
  double TINYPDF;

  // Main Vincia ISR on/off switches.
  bool doII, doIF, doQED;

  // Map of which systems ISR::prepare() has treated.
  map<int, bool> hasPrepared;

  // Shower parameters.
  bool helicityShower, sectorShower, convGluonToQuarkI, convQuarkToGluonI;
  bool kineMapIFretry;
  int nGluonToQuarkI, nGluonToQuarkF;
  double cutoffScaleII, cutoffScaleIF;
  int nFlavZeroMass;

  // Factorization scale and shower starting settings.
  int    pTmaxMatch;
  double pTmaxFudge, pT2maxFudge, pT2maxFudgeMPI;

  // AlphaS parameters.
  bool useCMW;
  int alphaSorder;
  double alphaSvalue, alphaSmax, alphaSmuFreeze, alphaSmuMin;
  double aSkMu2EmitI, aSkMu2SplitI, aSkMu2SplitF, aSkMu2Conv;
  double mt, mtb, ms, mb, mc;
  // Calculated values.
  double mu2freeze, mu2min;

  // Trial generators.
  TrialIISoft    trialIISoft;
  TrialIIGCollA  trialIIGCollA;
  TrialIIGCollB  trialIIGCollB;
  TrialIISplitA  trialIISplitA;
  TrialIISplitB  trialIISplitB;
  TrialIIConvA   trialIIConvA;
  TrialIIConvB   trialIIConvB;
  TrialIFSoft    trialIFSoft;
  TrialVFSoft    trialVFSoft;
  TrialIFGCollA  trialIFGCollA;
  TrialIFSplitA  trialIFSplitA;
  TrialIFSplitK  trialIFSplitK;
  TrialIFConvA   trialIFConvA;

  // Trial generators for the sector shower.
  TrialIFGCollK  trialIFGCollK;

  // Enhancing switches and parameters.
  bool enhanceInHard, enhanceInResDec, enhanceInMPI;
  double enhanceAll, enhanceBottom, enhanceCharm, enhanceCutoff;

  // Pointer to VINCIA objects.
  AntennaSetISR*        antSetPtr{};
  MECs*                 mecsPtr{};
  VinciaColour*         colourPtr{};
  Resolution*           resolutionPtr{};
  shared_ptr<VinciaFSR> fsrPtr{};
  VinciaCommon*         vinComPtr{};
  VinciaWeights*        weightsPtr{};

  // Diagnostics and Profiling.
  shared_ptr<VinciaDiagnostics> diagnosticsPtr;

  // Electroweak shower pointers.
  VinciaModulePtr       ewShowerPtr;
  VinciaModulePtr       qedShowerHardPtr;
  VinciaModulePtr       qedShowerSoftPtr;

  // Total and MEC accept probability.
  vector<double> Paccept;

  // Evolution windows.
  vector<double> regMinScalesMtSav;
  vector<double> regMinScalesSav;
  vector<double> regMinScalesNow;

  // Vector of dipoles (with trial branchings, 4 at most).
  vector<BranchElementalISR > branchElementals;

  // Current winner.
  BranchElementalISR* winnerPtr{};
  int indxWin;
  int iSysWin;
  vector<Particle> stateNew;
  VinciaClustering minClus;

  // Flags to tell a few basic properties of each parton system.
  map<int, bool> isHardSys, isResonanceSys, polarisedSys, doMECsSys;

  // Saved particle state and number in event record.
  map<int, vector< Particle > > partsSav;
  map<int, vector< int      > > indexSav;

  // Save initial ISR starting scale system by system.
  map<int, double> Q2hat;

  // Count the number of branchings in the system.
  map<int, int> nBranch, nBranchISR;

  // Saved incoming guys.
  map<int, Particle> initialA;
  map<int, Particle> initialB;
  double eBeamAUsed, eBeamBUsed;

  // Count numbers of quarks and gluons.
  map<int, int> nG, nQQ;

  // Partons present in the Born (needed in sector shower).
  map<int, bool> savedBorn;
  map<int, bool> resolveBorn;
  map<int, map<int, int>> nFlavsBorn;

  // Flags used in merging
  bool doMerging, isTrialShower, isTrialShowerRes;

  // Rescue mechanism.
  bool doRescue;
  int nRescue;
  double rescueMin;

  // Verbose setting.
  int verbose;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaISR_H
