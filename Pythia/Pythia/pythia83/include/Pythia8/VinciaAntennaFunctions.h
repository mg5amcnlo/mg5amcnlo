// VinciaAntennaFunctions.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains header information for the AntennaFunction base
// class, its derived classes for FF, IF, and II antenna functions,
// the AntennaSetFSR and AntennaSetISR classes, and the MEC class for
// matrix- element corrections.

#ifndef Pythia8_VinciaAntennaFunctions_H
#define Pythia8_VinciaAntennaFunctions_H

// Pythia headers.
#include "Pythia8/Basics.h"
#include "Pythia8/Event.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/ShowerMEs.h"

// Vincia headers.
#include "Pythia8/VinciaCommon.h"

namespace Pythia8 {

//==========================================================================

// A class containing DGLAP splitting functions for limit checking.

class DGLAP {

public:

  // Constructor.
  DGLAP() {;}

  // Helicity-dependent Altarelli-Parisi kernels (mu = m/Q).  Note,
  // Pg2gg is written with standard factor 2 normalization convention.
  double Pg2gg(double z, int hA=9, int hB=9, int hC=9);
  double Pg2qq(double z, int hA=9, int hB=9, int hC=9, double mu=0.);
  double Pq2qg(double z, int hA=9, int hB=9, int hC=9, double mu=0.);
  double Pq2gq(double z, int hA=9, int hB=9, int hC=9, double mu=0.);

  // Wrappers to get unpolarized massive Altarelli-Pariis kernels (mu = m/Q).
  double Pg2qq(double z, double mu) {return Pg2qq(z, 9, 9, 9, mu);}
  double Pq2qg(double z, double mu) {return Pq2qg(z, 9, 9, 9, mu);}
  double Pq2gq(double z, double mu) {return Pq2gq(z, 9, 9, 9, mu);}

  // Altarelli-Parisi kernels with linear in/out polarizations for
  // gluons: pol = +1 for in-plane, -1 for out-of-plane.
  double Pg2ggLin(double z, int polA = 9, int polB = 9, int polC = 9);
  double Pg2qqLin(double z, int polA = 9, int hB = 9, int hC = 9,
                  double mu = 0.);
  double Pq2qgLin(double z, int hA = 9, int hB=9, int polC = 9,
                  double mu = 0.);
  double Pq2gqLin(double z, int hA = 9, int polB=9, int hC = 9,
                  double mu = 0.);

};

//==========================================================================

// The AntennaFunction base class. Base class implementation for all
// AntennaFunction objects.

class AntennaFunction {

public:

  // Constructor.
  AntennaFunction() = default;

  // Destructor.
  virtual ~AntennaFunction() {};

  // Names of this antenna, for VINCIA, and for humans.
  virtual string vinciaName() const = 0;

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA() const = 0;
  virtual int idB() const = 0;
  virtual int id1() const = 0;

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> mNew,
    vector<int> helBef, vector<int> helNew) = 0;

  // Optional implementation of the DGLAP kernels for collinear-limit checks
  // Defined as PI/sij + PK/sjk, i.e. equivalent to antennae.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew) = 0;

  // Default initialization.
  virtual bool init();

  // Construct baseName from idA, idB, and id1.
  virtual string baseName() const {
    return id2str(id1()) + "/" + id2str(idA()) + id2str(idB());}

  // Wrapper that can modify baseName to more human readable form if required.
  virtual string humanName() const {return baseName();}

  // Function to check singularities, positivity, etc.
  virtual bool check();

  // Method to intialise mass values.
  virtual void initMasses(vector<double>* masses) {
    if (masses->size() >= 3) {
      mi = masses->at(0); mj = masses->at(1); mk = masses->at(2);
    } else {mi = 0.0; mj = 0.0; mk = 0.0;}}

  // Method to initialise internal helicity variables.
  virtual int initHel(vector<int>* helBef, vector<int>* helNew);

  // Wrapper for helicity-summed/averaged antenna function.
  double antFun(vector<double> invariants, vector<double> masses) {
    return antFun(invariants, masses, hDum, hDum);}

  // Wrapper for massless, helicity-summed/averaged antenna function.
  double antFun(vector<double> invariants) {
    return antFun(invariants, mDum, hDum, hDum);}

  // Wrapper without helicity assignments.
  double AltarelliParisi(vector<double> invariants, vector<double> masses) {
    return AltarelliParisi(invariants, masses, hDum, hDum);}

  // Wrapper for massless helicity-summed/averaged DGLAP kernels.
  double AltarelliParisi(vector<double> invariants) {
    return AltarelliParisi(invariants, mDum, hDum, hDum);}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn);

  // Get parameters.
  double chargeFac()  {return chargeFacSav;}
  int    kineMap()    {return kineMapSav;}
  double alpha()      {return alphaSav;}
  double sectorDamp() {return sectorDampSav;}

  // Functions to get Altarelli-Parisi energy fractions from invariants.
  double zA(vector<double> invariants) {
    double yij = invariants[1]/invariants[0];
    double yjk = invariants[2]/invariants[0];
    return (1.-yjk)/(1.+yij);}
  double zB(vector<double> invariants) {
    double yij = invariants[1]/invariants[0];
    double yjk = invariants[2]/invariants[0];
    return (1.-yij)/(1.+yjk);}

  // Auxiliary function to translate an ID code to a string.
  string id2str(int id) const;

protected:

  // Is initialized.
  bool isInitPtr{false}, isInit{false};

  // Charge factor, kinematics map, and subleading-colour treatment.
  double chargeFacSav{0.0};
  int kineMapSav{0}, modeSLC{-1};
  bool sectorShower{false};

  // The alpha collinear-partitioning parameter.
  double alphaSav{0.0};

  // The sector-shower collinear dampening parameter.
  double sectorDampSav{0.0};

  // Shorthand for commonly used variable(s).
  double term{}, preFacFiniteTermSav{0.0}, antMinSav{0.0};
  bool   isMinVar{};

  // Variables for internal storage of masses and helicities.
  double mi{0.0}, mj{0.0}, mk{0.0};
  int hA{9}, hB{9}, hi{9}, hj{9}, hk{9};

  // Map to tell whether a given helicity value maps to L- and/or
  // R-handed. Defined by constructor and not to be changed
  // dynamically.
  map<int, bool> LH{{9, true}, {1, false}, {-1, true}};
  map<int, bool> RH{{9, true}, {1, true},  {-1, false}};

  // Verbosity level.
  int verbose{1};

  // Pointers to Pythia8 classes.
  Info*         infoPtr{};
  ParticleData* particleDataPtr{};
  Settings*     settingsPtr{};
  Rndm*         rndmPtr{};

  // Pointer to VINCIA DGLAP class.
  DGLAP* dglapPtr{};

  // Dummy vectors.
  vector<double> mDum{0, 0, 0, 0};
  vector<int> hDum{9, 9, 9, 9};

};

//==========================================================================

// Class AntQQemitFF, final-final antenna function.

class AntQQemitFF : public AntennaFunction {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const {return "Vincia:QQemitFF";};

  // Functions needed by soft- and collinear-limit checks (AB -> 0i 1j 2k).
  virtual int idA() const {return 1;}
  virtual int idB() const {return -1;}
  virtual int id1() const {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> mNew,
    vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  // Defined as PI/sij + PK/sjk, i.e. equivalent to antennae.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntQGemitFF, final-final antenna function.

class AntQGemitFF : public AntennaFunction {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const {return "Vincia:QGemitFF";};

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA() const {return 1;}
  virtual int idB() const {return 21;}
  virtual int id1() const {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double> /* mNew */, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGQemitFF, final-final antenna function.

class AntGQemitFF : public AntQGemitFF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const {return "Vincia:GQemitFF";};

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA() const {return 21;}
  virtual int idB() const {return -1;}
  virtual int id1() const {return 21;}

  // The antenna function [GeV^-2] (derived from AntQGemit by swapping).
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGQemitFF, final-final antenna function.

class AntGGemitFF : public AntennaFunction {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName()    const {return "Vincia:GGemitFF";};

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA()    const {return 21;}
  virtual int idB()    const {return 21;}
  virtual int id1()    const {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGXsplitFF, final-final antenna function.

class AntGXsplitFF : public AntennaFunction {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const {return "Vincia:GXsplitFF";};

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA() const {return 21;}
  virtual int idB() const {return  0;}
  virtual int id1() const {return -1;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntQQemitFFsec, sector final-final antenna function, identical
// to global one.

class AntQQemitFFsec : public AntQQemitFF {};

//==========================================================================

// Class AntQGemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntQGemitFF.

class AntQGemitFFsec : public AntQGemitFF {

public:

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGQemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGQemitFF.

class AntGQemitFFsec : public AntQGemitFFsec {

public:

  // Parton types (AB -> 0i 1j 2k): needed by soft- and collinear-limit checks.
  virtual int idA() const {return 21;}
  virtual int idB() const {return -1;}
  virtual int id1() const {return 21;}

  // The antenna function [GeV^-2] (derived from AntQGemitFFsec by swapping).
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew);

  // Function to give Altarelli-Parisi limits of this antenna.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGGemitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGGemitFF.

class AntGGemitFFsec : public AntGGemitFF {

public:

  // The dimensionless antenna function.
  virtual double antFun(vector<double> invariants, vector<double> mNew,
    vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntGXsplitFFsec, sector final-final antenna function, explicit
// symmetrisation of AntGXsplitFF.

class AntGXsplitFFsec : public AntGXsplitFF {

 public:

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> mNew,
    vector<int> helBef, vector<int> helNew);

};

//==========================================================================

// Class AntennaFunctionIX, base class for initial-initial and
// initial-final antenna functions which implements II. All derived
// classes for initial-initial antenna functions are the same for
// global and sector cases since even the global ones include sector
// terms representing "emission into the initial state".

class AntennaFunctionIX : public AntennaFunction {

public:

  // Method to initialise (can be different than that of the base class).
  virtual bool init() override;

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {
    return "Vincia:AntennaFunctionIX";}

  // Parton types AB -> 0a 1j 2b with A,B,a,b initial and j final.
  virtual int idA() const override {return 0;}
  virtual int idB() const override {return 0;}
  virtual int id0() const {return 0;}
  virtual int id1() const override {return 0;}
  virtual int id2() const {return 0;}

  // Functions to get Altarelli-Parisi energy fractions.
  virtual double zA(vector<double> invariants) {double sAB = invariants[0];
    double sjb = invariants[2]; return sAB/(sAB+sjb);}
  virtual double zB(vector<double> invariants) {double sAB = invariants[0];
    double saj = invariants[1]; return sAB/(sAB+saj);}

  // Function to tell if this is an II antenna.
  virtual bool isIIant() {return true;}

  // Function to check singularities, positivity, etc.
  virtual bool check() override;

};

//==========================================================================

// Class AntQQemitII, initial-initial antenna function.

class AntQQemitII : public AntennaFunctionIX {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:QQemitII";}

  // Parton types AB -> 0a 1j 2b with A,B,a,b initial and j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return -1;}
  virtual int id0() const override {return 1;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return -1;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // AP splitting kernel for collinear limit checks.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntGQemitII, initial-initial antenna function.

class AntGQemitII : public AntennaFunctionIX {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GQemitII";}

  // Parton types AB -> 0a 1j 2b with A,B,a,b initial and j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 1;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return 1;}

  // The antenna function.
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // AP splitting kernel for collinear limit checks.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntGGemitII, initial-initial antenna function.

class AntGGemitII : public AntennaFunctionIX {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GGemitII";}

  // Parton types AB -> 0a 1j 2b with A,B,a,b initial and j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 21;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // AP splitting kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntQXsplitII, initial-initial antenna function. splitting is in
// the forwards sense, i.e. quark backwards evolving to a gluon and
// emitting an antiquark in the final state.

class AntQXsplitII : public AntennaFunctionIX {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override { return "Vincia:QXsplitII";}

  // Parton types AB -> 0a 1j 2b with A,B, a,b initial and j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return 0;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return -1;}
  virtual int id2() const override {return 0;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // AP splitting kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

  // Mark that this function has no zB collinear limit.
  virtual double zB(vector<double>) override {return -1.0;}

};

//==========================================================================

// Class AntGXconvII, initial-initial antenna function. Gluon evolves
// backwards into a quark and emits a quark in the final state.

class AntGXconvII : public AntennaFunctionIX {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GXconvII";}

  // Parton types AB -> 0a 1j 2b with A,B,a,b initial and j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 0;}
  virtual int id0() const override {return 2;}
  virtual int id1() const override {return 2;}
  virtual int id2() const override {return 0;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // AP splitting kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

  // Mark that this function has no zB collinear limit.
  virtual double zB(vector<double>) override {return -1.0;}

};

//==========================================================================

// Class AntennaFunctionIF, base class for IF/RF antenna functions
// which implements AntQQemitIF. Derived classes are for global
// initial-final and resonance-final antenna functions. The method
// isRFant() distinguishes between the two.

class AntennaFunctionIF : public AntennaFunctionIX {

public:

  // Method to initialise (can be different than that of the base class).
  virtual bool init() override;

  // Function to check singularities, positivity, etc.
  virtual bool check() override;

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {
    return "Vincia:AntennaFunctionIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return -1;}
  virtual int id0() const override {return 1;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return -1;}

  // Functions to get Altarelli-Parisi energy fractions.
  virtual double zA(vector<double> invariants) override {
    double sAK(invariants[0]), sjk(invariants[2]); return sAK/(sAK+sjk);}
  virtual double zB(vector<double> invariants) override {
    double sAK(invariants[0]), saj(invariants[1]); return (sAK-saj)/sAK;}

  // Methods to tell II, IF, and RF apart.
  virtual bool isIIant() override {return false;}
  virtual bool isRFant() {return false;}

  // Check for resonances.
  virtual bool checkRes();

  // Create the test masses for the checkRes method.
  virtual void getTestMasses(vector<double> &masses) {masses.resize(4, 0.0);}

  // Create the test invariants for the checkRes method.
  virtual bool getTestInvariants(vector<double> &invariants,
    vector<double> masses, double yaj, double yjk);

protected:

  // Massive eikonal factor, n.b. is positive definite - proportional
  // to gram determinant.
  double massiveEikonal(double saj, double sjk, double sak,
    double m_a, double m_k) {return 2.0*sak/(saj*sjk) - 2.0*m_a*m_a/(saj*saj)
      - 2.0*m_k*m_k/(sjk*sjk);}

  // Massive eikonal factor, given invariants and masses.
  double massiveEikonal(vector<double> invariants, vector<double> masses) {
    return massiveEikonal(invariants[1], invariants[2], invariants[3],
                          masses[0], masses[2]);}

  // Return the Gram determinant.
  double gramDet(vector<double> invariants, vector<double> masses) {
    double saj(invariants[1]), sjk(invariants[2]), sak(invariants[3]),
      mares(masses[0]), mjres(masses[1]), mkres(masses[2]);
    return 0.25*(saj*sjk*sak - saj*saj*mkres*mkres -sak*sak*mjres*mjres
      - sjk*sjk*mares*mares + 4.0*mares*mares*mjres*mjres*mkres*mkres);}

  // Wrapper for comparing to AP functions, sums over flipped
  // invariants where appropriate.
  double antFunCollLimit(vector<double> invariants,vector<double> masses);

};

//==========================================================================

// Class AntQQemitIF, initial-final antenna function.

class AntQQemitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override { return "Vincia:QQemitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return -1;}
  virtual int id0() const override {return 1;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return -1;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

  // Functions to get Altarelli-Parisi energy fractions.
  virtual double zA(vector<double> invariants) override {
    double sAK(invariants[0]), sjk(invariants[2]); return sAK/(sAK+sjk);}
  virtual double zB(vector<double> invariants) override {
    double sAK(invariants[0]), saj(invariants[1]); return (sAK-saj)/sAK;}

};

//==========================================================================

// Class AntQGemitIF, initial-final antenna function.

class AntQGemitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:QGemitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return 21;}
  virtual int id0() const override {return 1;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntGQemitIF, initial-final antenna function.

class AntGQemitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GQemitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 1;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return 1;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntGGemitIF, initial-final antenna function.

class AntGGemitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GGemitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 21;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return 21;}
  virtual int id2() const override {return 21;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntQXsplitIF, initial-final antenna function. splitting is in
// the forwards sense, i.e. quark backwards evolving to a gluon and
// emitting an antiquark in the final state.

class AntQXsplitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:QXsplitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 1;}
  virtual int idB() const override {return 0;}
  virtual int id0() const override {return 21;}
  virtual int id1() const override {return -1;}
  virtual int id2() const override {return 0;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  virtual double AltarelliParisi(vector<double> invariants,
    vector<double> /*mNew*/, vector<int> helBef, vector<int> helNew) override;

  // Mark that this function does not have a zB collinear limit.
  virtual double zB(vector<double>) override {return -1.0;}

};

//==========================================================================

// Class AntGXconvIF, initial-final antenna function. Gluon evolves
// backwards into a quark and emits a quark in the final state.

class AntGXconvIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:GXconvIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 21;}
  virtual int idB() const override {return 0;}
  virtual int id0() const override {return 2;}
  virtual int id1() const override {return 2;}
  virtual int id2() const override {return 0;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

  // Mark that this function does not have a zB collinear limit.
  virtual double zB(vector<double>) override {return -1.0;}

};

//==========================================================================

// Class AntXGsplitIF, initial-final antenna function. Gluon splitting in
// the final state.

class AntXGsplitIF : public AntennaFunctionIF {

public:

  // Names (remember to redefine both for each inherited class).
  virtual string vinciaName() const override {return "Vincia:XGsplitIF";}

  // Parton types AB -> 0a 1j 2b with A,a initial and B,b,j final.
  virtual int idA() const override {return 0;}
  virtual int idB() const override {return 21;}
  virtual int id0() const override {return 0;}
  virtual int id1() const override {return -1;}
  virtual int id2() const override {return 1;}

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> masses,
    vector<int> helBef, vector<int> helNew) override;

  // The AP kernel, P(z)/Q2.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double>, vector<int> helBef, vector<int> helNew) override;

  // Mark that this function does not have a zA collinear limit.
  virtual double zA(vector<double>) override {return -1.0;}

};

//==========================================================================

// Class AntQGemitIFsec, derived class for sector initial-final antenna
// function. Note only the final-state leg needs to be symmetrised,
// as the global IF functions already contain sector terms on their
// initial-state legs to account for the absence of "emission into the
// initial state".

class AntQGemitIFsec : public AntQGemitIF {

public:

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntGGemitIFsec, sector initial-final antenna function.

class AntGGemitIFsec : public AntGGemitIF {

public:

  // The antenna function [GeV^-2].
  virtual double antFun(vector<double> invariants,
    vector<double> mNew, vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntXGsplitIFsec, sector initial-final antenna function. Gluon
// splitting in the final state.

class AntXGsplitIFsec : public AntXGsplitIF {

public:

  // The antenna function, just 2*global [GeV^-2].
  virtual double antFun(vector<double> invariants, vector<double> mNew,
    vector<int> helBef, vector<int> helNew) override;

};

//==========================================================================

// Class AntQQemitRF, resonance-final antenna function.

class AntQQemitRF : public AntQQemitIF {

public:

  // Names (remember to redefine both for each inherited class).
  string vinciaName() const override {return "Vincia:QQemitRF";}

  // Parton types AB -> ijk with A,i initial and B,k,j final.
  int idA() const override {return 6;}
  int idB() const override {return 5;}
  int id0() const override {return 6;}
  int id1() const override {return 21;}
  int id2() const override {return 5;}

  // Mark that this function does not have a zA collinear limit.
  double zA(vector<double>) override {return -1;}

  // Return this is a resonance-final antenna.
  bool isRFant() override {return true;}

  // Test masses (top, gluon, bottom, and W mass).
  void getTestMasses(vector<double> &masses) override {masses =
      {particleDataPtr->m0(6), 0.0, particleDataPtr->m0(5),
       particleDataPtr->m0(24)};}

  // AP with dummy helicities.
  virtual double AltarelliParisi(vector<double> invariants,
    vector<double> masses, vector<int>, vector<int>) override {
    double sjk(invariants[2]), mkres(masses[2]), z(zB(invariants)),
      mu2(mkres*mkres/sjk), Pz(dglapPtr->Pq2gq(z,9,9,9,mu2));
    return Pz/sjk;};

};

//==========================================================================

// Class AntQGemitRF, resonance-final antenna function.

class AntQGemitRF : public AntQGemitIF {

public:

  // Names (remember to redefine both for each inherited class).
  string vinciaName() const override {return "Vincia:QGemitRF";}

  // Parton types AB -> ijk with A,i initial and B,k,j final.
  int idA() const override {return 6;}
  int idB() const override {return 21;}
  int id1() const override {return 21;}
  int ida() const {return 6;}
  int idb() const {return 21;}

  // Mark that this function does not have a zA collinear limit.
  double zA(vector<double>) override {return -1;}

  // Return this is a resonance-final antenna.
  bool isRFant() override {return true;}

  // Test masses (top, gluon, gluon, X).
  void getTestMasses(vector<double> &masses) override {masses =
      {particleDataPtr->m0(6), 0.0, 0.0, 0.6*particleDataPtr->m0(6)};}

  // AP with dummy helicities and masses.
  virtual double AltarelliParisi(vector<double> invariants, vector<double>,
    vector<int>, vector<int>) override {
    double sjk(invariants[2]), z(zB(invariants)),
      Pz(dglapPtr->Pg2gg(z, 9, 9, 9));
    return Pz/sjk;}

};

//==========================================================================

// Class AntQGemitRF, resonance-final antenna function.

class AntQGemitRFsec : public AntQGemitIFsec {

public:

  // Names (remember to redefine both for each inherited class).
  string vinciaName() const override {return "Vincia:QGemitRF";}

  // Parton types AB -> ijk with A,i initial and B,k,j final.
  int idA() const override {return 6;}
  int idB() const override {return 21;}
  int id1() const override {return 21;}
  int ida() const {return 6;}
  int idb() const {return 21;}

  // Mark that this function does not have a zA collinear limit.
  double zA(vector<double>) override {return -1;}

  // Return this is a resonance-final antenna.
  bool isRFant() override {return true;}

  // Test masses (top, gluon, gluon, X).
  void getTestMasses(vector<double> &masses) override {masses =
      {particleDataPtr->m0(6), 0.0, 0.0, 0.6*particleDataPtr->m0(6)};}

  // AP with dummy helicities and masses.
  virtual double AltarelliParisi(vector<double> invariants, vector<double>,
    vector<int>, vector<int>) override {
    double sjk(invariants[2]), z(zB(invariants)),
      Pz(dglapPtr->Pg2gg(z, 9, 9, 9));
    return Pz/sjk;}

};

//==========================================================================

// Class AntXGsplitRF, resonance-final antenna function.

class AntXGsplitRF : public AntXGsplitIF {

public:

  // Names (remember to redefine both for each inherited class)
  string vinciaName() const override {return "Vincia:XGsplitRF";}

  // Mark that this function does not have a zA collinear limit.
  double zA(vector<double>) override {return -1;}

  // Return this is a resonance-final antenna.
  bool isRFant() override {return true;}

  // Parton types AB -> ijk with A,i initial and B,k,j final.
  int idA() const override {return 6;}
  int idB() const override {return 21;}
  int id1() const override {return -2;}
  int ida() const {return 6;}
  int idb() const {return 2;}

  // Test masses (top, gluon, gluon, X).
  void getTestMasses(vector<double> &masses) override {masses =
      {particleDataPtr->m0(6), 0.0, 0.0, 0.6*particleDataPtr->m0(6)};}

  // AP with dummy helicities.
  double AltarelliParisi(vector<double> invariants, vector<double> masses,
    vector<int>, vector<int>) override {
    double sAK(invariants[0]), saj(invariants[1]), sjk(invariants[2]),
      mkres(masses[2]), m2q(mkres*mkres), Q2(sjk + 2*m2q), mu2(m2q/Q2),
      z((sAK+saj-Q2)/sAK), Pz(dglapPtr->Pg2qq(z, 9, 9, 9, mu2));
    return Pz/Q2;}

};

//==========================================================================

// Class AntXGsplitRF, resonance-final antenna function.

class AntXGsplitRFsec : public AntXGsplitIFsec {

public:

  // Names (remember to redefine both for each inherited class)
  string vinciaName() const override {return "Vincia:XGsplitRF";}

  // Mark that this function does not have a zA collinear limit.
  double zA(vector<double>) override {return -1;}

  // Return this is a resonance-final antenna.
  bool isRFant() override {return true;}

  // Parton types AB -> ijk with A,i initial and B,k,j final.
  int idA() const override {return 6;}
  int idB() const override {return 21;}
  int id1() const override {return -2;}
  int ida() const {return 6;}
  int idb() const {return 2;}

  // Test masses (top, gluon, gluon, X).
  void getTestMasses(vector<double> &masses) override {masses =
      {particleDataPtr->m0(6), 0.0, 0.0, 0.6*particleDataPtr->m0(6)};}

  // AP with dummy helicities.
  double AltarelliParisi(vector<double> invariants, vector<double> masses,
    vector<int>, vector<int>) override {
    double sAK(invariants[0]), saj(invariants[1]), sjk(invariants[2]),
      mkres(masses[2]), m2q(mkres*mkres), Q2(sjk + 2*m2q), mu2(m2q/Q2),
      z((sAK+saj-Q2)/sAK), Pz(dglapPtr->Pg2qq(z, 9, 9, 9, mu2));
    return Pz/Q2;}

};


//==========================================================================

// The AntennaSetFSR class. Simple container of FF and RF antenna functions.

class AntennaSetFSR {

public:

  // Default constructor.
  AntennaSetFSR() = default;

  // Destructor, delete the antennae.
  virtual ~AntennaSetFSR() {
    for (auto it = antFunPtrs.begin(); it != antFunPtrs.end(); ++it)
      delete it->second;
    antFunPtrs.clear();}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn);

  // Initialize antenna set (optionally with min or max variation).
  void init();

  // Function to chek if an antenna with the given index exists.
  bool exists(enum AntFunType antFunType) {
    return antFunPtrs.count(antFunType);}

  // Gets an antenna from the AntennaSet.
  AntennaFunction* getAntFunPtr(enum AntFunType antFunType) {
    return (exists(antFunType)) ? antFunPtrs[antFunType] : nullptr;}

  // Get list of all AntFunTypes contained in this set.
  vector<enum AntFunType> getAntFunTypes();

  // Get Vincia name, e.g. "Vincia:QQemitFF".
  string vinciaName(enum AntFunType antFunType) {
    return exists(antFunType) ? antFunPtrs[antFunType]->vinciaName() :
      "noVinciaName";}

  // Get human name, e.g. "g/qq".
  string humanName(enum AntFunType antFunType) {
    return exists(antFunType) ? antFunPtrs[antFunType]->humanName() :
      "noHumanName";}

private:

  // Use a map of AntennaFunction pointers, create them with new on
  // initialization.
  map<enum AntFunType, AntennaFunction*> antFunPtrs{};

  // Pointers to Pythia8 classes, needed to initialise antennae.
  bool isInitPtr{false}, isInit{false};
  Info*         infoPtr{};
  ParticleData* particleDataPtr{};
  Settings*     settingsPtr{};
  Rndm*         rndmPtr{};

  // Pointer to VINCIA DGLAP class.
  DGLAP* dglapPtr{};

  // Verbosity level
  int verbose{};

};

//==========================================================================

// The AntennaSetISR class. Simple container of II and IF antenna functions.

class AntennaSetISR {

 public:

  // Default constructor.
  AntennaSetISR() = default;

  // Destructor, delete the antennae.
  ~AntennaSetISR() {
    for (auto it = antFunPtrs.begin(); it != antFunPtrs.end(); ++it)
      delete it->second;
    antFunPtrs.clear();}

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, DGLAP* dglapPtrIn);

  // Initialize antenna set.
  void init();

  // Function to chek if an antenna with the given index exists.
  bool exists(enum AntFunType antFunType) {
    return antFunPtrs.count(antFunType);}

  // Gets an antenna from the AntennaSetISR.
  AntennaFunctionIX* getAntFunPtr(enum AntFunType antFunType) {
    return (exists(antFunType)) ? antFunPtrs[antFunType] : nullptr;}

  // Get list of all AntFunTypes contained in this set.
  vector<enum AntFunType> getAntFunTypes();

  // Get Vincia name, e.g. "Vincia:QQemitII".
  string vinciaName(enum AntFunType antFunType) {
    return exists(antFunType) ? antFunPtrs[antFunType]->vinciaName()
      : "noVinciaName";}

  // Get human name, e.g. "g/qq".
  string humanName(enum AntFunType antFunType) {
    return exists(antFunType) ? antFunPtrs[antFunType]->humanName() :
      "noHumanName";}

private:

  // Use a map of AntennaFunction pointers, create them with new on
  // initialization.
  map<enum AntFunType, AntennaFunctionIX*> antFunPtrs{};

  // Pointers to Pythia 8 classes, needed to initialise antennae.
  bool isInitPtr{false}, isInit{false};
  Info*         infoPtr{};
  ParticleData* particleDataPtr{};
  Settings*     settingsPtr{};
  Rndm*         rndmPtr{};

  // Pointer to VINCIA DGLAP class
  DGLAP* dglapPtr{};

  // Verbosity level
  int verbose{};

};

//==========================================================================

// Class MECs, for computing matrix-element corrections to antenna
// functions.

class MECs {

public:

  // Constructor.
  MECs() {isInitPtr = false; isInit = false;}

  // Destructor.
  virtual ~MECs() {};

  // Initialize pointers.
  void initPtr(Info* infoPtrIn, ShowerMEs* mg5mesPtrIn,
    VinciaCommon* vinComPtrIn, Resolution* resPtrIn);

  // Initialize pointers to antenna sets.
  void initAntPtr(AntennaSetFSR* antFSRusr, AntennaSetISR* antISRusr) {
    antSetFSR = antFSRusr; antSetISR = antISRusr;}

  // Initialize.
  void init();

  // Function to return ME class (Born type) for a parton
  // configuration. Can be called from either of the ISR::prepare() or
  // FSR::prepare() functions, or from the ISR::branch() or
  // FSR::branch() functions. Returns >= 0 if there an ME for this
  // configuration, associated with the (arbitrary) integer code label
  // denoted by the return value. If return < 0 we don't have an ME /
  // no ME should be used for this system.
  bool prepare(const int iSys, Event& event);

  // Function to assign helicities to particles (using MEs).
  bool polarise(const int iSys, Event& event, const bool force = false);
  bool polarise(vector<Particle>& state, const bool force = false);

  // Check if state already has helicities.
  // checkAll = true : only return true if all particles have helicities.
  // checkAll = false : return true if *any* particle is polarised.
  bool isPolarised(int iSys, Event& event, bool checkAll = true);

  // Wrapper function to return a specific antenna function.
  AntennaFunction* getAntFunPtrFSR(const enum AntFunType antFunType) {
    return antSetFSR->getAntFunPtr(antFunType); }
  AntennaFunctionIX* getAntFunPtrISR(const enum AntFunType antFunType) {
    return antSetISR->getAntFunPtr(antFunType); }

  // Function to determine if MECs are requested at this order for this system.
  bool doMEC(const int iSys, const int nBranch);

  // Check whether we have a matrix element for this configuration.
  bool meAvailable(int iSys, const Event& event);
  bool meAvailable(const vector<Particle>& state);

  // Get squared matrix element.
  double getME2(const vector<Particle>& state, int nIn);
  double getME2(int iSys, const Event& event);

  // Get matrix element correction factor for sector shower.
  double getMECSector(int iSys, const vector<Particle>& stateNow,
    const vector<Particle>& statePost, VinciaClustering& clus);
  //TODO: Matrix element corrections for global shower?
  //   double getMECGlobal(const vector<Particle>& statePost,
  //     const VinciaClustering& clus, int nIn);

  // Communicate that the trial was accepted and we branched.
  void hasBranched(int iSys);

  // Set whether we need to calculate a new matrix element for
  // current configuration (e.g. due to an EW decay).
  void needsNewME2(int iSys, bool needsNewIn) {
    hasME2now[iSys] = !needsNewIn; }

  // Return number of partons added since Born (as defined by prepare).
  int sizeOutBorn(const int iSys) {return sizeOutBornSav[iSys];}

  // Function to set level of verbose output.
  void setVerbose(const int verboseIn) {verbose = verboseIn;}

  // Header.
  void header();

  // Is initalised?
  bool isInitialised() { return isInit; }

private:

  // Save hard scale in current system.
  bool saveHardScale(int iSys, Event& /*event*/);

  // Regularise this correction?
  bool doRegMatch(int iSys, const vector<Particle>& state);

  // Get matching regulator.
  double getMatchReg(int iSys, const VinciaClustering& clus);

  // Get antenna approximation.
  double getAntApprox(const VinciaClustering& clus);

  // Get colour weight.
  double getColWeight(const vector<Particle>& state);

  // Verbosity level.
  int verbose;

  // Is initialised.
  bool isInitPtr, isInit;

  // Pointers to PYTHIA objects.
  Info*          infoPtr;
  Rndm*          rndmPtr;
  ParticleData*  particleDataPtr;
  PartonSystems* partonSystemsPtr;
  Settings*      settingsPtr;
  ShowerMEs*     mg5mesPtr;

  // Pointers to VINCIA objects.
  Resolution*    resolutionPtr;
  VinciaCommon*  vinComPtr;

  // Antenna sets.
  AntennaSetFSR* antSetFSR;
  AntennaSetISR* antSetISR;

  // Matching settings.
  bool matchingFullColour, matchingScaleIsAbs;
  int  modeMECs;
  int  matchingRegOrder, matchingRegShape;
  int  maxMECs2to1, maxMECs2to2, maxMECs2toN, maxMECsResDec, maxMECsMPI;
  int  nFlavZeroMass;
  double matchingIRcutoff, matchingScale, q2Match;

  // Map from iSys to Born multiplicity and ME class; set in prepare().
  map<int,int> sizeOutBornSav;

  // Map from iSys to number of QCD particles in Born; set in prepare().
  map<int,int> sysToBornMultQCD;

  // Map from iSys to hard scale; set in prepare();
  map<int,double> sysToHardScale;

  // Map from iSys to matrix element of current state.
  map<int,double> me2now;
  map<int,bool>   hasME2now;

  // Maps from iSys to matrix element of post-branching state.
  map<int,double> me2post;
  map<int,bool>   hasME2post;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_VinciaAntennaFunctions_H
