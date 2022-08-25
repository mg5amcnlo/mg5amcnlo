// VinciaTrialGenerators.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

#ifndef Pythia8_VinciaTrial_H
#define Pythia8_VinciaTrial_H

// Pythia headers.
#include "Pythia8/Basics.h"
#include "Pythia8/PythiaStdlib.h"

// Vincia headers.
#include "Pythia8/VinciaCommon.h"

namespace Pythia8 {

// Helpful enums.
enum class TrialGenType { Void = 0, FF = 1, RF = 2, IF = 3, II = 4 };
// (Default is used for soft, global, or splittings as appropriate.)
enum class BranchType { Void = -1, Emit = 0, SplitF = 1, SplitI = 2,
  Conv = 3 };
enum class Sector { Void = -99, ColI = -1, Default = 0, ColK = 1 };

// Forward declarations.
class ZetaGenerator;
class ZetaGeneratorSet;

//==========================================================================

// Helper struct for passing trial-alphaS information.

struct EvolutionWindow {

  int runMode{};
  double alphaSmax{}, b0{}, kMu2{}, lambda2{}, qMin{};
  map<int, double> mass;

};

//==========================================================================

// Base class for trial generators.

class TrialGenerator {

 public:

  // Main constructor.
  TrialGenerator(bool isSectorIn, TrialGenType trialGenTypeIn,
    BranchType branchTypeIn, ZetaGeneratorSet& zetaGenSet)
    : isSector(isSectorIn), trialGenTypeSav(trialGenTypeIn),
        branchType(branchTypeIn) { setupZetaGens(zetaGenSet); }

  // Destructor.
    virtual ~TrialGenerator() = default;

  // Set pointers to zetaGenerators.
  void setupZetaGens(ZetaGeneratorSet& zetaGenSet);

  // Re-calculate the current zeta limits and integrals.
  virtual void reset(double Q2min, double s, const vector<double> & masses,
    enum AntFunType antFunType, double xA = 1., double xB = 1.);

  // Generate the next trial scale.
  virtual double genQ2(double Q2MaxNow, Rndm* rndmPtr,
    const EvolutionWindow* evWindowPtrIn, double colFac,
    double wtIn, Info* infoPtr, int verboseIn);

  // Get the invariants.
  virtual bool genInvariants(double sAnt, const vector<double>& masses,
    vector<double>& invariants, Rndm* rndmPtr, Info* infoPtr, int verboseIn);

  // Calculate the trial based on invariants and saved quantities.
  virtual double aTrial(vector<double>& invariants,
    const vector<double>& masses, int verboseIn);

  // Calculate the colour and coupling stripped antenna function.
  virtual double aTrialStrip(vector<double>& invariants,
    const vector<double>& masses, int verboseIn);

  // Delete the current trial.
  virtual void resetTrial();

  // Mark trial as used.
  virtual void needsNewTrial();

  // Return the sector.
  int getSector() {return (int)sectorSav;}

 protected:

  // Calculate the Kallen factor.
  virtual void calcKallenFac(double, const vector<double>&) {
    kallenFacSav = 1.0;}

  // Calculate the PDF ratio.
  virtual void calcRpdf(const vector<double>&) {Rpdf = 1.0;}

  void addGenerator(ZetaGeneratorSet& zetaGenSet,
    Sector sector = Sector::Default);

  // True when init succeeds.
  bool isInit{false};

  // Information set at construction.
  const bool isSector;
  const TrialGenType trialGenTypeSav;
  const BranchType branchType;

  // Common prefactors to the trial integral.
  double kallenFacSav{1.};
  double Rpdf{1.};

  // Information about the antenna.
  double sAntSav{};
  vector<double> massesSav;

  // Information about the trial.
  bool hasTrial{false};
  double q2Sav{}, colFacSav{};
  const EvolutionWindow* evWindowSav{};
  Sector sectorSav;

  // Map from sector to the correct zeta generator.
  // (note these live inside a ZetaGeneratorSet)
  map<Sector, ZetaGenerator*> zetaGenPtrs;

  // Map from sector to the corresponding zeta phase-space limits.
  map<Sector, pair<double, double>> zetaLimits;

  // Save the zeta integrals.
  map<Sector, double> IzSav;

  // Save which sectors are currently active.
  map<Sector, bool> isActiveSector;

};

//==========================================================================

// Trial generator for final-final branchings.

class TrialGeneratorFF : public TrialGenerator {

 public:

  // Default constructor.
  TrialGeneratorFF(bool isSectorIn, BranchType branchTypeIn,
    ZetaGeneratorSet& zetaGenSet) : TrialGenerator(isSectorIn,
      TrialGenType::FF, branchTypeIn, zetaGenSet) {;}

 private:

  void calcKallenFac(double sIK, const vector<double>& masses);

};

//==========================================================================

// Trial generator for resonance-final branchings.

class TrialGeneratorRF : public TrialGenerator{

 public:

  // Default constructor.
  TrialGeneratorRF(bool isSectorIn, BranchType branchTypeIn,
    ZetaGeneratorSet& zetaGenSet) : TrialGenerator(isSectorIn,
      TrialGenType::RF, branchTypeIn, zetaGenSet) {;}

 private:

  void calcKallenFac(double sAK, const vector<double>& masses);

};

//==========================================================================

// Trial generator for initial-final branchings.

class TrialGeneratorIF : public TrialGenerator {

 public:

  // Default constructor.
  TrialGeneratorIF(bool isSectorIn, BranchType branchTypeIn,
    ZetaGeneratorSet& zetaGenSet) : TrialGenerator(isSectorIn,
      TrialGenType::IF, branchTypeIn, zetaGenSet) {;}

};

//==========================================================================

// Trial generator for initial-initial branchings.

class TrialGeneratorII : public TrialGenerator {

 public:

  // Default constructor.
  TrialGeneratorII(bool isSectorIn, BranchType branchTypeIn,
    ZetaGeneratorSet& zetaGenSet) : TrialGenerator(isSectorIn,
      TrialGenType::II, branchTypeIn, zetaGenSet) {;}

};

//==========================================================================

// Place to store all types of zeta trial generators.
// To live in VinicaFSR, VinciaISR.

class ZetaGeneratorSet {

 public:

  // Construct all zeta generators for a given type.
  ZetaGeneratorSet(TrialGenType trialGenTypeIn);

  // Clear list.
  ~ZetaGeneratorSet();

  // Get ptr to ZetaGenerator for a sector.
  ZetaGenerator* getZetaGenPtr(BranchType branchType, Sector sectIn);

  TrialGenType getTrialGenType() {return trialGenTypeSav;}

 protected :

  const TrialGenType trialGenTypeSav;

  void addGenerator(ZetaGenerator* zGenPtr);

  map<pair<BranchType, Sector>, ZetaGenerator*> zetaGenPtrs;

};

//==========================================================================

// Base class for zeta trial generators.

class ZetaGenerator {

 public:

  // Constructor and destructor.
  ZetaGenerator(TrialGenType trialGenTypeIn, BranchType branchTypeIn,
    Sector sectorIn, double globalIn) : trialGenType(trialGenTypeIn),
    branchType(branchTypeIn), sector(sectorIn), globalFactSav(globalIn) {;}
  virtual ~ZetaGenerator() = default;

  // Get (best/physical) limits given a set of input parameters.
  virtual double getzMin(double Q2min,double sAnt,
    const vector<double>& masses, double xA = 1., double xB = 1.) = 0;
  virtual double getzMax(double Q2min,double sAnt,
    const vector<double>& masses, double xA = 1., double xB = 1.) = 0;

  // Get hull of physical phase space in zeta.
  virtual double getzMinHull(double Q2min,double sAnt,
    const vector<double>& masses, double xA = 1., double xB = 1.) {
    return getzMin(Q2min, sAnt, masses, xA, xB);}
  virtual double getzMaxHull(double Q2min,double sAnt,
    const vector<double>& masses, double xA = 1., double xB = 1.) {
    return getzMax(Q2min, sAnt, masses, xA, xB);}

  // Get constant factor for zeta integral.
  // NOTE: only used in II conversion trial.
  virtual double getConstFactor(double,
    const vector<double>&) {return 1.;}

  // Set the invariants for the current value of the evolution variables.
  virtual void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) = 0;

  // Evaluate the trial antenna given invariants and masses.
  virtual double aTrial(const vector<double>& invariants,
    const vector<double>& masses) = 0;

  // Check if this trial is active for specific AntFunType.
  virtual bool isActive(enum AntFunType) {return false;}

  // Return information about this generator.
  TrialGenType getTrialGenType() {return trialGenType;}
  Sector getSector() {return sector;}
  BranchType getBranchType() {return branchType;}

  // Return multiplier to convert to global.
  double globalMultiplier() {return globalFactSav;}

  // The zeta integral.
  // Optionally with exponent gamma for PDF overestimate.
  double getIz(double zMinIn, double zMaxIn, double gammaPDF = 1.) {
    return zetaIntSingleLim(zMaxIn, gammaPDF)
      -zetaIntSingleLim(zMinIn, gammaPDF);}

  // Generate a value of zeta.
  double genZeta(Rndm* rndmPtr, double zMinIn, double zMaxIn,
    double gammaPDF = 1.);

  // Print the trial generator.
  void print();

 protected:

  // The functional form of the zeta integral.
  // Optionally with exponent gamma for PDF overestimate.
  virtual double zetaIntSingleLim(double z, double gammaPDF = 1.) = 0;

  // The function form of the inverse of the zeta integral.
  // Optionally with exponent gamma for PDF overestimate.
  virtual double inverseZetaIntegral(double Iz, double gammaPDF = 1.) = 0;

  // Check if invariants are valid.
  bool valid(const string& method, Info* infoPtr, int verbose, double zIn);
  bool valid(const string& method, Info* infoPtr, int verbose, double zIn,
    const double& Q2In);

  // Labels to define this trial generator (set in derived constructors).
  const TrialGenType trialGenType{TrialGenType::Void};
  const BranchType branchType{BranchType::Void};
  const Sector sector{Sector::Void};

  // Multiplier to convert trial to global.
  const double globalFactSav;

};
//==========================================================================

// Final-final trial generators.

//==========================================================================

// The final-final default sector generator.

class ZGenFFEmitSoft : public ZetaGenerator {

 public:

  // Constructor.
  ZGenFFEmitSoft() : ZetaGenerator(TrialGenType::FF , BranchType::Emit,
    Sector::Default, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QQemitFF || antFunType == QGemitFF ||
      antFunType == GQemitFF || antFunType == GGemitFF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The final-final ColI sector emission generator.

class ZGenFFEmitColI: public ZetaGenerator {

 public:

  // Constructor.
  ZGenFFEmitColI() : ZetaGenerator(TrialGenType::FF, BranchType::Emit,
    Sector::ColI,1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt,const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt,const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == GQemitFF || antFunType == GGemitFF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The final-final ColK sector emission generator.

class ZGenFFEmitColK : public ZetaGenerator {

 public:

  // Constructor.
  ZGenFFEmitColK() : ZetaGenerator(TrialGenType::FF, BranchType::Emit,
    Sector::ColK, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt,const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt,const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QGemitFF || antFunType == GGemitFF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The final-final default sector splitting generator.

class ZGenFFSplit : public ZetaGenerator {

 public:

  // Constructor.
  ZGenFFSplit() : ZetaGenerator(TrialGenType::FF , BranchType::SplitF,
    Sector::Default, 0.5) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == GXsplitFF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// Resonance-final trial generators.

//==========================================================================

// The resonance-final default sector generator.

class ZGenRFEmitSoft : public ZetaGenerator {

 public:

  // Constructor.
  ZGenRFEmitSoft() : ZetaGenerator(TrialGenType::RF, BranchType::Emit,
    Sector::Default, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QQemitRF || antFunType == QGemitRF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;
};

//==========================================================================

// The resonance-final default sector alternate generator.

class ZGenRFEmitSoftAlt : public ZetaGenerator {

 public:

  // Constructor.
  ZGenRFEmitSoftAlt() : ZetaGenerator(TrialGenType::RF, BranchType::Emit,
    Sector::Default, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA=1., double xB=1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA=1., double xB=1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses ) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QQemitRF || antFunType == QGemitRF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The resonance-final ColK sector generator.

class ZGenRFEmitColK : public ZetaGenerator {

 public:

  // Constructor.
  ZGenRFEmitColK() : ZetaGenerator(TrialGenType::RF, BranchType::Emit,
    Sector::ColK, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial( const vector<double>& invariants,
    const vector<double>& masses ) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QGemitRF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The resonance-final default sector splitting generator.

class ZGenRFSplit : public ZetaGenerator {

 public:

  // Constructor.
  ZGenRFSplit() : ZetaGenerator(TrialGenType::RF, BranchType::SplitF,
    Sector::Default, 0.5) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses ) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == XGsplitRF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// Intial-final trial generators.

//==========================================================================

// The initial-final default sector generator.

class ZGenIFEmitSoft : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFEmitSoft() :
    ZetaGenerator(TrialGenType::IF, BranchType::Emit, Sector::Default, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {return
      antFunType == QQemitIF || antFunType == QGemitIF ||
      antFunType == GQemitIF || antFunType == GGemitIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-final ColI sector generator.

class ZGenIFEmitColA : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFEmitColA() :
    ZetaGenerator(TrialGenType::IF, BranchType::Emit, Sector::ColI, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType  == GQemitIF || antFunType == GGemitIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-final ColK sector generator.

class ZGenIFEmitColK : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFEmitColK() : ZetaGenerator(TrialGenType::IF, BranchType::Emit,
    Sector::ColK, 1.0) {;}

  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA=1., double xB=1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA=1., double xB=1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {return
      antFunType == QGemitIF || antFunType == GGemitIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-final initial antenna splitting generator.

class ZGenIFSplitA: public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFSplitA() : ZetaGenerator(TrialGenType::IF, BranchType::SplitI,
    Sector::Default, 1.) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial( const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QXsplitIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-final final antenna splitting generator.

class ZGenIFSplitK : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFSplitK() : ZetaGenerator(TrialGenType::IF, BranchType::SplitF,
    Sector::Default, .5) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double> & invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == XGsplitIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-final splitting generator.

class ZGenIFConv : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIFConv() : ZetaGenerator(TrialGenType::IF, BranchType::Conv,
    Sector::Default, 1.) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == GXconvIF;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-initial trial generators.

//==========================================================================

// The initial-initial default sector generator.

class ZGenIIEmitSoft : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIIEmitSoft() : ZetaGenerator(TrialGenType::II, BranchType::Emit,
    Sector::Default, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QQemitII || antFunType == GQemitII ||
      antFunType == GGemitII;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-initial ColI sector generator.

class ZGenIIEmitCol : public ZetaGenerator {

 public:

  // Constructor
  ZGenIIEmitCol() : ZetaGenerator(TrialGenType::II, BranchType::Emit,
    Sector::ColI, 1.0) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses ) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType  == GQemitII || antFunType == GGemitII;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-initial initial splitting generator.

class ZGenIISplit : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIISplit() : ZetaGenerator(TrialGenType::II, BranchType::SplitI,
    Sector::Default, 1.) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == QXsplitII;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

//==========================================================================

// The initial-initial splitting generator.

class ZGenIIConv : public ZetaGenerator {

 public:

  // Constructor.
  ZGenIIConv() : ZetaGenerator(TrialGenType::II, BranchType::Conv,
    Sector::Default, 1.) {;}

  // Overridden methods.
  double getzMin(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getzMax(double Q2,double sAnt, const vector<double>& masses,
    double xA = 1., double xB = 1.) override;
  double getConstFactor(double sAnt, const vector<double>& masses) override;
  void genInvariants(double Q2In, double zIn, double sAnt,
    const vector<double>& masses, vector<double>& invariants,
    Info* infoPtr, int verboseIn) override;
  double aTrial(const vector<double>& invariants,
    const vector<double>& masses) override;
  bool isActive(enum AntFunType antFunType) override {
    return antFunType == GXconvII;}

 private:

  double zetaIntSingleLim(double z, double gammaPDF = 1.) override;
  double inverseZetaIntegral(double Iz, double gammaPDF = 1.) override;

};

} // end namespace Pythia8

#endif // end Pythia8_VinciaTrial_H
