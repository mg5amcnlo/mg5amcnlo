// DireSplitInfo.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Stefan Prestel, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for splitting information storage containers.
// DireSplitParticle:   Quantum numbers and other information of one of the
//                      particles participating in current branching.
// DireSplitKinematics: Kinematical variables from which to construct a
//                      branching.
// DireSplitInfo:       Particles participating in the current branching.

#ifndef Pythia8_DireSplitInfo_H
#define Pythia8_DireSplitInfo_H

#define DIRE_SPLITINFO_VERSION "2.002"

#include "Pythia8/Event.h"
#include "Pythia8/PartonSystems.h"
#include "Pythia8/DireBasics.h"

namespace Pythia8 {

//==========================================================================

// Definition of color chains.

class DireSingleColChain {

public:

  DireSingleColChain() {}
  DireSingleColChain(const DireSingleColChain& chainIn) : chain(chainIn.chain),
    original_chain(chainIn.original_chain) {}
  DireSingleColChain & operator=(const DireSingleColChain& c) { if (this != &c)
    { chain = c.chain;  original_chain = c.original_chain;} return *this; }
  DireSingleColChain(int iPos, const Event& state,
                     PartonSystems* partonSysPtr);

  void addToChain(const int iPos, const Event& state);

  int  iPosEnd()    { return chain.back().first; }
  int  colEnd ()    { return chain.back().second.first; }
  int  acolEnd()    { return chain.back().second.second; }
  int  size() const { return chain.size(); }
  bool isInChain  ( int iPos);
  int  posInChain ( int iPos);
  bool colInChain ( int col);

  DireSingleColChain chainFromCol(int iPos, int col, int nSteps,
    const Event& state);

  string listPos() const;

  // List functions by N. Fischer.
  void print() const;
  void list () const;
  string list2 () const;

  void clear() { chain.resize(0); original_chain.resize(0); }

  vector<pair<int,pair<int,int> > > chain;
  vector<pair<int,pair<int,int> > > original_chain;

  // Overload index operator to access element of event record.
  pair<int,pair<int,int> >& operator[](int i) {return chain[i];}
  const pair<int,pair<int,int> >& operator[](int i) const {return chain[i];}

};

//==========================================================================

// Container for multiple color chains.

class DireColChains {

public:

  DireColChains() {}

  void addChain(DireSingleColChain chain) { chains.push_back( chain); }

  int size() const { return chains.size(); }

  DireSingleColChain chainOf (int iPos);

  DireSingleColChain chainFromCol (int iPos, int col, int nSteps,
    const Event& state);

  int check(int iSys, const Event& state, PartonSystems* partonSysPtr);

  void list();

  vector<DireSingleColChain> chains;

};

//==========================================================================

class DireSplitParticle {

public:

  DireSplitParticle() : id(0), col(-1), acol(-1), charge(0), spin(-9), m2(-1.),
    isFinal(false) {}
  DireSplitParticle( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn) : id(idIn), col(colIn),
    acol(acolIn), charge(chargeIn), spin(spinIn), m2(m2In), isFinal(isFinalIn)
  {}
  DireSplitParticle ( const Particle& in) : id(in.id()),
    col(in.col()), acol(in.acol()), charge(in.charge()), spin(in.pol()),
    m2(pow2(in.m())), isFinal(in.isFinal()) {}

  void store( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn) { id = idIn; col = colIn;
    acol = acolIn; charge = chargeIn; spin = spinIn; m2 = m2In;
    isFinal = isFinalIn; }
  void store( const Particle in) { id = in.id(); col = in.col();
    acol = in.acol(); charge = in.charge(); spin = in.pol();
    m2 = pow2(in.m()); isFinal = in.isFinal(); }

  void clear() { col = acol = -1; id = charge = 0; spin = -9; m2 = -1.;
    isFinal = false; }

  // Quantum numbers.
  int id, col, acol, charge, spin;

  // Mass.
  double m2;

  // Incoming/final
  bool isFinal;

};

//==========================================================================

class DireSplitKinematics {

public:

  DireSplitKinematics() : m2Dip(-1.), pT2(-1.), pT2Old(-1.), z(-1.), phi(-9.),
    sai(0.), xa(-1), phi2(-9.), m2RadBef(-1.), m2Rec(-1.), m2RadAft(-1.),
    m2EmtAft(-1.), m2EmtAft2(-1.), xBef(-1.), xAft(-1.) {}

  DireSplitKinematics( double m2DipIn, double pT2In, double zIn, double phiIn,
    double saiIn, double xaIn, double phi2In, double m2RadBefIn,
    double m2RecIn, double m2RadAftIn, double m2EmtAftIn, double m2EmtAft2In,
    double xBefIn, double xAftIn )
    : m2Dip(m2DipIn), pT2(pT2In), pT2Old(-1.), z(zIn), phi(phiIn),
    sai(saiIn), xa(xaIn), phi2(phi2In), m2RadBef(m2RadBefIn), m2Rec(m2RecIn),
    m2RadAft(m2RadAftIn), m2EmtAft(m2EmtAftIn), m2EmtAft2(m2EmtAft2In),
      xBef(xBefIn), xAft(xAftIn) {}

  void store2to3kine( double m2DipIn, double pT2In, double zIn, double phiIn,
    double xAftIn = -1.) {
    m2Dip = m2DipIn; pT2 = pT2In; z = zIn; phi = phiIn; xAft = xAftIn; }
  void store2to3mass( double m2RadBefIn, double m2RecIn, double m2RadAftIn,
    double m2EmtAftIn) { m2RadBef = m2RadBefIn; m2Rec = m2RecIn;
    m2RadAft = m2RadAftIn; m2EmtAft = m2EmtAftIn; }

  void store2to4kine( double m2DipIn, double pT2In, double zIn, double phiIn,
    double saiIn, double xaIn, double phi2In, double xAftIn = -1.) {
    m2Dip = m2DipIn;
    pT2 = pT2In; z = zIn; phi = phiIn; sai = saiIn; xa = xaIn; phi2 = phi2In;
    xAft = xAftIn;}
  void store2to4mass( double m2RadBefIn, double m2RecIn, double m2RadAftIn,
    double m2EmtAftIn, double m2EmtAft2In) {
    m2RadBef = m2RadBefIn; m2Rec = m2RecIn; m2RadAft = m2RadAftIn;
    m2EmtAft = m2EmtAftIn; m2EmtAft2 = m2EmtAft2In; }

  void set_m2Dip     ( double in) {m2Dip=(in);}
  void set_pT2       ( double in) {pT2=(in);}
  void set_pT2Old    ( double in) {pT2Old=(in);}
  void set_z         ( double in) {z=(in);}
  void set_phi       ( double in) {phi=(in);}
  void set_sai       ( double in) {sai=(in);}
  void set_xa        ( double in) {xa=(in);}
  void set_phi2      ( double in) {phi2=(in);}
  void set_m2RadBef  ( double in) {m2RadBef=(in);}
  void set_m2Rec     ( double in) {m2Rec=(in);}
  void set_m2RadAft  ( double in) {m2RadAft=(in);}
  void set_m2EmtAft  ( double in) {m2EmtAft=(in);}
  void set_m2EmtAft2 ( double in) {m2EmtAft2=(in);}
  void set_xBef      ( double in) {xBef=(in);}
  void set_xAft      ( double in) {xAft=(in);}

  void clear() { m2Dip = pT2 = pT2Old = z = xa = m2RadBef = m2Rec = m2RadAft
     = m2EmtAft = m2EmtAft2 = xBef = xAft = -1.; sai = 0.; phi = phi2 = -9.; }

  void store ( const DireSplitKinematics& k);

  void list();

  unordered_map<string,double> getKinInfo() {
    return create_unordered_map<string,double> ("m2Dip",m2Dip)("pT2",pT2)
      ("pT2Old",pT2Old)
      ("z",z)("phi",phi)("sai",sai)("xa",xa)("phi2",phi2)("m2RadBef",m2RadBef)
      ("m2Rec",m2Rec)("m2RadAft",m2RadAft)("m2EmtAft",m2EmtAft)
      ("m2EmtAft2",m2EmtAft2)("xBef",xBef) ("xAft",xAft);
  }

  // Kinematic variable to enable branching.
  double m2Dip, pT2, pT2Old, z, phi, sai, xa, phi2,
         m2RadBef, m2Rec, m2RadAft, m2EmtAft, m2EmtAft2;
  double xBef, xAft;

};

//==========================================================================

class DireSplitInfo {

public:

  DireSplitInfo() : iRadBef(0), iRecBef(0), iRadAft(0), iRecAft(0), iEmtAft(0),
    iEmtAft2(0), side(0), type(0), system(0), systemRec(0),
    splittingSelName(""), useForBranching(false), terminateEvolution(false),
    iRadBefStore(-1), iRecBefStore(-1), iRadAftStore(-1), iRecAftStore(-1),
    iEmtAftStore(-1), iEmtAft2Store(-1), sideStore(-1), typeStore(-1),
    systemStore(-1), systemRecStore(-1), splittingSelNameStore(""),
    useForBranchingStore(false), terminateEvolutionStore(false) {
    init(); }
  DireSplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn,
    int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    double m2DipIn = -1., double pT2In = -1., double zIn = -1.,
    double phiIn = -9., double saiIn = 0., double xaIn = -1.,
    double phi2In = -9., double m2RadBefIn = -1., double m2RecIn = -1.,
    double m2RadAftIn = -1., double m2EmtAftIn = -1., double m2EmtAft2In = -1.,
    double xBefIn = -1., double xAftIn = -1.,
    int sideIn = 0, int typeIn = 0, int systemIn = 0, int systemRecIn = 0,
    string splittingSelNameIn = "", bool useForBranchingIn = false,
    DireSingleColChain iSiblingsIn = DireSingleColChain() ) :
      iRadBef(iRadBefIn), iRecBef(iRecBefIn),
      iRadAft(iRadAftIn), iRecAft(iRecAftIn), iEmtAft(iEmtAftIn),
      kinSave(m2DipIn, pT2In, zIn, phiIn, saiIn, xaIn, phi2In, m2RadBefIn,
        m2RecIn, m2RadAftIn, m2EmtAftIn, m2EmtAft2In, xBefIn, xAftIn),
      side(sideIn), type(typeIn), system(systemIn), systemRec(systemRecIn),
      splittingSelName(splittingSelNameIn), useForBranching(useForBranchingIn),
      terminateEvolution(false), iSiblings(iSiblingsIn)
      { init(state); }

  DireSplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn,
    string splittingSelNameIn) : iRadBef(iRadBefIn), iRecBef(iRecBefIn),
    splittingSelName(splittingSelNameIn),
    iRadBefStore(-1), iRecBefStore(-1), iRadAftStore(-1), iRecAftStore(-1),
    iEmtAftStore(-1), iEmtAft2Store(-1), sideStore(-1), typeStore(-1),
    systemStore(-1), systemRecStore(-1), splittingSelNameStore(""),
    useForBranchingStore(false), terminateEvolutionStore(false) {
    iRadAft = iRecAft = iEmtAft = side = type = system = systemRec = 0;
    useForBranching = terminateEvolution = false; init(state); }

  DireSplitInfo ( const Event& state, int iRadAftIn, int iRecAftIn,
    int iEmtAftIn, string splittingSelNameIn) :
    iRadAft(iRadAftIn), iRecAft(iRecAftIn),
    iEmtAft(iEmtAftIn), splittingSelName(splittingSelNameIn),
    iRadBefStore(-1), iRecBefStore(-1), iRadAftStore(-1), iRecAftStore(-1),
    iEmtAftStore(-1), iEmtAft2Store(-1), sideStore(-1), typeStore(-1),
    systemStore(-1), systemRecStore(-1), splittingSelNameStore(""),
    useForBranchingStore(false), terminateEvolutionStore(false) {
    splittingSelName = ""; iRadBef = iRecBef = side = type = system
      = systemRec = 0;
    useForBranching = terminateEvolution = false; init(state); }

  DireSplitInfo ( const DireSplitInfo& s) : iRadBefStore(-1), iRecBefStore(-1),
    iRadAftStore(-1), iRecAftStore(-1), iEmtAftStore(-1), iEmtAft2Store(-1),
    sideStore(-1), typeStore(-1), systemStore(-1), systemRecStore(-1),
    particleSaveStore(), kinSaveStore(), splittingSelNameStore(),
    extrasStore(), useForBranchingStore(false), terminateEvolutionStore(false),
    iSiblingsStore() {
    iRadBef = s.iRadBef;
    iRecBef = s.iRecBef;
    iRadAft = s.iRadAft;
    iRecAft = s.iRecAft;
    iEmtAft = s.iEmtAft;
    iEmtAft2 = s.iEmtAft2;
    for (int i=0; i < int(s.particleSave.size()); ++i)
      particleSave.push_back(s.particleSave[i]);
    kinSave.store(s.kinSave);
    side = s.side;
    type = s.type;
    system = s.system;
    systemRec = s.systemRec;
    splittingSelName = s.splittingSelName;
    for ( unordered_map<string,double>::const_iterator it = s.extras.begin();
      it != s.extras.end(); ++it )
      extras.insert(make_pair(it->first,it->second));
    useForBranching = s.useForBranching;
    terminateEvolution = s.terminateEvolution;
    iSiblings       = s.iSiblings;
  }

  void init(const Event& state = Event() );

  void store (const DireSplitInfo& s);

  void save ();

  void restore ();

  const DireSplitParticle* radBef() const { return &particleSave[0]; }
  const DireSplitParticle* recBef() const { return &particleSave[1]; }
  const DireSplitParticle* radAft() const { return &particleSave[2]; }
  const DireSplitParticle* recAft() const { return &particleSave[3]; }
  const DireSplitParticle* emtAft() const { return &particleSave[4]; }
  const DireSplitParticle* emtAft2() const { return &particleSave[5]; }

  const DireSplitKinematics* kinematics() const { return &kinSave; }

  void set2to3kin( double m2DipIn, double pT2In, double zIn, double phiIn,
    double m2RadBefIn, double m2RecIn, double m2RadAftIn, double m2EmtAftIn) {
    kinSave.store2to3kine(m2DipIn, pT2In, zIn, phiIn);
    kinSave.store2to3mass(m2RadBefIn, m2RecIn, m2RadAftIn, m2EmtAftIn); }

  void set2to4kin( double m2DipIn, double pT2In, double zIn, double phiIn,
    double saiIn, double xaIn, double phi2In, double m2RadBefIn,
    double m2RecIn, double m2RadAftIn, double m2EmtAftIn, double m2EmtAft2In) {
    kinSave.store2to4kine(m2DipIn, pT2In, zIn, phiIn, saiIn, xaIn, phi2In);
    kinSave.store2to4mass(m2RadBefIn, m2RecIn, m2RadAftIn, m2EmtAftIn,
      m2EmtAft2In); }

  void set_m2Dip     ( double in) {kinSave.set_m2Dip(in);}
  void set_pT2       ( double in) {kinSave.set_pT2(in);}
  void set_pT2Old    ( double in) {kinSave.set_pT2Old(in);}
  void set_z         ( double in) {kinSave.set_z(in);}
  void set_phi       ( double in) {kinSave.set_phi(in);}
  void set_sai       ( double in) {kinSave.set_sai(in);}
  void set_xa        ( double in) {kinSave.set_xa(in);}
  void set_phi2      ( double in) {kinSave.set_phi2(in);}
  void set_m2RadBef  ( double in) {kinSave.set_m2RadBef(in);}
  void set_m2Rec     ( double in) {kinSave.set_m2Rec(in);}
  void set_m2RadAft  ( double in) {kinSave.set_m2RadAft(in);}
  void set_m2EmtAft  ( double in) {kinSave.set_m2EmtAft(in);}
  void set_m2EmtAft2 ( double in) {kinSave.set_m2EmtAft2(in);}
  void set_xBef      ( double in) {kinSave.set_xBef(in);}
  void set_xAft      ( double in) {kinSave.set_xAft(in);}

  void storeRadBef(const Particle& in)
    { particleSave[0].store(in); }
  void storeRecBef(const Particle& in)
    { particleSave[1].store(in); }
  void storeRadAft(const Particle& in)
    { particleSave[2].store(in); }
  void storeRecAft(const Particle& in)
    { particleSave[3].store(in); }
  void storeEmtAft(const Particle& in)
    { particleSave[4].store(in); }
  void storeEmtAft2(const Particle& in)
    { particleSave[5].store(in); }

  void setRadBef( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(0, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRecBef( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(1, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRadAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(2, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRecAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(3, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setEmtAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(4, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setEmtAft2( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { setParticle(5, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void clearRadBef()  { setParticle(0, 0, -1, -1, 0, -9, -1.0, false); }
  void clearRecBef()  { setParticle(1, 0, -1, -1, 0, -9, -1.0, false); }
  void clearRadAft()  { setParticle(2, 0, -1, -1, 0, -9, -1.0, false); }
  void clearRecAft()  { setParticle(3, 0, -1, -1, 0, -9, -1.0, false); }
  void clearEmtAft()  { setParticle(4, 0, -1, -1, 0, -9, -1.0, false); }
  void clearEmtAft2() { setParticle(5, 0, -1, -1, 0, -9, -1.0, false); }
  void setParticle( int iPos, int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false) { particleSave[iPos].store(
    idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }

  void storeName (string name) { splittingSelName = name; }
  void storeType ( int in) { type = in; }
  void storeSystem ( int in) { system = in; }
  void storeSystemRec ( int in) { systemRec = in; }
  void storeSide ( int in) { side = in; }
  void storeExtras ( unordered_map<string,double> in) { extras = in; }
  void storeRadRecBefPos (int rad, int rec) { iRadBef = rad; iRecBef = rec; }
  void canUseForBranching (bool in) { useForBranching = in;}

  void addExtra(string key, double value) {
    unordered_map<string, double>::iterator it = extras.find(key);
    if (it == extras.end()) extras.insert(make_pair(key,value));
    else                    it->second = value;
  }

  void storeInfo(string name, int typeIn, int systemIn, int systemRecIn,
    int sideIn, int iPosRadBef, int iPosRecBef,
    const Event& state, int idEmtAft,
    int idRadAft, int nEmissions, double m2Dip, double pT2, double pT2Old,
    double z, double phi, double m2Bef, double m2s, double m2r, double m2i,
    double sa1, double xa, double phia1, double m2j, double xBef, double xAft);

  unordered_map<string,double> getKinInfo() { return kinSave.getKinInfo();}

  void storePosAfter( int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    int iEmtAft2In);

  void clear();

  void list();

  // Information to enable branching.
  int iRadBef, iRecBef;

  // Information to enable clustering.
  int    iRadAft, iRecAft, iEmtAft, iEmtAft2;

  vector<DireSplitParticle> particleSave;
  DireSplitKinematics kinSave;

  // Auxiliary information.
  int side, type, system, systemRec;
  string splittingSelName;
  unordered_map<string,double> extras;

  bool useForBranching, terminateEvolution;

  // Information to enable branching.
  int iRadBefStore, iRecBefStore, iRadAftStore, iRecAftStore, iEmtAftStore,
      iEmtAft2Store, sideStore, typeStore, systemStore, systemRecStore;
  vector<DireSplitParticle> particleSaveStore;
  DireSplitKinematics kinSaveStore;
  string splittingSelNameStore;
  unordered_map<string,double> extrasStore;
  bool useForBranchingStore, terminateEvolutionStore;

  DireSingleColChain iSiblings, iSiblingsStore;
  void setSiblings(DireSingleColChain s) { clearSiblings(); iSiblings = s; }
  void clearSiblings() { iSiblings.clear(); }

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_DireSplitInfo_H
