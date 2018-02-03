// SplitInfo.h is a part of the DIRE plugin to the PYTHIA event generator.
// Copyright (C) 2018 Stefan Prestel.

// Header file for splitting information storage containers.
// SplitParticle:   Quantum numbers and other information of one of the
//                  particles participating in current branching.
// SplitKinematics: Kinematical variables from which to construct a
//                  branching.
// SplitInfo:       Particles participating in the current branching.

#ifndef Pythia8_SplitInfo_H
#define Pythia8_SplitInfo_H

#define DIRE_SPLITINFO_VERSION "2.002"

#include "Pythia8/Event.h"
#include "Dire/Basics.h"

namespace Pythia8 {

//==========================================================================

class SplitParticle {

public:

  SplitParticle() : id(0), col(-1), acol(-1), charge(0), spin(-9), m2(-1.),
    isFinal(false), isSoft(false) {}
  SplitParticle( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn, bool isSoftIn) : id(idIn), col(colIn),
    acol(acolIn), charge(chargeIn), spin(spinIn), m2(m2In), isFinal(isFinalIn),
    isSoft(isSoftIn) {} 
  SplitParticle ( const Particle& in, bool isSoftIn) : id(in.id()),
    col(in.col()), acol(in.acol()), charge(in.charge()), spin(in.pol()),
    m2(pow2(in.m())), isFinal(in.isFinal()), isSoft(isSoftIn) {}

  void store( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn, bool isSoftIn) { id = idIn; col = colIn;
    acol = acolIn; charge = chargeIn; spin = spinIn; m2 = m2In;
    isFinal = isFinalIn; isSoft = isSoftIn; }
  void store( const Particle in, bool isSoftIn) { id = in.id(); col = in.col();
    acol = in.acol(); charge = in.charge(); spin = in.pol();
    m2 = pow2(in.m()); isFinal = in.isFinal(); isSoft = isSoftIn;}

  void clear() { col = acol = -1; id = charge = 0; spin = -9; m2 = -1.;
    isFinal = isSoft = false; }

  // Quantum numbers.
  int id, col, acol, charge, spin;

  // Mass.
  double m2;

  // Incoming/final
  bool isFinal;

  // Soft particle, i.e. soft unidentified emission in previous splitting.
  bool isSoft;

};

//==========================================================================

class SplitKinematics {

public:

  SplitKinematics() : m2Dip(-1.), pT2(-1.), pT2Old(-1.), z(-1.), phi(-9.),
    sai(0.), xa(-1), phi2(-9.), m2RadBef(-1.), m2Rec(-1.), m2RadAft(-1.),
    m2EmtAft(-1.), m2EmtAft2(-1.) {}

  SplitKinematics( double m2DipIn, double pT2In, double zIn, double phiIn,
    double saiIn, double xaIn, double phi2In, double m2RadBefIn,
    double m2RecIn, double m2RadAftIn, double m2EmtAftIn, double m2EmtAft2In )
    : m2Dip(m2DipIn), pT2(pT2In), z(zIn), phi(phiIn),
    sai(saiIn), xa(xaIn), phi2(phi2In), m2RadBef(m2RadBefIn), m2Rec(m2RecIn),
    m2RadAft(m2RadAftIn), m2EmtAft(m2EmtAftIn), m2EmtAft2(m2EmtAft2In) {}

  void store2to3kine( double m2DipIn, double pT2In, double zIn, double phiIn) {
    m2Dip = m2DipIn; pT2 = pT2In; z = zIn; phi = phiIn; }
  void store2to3mass( double m2RadBefIn, double m2RecIn, double m2RadAftIn,
    double m2EmtAftIn) { m2RadBef = m2RadBefIn; m2Rec = m2RecIn;
    m2RadAft = m2RadAftIn; m2EmtAft = m2EmtAftIn; }

  void store2to4kine( double m2DipIn, double pT2In, double zIn, double phiIn,
    double saiIn, double xaIn, double phi2In) { m2Dip = m2DipIn;
    pT2 = pT2In; z = zIn; phi = phiIn; sai = saiIn; xa = xaIn; phi2 = phi2In; }
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

  void clear() { m2Dip = pT2 = pT2Old = z = xa = m2RadBef = m2Rec = m2RadAft
     = m2EmtAft = m2EmtAft2 = -1.; sai = 0.;  phi = phi2 = -9.; }

  void store ( const SplitKinematics& k) {
    m2Dip     = k.m2Dip;
    pT2       = k.pT2;
    pT2Old    = k.pT2Old;
    z         = k.z;
    phi       = k.phi;
    sai       = k.sai;
    xa        = k.xa;
    phi2      = k.phi2;
    m2RadBef  = k.m2RadBef;
    m2Rec     = k.m2Rec;
    m2RadAft  = k.m2RadAft;
    m2EmtAft  = k.m2EmtAft;
    m2EmtAft2 = k.m2EmtAft2;
  }

  void list() {
    cout << "List SplitKinematics:"
         << scientific << setprecision(3) << "\n"
         << " m2Dip = " << m2Dip << "\n"
         << " pT2 = "   << pT2 << "\t" 
         << " z = "     << z << "\t"
         << " phi = "   << phi << "\n"
         << " sai = "   << sai << "\t"
         << " xa = "    << xa << "\t"
         << " phi2 = "  << phi2 << "\n"
         << " m2RadBef = "   << m2RadBef << " "
         << " m2Rec = "      << m2Rec << " "
         << " m2RadAft = "   << m2RadAft << " "
         << " m2EmtAft = "   << m2EmtAft << " "
         << " m2EmtAft2t = " << m2EmtAft2 << "\n";
  }

  map<string,double> getKinInfo() {
    return createmap<string,double> ("m2Dip",m2Dip)("pT2",pT2)("pT2Old",pT2Old)
      ("z",z)("phi",phi)("sai",sai)("xa",xa)("phi2",phi2)("m2RadBef",m2RadBef)
      ("m2Rec",m2Rec)("m2RadAft",m2RadAft)("m2EmtAft",m2EmtAft)
      ("m2EmtAft2",m2EmtAft2);
  }

  // Kinematic variable to enable branching.
  double m2Dip, pT2, pT2Old, z, phi, sai, xa, phi2,
         m2RadBef, m2Rec, m2RadAft, m2EmtAft, m2EmtAft2;

};

//==========================================================================

class SplitInfo {

public:

  SplitInfo() : iRadBef(0), iRecBef(0), iRadAft(0), iRecAft(0), iEmtAft(0),
    iEmtAft2(0), side(0), type(0), system(0), systemRec(0), 
    splittingSelName(""), useForBranching(false) { init(); }
  SplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn, 
    int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    double m2DipIn = -1., double pT2In = -1., double zIn = -1.,
    double phiIn = -9., double saiIn = 0., double xaIn = -1.,
    double phi2In = -9., double m2RadBefIn = -1., double m2RecIn = -1.,
    double m2RadAftIn = -1., double m2EmtAftIn = -1., double m2EmtAft2In = -1.,
    int sideIn = 0, int typeIn = 0, int systemIn = 0, int systemRecIn = 0,
    string splittingSelNameIn = "", bool useForBranchingIn = false) :
      iRadBef(iRadBefIn), iRecBef(iRecBefIn), 
      iRadAft(iRadAftIn), iRecAft(iRecAftIn), iEmtAft(iEmtAftIn),
      kinSave(m2DipIn, pT2In, zIn, phiIn, saiIn, xaIn, phi2In, m2RadBefIn,
        m2RecIn, m2RadAftIn, m2EmtAftIn, m2EmtAft2In),
      side(sideIn), type(typeIn), system(systemIn), systemRec(systemRecIn),
      splittingSelName(splittingSelNameIn), useForBranching(useForBranchingIn)
      { init(state); }

  SplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn,
    string splittingSelNameIn) : iRadBef(iRadBefIn), iRecBef(iRecBefIn),
    splittingSelName(splittingSelNameIn) {
    iRadAft = iRecAft = iEmtAft = side = type = system = systemRec = 0; 
    useForBranching = false; init(state); }

  SplitInfo ( const Event& state, int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    string splittingSelNameIn) : iRadAft(iRadAftIn), iRecAft(iRecAftIn),
    iEmtAft(iEmtAftIn), splittingSelName(splittingSelNameIn) {
    splittingSelName = ""; iRadBef = iRecBef = side = type = system
      = systemRec = 0;
    useForBranching = false; init(state); }

  void init(const Event& state = Event() ) {
    if (iRadBef>0) particleSave.push_back(SplitParticle(state[iRadBef],false));
    else           particleSave.push_back(SplitParticle()); 
    if (iRecBef>0) particleSave.push_back(SplitParticle(state[iRecBef],false));
    else           particleSave.push_back(SplitParticle()); 
    if (iRadAft>0) particleSave.push_back(SplitParticle(state[iRadAft],false));
    else           particleSave.push_back(SplitParticle()); 
    if (iRecAft>0) particleSave.push_back(SplitParticle(state[iRecAft],false));
    else           particleSave.push_back(SplitParticle()); 
    if (iEmtAft>0) particleSave.push_back(SplitParticle(state[iEmtAft],false));
    else           particleSave.push_back(SplitParticle()); 
    if (iEmtAft2>0)particleSave.push_back(SplitParticle(state[iEmtAft2],false));
    else           particleSave.push_back(SplitParticle()); 
  }

  void store (const SplitInfo& s) {
    clear();
    kinSave.clear();
    particleSave.resize(0);
    extras.clear();
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
    for ( map<string,double>::const_iterator it = s.extras.begin();
      it != s.extras.end(); ++it )
      extras.insert(make_pair(it->first,it->second));
    useForBranching = s.useForBranching;
  }

  void save () {
    kinSaveStore = kinSave;
    particleSaveStore = particleSave;
    extrasStore = extras;
    iRadBefStore = iRadBef;
    iRecBefStore = iRecBef;
    iRadAftStore = iRadAft;
    iRecAftStore = iRecAft;
    iEmtAftStore = iEmtAft;
    iEmtAft2Store = iEmtAft2;
    sideStore = side;
    typeStore = type;
    systemStore = system;
    systemRecStore = systemRec;
    splittingSelNameStore = splittingSelName;
    useForBranchingStore = useForBranching;
  }

  void restore () {
    kinSave = kinSaveStore;
    particleSave = particleSaveStore;
    extras = extrasStore;
    iRadBef = iRadBefStore;
    iRecBef = iRecBefStore;
    iRadAft = iRadAftStore;
    iRecAft = iRecAftStore;
    iEmtAft = iEmtAftStore;
    iEmtAft2 = iEmtAft2Store;
    side = sideStore;
    type = typeStore;
    system = systemStore;
    systemRec = systemRecStore;
    splittingSelName = splittingSelNameStore;
    useForBranching = useForBranchingStore;
  }

  const SplitParticle* radBef() const { return &particleSave[0]; }
  const SplitParticle* recBef() const { return &particleSave[1]; }
  const SplitParticle* radAft() const { return &particleSave[2]; }
  const SplitParticle* recAft() const { return &particleSave[3]; }
  const SplitParticle* emtAft() const { return &particleSave[4]; }
  const SplitParticle* emtAft2() const { return &particleSave[5]; }

  const SplitKinematics* kinematics() const { return &kinSave; }

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

  void storeRadBef(const Particle& in, bool isSoft)
    { particleSave[0].store(in, isSoft); }
  void storeRecBef(const Particle& in, bool isSoft)
    { particleSave[1].store(in, isSoft); }
  void storeRadAft(const Particle& in, bool isSoft)
    { particleSave[2].store(in, isSoft); }
  void storeRecAft(const Particle& in, bool isSoft)
    { particleSave[3].store(in, isSoft); }
  void storeEmtAft(const Particle& in, bool isSoft)
    { particleSave[4].store(in, isSoft); }
  void storeEmtAft2(const Particle& in, bool isSoft)
    { particleSave[5].store(in, isSoft); }

  void setRadBef( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, 
    bool isFinalIn = false, bool isSoft = false) { setParticle(0, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void setRecBef( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { setParticle(1, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void setRadAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { setParticle(2, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void setRecAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { setParticle(3, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void setEmtAft( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { setParticle(4, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void setEmtAft2( int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { setParticle(5, idIn, colIn,
    acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }
  void clearRadBef()  { setParticle(0, 0, -1, -1, 0, -9, -1.0, false, false); }
  void clearRecBef()  { setParticle(1, 0, -1, -1, 0, -9, -1.0, false, false); }
  void clearRadAft()  { setParticle(2, 0, -1, -1, 0, -9, -1.0, false, false); }
  void clearRecAft()  { setParticle(3, 0, -1, -1, 0, -9, -1.0, false, false); }
  void clearEmtAft()  { setParticle(4, 0, -1, -1, 0, -9, -1.0, false, false); }
  void clearEmtAft2() { setParticle(5, 0, -1, -1, 0, -9, -1.0, false, false); }
  void setParticle( int iPos, int idIn = 0, int colIn = -1, int acolIn = -1,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0,
    bool isFinalIn = false, bool isSoft = false) { particleSave[iPos].store( 
    idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn, isSoft); }

  void storeName (string name) { splittingSelName = name; }
  void storeType ( int in) { type = in; }
  void storeSystem ( int in) { system = in; }
  void storeSystemRec ( int in) { systemRec = in; }
  void storeSide ( int in) { side = in; }
  void storeExtras ( map<string,double> in) { extras = in; }
  void storeRadRecBefPos (int rad, int rec) { iRadBef = rad; iRecBef = rec; }
  void canUseForBranching (bool in) { useForBranching = in;}

  void storeInfo(string name, int typeIn, int systemIn, int systemRecIn, 
    int sideIn, int iPosRadBef, bool softRadBef, int iPosRecBef, 
    bool softRecBef, const Event& state, int idEmtAft,
    int idRadAft, int nEmissions, double m2Dip, double pT2, double z,
    double phi, double m2Bef, double m2s, double m2r, double m2i, double sa1,
    double xa, double phia1, double m2j) {
    clear();
    storeName(name);
    storeType(typeIn);
    storeSystem(systemIn);
    storeSystemRec(systemRecIn);
    storeSide(sideIn);
    storeRadRecBefPos(iPosRadBef, iPosRecBef);
    storeRadBef(state[iPosRadBef], softRadBef);
    storeRecBef(state[iPosRecBef], softRecBef);
    setEmtAft(idEmtAft);
    setRadAft(idRadAft);
    if (nEmissions == 2) set2to4kin( m2Dip, pT2, z, phi, sa1, xa, phia1,
      m2Bef, m2s, m2r, m2i, m2j);
    else set2to3kin( m2Dip, pT2, z, phi, m2Bef, m2s, m2r, m2i);
    storeExtras(
     map<string,double>(createmap<string,double>("iRadBef",iPosRadBef)
       ("iRecBef",iPosRecBef)("idRadAft",idRadAft)) );
  }

  map<string,double> getKinInfo() { return kinSave.getKinInfo();}

  void storePosAfter( int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    int iEmtAft2In) {
    iRadAft = iRadAftIn;
    iRecAft = iRecAftIn;
    iEmtAft = iEmtAftIn;
    iEmtAft2 = iEmtAft2In;
  }

  void clear() {
    iRadBef = iRecBef = iRadAft = iRecAft = iEmtAft = iEmtAft2 = side
      = type = system = systemRec = 0;
    splittingSelName = "";
    useForBranching = false;
    for (int i= 0; i < int(particleSave.size()); ++i) particleSave[i].clear();
    kinSave.clear();
    extras.clear();
  }

  void list() {
    cout << "List SplitInfo: "
         << " name = " << splittingSelName << "\n"
         << " [ id(radBef)= " << radBef()->id
         << " id(recBef)= "   << recBef()->id << " ] --> "
         << " { id(radAft)= " << radAft()->id
         << " id(emtAft)= "   << emtAft()->id
         << " id(emtAft2)= "  << emtAft2()->id
         << " id(recAft)= "   << recAft()->id
         << " } \n";
    kinSave.list();
    cout << "\n";
  }

  // Information to enable branching.
  int iRadBef, iRecBef;

  // Information to enable clustering. 
  int    iRadAft, iRecAft, iEmtAft, iEmtAft2;

  vector<SplitParticle> particleSave;
  SplitKinematics kinSave;

  // Auxiliary information.
  int side, type, system, systemRec;
  string splittingSelName;
  map<string,double> extras;

  bool useForBranching;

  // Information to enable branching.
  int iRadBefStore, iRecBefStore, iRadAftStore, iRecAftStore, iEmtAftStore,
      iEmtAft2Store, sideStore, typeStore, systemStore, systemRecStore;
  vector<SplitParticle> particleSaveStore;
  SplitKinematics kinSaveStore;
  string splittingSelNameStore;
  map<string,double> extrasStore;
  bool useForBranchingStore;


};

//==========================================================================

} // end namespace Pythia8

#endif
