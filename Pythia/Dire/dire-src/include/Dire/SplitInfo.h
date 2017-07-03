// SplitInfo.h is a part of the DIRE plugin to the PYTHIA event generator.
// Copyright (C) 2016 Stefan Prestel.

// Header file for splitting information storage containers.
// SplitParticle:   Quantum numbers and other information of one of the
//                  particles participating in current branching.
// SplitKinematics: Kinematical variables from which to construct a
//                  branching.
// SplitInfo:       Particles participating in the current branching.

#ifndef Pythia8_SplitInfo_H
#define Pythia8_SplitInfo_H

#define DIRE_SPLITINFO_VERSION "2.000"

#include "Pythia8/Event.h"
#include "Dire/Basics.h"

namespace Pythia8 {

//==========================================================================

class SplitParticle {

public:

  SplitParticle() : id(0), col(0), acol(0), charge(0), spin(-9), m2(-1.),
    isFinal(false) {}
  SplitParticle( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn) : id(idIn), col(colIn), acol(acolIn),
    charge(chargeIn), spin(spinIn), m2(m2In), isFinal(isFinalIn) {} 
  SplitParticle ( const Particle& in) : id(in.id()), col(in.col()),
    acol(in.acol()), charge(in.charge()), spin(in.pol()), m2(pow2(in.m())),
    isFinal(in.isFinal()) {}

  void store( int idIn, int colIn, int acolIn, int chargeIn, int spinIn,
    double m2In, bool isFinalIn) { id = idIn; col = colIn; acol = acolIn;
    charge = chargeIn; spin = spinIn; m2 = m2In; isFinal = isFinalIn; }
  void store( const Particle in) { id = in.id(); col = in.col();
    acol = in.acol(); charge = in.charge(); spin = in.pol();
    m2 = pow2(in.m()); isFinal = in.isFinal(); }

  void clear() { id = col = acol = charge = 0; spin = -9; m2 = -1.; }

  // Quantum numbers.
  int id, col, acol, charge, spin;

  // Mass.
  double m2;

  // Incoming/final
  bool isFinal;

};

//==========================================================================

class SplitKinematics {

public:

  SplitKinematics() : m2Dip(-1.), pT2(-1.), z(-1.), phi(-9.), sai(0.), xa(-1),
    phi2(-9.), m2RadBef(-1.), m2Rec(-1.), m2RadAft(-1.), m2EmtAft(-1.),
    m2EmtAft2(-1.) {}

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

  void clear() { m2Dip = pT2 = z = xa = m2RadBef = m2Rec = m2RadAft = m2EmtAft
     = m2EmtAft2 = -1.; sai = 0.;  phi = phi2 = -9.; }

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

  // Kinematic variable to enable branching.
  double m2Dip, pT2, z, phi, sai, xa, phi2,
         m2RadBef, m2Rec, m2RadAft, m2EmtAft, m2EmtAft2;

};

//==========================================================================

class SplitInfo {

public:

  SplitInfo() : iRadBef(0), iRecBef(0), iRadAft(0), iRecAft(0), iEmtAft(0),
    iEmtAft2(0), side(0), type(0), system(0), splittingSelName("") { init(); }
  SplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn, 
    int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    double m2DipIn = -1., double pT2In = -1., double zIn = -1.,
    double phiIn = -9., double saiIn = 0., double xaIn = -1.,
    double phi2In = -9., double m2RadBefIn = -1., double m2RecIn = -1.,
    double m2RadAftIn = -1., double m2EmtAftIn = -1., double m2EmtAft2In = -1.,
    int sideIn = 0, int typeIn = 0, int systemIn = 0,
    string splittingSelNameIn = "") :
      iRadBef(iRadBefIn), iRecBef(iRecBefIn), 
      iRadAft(iRadAftIn), iRecAft(iRecAftIn), iEmtAft(iEmtAftIn),
      kinSave(m2DipIn, pT2In, zIn, phiIn, saiIn, xaIn, phi2In, m2RadBefIn,
        m2RecIn, m2RadAftIn, m2EmtAftIn, m2EmtAft2In),
      side(sideIn), type(typeIn), system(systemIn),
      splittingSelName(splittingSelNameIn)
      { init(state); }

  SplitInfo ( const Event& state, int iRadBefIn, int iRecBefIn,
    string splittingSelNameIn) : iRadBef(iRadBefIn), iRecBef(iRecBefIn),
    splittingSelName(splittingSelNameIn) {
    iRadAft = iRecAft = iEmtAft = side = type = system = 0; init(state); }

  SplitInfo ( const Event& state, int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    string splittingSelNameIn) : iRadAft(iRadAftIn), iRecAft(iRecAftIn),
    iEmtAft(iEmtAftIn), splittingSelName(splittingSelNameIn) {
    splittingSelName = ""; iRadBef = iRecBef = side = type = system = 0;
    init(state); }

  void init(const Event& state = Event() ) {
    if (iRadBef > 0) particleSave.push_back(SplitParticle(state[iRadBef]));
    else             particleSave.push_back(SplitParticle()); 
    if (iRecBef)     particleSave.push_back(SplitParticle(state[iRecBef]));
    else             particleSave.push_back(SplitParticle()); 
    if (iRadAft > 0) particleSave.push_back(SplitParticle(state[iRadAft]));
    else             particleSave.push_back(SplitParticle()); 
    if (iRecAft > 0) particleSave.push_back(SplitParticle(state[iRecAft]));
    else             particleSave.push_back(SplitParticle()); 
    if (iEmtAft > 0) particleSave.push_back(SplitParticle(state[iEmtAft]));
    else             particleSave.push_back(SplitParticle()); 
    if (iEmtAft2 > 0) particleSave.push_back(SplitParticle(state[iEmtAft2]));
    else             particleSave.push_back(SplitParticle()); 
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

  void storeRadBef(const Particle& in) { particleSave[0].store(in); }
  void storeRecBef(const Particle& in) { particleSave[1].store(in); }
  void storeRadAft(const Particle& in) { particleSave[2].store(in); }
  void storeRecAft(const Particle& in) { particleSave[3].store(in); }
  void storeEmtAft(const Particle& in) { particleSave[4].store(in); }
  void storeEmtAft2(const Particle& in) { particleSave[5].store(in); }

  void setRadBef( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(0, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRecBef( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(1, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRadAft( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(2, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setRecAft( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(3, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setEmtAft( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(4, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setEmtAft2( int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    setParticle(5, idIn, colIn, acolIn, chargeIn, spinIn, m2In, isFinalIn); }
  void setParticle( int iPos, int idIn = 0, int colIn = 0, int acolIn = 0,
    int chargeIn = 0, int spinIn = -9, double m2In = -1.0, bool isFinalIn = false) {
    particleSave[iPos].store( idIn, colIn, acolIn, chargeIn, spinIn, m2In,
      isFinalIn); }

  void storeName (string name) { splittingSelName = name; }
  void storeType ( int in) { type = in; }
  void storeSystem ( int in) { system = in; }
  void storeSide ( int in) { side = in; }
  void storeExtras ( map<string,double> in) { extras = in; }
  void storeRadRecBefPos (int rad, int rec) { iRadBef = rad; iRecBef = rec; }

  void storeInfo(string name, int typeIn, int systemIn, int sideIn,
    int iPosRadBef, int iPosRecBef, const Event& state, int idEmtAft,
    int idRadAft, int nEmissions, double m2Dip, double pT2, double z,
    double phi, double m2Bef, double m2s, double m2r, double m2i, double sa1,
    double xa, double phia1, double m2j) {
    clear();
    storeName(name);
    storeType(typeIn);
    storeSystem(systemIn);
    storeSide(sideIn);
    storeRadRecBefPos(iPosRadBef, iPosRecBef);
    storeRadBef(state[iPosRadBef]);
    storeRecBef(state[iPosRecBef]);
    setEmtAft(idEmtAft);
    setRadAft(idRadAft);
    if (nEmissions == 2) set2to4kin( m2Dip, pT2, z, phi, sa1, xa, phia1,
      m2Bef, m2s, m2r, m2i, m2j);
    else set2to3kin( m2Dip, pT2, z, phi, m2Bef, m2s, m2r, m2i);
    storeExtras(
     map<string,double>(createmap<string,double>("iRadBef",iPosRadBef)
       ("iRecBef",iPosRecBef)("idRadAft",idRadAft)) );
  }

  void storePosAfter( int iRadAftIn, int iRecAftIn, int iEmtAftIn,
    int iEmtAft2In) {
    iRadAft = iRadAftIn;
    iRecAft = iRecAftIn;
    iEmtAft = iEmtAftIn;
    iEmtAft2 = iEmtAft2In;
  }

  void clear() {
    iRadBef = iRecBef = iRadAft = iRecAft = iEmtAft = iEmtAft2 = side
      = type = system = 0;
    splittingSelName = "";
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
  int side, type, system;
  string splittingSelName;
  map<string,double> extras;


};

//==========================================================================

} // end namespace Pythia8

#endif
