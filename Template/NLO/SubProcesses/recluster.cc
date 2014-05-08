
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

// LHEF class to parse Les Houches events (as testing case).
//#include "LHEF.h"

using namespace std;

// Functions for 4-vector manipulations.
#include "Boosts.h"

// Container class for a single particle.
class Particle {

public:

 ~Particle() {};
  Particle() : idSave(0), statusSave(0), colSave(0), acolSave(0),
    mSave(0.0), m0Save(0.0), pSave(Vec4()) {};
  Particle(const int idIn, const int statusIn, const int colIn,
      const int acolIn, const double mIn, const double m0In,
      double eI, double pxI, double pyI, double pzI) 
    : idSave(idIn), statusSave(statusIn),  colSave(colIn), acolSave(acolIn),
      mSave(mIn), m0Save(m0In), pSave(Vec4()) {
      e(eI); px(pxI);  py(pyI);  pz(pzI); };
  Particle(const Particle& particleIn) 
    : idSave(particleIn.idSave), statusSave(particleIn.statusSave),
      colSave(particleIn.colSave), acolSave(particleIn.acolSave),
      mSave(particleIn.mSave), m0Save(particleIn.m0Save),
      pSave(particleIn.pSave) {};

  int     idSave, statusSave, colSave, acolSave;
  double  mSave, m0Save;
  Vec4    pSave;

  // Member functions for input.
  void id(int idIn) {idSave = idIn;}
  void status(int statusIn) {statusSave = statusIn;}
  void p(Vec4 pIn) {pSave = pIn;}
  void p(double pxIn, double pyIn, double pzIn, double eIn) 
    {pSave.p(pxIn, pyIn, pzIn, eIn);}
  void px(double pxIn) {pSave.px(pxIn);}
  void py(double pyIn) {pSave.py(pyIn);}
  void pz(double pzIn) {pSave.pz(pzIn);}
  void e(double eIn) {pSave.e(eIn);}
  void m(double mIn) {mSave = mIn;}
  void col(int colIn) {colSave = colIn;}
  void acol(int acolIn) {acolSave = acolIn;}

  // Member functions for output.
  int    id()        const {return idSave;}
  int    status()    const {return statusSave;}
  Vec4   p()         const {return pSave;}
  double px()        const {return pSave.px();}
  double py()        const {return pSave.py();}
  double pz()        const {return pSave.pz();}
  double e()         const {return pSave.e();}
  double m()         const {return mSave;}
  double m0()        const {return m0Save;}

  // Member functions for output; derived int and bool quantities.
  bool   isFinal()   const {return (statusSave > 0);}

  double mCalc()     const {return pSave.mCalc();}
  double m2Calc()    const {return pSave.m2Calc();}
  double theta()     const {return pSave.theta();}
  double phi()       const {return pSave.phi();}

  int    col()       const {return colSave;}
  int    acol()      const {return acolSave;}

  // Member functions that perform operations.
  void rot(double thetaIn, double phiIn) {pSave.rot(thetaIn, phiIn);} 
  void bst(double betaX, double betaY, double betaZ) { pSave.bst(betaX, betaY, betaZ);}
  void bst(double betaX, double betaY, double betaZ, double gamma) {pSave.bst(betaX, betaY, betaZ, gamma);}
  void bst(const Vec4& pBst) {pSave.bst(pBst);}
  void bst(const Vec4& pBst, double mBst) {pSave.bst(pBst, mBst);}
  void rotbst(const RotBstMatrix& M) {pSave.rotbst(M);} 

  friend class State;

};

// Container class for the whole state. Also includes clustering routines.
class State {

public:

 ~State() {} 
  State() { entry.reserve(100);} 
  State(const vector<int> id, const vector<int> status) { entry.reserve(100);}

  // Overload index operator to access element of event record.
  Particle& operator[](int i) {return entry[i];}
  const Particle& operator[](int i) const {return entry[i];}

  vector<Particle> entry;
  int size() const {return entry.size();}
  int append(Particle entryIn) { entry.push_back(entryIn); return entry.size() - 1;}

  void list() const; 

};

// Function to list state.
void State::list() const {

  // Header.
  std::cout << "\n --------  State listing ----------"
     << "-------------------------------------------------\n \n    no    "
     << "    id    status    colours"
     << "      p_x        p_y        p_z         e          m \n"; 

  // At high energy switch to scientific format for momenta.
  bool useFixed = (entry[0].e() < 1e5);

  // Listing of complete event.
  Vec4 pSum;
  double chargeSum = 0.;
  for (int i = 0; i < int(entry.size()); ++i) {
    const Particle& pt = entry[i];

    // Basic line for a particle, always printed.
    std::cout << setw(6) << i << setw(6) << pt.id() << "   " << setw(6) 
       << pt.status() << setw(6) << pt.col() << setw(6) << pt.acol()
       << scientific << setprecision(5) 
       << setw(14) << pt.px() << setw(14) << pt.py() << setw(14) 
       << pt.pz() << setw(14) << pt.e() << setw(14) << pt.m() << "\n";

    // Statistics on momentum and charge.
    if (entry[i].status() > 0) {
      pSum += entry[i].p(); 
    }
  }

  // Line with sum charge, momentum, energy and invariant mass.
  std::cout << fixed << setprecision(3) << "                                   "
     << "Momentum sum:" 
     << scientific << setprecision(3) 
     << setw(11) << pSum.px() << setw(11) << pSum.py() << setw(11) 
     << pSum.pz() << setw(11) << pSum.e() << setw(11) << pSum.mCalc() 
     << "\n";

  // Listing finished.
  std::cout << "\n-------------------------------------------------------------------"
     << endl;
}

// Function to get the flavour of the radiator before the splitting
// for clustering (helper for State::cluster)
int getRadBeforeFlav(const int RadAfter, const int EmtAfter,
      const State& event) {
  int type = event[RadAfter].isFinal() ? 1 :-1;
  int emtID  = event[EmtAfter].id();
  int radID  = event[RadAfter].id();
  int emtCOL = event[EmtAfter].col();
  int radCOL = event[RadAfter].col();
  int emtACL = event[EmtAfter].acol();
  int radACL = event[RadAfter].acol();
  bool colConnected = ((type == 1) && ( (emtCOL !=0 && (emtCOL ==radACL))
                                     || (emtACL !=0 && (emtACL ==radCOL)) ))
                    ||((type ==-1) && ( (emtCOL !=0 && (emtCOL ==radCOL))
                                     || (emtACL !=0 && (emtACL ==radACL)) ));
  // QCD splittings
  // Gluon radiation
  if ( emtID == 21 )
    return radID;
  // Final state gluon splitting
  if ( type == 1 && emtID == -radID && !colConnected )
    return 21;
  // Initial state s-channel gluon splitting
  if ( type ==-1 && radID == 21 )
    return -emtID;
  // Initial state t-channel gluon splitting
  if ( type ==-1 && !colConnected
    && emtID != 21 && radID != 21 && abs(emtID) < 10 && abs(radID) < 10)
    return 21;
  // Done.
  return 0;

}

// Function to get the colour of the radiator before the splitting
// for clustering (helper for State::cluster)
int getRadBeforeCol(const int rad, const int emt,
      const State& event) {
  // Save type of splitting
  int type = (event[rad].isFinal()) ? 1 :-1;
  // Get flavour of radiator after potential clustering
  int radBeforeFlav = getRadBeforeFlav(rad,emt,event);
  // Get colours of the radiator before the potential clustering
  int radBeforeCol = -1;
  // Get reconstructed gluon colours
  if (radBeforeFlav == 21) {
    // Start with quark emissions in FSR
    if (type == 1 && event[emt].id() != 21) {
      radBeforeCol = (event[rad].col()  > 0)
                   ? event[rad].col() : event[emt].col();
    // Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      radBeforeCol = (event[rad].col()  > 0)
                   ? event[rad].col() : event[emt].acol();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : event[rad].col();
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : event[rad].acol();
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    }
  // Get reconstructed quark colours
  } else if ( radBeforeFlav != 21 && radBeforeFlav > 0) {
    // Quark emission in FSR
    if (type == 1 && event[emt].id() != 21) {
      // If radiating is a quark, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].col() : event[rad].col();
    //Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      // If emitted is a quark, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : 0;
      radBeforeCol  = (event[rad].col()  == colRemove)
                    ? event[emt].acol() : event[rad].col();
    }
  // Other particles are assumed uncoloured
  } else {
    radBeforeCol = 0;
  }
  return radBeforeCol;
}

//--------------------------------------------------------------------------

// Function to get the anticolour of the radiator before the splitting
// for clustering  (helper for State::cluster)
int getRadBeforeAcol(const int rad, const int emt,
      const State& event) {
  // Save type of splitting
  int type = (event[rad].isFinal()) ? 1 :-1;
  // Get flavour of radiator after potential clustering
  int radBeforeFlav = getRadBeforeFlav(rad,emt,event);
  // Get colours of the radiator before the potential clustering
  int radBeforeAcl = -1;
  // Get reconstructed gluon colours
  if (radBeforeFlav == 21) {
    // Start with quark emissions in FSR
    if (type == 1 && event[emt].id() != 21) {
      radBeforeAcl = (event[rad].acol() > 0)
                   ? event[rad].acol() : event[emt].acol();
    // Quark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      radBeforeAcl = (event[rad].acol() > 0)
                   ? event[rad].acol() : event[emt].col();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : event[rad].col();
      radBeforeAcl  = (event[rad].acol() == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].col())
                    ? event[rad].col() : event[rad].acol();
      radBeforeAcl  = (event[rad].acol() == colRemove)
                    ? event[emt].col() : event[rad].acol();
    }
  // Get reconstructed anti-quark colours
  } else if ( radBeforeFlav != 21 && radBeforeFlav < 0) {
    // Antiquark emission in FSR
    if (type == 1 && event[emt].id() != 21) {
      // If radiating is a antiquark, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].col() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Gluon emissions in FSR
    } else if (type == 1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].acol() == event[emt].col())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].acol() : event[rad].acol();
    //Antiquark emissions in ISR
    } else if (type == -1 && event[emt].id() != 21) {
      // If emitted is an antiquark, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].acol() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].col() : event[rad].acol();
    //Gluon emissions in ISR
    } else if (type == -1 && event[emt].id() == 21) {
      // If emitted is a gluon, remove the repeated index, and take
      // the remaining indices as colour and anticolour 
      int colRemove = (event[rad].acol() == event[emt].acol())
                    ? event[rad].acol() : 0;
      radBeforeAcl  = (event[rad].acol()  == colRemove)
                    ? event[emt].col() : event[rad].acol();
    }
  // Other particles are considered uncoloured
  } else {
    radBeforeAcl = 0;
  }
  return radBeforeAcl;
}

// Function to cluster state.
State cluster(int Rad, int Rec, int Emt, double eCM, const State& state) {

  // Flags for type of radiation
  int radType = state[Rad].isFinal() ? 1 : -1;
  int recType = state[Rec].isFinal() ? 1 : -1;

  // Construct the clustered event
  State NewEvent = State();
  // Copy all unchanged particles to NewEvent
  for (int i = 0; i < state.size(); ++i)
    if ( i != Rad && i != Rec && i != Emt )
      NewEvent.append( state[i] );

  // Set properties of radiator/recoiler after the clustering
  // Recoiler properties will be unchanged
  Particle RecBefore = Particle( state[Rec] );

  // Find flavour of radiator before splitting
  int radID = 21;  //getRadBeforeFlav(Rad, Emt, state);
  int radCol1 = 0; //getRadBeforeCol(Rad, Emt, state);
  int radCol2 = 0; //getRadBeforeAcol(Rad, Emt, state);

  int recID = state[Rec].id();
  Particle RadBefore = Particle( state[Rad] );
  RadBefore.id(radID);

  // Put mass for radiator and recoiler
  double radMass = state[Rad].m0();
  double recMass = state[Rec].m0();
  if (radType == 1 ) RadBefore.m(radMass);
  else RadBefore.m(0.0);
  if (recType == 1 ) RecBefore.m(recMass);
  else RecBefore.m(0.0);

  // Construct momenta and  colours of clustered particles
  // ISR/FSR splittings are treated differently
  if ( radType + recType == 2 ) {
    // Clustering of final(rad)/final(rec) dipole splitting
    // Get eCM of (rad,rec,emt) triple
    Vec4   sum     = state[Rad].p() + state[Rec].p() + state[Emt].p();
    double eCMME   = sum.mCalc();

    // Define radiator and recoiler back-to-back in the dipole
    // rest frame [=(state[rad]+state[emt])+state[rec] rest frame]
    Vec4 Rad4mom;
    Vec4 Rec4mom;
    double mDsq   = pow(eCMME,2);
    // If possible, keep the invariant mass of the radiator.
    double mRsq   = (radID == state[Rad].id() )
                  ? abs(state[Rad].p().m2Calc())
                  : pow(state[Rad].m0(),2);
    double mSsq   = abs(state[Rec].p().m2Calc());
    double a = 0.5*sqrt(mDsq);
    double b = 0.25*(mRsq - mSsq) / a;
    double c = sqrt(pow(a,2) + pow(b,2) - 2.*a*b - mSsq);

    Rad4mom.p( 0., 0., c, a+b);
    Rec4mom.p( 0., 0.,-c, a-b);

    // Find boost from Rad4mom+Rec4mom rest frame to event cm frame
    Vec4 old1 = Vec4(state[Rad].p() + state[Emt].p());
    Vec4 old2 = Vec4(state[Rec].p());
    RotBstMatrix fromCM;
    fromCM.fromCMframe(old1, old2);
    // Transform momenta
    Rad4mom.rotbst(fromCM);
    Rec4mom.rotbst(fromCM);

    RadBefore.p(Rad4mom);
    RecBefore.p(Rec4mom);
    RadBefore.m(sqrt(mRsq));
    RecBefore.m(sqrt(mSsq));

    RecBefore.status(1);
    RadBefore.status(1);
    RadBefore.col(radCol1);
    RadBefore.acol(radCol2);

  } else if ( radType + recType == 0 ) {
    // Clustering of final(rad)/initial(rec) dipole splitting
    // Get eCM of (rad,rec,emt) triple
    Vec4   sum     = state[Rad].p() + state[Rec].p() + state[Emt].p();
    double eCMME   = sum.mCalc();
    // Define radiator and recoiler back-to-back in the dipole
    // rest frame [=(state[rad]+state[emt])+state[rec] rest frame]
    Vec4 Rad4mom;
    Vec4 Rec4mom;
    double mDsq   = pow(eCMME,2);
    // If possible, keep the invariant mass of the radiator.
    double mRsq   = (radID == state[Rad].id() )
                  ? abs(state[Rad].p().m2Calc())
                  : pow(state[Rad].m0(),2);
    double mSsq   = abs(state[Rec].p().m2Calc());
    double a = 0.5*sqrt(mDsq);
    double b = 0.25*(mRsq - mSsq) / a;
    double c = sqrt(pow(a,2) + pow(b,2) - 2.*a*b - mSsq);

    Rad4mom.p( 0., 0., c, a+b);
    Rec4mom.p( 0., 0.,-c, a-b);

    // Find boost from Rad4mom+Rec4mom rest frame to event cm frame
    Vec4 old1 = Vec4(state[Rad].p() + state[Emt].p());
    Vec4 old2 = Vec4(state[Rec].p());
    RotBstMatrix fromCM;
    fromCM.fromCMframe(old1, old2);
    // Transform momenta
    Rad4mom.rotbst(fromCM);
    Rec4mom.rotbst(fromCM);

    // Rescale recoiler momentum
    Rec4mom = 2.*state[Rec].p() - Rec4mom;

    RadBefore.p(Rad4mom);
    RecBefore.p(Rec4mom);
    RadBefore.m(sqrt(mRsq));

    // Set mass of initial recoiler to zero
    RecBefore.m( 0.0 );

    RecBefore.status(-1);
    RadBefore.status(1);
    RadBefore.col(radCol1);
    RadBefore.acol(radCol2);

  } else {
    // Clustering of initial(rad)/initial(rec) dipole splitting
    // We want to cluster: Meaning doing the inverse of a process
    //            ( pDaughter + pRecoiler -> pOut )
    //        ==> ( pMother + pPartner -> pOut' + pSister )
    // produced by an initial state splitting. The matrix element
    // provides us with pMother, pPartner, pSister and pOut'
    Vec4 pMother( state[Rad].p() );
    Vec4 pSister( state[Emt].p() );
    Vec4 pPartner( state[Rec].p() );
    Vec4 pDaughter( 0.,0.,0.,0. );
    Vec4 pRecoiler( 0.,0.,0.,0. );

    // Find side that radiates event (mother moving in sign * p_z direction).
    int sign = (state[Rad].pz() > 0.) ? 1 : -1;

    // Find rotation by phi that would have been done for a
    // splitting daughter -> mother + sister
    double phi = pSister.phi();
    // Find rotation with -phi
    RotBstMatrix rot_by_mphi;
    rot_by_mphi.rot(0.,-phi);
    // Find rotation with +phi
    RotBstMatrix rot_by_pphi;
    rot_by_pphi.rot(0.,phi);

    // Transform pMother and outgoing momenta
    pMother.rotbst( rot_by_mphi );
    pSister.rotbst( rot_by_mphi );
    pPartner.rotbst( rot_by_mphi );
    for(int i=0; i< NewEvent.size(); ++i)
      NewEvent[i].rotbst( rot_by_mphi );

    // Get mother and partner x values
    // x1 after isr
    double x1 = 2. * pMother.e() / eCM;
    // x2 after isr
    double x2 = 2. * pPartner.e() / eCM;

    pDaughter.p( pMother - pSister);
    pRecoiler.p( pPartner );

    // Find boost from event cm frame to rest frame of
    // of-shell daughter + on-shell recoiler
    RotBstMatrix from_CM_to_DR;
    if (sign == 1)
      from_CM_to_DR.toCMframe(pDaughter, pRecoiler);
    else
      from_CM_to_DR.toCMframe(pRecoiler, pDaughter);

    // Transform all momenta
    pMother.rotbst( from_CM_to_DR );
    pPartner.rotbst( from_CM_to_DR );
    pSister.rotbst( from_CM_to_DR );
    for(int i=0; i< NewEvent.size(); ++i)
      NewEvent[i].rotbst( from_CM_to_DR );

    // Find theta angle between pMother and z-axis and undo
    // rotation that would have been done by shower
    double theta = pMother.theta();
    if ( pMother.px() < 0. ) theta *= -1.;
    if (sign == -1) theta += M_PI;


    // Find rotation by +theta
    RotBstMatrix rot_by_ptheta;
    rot_by_ptheta.rot(theta, 0.);

    // Transform all momenta
    pMother.rotbst( rot_by_ptheta );
    pPartner.rotbst( rot_by_ptheta );
    pSister.rotbst( rot_by_ptheta );
    for(int i=0; i< NewEvent.size(); ++i)
      NewEvent[i].rotbst( rot_by_ptheta );

    // Find z of the splitting
    Vec4 qDip( pMother - pSister);
    Vec4 qAfter(pMother + pPartner);
    Vec4 qBefore(qDip + pPartner);
    double z = qBefore.m2Calc() / qAfter.m2Calc();

    // Calculate new e_CM^2
    double x1New = z*x1; // x1 before isr
    double x2New = x2;   // x2 before isr
    double sHat = x1New*x2New*eCM*eCM;

    // Construct daughter and recoiler momenta
    pDaughter.p( 0., 0.,  sign*0.5*sqrt(sHat), 0.5*sqrt(sHat));
    pRecoiler.p( 0., 0., -sign*0.5*sqrt(sHat), 0.5*sqrt(sHat));

    // Find boost from current (daughter+recoiler rest frame)
    // frame to rest frame of daughter+unchanged recoiler to
    // recover the old x2 value
    RotBstMatrix from_DR_to_CM;
    from_DR_to_CM.bst( 0., 0., sign*( x1New - x2New ) / ( x1New + x2New ) );

    // Correct for momentum mismatch by transforming all momenta
    pMother.rotbst( from_DR_to_CM );
    pPartner.rotbst( from_DR_to_CM );
    pSister.rotbst( from_DR_to_CM );
    pDaughter.rotbst( from_DR_to_CM );
    pRecoiler.rotbst( from_DR_to_CM );
    for(int i=0; i< NewEvent.size(); ++i)
      NewEvent[i].rotbst( from_DR_to_CM );

    // Transform pMother and outgoing momenta
    pMother.rotbst( rot_by_pphi );
    pPartner.rotbst( rot_by_pphi );
    pSister.rotbst( rot_by_pphi );
    pDaughter.rotbst( rot_by_pphi );
    pRecoiler.rotbst( rot_by_pphi );
    for(int i=0; i< NewEvent.size(); ++i)
      NewEvent[i].rotbst( rot_by_pphi );

    // Set momenta of particles to be attached to new event record
    RecBefore.p( pRecoiler );
    RadBefore.p( pDaughter );

    RecBefore.status(-1);
    RadBefore.status(-1);
    RadBefore.col(radCol1);
    RadBefore.acol(radCol2);

  }

  // Append new recoiler and find new radiator colour
  NewEvent.append(RecBefore);
  NewEvent.append(RadBefore);

  // Done
  return NewEvent;

}

// Function to compute "pythia pT separation" from Particle input

double pTpythia(const Particle& RadAfterBranch,
              const Particle& EmtAfterBranch,
              const Particle& RecAfterBranch, int ShowerType){

  // Save type: 1 = FSR pT definition, else ISR definition
  int Type   = ShowerType;
  // Calculate virtuality of splitting
  int sign = (Type==1) ? 1 : -1;
  Vec4 Q(RadAfterBranch.p() + sign*EmtAfterBranch.p());
  double Qsq = sign * Q.m2Calc();
  // Mass term of radiator
  double m2Rad = ( true
               && abs(RadAfterBranch.id()) >= 4
               && abs(RadAfterBranch.id()) < 7)
               ? pow(RadAfterBranch.m0(), 2)
               : 0.;

  // Construct 2->3 variables for FSR
  Vec4   sum     = RadAfterBranch.p() + RecAfterBranch.p()
                 + EmtAfterBranch.p();
  double m2Dip = sum.m2Calc();
  double x1 = 2. * (sum * RadAfterBranch.p()) / m2Dip;
  double x3 = 2. * (sum * EmtAfterBranch.p()) / m2Dip;
  // Construct momenta of dipole before/after splitting for ISR 
  Vec4 qBR(RadAfterBranch.p() - EmtAfterBranch.p() + RecAfterBranch.p());
  Vec4 qAR(RadAfterBranch.p() + RecAfterBranch.p());
  // Calculate z of splitting, different for FSR and ISR
  double z = (Type==1) ? x1/(x1+x3)
                     : (qBR.m2Calc())/( qAR.m2Calc());
  // Separation of splitting, different for FSR and ISR
  double pTpyth = (Type==1) ? z*(1.-z) : (1.-z);
  // pT^2 = separation*virtuality
  pTpyth *= (Qsq - sign*m2Rad);
  if(pTpyth < 0.) pTpyth = 0.;

  // Return pT
  return sqrt(pTpyth);
}

// Find the minimal Lund pT between coloured partons in the input
// inEvent
double minPythiaSep( const State& inEvent, int& iradMin, int& iemtMin, int& irecMin, int& typeMin ){


  // Find all electroweak decayed bosons in the state.
  vector<int> ewResonancePos;
  ewResonancePos.clear();
  for (int i=0; i < inEvent.size(); ++i)
    if( abs(inEvent[i].status()) == 2
      && ( abs(inEvent[i].id()) == 22
        || abs(inEvent[i].id()) == 23
        || abs(inEvent[i].id()) == 24
        || abs(inEvent[i].id()) == 25 ) )
      ewResonancePos.push_back(i);

  // Declare final parton vectors
  vector <int> FinalPartPos;
  FinalPartPos.clear();
  // Search inEvent record for final state partons.
  // Exclude decay products of ew resonance.
  for (int i=0; i < inEvent.size(); ++i){
    if( inEvent[i].isFinal() && (inEvent[i].id() == 21 || abs(inEvent[i].id()) < 10) ){
      bool isDecayProduct = false;
//      for(int j=0; j < int(ewResonancePos.size()); ++j)
//        if( inEvent.isAncestor(i, ewResonancePos[j]) )
//          isDecayProduct = true;
      if(!isDecayProduct)
        FinalPartPos.push_back(i);
    }
  }

  // Get index of first incoming
  int in1 = 0;
  for (int i=0; i < inEvent.size(); ++i)
    if(inEvent[i].status() == -1 && inEvent[i].pz() > 0.){
      in1 = i;
      break;
    }

  // Get index of second incoming
  int in2 = 0;
  for (int i=0; i < inEvent.size(); ++i)
    if(inEvent[i].status() == -1 && inEvent[i].pz() < 0.){
      in2 = i;
      break;
    }

  // Find minimal pythia pt in inEvent
  double ptmin = 1e15;
  int irad, iemt, irec, itype;
  irad = iemt = irec = itype = 0;
  
  for(int i=0; i < int(FinalPartPos.size()); ++i){

    double pt12  = ptmin;
    // Compute pythia ISR separation i-jet and first incoming
    if(inEvent[in1].id() == 21 || abs(inEvent[in1].id()) < 10) {
      double temp = pTpythia( inEvent[in1],
                      inEvent[FinalPartPos[i]], inEvent[in2], -1 );
      if (temp > 0e0 && pt12 > temp) { irad = in1; iemt = FinalPartPos[i]; irec = in2; itype = -1;}
      pt12 = min(pt12, temp);
    }
    // Compute pythia ISR separation i-jet and second incoming
    if(inEvent[in2].id() == 21 || abs(inEvent[in2].id()) < 10) {
      double temp = pTpythia( inEvent[in2], inEvent[FinalPartPos[i]], inEvent[in1], -1 );
      if (temp > 0e0 && pt12 > temp) { irad = in2; iemt = FinalPartPos[i]; irec = in1; itype = -1;}
      pt12 = min(pt12, temp);
    }

    // Compute pythia FSR separation between two jets,
    // without any knowledge of colour connections
    for(int j=0; j < int(FinalPartPos.size()); ++j) {
      for(int k=0; k < int(FinalPartPos.size()); ++k) {
        // Allow any parton as recoiler
        if( (i != j) && (i != k) && (j != k) ){

          double temp = 0.;
          // Only check splittings allowed by flavour, e.g.
          // only q -> qg and g -> qqbar
          temp = pTpythia( inEvent[FinalPartPos[i]],
                            inEvent[FinalPartPos[j]],
                            inEvent[FinalPartPos[k]], 1 );
          if (temp > 0e0 && pt12 > temp) { irad = FinalPartPos[i]; iemt = FinalPartPos[j]; irec = FinalPartPos[k]; itype = 1;}
          pt12 = min(pt12, temp);
        } // if allowed triple
      } // loop over k
    } // loop over j

    // Compute pythia FSR separation between two jets, with initial recoiler
    // without any knowledge of colour connections
    if ( (inEvent[in1].id() == 21 || abs(inEvent[in1].id()) < 10)
      && (inEvent[in2].id() == 21 || abs(inEvent[in2].id()) < 10)) {
      for(int j=0; j < int(FinalPartPos.size()); ++j) {
        // Allow both initial partons as recoiler
        if( i != j ){
          // Check with first initial as recoiler
          double temp = pTpythia( inEvent[FinalPartPos[i]],
                                  inEvent[FinalPartPos[j]],
                                  inEvent[in1], 1 );
          if (temp > 0e0 && pt12 > temp) { irad = FinalPartPos[i]; iemt = FinalPartPos[j]; irec = in1; itype = 1;}
          pt12 = min(pt12, temp);
          // Check with second initial as recoiler
          temp        = pTpythia( inEvent[FinalPartPos[i]],
                                  inEvent[FinalPartPos[j]],
                                  inEvent[in2], 1 );
          if (temp > 0e0 && pt12 > temp) { irad = FinalPartPos[i]; iemt = FinalPartPos[j]; irec = in2; itype = 1;}
          pt12 = min(pt12, temp);
        }
      } // loop over j
    } // if initial partons

    // Reset minimal y separation
    ptmin = min(ptmin,pt12);

  } // loop over i

  // Return indices of minimal-pT triplett
  iradMin = irad;
  iemtMin = iemt;
  irecMin = irec;
  typeMin = itype; 

  // Done
  return ptmin;

}



// Small values used to cut off momenta
const double TINY      = 1e-15;
const double TINYMASS  = 1e-8;



extern "C" {   
//Fortran interface

  void pythia_unlops_cluster_(const double & eCM, const double * p, const int & npart,
			      const int * id, const int * ist, double & pTmin1,
			      double & pTmin2 ){
    // This is our working copy.
    State* originalState = new State();

    for (int i=0; i<npart*5; i=i+5) {
      // Store information in a State.
      double pup0 = (abs(p[i+0]) < TINY) ? 0.: p[i+0];
      double pup1 = (abs(p[i+1]) < TINY) ? 0.: p[i+1];
      double pup2 = (abs(p[i+2]) < TINY) ? 0.: p[i+2];
      double pup3 = (abs(p[i+3]) < TINY) ? 0.: p[i+3];
      double pup4 = (abs(p[i+4]) < TINYMASS) ? 0.: p[i+4];
      // color information not known
      int col1 = 0;
      int col2 = 0;
      if ( abs(ist[i/5]) == 1)
        originalState->append(Particle(id[i/5], ist[i/5], col1, col2,
          pup4, 0.0, pup3, pup0, pup1, pup2));
    }

    // Get minimal pT for first clustering
    int irad, iemt, irec, type;
    irad = iemt = irec = type = 0;
    pTmin1 = minPythiaSep(*originalState,irad,iemt,irec,type);
    // Perform clustering of lowest-pT triple.
    // Clustering inputs: Radiator, Recoiler, Emitted, eCM, State to be clustered.
    State clusteredState = State(cluster(irad,irec,iemt,eCM,*originalState));
    // Get minimal pT for second clustering
    irad = iemt = irec = type = 0;
    pTmin2 = minPythiaSep(clusteredState,irad,iemt,irec,type);
    delete originalState;
  }
}

