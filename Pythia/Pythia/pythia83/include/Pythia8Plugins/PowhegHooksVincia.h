// PowhegHooks.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Richard Corke, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Christian T Preuss, based on PowhegHooks.h
// This class is used to perform a vetoed Vincia shower, where emissions
// already covered in a POWHEG NLO generator should be omitted.
// To first approximation the handover should happen at the SCALE
// of the LHA, but since the POWHEG-BOX uses a different pT definition
// than PYTHIA, both for ISR and FSR, a more sophisticated treatment
// is needed. See the online manual on POWHEG matching for details.

#ifndef Pythia8_PowhegHooksVincia_H
#define Pythia8_PowhegHooksVincia_H

// Includes
#include "Pythia8/Pythia.h"

namespace Pythia8 {

//==========================================================================

// Use userhooks to veto Vincia emissions above the POWHEG scale.

class PowhegHooksVincia : public UserHooks {

public:

  // Constructor and destructor.
   PowhegHooksVincia() {}
  ~PowhegHooksVincia() {}

  //--------------------------------------------------------------------------

  // Initialize settings, detailing merging strategy to use.
  bool initAfterBeams() {
    nFinal      = settingsPtr->mode("POWHEG:nFinal");
    vetoMode    = settingsPtr->mode("POWHEG:veto");
    vetoCount   = settingsPtr->mode("POWHEG:vetoCount");
    pThardMode  = settingsPtr->mode("POWHEG:pThard");
    pTemtMode   = settingsPtr->mode("POWHEG:pTemt");
    emittedMode = settingsPtr->mode("POWHEG:emitted");
    pTdefMode   = settingsPtr->mode("POWHEG:pTdef");
    MPIvetoMode = settingsPtr->mode("POWHEG:MPIveto");
    QEDvetoMode = settingsPtr->mode("POWHEG:QEDveto");
    return true;
  }

  //--------------------------------------------------------------------------

  // Routines to calculate the pT (according to pTdefMode) in a branching.
  //   ISR: i (radiator after)  -> j (emitted after) k (radiator before)
  //   FSR: i (radiator before) -> j (emitted after) k (radiator after)
  // For Vincia, the shower pT definition is used.

  // Compute the Vincia pT as in Eq. (2.63)-(2.66) in arXiv:2003.00702.
  // Branching is assumed to be {13} {23} -> 1 3 2.
  inline double pTvincia(const Event& event, int i1,
    int i3, int i2, bool /*isFSR*/) {

    // Shorthands.
    Vec4 p1 = event[i1].p();
    Vec4 p3 = event[i3].p();
    Vec4 p2 = event[i2].p();

    // Fetch mothers of 1 and 2.
    int iMoth1 = event[i1].mother1();
    int iMoth2 = event[i2].mother1();
    if (iMoth1 == 0 || iMoth2 == 0) {
      cout << " Error: could not find mothers of particles." << endl;
      exit(1);
    }

    // Invariants defined as in Eq. (5) in arXiv:2008.09468.
    double mMoth1Sq = pow2(event[iMoth1].m());
    double mMoth2Sq = pow2(event[iMoth2].m());
    double sgn1 = event[i1].isFinal() ? 1. : -1.;
    double sgn2 = event[i2].isFinal() ? 1. : -1.;
    double qSq13 = sgn1*(m2(sgn1*p1+p3) - mMoth1Sq);
    double qSq23 = sgn2*(m2(sgn2*p2+p3) - mMoth2Sq);

    // Normalisation as in Eq. (6) in arXiv:2008.09468.
    double sMax = -1.;
    if (event[i1].isFinal() && event[i2].isFinal()) {
      // FF.
      sMax = m2(p1+p2+p3) - mMoth1Sq - mMoth2Sq;
    } else if ((event[i1].isResonance() && event[i2].isFinal())
      || (!event[i1].isFinal() && event[i2].isFinal())) {
      // RF or IF.
      sMax = 2.*p1*p3 + 2.*p1*p2;
    } else if ((event[i1].isFinal() && event[i2].isResonance())
      || (event[i1].isFinal() && !event[i2].isFinal())) {
      // FR or FI.
      sMax = 2.*p2*p3 + 2.*p1*p2;
    } else if (!event[i1].isFinal() || !event[i2].isFinal()) {
      // II.
      sMax = 2.*p1*p2;
    } else {
      cout << " Error: could not determine branching type." << endl;
      exit(1);
    }

    // Calculate pT2 as in Eq. (5) in arXiv:2008.09468.
    double pT2now = qSq13*qSq23/sMax;

    // Sanity check.
    if (pT2now < 0.) {
      cout << "Warning: negative pTvincia" << endl;
      return -1.;
    }

    // Return pT
    return sqrt(pT2now);
  }

  // Compute the POWHEG pT separation between i and j
  inline double pTpowheg(const Event &e, int i, int j, bool FSR) {

    // pT value for FSR and ISR
    double pTnow = 0.;
    if (FSR) {
      // POWHEG d_ij (in CM frame). Note that the incoming beams have not
      // been updated in the parton systems pointer yet (i.e. prior to any
      // potential recoil).
      int iInA = partonSystemsPtr->getInA(0);
      int iInB = partonSystemsPtr->getInB(0);
      double betaZ = - ( e[iInA].pz() + e[iInB].pz() ) /
                       ( e[iInA].e()  + e[iInB].e()  );
      Vec4 iVecBst(e[i].p()), jVecBst(e[j].p());
      iVecBst.bst(0., 0., betaZ);
      jVecBst.bst(0., 0., betaZ);
      pTnow = sqrt( (iVecBst + jVecBst).m2Calc() *
                    iVecBst.e() * jVecBst.e() /
                    pow2(iVecBst.e() + jVecBst.e()) );

    } else {
      // POWHEG pT_ISR is just kinematic pT
      pTnow = e[j].pT();
    }

    // Check result
    if (pTnow < 0.) {
      cout << "Warning: negative pTpowheg" << endl;
      return -1.;
    }

    return pTnow;
  }

  // Calculate pT for a splitting based on pTdefMode.
  // If j is -1, all final-state partons are tried.
  // If i, k, r and xSR are -1, then all incoming and outgoing
  // partons are tried.
  // xSR set to 0 means ISR, while xSR set to 1 means FSR
  inline double pTcalc(const Event& e, int i, int j, int k, int r, int xSRin) {

    // Loop over ISR and FSR if necessary
    double pTemt = -1., pTnow;
    int xSR1 = (xSRin == -1) ? 0 : xSRin;
    int xSR2 = (xSRin == -1) ? 2 : xSRin + 1;
    for (int xSR = xSR1; xSR < xSR2; xSR++) {
      // FSR flag
      bool FSR = (xSR == 0) ? false : true;

      // If all necessary arguments have been given, then directly calculate.
      // POWHEG ISR and FSR, need i and j.
      if ((pTdefMode == 0 || pTdefMode == 1) && i > 0 && j > 0) {
        pTemt = pTpowheg(e, i, j, (pTdefMode == 0) ? false : FSR);

      // Vincia ISR, need i, j and r.
      } else if (!FSR && pTdefMode == 2 && i > 0 && j > 0 && r > 0) {
        pTemt = pTvincia(e, i, j, r, FSR);

      // Vincia FSR, need k, j and r.
      } else if (FSR && pTdefMode == 2 && j > 0 && k > 0 && r > 0) {
        pTemt = pTvincia(e, k, j, r, FSR);

      // Otherwise need to try all possible combinations.
      } else {
        // Start by finding incoming legs to the hard system after
        // branching (radiator after branching, i for ISR).
        // Use partonSystemsPtr to find incoming just prior to the
        // branching and track mothers.
        int iInA = partonSystemsPtr->getInA(0);
        int iInB = partonSystemsPtr->getInB(0);
        while (e[iInA].mother1() != 1) { iInA = e[iInA].mother1(); }
        while (e[iInB].mother1() != 2) { iInB = e[iInB].mother1(); }

        // If we do not have j, then try all final-state partons
        int jNow = (j > 0) ? j : 0;
        int jMax = (j > 0) ? j + 1 : e.size();
        for (; jNow < jMax; jNow++) {

          // Final-state only
          if (!e[jNow].isFinal()) continue;
          // Exclude photons (and W/Z!)
          if (QEDvetoMode==0 && e[jNow].colType() == 0) continue;

          // POWHEG
          if (pTdefMode == 0 || pTdefMode == 1) {

            // ISR - only done once as just kinematical pT
            if (!FSR) {
              pTnow = pTpowheg(e, iInA, jNow, (pTdefMode == 0) ? false : FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : min(pTemt, pTnow);

            // FSR - try all outgoing partons from system before branching
            // as i. Note that for the hard system, there is no
            // "before branching" information.
            } else {

              int outSize = partonSystemsPtr->sizeOut(0);
              for (int iMem = 0; iMem < outSize; iMem++) {
                int iNow = partonSystemsPtr->getOut(0, iMem);

                // i != jNow and no carbon copies
                if (iNow == jNow ) continue;
                // Exclude photons (and W/Z!)
                if (QEDvetoMode==0 && e[iNow].colType() == 0) continue;
                if (jNow == e[iNow].daughter1()
                  && jNow == e[iNow].daughter2()) continue;

                pTnow = pTpowheg(e, iNow, jNow, (pTdefMode == 0)
                  ? false : FSR);
                if (pTnow > 0.) pTemt = (pTemt < 0)
                  ? pTnow : min(pTemt, pTnow);
              }
             // for (iMem)
            }
            // if (!FSR)
          // Vincia
          } else if (pTdefMode == 2) {

            // ISR - other incoming as recoiler
            if (!FSR) {
              pTnow = pTvincia(e, iInA, jNow, iInB, FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : min(pTemt, pTnow);
              pTnow = pTvincia(e, iInB, jNow, iInA, FSR);
              if (pTnow > 0.) pTemt = (pTemt < 0) ? pTnow : min(pTemt, pTnow);

            // FSR - try all final-state coloured partons as radiator
            //       after emission (k).
            } else {
              for (int kNow = 0; kNow < e.size(); kNow++) {
                if (kNow == jNow || !e[kNow].isFinal()) continue;
                if (QEDvetoMode==0 && e[kNow].colType() == 0) continue;

                // For this kNow, need to have a recoiler.
                // Try two incoming.
                pTnow = pTvincia(e, kNow, jNow, iInA, FSR);
                if (pTnow > 0.) pTemt = (pTemt < 0)
                  ? pTnow : min(pTemt, pTnow);
                pTnow = pTvincia(e, kNow, jNow, iInB, FSR);
                if (pTnow > 0.) pTemt = (pTemt < 0)
                  ? pTnow : min(pTemt, pTnow);

                // Try all other outgoing.
                for (int rNow = 0; rNow < e.size(); rNow++) {
                  if (rNow == kNow || rNow == jNow ||
                      !e[rNow].isFinal()) continue;
                  if(QEDvetoMode==0 && e[rNow].colType() == 0) continue;
                  pTnow = pTvincia(e, kNow, jNow, rNow, FSR);
                  if (pTnow > 0.) pTemt = (pTemt < 0)
                    ? pTnow : min(pTemt, pTnow);
                }
              // for (rNow)
              }
            // for (kNow)
            }
          // if (!FSR)
          }
        // if (pTdefMode)
        }
      // for (j)
      }
    }
    // for (xSR)

    return pTemt;
  }

  //--------------------------------------------------------------------------

  // Extraction of pThard based on the incoming event.
  // Assume that all the final-state particles are in a continuous block
  // at the end of the event and the final entry is the POWHEG emission.
  // If there is no POWHEG emission, then pThard is set to SCALUP.

  inline bool canVetoMPIStep()    { return true; }
  inline int  numberVetoMPIStep() { return 1; }
  inline bool doVetoMPIStep(int nMPI, const Event &e) {
    // Extra check on nMPI
    if (nMPI > 1) return false;

    // Find if there is a POWHEG emission. Go backwards through the
    // event record until there is a non-final particle. Also sum pT and
    // find pT_1 for possible MPI vetoing
    int    count = 0;
    double pT1 = 0., pTsum = 0.;
    for (int i = e.size() - 1; i > 0; i--) {
      if (e[i].isFinal()) {
        count++;
        pT1    = e[i].pT();
        pTsum += e[i].pT();
      } else break;
    }
    // Extra check that we have the correct final state
    if (count != nFinal && count != nFinal + 1) {
      cout << "Error: wrong number of final state particles in event" << endl;
      exit(1);
    }
    // Flag if POWHEG radiation present and index
    isEmt = (count == nFinal) ? false : true;
    int  iEmt  = (isEmt) ? e.size() - 1 : -1;

    // If there is no radiation or if pThardMode is 0 then set pThard = SCALUP.
    if (!isEmt || pThardMode == 0) {
      pThard = infoPtr->scalup();

    // If pThardMode is 1 then the pT of the POWHEG emission is checked against
    // all other incoming and outgoing partons, with the minimal value taken
    } else if (pThardMode == 1) {
      pThard = pTcalc(e, -1, iEmt, -1, -1, -1);

    // If pThardMode is 2, then the pT of all final-state partons is checked
    // against all other incoming and outgoing partons, with the minimal value
    // taken
    } else if (pThardMode == 2) {
      pThard = pTcalc(e, -1, -1, -1, -1, -1);
    }

    // Find MPI veto pT if necessary
    if (MPIvetoMode == 1) {
      pTMPI = (isEmt) ? pTsum / 2. : pT1;
    }

    // Initialise other variables
    accepted   = false;
    nAcceptSeq = nISRveto = nFSRveto = 0;

    // Do not veto the event
    return false;
  }

  //--------------------------------------------------------------------------

  // ISR veto

  inline bool canVetoISREmission() { return (vetoMode == 0) ? false : true; }
  inline bool doVetoISREmission(int, const Event& e, int iSys) {
    // Must be radiation from the hard system
    if (iSys != 0) return false;

    // If we already have accepted 'vetoCount' emissions in a row, do nothing
    if (vetoMode == 1 && nAcceptSeq >= vetoCount) return false;

    // Vincia radiator after, emitted and recoiler after.
    int iRadAft = -1, iEmt = -1, iRecAft = -1;
    for (int i(e.size()-1); i > 0; --i) {
      if      (iRadAft == -1 && e[i].status() == -41) iRadAft = i;
      else if (iEmt    == -1 && e[i].status() ==  43) iEmt    = i;
      else if (iRecAft == -1
        && (e[i].status() == -41 || e[i].status() == 44)) iRecAft = i;
      if (iRadAft != -1 && iEmt != -1 && iRecAft != -1) break;
    }
    if (iRadAft == -1 || iEmt == -1 || iRecAft == -1) {
      e.list();
      cout << "Error: couldn't find Vincia ISR emission" << endl;
      cout << " iRadAft = " << iRadAft << " iEmt = " << iEmt
           << " iRecAft = " << iRecAft << endl;
      exit(1);
    }

    // pTemtMode == 0: pT of emitted w.r.t. antenna.
    // pTemtMode == 1: min(pT of emitted w.r.t. all incoming/outgoing)
    // pTemtMode == 2: min(pT of all outgoing w.r.t. all incoming/outgoing)
    int xSR      = (pTemtMode == 0) ? 0       : -1;
    int i        = (pTemtMode == 0) ? iRadAft : -1;
    int j        = (pTemtMode != 2) ? iEmt    : -1;
    int k        = -1;
    int r        = (pTemtMode == 0) ? iRecAft : -1;
    double pTemt = pTcalc(e, i, j, k, r, xSR);

    // If a Born configuration, and a photon, and QEDvetoMode=2,
    //  then don't veto photons, W, or Z harder than pThard
    bool vetoParton = (!isEmt && e[iEmt].colType()==0 && QEDvetoMode==2)
      ? false: true;

    // Veto if pTemt > pThard
    if (pTemt > pThard) {
      if(!vetoParton) {
        // Don't veto ANY emissions afterwards
        nAcceptSeq = vetoCount-1;
      } else {
        nAcceptSeq = 0;
        nISRveto++;
        return true;
      }
    }

    // Else mark that an emission has been accepted and continue
    nAcceptSeq++;
    accepted = true;
    return false;
  }

  //--------------------------------------------------------------------------

  // FSR veto

  inline bool canVetoFSREmission() { return (vetoMode == 0) ? false : true; }
  inline bool doVetoFSREmission(int, const Event &e, int iSys, bool) {
    // Must be radiation from the hard system
    if (iSys != 0) return false;

    // If we already have accepted 'vetoCount' emissions in a row, do nothing
    if (vetoMode == 1 && nAcceptSeq >= vetoCount) return false;

    // Vincia radiator (before and after), emitted and recoiler (after)
    int iRecAft = e.size() - 1;
    int iEmt    = e.size() - 2;
    int iRadAft = e.size() - 3;
    int iRadBef = e[iEmt].mother1();
    if ( (e[iRecAft].status() != 51 && e[iRecAft].status() != 52) ||
      e[iEmt].status() != 51
      || e[iRadAft].status() != 51) {
      e.list();
      cout << "Error: couldn't find Vincia FSR emission" << endl;
      cout << " iRadAft = " << iRadAft << " iEmt = "
           << iEmt << " iRecAft = " << iRecAft << endl;
      exit(1);
    }

    // Behaviour based on pTemtMode:
    //  0 - pT of emitted w.r.t. radiator before
    //  1 - min(pT of emitted w.r.t. all incoming/outgoing)
    //  2 - min(pT of all outgoing w.r.t. all incoming/outgoing)
    int xSR = (pTemtMode == 0) ? 1       : -1;
    int i   = (pTemtMode == 0) ? iRadBef : -1;
    int k   = (pTemtMode == 0) ? iRadAft : -1;
    int r   = (pTemtMode == 0) ? iRecAft : -1;

    // When pTemtMode is 0 or 1, iEmt has been selected
    double pTemt = 0.;
    if (pTemtMode == 0 || pTemtMode == 1) {
      // Which parton is emitted, based on emittedMode:
      //  0 - Vincia definition of emitted
      //  1 - Vincia definition of radiated after emission
      //  2 - Random selection of emitted or radiated after emission
      //  3 - Try both emitted and radiated after emission
      int j = iRadAft;
      if (emittedMode == 0 || (emittedMode == 2 && rndmPtr->flat() < 0.5)) j++;

      for (int jLoop = 0; jLoop < 2; jLoop++) {
        if      (jLoop == 0) pTemt = pTcalc(e, i, j, k, r, xSR);
        else if (jLoop == 1) pTemt = min(pTemt, pTcalc(e, i, j, k, r, xSR));

        // For emittedMode == 3, have tried iRadAft, now try iEmt
        if (emittedMode != 3) break;
        if (k != -1) swap(j, k); else j = iEmt;
      }

    // If pTemtMode is 2, then try all final-state partons as emitted
    } else if (pTemtMode == 2) {
      pTemt = pTcalc(e, i, -1, k, r, xSR);

    }

    // If a Born configuration, and a photon, and QEDvetoMode=2,
    //  then don't veto photons, W's or Z's harder than pThard
    bool vetoParton = (!isEmt && e[iEmt].colType()==0 && QEDvetoMode==2)
      ? false: true;

    // Veto if pTemt > pThard
    if (pTemt > pThard) {
      if(!vetoParton) {
        // Don't veto ANY emissions afterwards
        nAcceptSeq = vetoCount-1;
      } else {
        nAcceptSeq = 0;
        nFSRveto++;
        return true;
      }
    }

    // Else mark that an emission has been accepted and continue
    nAcceptSeq++;
    accepted = true;
    return false;
  }

  //--------------------------------------------------------------------------

  // MPI veto

  inline bool canVetoMPIEmission() {return (MPIvetoMode == 0) ? false : true;}
  inline bool doVetoMPIEmission(int, const Event &e) {
    if (MPIvetoMode == 1) {
      if (e[e.size() - 1].pT() > pTMPI) return true;
    }
    return false;
  }

  //--------------------------------------------------------------------------

  // Functions to return information

  inline int    getNISRveto() { return nISRveto; }
  inline int    getNFSRveto() { return nFSRveto; }

  //--------------------------------------------------------------------------

 private:
  int nFinal, vetoMode, vetoCount, pThardMode, pTemtMode,
    emittedMode, pTdefMode, MPIvetoMode, QEDvetoMode;
  double pThard, pTMPI;
  bool   accepted, isEmt;
  // The number of accepted emissions (in a row)
  // Flag for PowHeg Born or Radiation
  int nAcceptSeq;
  // Statistics on vetos
  unsigned long int nISRveto, nFSRveto;

};

//==========================================================================

} // end namespace Pythia8

#endif // end Pythia8_PowhegHooksVincia_H
