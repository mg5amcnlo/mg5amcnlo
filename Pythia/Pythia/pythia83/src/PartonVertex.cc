// PartonVertex.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the PartonVertex class.

#include "Pythia8/PartonVertex.h"

namespace Pythia8 {

//==========================================================================

// The PartonVertex class.

//--------------------------------------------------------------------------

// Find relevant settings.

void PartonVertex::init() {

    doVertex      = flag("PartonVertex:setVertex");
    modeVertex    = mode("PartonVertex:modeVertex");
    epsPhi        = parm("PartonVertex:phiAsym");
    epsRat        = sqrt( (1. + epsPhi) / (1. - epsPhi) );
    rProton       = parm("PartonVertex:ProtonRadius");
    rProton2      = rProton * rProton;
    pTmin         = parm("PartonVertex:pTmin");
    widthEmission = parm("PartonVertex:EmissionWidth");

}

//--------------------------------------------------------------------------

// Set vertices for a beam particle and the beam remnants.

void PartonVertex::vertexBeam( int iBeam, vector<int>& iRemn,
  vector<int>& iInit, Event& event) {

  // Set vertex for incoming beam particle itself.
  double xBeam = (iBeam == 0) ? bHalf : -bHalf;
  Vec4   vBeam( xBeam, 0., 0., 0.);
  event[iBeam + 1].vProd( FM2MM * vBeam );

  // Variables for further use.
  vector<Vec4>   vNow;
  Vec4           vSum;
  vector<double> wtNow;
  double         wtSum = 0.;
  double         x, y, r, phi, sthe, xSgn, xWt;
  pair<double,double> xy;

  // Loop over all remnants and set their location relative to the beam.
  for (int i = 0; i < int(iRemn.size()); ++i) {

    // Sample according to sphere.
    if (modeVertex < 2) {
      r    = rProton * pow(rndmPtr->flat(), 1./3.);
      phi  = 2. * M_PI * rndmPtr->flat();
      sthe = sqrtpos( 1. - pow2(2. * rndmPtr->flat() - 1.) );
      x    = r * sthe * cos(phi);
      y    = r * sthe * sin(phi);

    // Sample according to Gaussian.
    } else {
      xy = rndmPtr->gauss2();
      x  = xy.first  * rProton / sqrt(3.);
      y  = xy.second * rProton / sqrt(3.);
    }

    // Save. Calculate energy-weighted center and displacement weight.
    vNow.push_back( Vec4( x, y, 0., 0.) );
    vSum  += event[iRemn[i]].e() * vNow[i];
    xSgn   = (iBeam == 0) ? x : -x;
    xWt    =  1. / ( 1. + (bNow / rProton) * exp(xSgn / rProton) );
    wtNow.push_back( xWt );
    wtSum += event[iRemn[i]].e() * xWt;
  }

  // Add initiator energy-weighted center relative to proton center.
  for (int i = 0; i < int(iInit.size()); ++i)
    vSum += event[iInit[i]].e() * (MM2FM * event[iInit[i]].vProd() - vBeam);

  // Distribute recoil among remnants to ensure that center is centered.
  // (But scale down if suspiciously large shift.)
  Vec4 vShift;
  for (int i = 0; i < int(iRemn.size()); ++i) {
    vShift = vSum * wtNow[i] / wtSum;
    if (vShift.pT2() > rProton2) vShift *= rProton / vShift.pT();
    event[iRemn[i]].vProd( FM2MM * (vNow[i] - vShift + vBeam) );
  }

}

//--------------------------------------------------------------------------

// Select vertex for an MPI.

void PartonVertex::vertexMPI( int iBeg, int nAdd, double bNowIn,
  Event& event) {

  // Convert the impact parameter to physical units. Prepare selection.
  bNow  = bNowIn * rProton;
  bHalf = 0.5 * bNow;
  if (modeVertex < 2) {
    if (bHalf > 0.95 * rProton) {
      infoPtr->errorMsg("Warning in PartonVertex::vertexMPI: large b value");
      bHalf = 0.95 * rProton;
    }
    xMax   = rProton - bHalf;
    yMax   = sqrt( rProton2 - bHalf * bHalf);
    zWtMax = yMax * yMax;
  }
  double x = 0.;
  double y = 0.;

  // Sample x and y inside a box, and then require it to be within sphere.
  if (modeVertex < 2) {
    bool accept = false;
    while (!accept) {
      x   = (2. * rndmPtr->flat() - 1.) * xMax;
      y   = (2. * rndmPtr->flat() - 1.) * yMax;
      double rA2 = pow2(x - bHalf) + y * y;
      double rB2 = pow2(x + bHalf) + y * y;
      if ( max( rA2, rB2) < rProton2 ) accept = true;
      if (accept && sqrtpos(rProton2 - rA2) * sqrtpos(rProton2 - rB2)
        < rndmPtr->flat() * zWtMax) accept = false;
    }

  // Sample x and y according to two-dimensional Gaussian.
  } else {
    bool accept = false;
    while (!accept) {
      pair<double,double> xy = rndmPtr->gauss2();
      x = xy.first  * rProton / sqrt(6.);
      y = xy.second * rProton / sqrt(6.);
      if (modeVertex == 2) accept = true;
      // Option with elliptic shape.
      else if (modeVertex == 3) { x *= epsRat; y /= epsRat; accept = true;}
      // Option with azimuthal distribution 1 + epsilon * cos(2 * phi).
      else if ( 1. + epsPhi * (x*x - y*y)/(x*x + y*y)
        > rndmPtr->flat() * (1. + abs(epsPhi)) ) accept = true;
    }
  }

  // Set production vertices.
  for (int iNow = iBeg; iNow < iBeg + nAdd; ++iNow)
    event[iNow].vProd( x * FM2MM, y * FM2MM, 0., 0.);

}

//--------------------------------------------------------------------------

// Select vertex for an FSR branching.

void PartonVertex::vertexFSR( int iNow, Event& event) {

  // Start from known vertex, or mother one.
  int iMo = event[iNow].mother1();
  Vec4 vStart = event[iNow].hasVertex() ? event[iNow].vProd()
              : event[iMo].vProd();

  // Add Gaussian smearing.
  double pT = max( event[iNow].pT(), pTmin);
  pair<double, double> xy = rndmPtr->gauss2();
  Vec4 vSmear = (widthEmission / pT) * Vec4( xy.first, xy.second, 0., 0.);
  event[iNow].vProd( vStart + vSmear * FM2MM);

}

//--------------------------------------------------------------------------

// Select vertex for an ISR branching.

void PartonVertex::vertexISR( int iNow, Event& event) {

  // Start from known vertex or mother/daughter one.
  int iMoDa = event[iNow].mother1();
  if (iMoDa == 0) iMoDa = event[iNow].daughter1();
  Vec4 vStart = event[iNow].vProd();
  if (!event[iNow].hasVertex() && iMoDa != 0) vStart = event[iMoDa].vProd();

  // Add Gaussian smearing.
  double pT = max( event[iNow].pT(), pTmin);
  pair<double, double> xy = rndmPtr->gauss2();
  Vec4 vSmear = (widthEmission / pT) * Vec4( xy.first, xy.second, 0., 0.);
  event[iNow].vProd( vStart + vSmear * FM2MM);

}

//--------------------------------------------------------------------------

// Propagate parton vertex information to hadrons.
// Still to be improved for closed gluon loops and junction topologies.

void PartonVertex::vertexHadrons( int nBefFrag, Event& event) {

  // Identify known cases and else return.
  int iFirst = event[nBefFrag].mother1();
  int iLast  = event[nBefFrag].mother2();
  vector<int> iNotG;
  for (int i = iFirst; i <= iLast; ++i)
    if (!event[i].isGluon()) iNotG.push_back(i);
  if ( iNotG.size() == 2 && event[iFirst].col() * event[iLast].col() == 0
    && event[iFirst].acol() * event[iLast].acol() == 0) {
  } else if (iNotG.size() == 0 || iNotG.size() == 3) {
  } else {
    infoPtr->errorMsg("Error in PartonVertex::vertexHadrons: "
      "unknown colour topology not handled");
    return;
  }

  // Use middle of endpoints for collapse to single hadron.
  if (event[iFirst].daughter2() == event[iFirst].daughter1()) {
    Vec4 vShift = 0.5 * (event[iFirst].vProd() + event[iLast].vProd());
    event[nBefFrag].vProdAdd( vShift);
    return;
  }

  // Simple qqbar strings or closed gluon loops.
  // Warning: for the latter still to transfer info on first breakup,
  // so not correct as is, but still good enough for smearing purposes?
  if (iNotG.size() == 2 || iNotG.size() == 0) {

    // Initial values and variables.
    int iBeg = iFirst;
    int iEnd = iBeg + 1;
    double eBeg = event[iBeg].e();
    double eEnd = (iEnd < iLast && event[iEnd].isGluon() ? 0.5 : 1.)
      * event[iEnd].e();
    double ePartonSum = eBeg + eEnd;
    double eHadronSum = 0.;

    // Loop over new primary hadrons. Midpoint energy of new hadron.
    for (int i = nBefFrag; i < event.size(); ++i) {
      eHadronSum += 0.5 * event[i].e();

      // Step up to parton pair that spans hadron midpoint energy.
      while (eHadronSum > ePartonSum && iEnd < iLast) {
        eHadronSum -= ePartonSum;
        ++iBeg;
        ++iEnd;
        eBeg = eEnd;
        eEnd = (event[iEnd].isGluon() ? 0.5 : 1.) * event[iEnd].e();
        ePartonSum = eBeg + eEnd;
      }

      // Add weighted average of parton vertices to hadron vertex.
      double eFrac = clamp( eHadronSum / ePartonSum, 0., 1.);
      Vec4 vShift  = (1. - eFrac) * event[iBeg].vProd()
                   + eFrac * event[iEnd].vProd();
      event[i].vProdAdd( vShift);
      eHadronSum  += 0.5 * event[i].e();
    }

    // Done for simple cases.
    return;
  }

  // Junction systems: identify order of hadrons produced.
  int iOrdSys[3] = {5, 5, 5};
  for (int k = 0; k < 3; ++k) {
    int col = max( event[iNotG[k]].col(), event[iNotG[k]].acol());
    for (int i = 0; i < event.sizeJunction(); ++i)
    for (int j = 0; j < 3; ++j)
    if (event.endColJunction( i, j) == col) {
      int statusLeg = event.statusJunction( i, j);
      if (statusLeg == 85) iOrdSys[0] = k;
      else if (statusLeg == 86) iOrdSys[1] = k;
      else iOrdSys[2] = k;
    }
  }

  // Repair cases where one colour end is not identified.
  if (iOrdSys[0] + iOrdSys[1] +iOrdSys[2] == 3) ;
  else if (iOrdSys[0] == 5 && iOrdSys[1] + iOrdSys[2] < 4)
    iOrdSys[0] = 3 - iOrdSys[1] - iOrdSys[2];
  else if (iOrdSys[1] == 5 && iOrdSys[0] + iOrdSys[2] < 4)
    iOrdSys[1] = 3 - iOrdSys[0] - iOrdSys[2];
  else if (iOrdSys[2] == 5 && iOrdSys[0] + iOrdSys[1] < 4)
    iOrdSys[2] = 3 - iOrdSys[0] - iOrdSys[1];
  else {
    infoPtr->errorMsg("Warning in PartonVertex::vertexHadrons: "
      "unidentified junction topology not handled");
    return;
  }

  // Initial values for the two lowest-energy legs.
  int nDone = nBefFrag;
  for (int leg = 0; leg < 2; ++leg) {
    int nStop = (iOrdSys[leg] == 0) ? iFirst : iNotG[iOrdSys[leg] - 1] + 1;
    int iBeg  = iNotG[iOrdSys[leg]];
    int iEnd  = max(iBeg - 1, nStop);
    double eBeg = event[iBeg].e();
    double eEnd = (event[iEnd].isGluon() ? 0.5 : 1.) * event[iEnd].e();
    double ePartonSum = eBeg + eEnd;
    double eHadronSum = 0.;
    int statusNow = (leg == 0) ? 85 : 86;

    // Loop over primary hadrons in two lowest-energy legs.
    for (int i = nDone; i < event.size(); ++i) {
      if (event[i].status() != statusNow) {
        nDone = i;
        break;
      }
      eHadronSum += 0.5 * event[i].e();

      // Step up to parton pair that spans hadron midpoint energy.
      while (eHadronSum > ePartonSum && iEnd > nStop) {
        eHadronSum -= ePartonSum;
        --iBeg;
        --iEnd;
        eBeg = eEnd;
        eEnd = (event[iEnd].isGluon() ? 0.5 : 1.) * event[iEnd].e();
        ePartonSum = eBeg + eEnd;
      }

      // If only one parton left then set entirely by it, else weight.
      if (eHadronSum > ePartonSum || iBeg == nStop) {
        event[i].vProdAdd( event[nStop].vProd() );
      } else {
        double eFrac = clamp( eHadronSum / ePartonSum, 0., 1.);
        Vec4 vShift  = (1. - eFrac) * event[iBeg].vProd()
                     + eFrac * event[iEnd].vProd();
        event[i].vProdAdd( vShift);
      }
      eHadronSum  += 0.5 * event[i].e();
    }
  }

  // Initial values for last leg.
  int nStop = (iOrdSys[2] == 0) ? iFirst : iNotG[iOrdSys[2] - 1] + 1;
  int iBeg  = iNotG[iOrdSys[2]];
  int iEnd  = max(iBeg - 1, nStop);
  double eBeg = event[iBeg].e();
  double eEnd = (event[iEnd].isGluon() ? 0.5 : 1.) * event[iEnd].e();
  double ePartonSum = eBeg + eEnd;
  double eHadronSum = 0.;

  // Loop over primary hadrons in last leg.
  for (int i = nDone; i < event.size(); ++i) {
    eHadronSum += 0.5 * event[i].e();

    // Step up to parton pair that spans hadron midpoint energy.
    while (eHadronSum > ePartonSum && iEnd > nStop) {
      eHadronSum -= ePartonSum;
      --iBeg;
      --iEnd;
      eBeg = eEnd;
      eEnd = (event[iEnd].isGluon() ? 0.5 : 1.) * event[iEnd].e();
      ePartonSum = eBeg + eEnd;
    }

    // If only one parton left then set entirely by it, else weight.
    // Warning: could do better for junction baryon and hadron after it.
    if (eHadronSum > ePartonSum) {
      event[i].vProdAdd( event[nStop].vProd() );
    } else {
      double eFrac = clamp( eHadronSum / ePartonSum, 0., 1.);
      Vec4 vShift  = (1. - eFrac) * event[iBeg].vProd()
                   + eFrac * event[iEnd].vProd();
      event[i].vProdAdd( vShift);
    }
    eHadronSum  += 0.5 * event[i].e();
  }

}


//==========================================================================

} // end namespace Pythia8
