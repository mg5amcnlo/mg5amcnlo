// PartonVertex.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for vertex information during the parton-level evolution.

#ifndef Pythia8_PartonVertex_H
#define Pythia8_PartonVertex_H

#include "Pythia8/Basics.h"
#include "Pythia8/Event.h"
#include "Pythia8/Info.h"
#include "Pythia8/PhysicsBase.h"
#include "Pythia8/PythiaStdlib.h"
#include "Pythia8/Settings.h"

namespace Pythia8 {

//==========================================================================

// The PartonVertex class sets parton-level vertex information.

class PartonVertex : public PhysicsBase {

public:

  // Constructor.
  PartonVertex() : doVertex(), modeVertex(), epsPhi(), epsRat(), rProton(),
    rProton2(), pTmin(), widthEmission(), bNow(), bHalf(), xMax(), yMax(),
    zWtMax() {}

  // Destructor.
  virtual ~PartonVertex() {}

  // Initialize a few parameters from Settings.
  virtual void init();

  // Select vertex for a beam particle.
  virtual void vertexBeam( int iBeam, vector<int>& iRemn, vector<int>& iInit,
    Event& event);

  // Select vertex for an MPI.
  virtual void vertexMPI( int iBeg, int nAdd, double bNowIn, Event& event);

  // Select vertex for an FSR branching.
  virtual void vertexFSR( int iNow, Event& event);

  // Select vertex for an ISR branching.
  virtual void vertexISR( int iNow, Event& event);

  // Propagate parton vertex information to hadrons.
  virtual void vertexHadrons( int nBefFrag, Event& event);

private:

  // Data related to currently implemented models.
  bool   doVertex;
  int    modeVertex;
  double epsPhi, epsRat, rProton, rProton2, pTmin, widthEmission;

  // Current values.
  double bNow, bHalf, xMax, yMax, zWtMax;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_PartonVertex_H
