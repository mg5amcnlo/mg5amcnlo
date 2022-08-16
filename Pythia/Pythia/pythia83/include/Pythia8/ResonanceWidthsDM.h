// ResonanceWidthsDM.h is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for DM resonance properties: dynamical widths etc.
// ResonanceS, ResonanceZp, ResonanceS1, ResonanceCha, ResonanceDM2,
// ResonanceChaD: derived classes for individual resonances.

#ifndef Pythia8_ResonanceWidthsDM_H
#define Pythia8_ResonanceWidthsDM_H

#include "Pythia8/Settings.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8/ResonanceWidths.h"

namespace Pythia8 {

//==========================================================================

// The ResonanceS class. (S a.k.a. DMmed(s=0), PDG id 54.)

class ResonanceS : public ResonanceWidths {

public:

  // Constructor and destructor.
  ResonanceS(int idResIn) : ResonanceWidths(), gq(), gX(), pScalar()
    {initBasic(idResIn);}
  virtual ~ResonanceS() {}

private:

  // Couplings etc.
  double gq, gX;
  bool pScalar;

  // Initialize constants.
  virtual void initConstants() override;

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

  // Loop integral for H -> gg coupling.
  virtual double eta2gg();

};

//==========================================================================

// The ResonanceZp class. (Zp a.k.a. DMmed(s=1), PDG id 55.)

class ResonanceZp : public ResonanceWidths {

public:

  // Constructor.
  ResonanceZp(int idResIn) : ResonanceWidths(), kinMix(), gZp(), eps(), vX(),
    aX(), vu(), vd(), vl(), vv(), au(), ad(), al(), av() {initBasic(idResIn);}

private:

  // Couplings etc.
  bool kinMix;
  double gZp, eps, vX, aX, vu, vd, vl, vv, au, ad, al, av;

  // Initialize constants.
  virtual void initConstants() override;

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

};

//==========================================================================

// Charged scalar partner of DM (PDG id 56.)

class ResonanceSl : public ResonanceWidths {

public:

  // Constructor.
  ResonanceSl(int idResIn) : yuk() {initBasic(idResIn);}

private:

  // Couplings etc.
  double yuk[4];

  // Initialize constants.
  virtual void initConstants() override;

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

};

//==========================================================================

// Charged partner of DM (PDG id 57.)

class ResonanceCha : public ResonanceWidths {

public:

  // Constructor.
  ResonanceCha(int idResIn) : ResonanceWidths(), mixN1(), mixN2(), mixing(),
    doDY() {initBasic(idResIn);}

protected:

  // Couplings etc.
  double mixN1, mixN2, mixing;
  bool   doDY;

  // Set masses and mixing from settings.
  void setMassMix();

private:

  // Initialize constants.
  virtual void initConstants() override {setMassMix();}

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

};

//==========================================================================

// Neutral Charged partner of DM (PDG id 58.)
// Not yet implemented.

class ResonanceDM2 : public ResonanceCha {

public:

  // Constructor.
  ResonanceDM2(int idResIn) : ResonanceCha(idResIn), mHiggs(), wHiggs()
    {initBasic(idResIn);}

private:

  // Couplings etc.
  double mHiggs, wHiggs;

  // Initialize constants.
  virtual void initConstants() override;

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

};

//==========================================================================

// Doubly Charged partner of DM (PDG id 59.)

class ResonanceChaD : public ResonanceCha {

public:

  // Constructor.
  ResonanceChaD(int idResIn) : ResonanceCha(idResIn) {initBasic(idResIn);}

private:

  // Initialize constants.
  virtual void initConstants() override {setMassMix();}

  // Calculate various common prefactors for the current mass.
  virtual void calcPreFac(bool = false) override;

  // Caclulate width for currently considered channel.
  virtual void calcWidth(bool calledFromInit = false) override;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_ResonanceWidthsDM_H
