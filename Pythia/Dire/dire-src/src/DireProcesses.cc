
#include "Dire/DireProcesses.h"

namespace Pythia8 {

//--------------------------------------------------------------------------
 
void SigmaDire2gg2gg::store2Kin( double x1in, double x2in, double sHin,
  double tHin, double m3in, double m4in, double runBW3in, double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2qqbar2qqbarNew::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2qqbar2gg::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2qq2qq::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2qg2qg::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2gg2qqbar::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2gg2QQbar::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

//==========================================================================
 
  void SigmaDire2qqbar2QQbar::store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in)
  {

  // Store inputs.
  // Default ordering of particles 3 and 4.
  swapTU   = false;
  // Incoming parton momentum fractions.
  x1Save   = x1in;
  x2Save   = x2in;
  // Incoming masses and their squares.
  bool masslessKin = (id3Mass() == 0) && (id4Mass() == 0);
  if (masslessKin) {
    m3     = 0.;
    m4     = 0.;
  } else {
    m3     = m3in;
    m4     = m4in;
  }
  mSave[3] = m3;
  mSave[4] = m4;
  s3       = m3 * m3;
  s4       = m4 * m4;
  // Standard Mandelstam variables and their squares.
  sH       = sHin;
  tH       = tHin;
  uH       = (masslessKin) ? -(sH + tH) : s3 + s4 - (sH + tH);
  mH       = sqrt(sH);
  sH2      = sH * sH;
  tH2      = tH * tH;
  uH2      = uH * uH;
  // The nominal Breit-Wigner factors with running width.
  runBW3   = runBW3in;
  runBW4   = runBW4in;
  // Calculate squared transverse momentum.
  pT2 = (masslessKin) ?  tH * uH / sH : (tH * uH - s3 * s4) / sH;

  // Scale setting for dijet process different from default Pythia.
  double mu2 = -1./ (1/sH + 1/tH + 1/uH) / 2; 
  Q2RenSave = renormMultFac*mu2;
  Q2FacSave = factorMultFac*mu2;

  // Evaluate alpha_strong and alpha_EM.
  alpS  = couplingsPtr->alphaS(Q2RenSave);
  alpEM = couplingsPtr->alphaEM(Q2RenSave);

}

}
