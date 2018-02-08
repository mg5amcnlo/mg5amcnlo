
// Pythia includes.
#include "Pythia8/Pythia.h"
#include "Pythia8/SigmaQCD.h"

namespace Pythia8 {

//==========================================================================
 
class SigmaDire2gg2gg : public Pythia8::Sigma2gg2gg {
public:
  SigmaDire2gg2gg() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2qqbar2qqbarNew : public Pythia8::Sigma2qqbar2qqbarNew {
public:
  SigmaDire2qqbar2qqbarNew() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2qqbar2gg : public Pythia8::Sigma2qqbar2gg {
public:
  SigmaDire2qqbar2gg() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2qq2qq : public Pythia8::Sigma2qq2qq {
public:
  SigmaDire2qq2qq() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2qg2qg : public Pythia8::Sigma2qg2qg {
public:
  SigmaDire2qg2qg() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2gg2qqbar : public Pythia8::Sigma2gg2qqbar {
public:
  SigmaDire2gg2qqbar() {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2gg2QQbar : public Pythia8::Sigma2gg2QQbar {
public:
  SigmaDire2gg2QQbar(int in1, int in2) : Pythia8::Sigma2gg2QQbar(in1,in2) {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

//==========================================================================
 
class SigmaDire2qqbar2QQbar : public Pythia8::Sigma2qqbar2QQbar {
public:
  SigmaDire2qqbar2QQbar(int in1, int in2) : Pythia8::Sigma2qqbar2QQbar(in1,in2) {}
  void store2Kin( double x1in, double x2in, double sHin,
    double tHin, double m3in, double m4in, double runBW3in,
    double runBW4in);
};

}
