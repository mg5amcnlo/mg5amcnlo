#include "Pythia8/Pythia.h"

class Analysis {

  public:

  Analysis() : sumOfWeights(0.) {}

  map<std::string, Pythia8::Hist> histograms;
  double sumOfWeights;

  virtual void init() = 0;
  virtual void fill(const Event&, double) = 0;
  virtual void fill(const Event&, const Event&, double) = 0;
  virtual void fill(const Event&, const vector<double>) = 0;
  virtual void fill(const Event&, const Event&,
    const vector<double>) = 0;
  virtual void finalize() = 0;
  virtual void print() = 0;

};
