// Independent massless-II quadrature using the installed PYTHIA PDF and
// AlphaStrong. This diagnoses the forward-table / backward-shower difference;
// it is not part of event generation. Nodes and weights are Gauss-Legendre.
#include "Pythia8/Pythia.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

static std::unique_ptr<Pythia8::Pythia> engine;
static Pythia8::PDFPtr pdf;
static Pythia8::AlphaStrong coupling;

extern "C" int lowpt_init(const char* xml, const char* set) {
  engine.reset(new Pythia8::Pythia(xml, false));
  if (!engine->readString(std::string("PDF:pSet = ") + set)) return 1;
  pdf = engine->getPDFPtr(2212);
  coupling.init(.1365, 1, 5, false);
  return !pdf || !pdf->isSetup();
}

extern "C" double lowpt_xf(int pid, double x, double q) {
  return x >= 1. ? 0. : pdf->xf(pid, x, q*q);
}

// Integrate the backward emission exponent B, PDF-free forward exponent F,
// and finite-z boundary term C. A gluon has two colour lines in the table;
// these exponents always refer to the whole leg.
extern "C" void lowpt_leg(int pid, double x, double q, double Q, double M,
                           int n, const double* nodes, const double* weights,
                           double* result) {
  const double pi = std::acos(-1.), ca = 3., cf = 4./3., tr = .5;
  std::fill(result, result+4, 0.);
  if (q >= Q) { result[3] = 1.; return; }
  std::vector<double> edges = {q};
  for (double a : {1.5, 1.65, 4.8}) if (q < a && a < Q) edges.push_back(a);
  edges.push_back(Q);
  for (std::size_t panel=1; panel<edges.size(); ++panel) {
    const double lo=2.*std::log(edges[panel-1]), hi=2.*std::log(edges[panel]);
    // Split long logarithmic intervals; the z integral uses a logit variable.
    const int panels = std::max(1, int(std::ceil((hi-lo)/2.)));
    for (int p=0; p<panels; ++p) {
      const double a=lo+(hi-lo)*p/panels, b=lo+(hi-lo)*(p+1)/panels;
      for (int i=0; i<n; ++i) {
        const double t=std::exp(.5*(a+b)+.5*(b-a)*nodes[i]);
        const double r=t/(M*M), omz=.5*r*(std::sqrt(1.+4./r)-1.), Z=1.-omz;
        const double den=pdf->xf(pid,x,t), alpha=coupling.alphaS(t)/(2.*pi);
        double backward=0., boundary=0.;
        auto pgg=[&](double z) { return 2.*ca*std::pow(1.-z+z*z,2)/(z*(1.-z)); };
        auto pqq=[&](double z) { return cf*(1.+z*z)/(1.-z); };
        auto real=[&](double z) {
          if (pid!=21) return (pqq(z)*pdf->xf(pid,x/z,t)
            +tr*(z*z+(1.-z)*(1.-z))*pdf->xf(21,x/z,t))/den;
          double sum=0.;
          for (int f=-5; f<=5; ++f) if (f) sum+=pdf->xf(f,x/z,t);
          return (pgg(z)*pdf->xf(21,x/z,t)+cf*(1.+(1.-z)*(1.-z))/z*sum)/den;
        };
        if (Z>x) {
          const double l=std::log(x/(1.-x)), u=std::log(Z/omz);
          for (int j=0; j<n; ++j) {
            const double v=.5*(l+u)+.5*(u-l)*nodes[j], z=1./(1.+std::exp(-v));
            backward+=.5*(u-l)*weights[j]*z*(1.-z)*real(z);
          }
        }
        const double lower=std::max(x,Z);
        for (int j=0; j<n; ++j) {
          const double z=1.-.5*(1.-lower)*(nodes[j]+1.);
          boundary+=.5*(1.-lower)*weights[j]*(real(z)-(pid==21?pgg(z):pqq(z)));
        }
        const int nf=t>4.8*4.8?5:t>1.5*1.5?4:3;
        double forward;
        if (pid==21) {
          forward=2.*ca*(-std::log(omz)-Z*Z+Z*Z*Z/3.-std::pow(Z,4)/4.)
            +2.*nf*tr*(Z*Z/2.-2.*Z*Z*Z/3.+std::pow(Z,4)/2.);
          auto primitive=[&](double z) {return 2.*ca*(std::log(z)-std::log(1.-z)-2.*z+.5*z*z-z*z*z/3.);};
          if (Z<x) boundary-=primitive(x)-primitive(Z);
          boundary+=2.*ca*(-std::log(Z)-2.*(1.-Z)+1.5*(1.-Z*Z)
            -2./3.*(1.-Z*Z*Z)+.25*(1.-std::pow(Z,4)))
            -2.*nf*tr*(1./3.-Z*Z/2.+2.*Z*Z*Z/3.-std::pow(Z,4)/2.);
        } else {
          forward=cf*(-2.*std::log(omz)-1.5*Z*Z);
          auto primitive=[&](double z) {return cf*(-2.*std::log(1.-z)-z-.5*z*z);};
          if (Z<x) boundary-=primitive(x)-primitive(Z);
          boundary+=cf*Z*omz;
        }
        const double weight=.5*(b-a)*weights[i]*alpha;
        result[0]+=weight*backward;
        result[1]+=weight*forward;
        result[2]+=weight*boundary;
      }
    }
  }
  result[3]=pdf->xf(pid,x,q*q)/pdf->xf(pid,x,Q*Q);
}
