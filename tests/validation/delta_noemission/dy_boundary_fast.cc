// Prototype fixed-quadrature light-quark PDF boundary correction.
// Diagnostic only: massless II kinematics and a five-flavour one-loop coupling.
#include "Pythia8/Pythia.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
static std::unique_ptr<Pythia8::Pythia> pythia;
static Pythia8::PDFPtr pdf;
extern "C" int boundary_init(const char* xml) {
  pythia.reset(new Pythia8::Pythia(xml,false));
  if (!pythia->readString("PDF:pSet = 13")) return 1;
  pdf=pythia->getPDFPtr(2212);
  return (!pdf || !pdf->isSetup()) ? 2 : 0;
}
extern "C" double boundary_xf(int pid,double x,double t) {
  return x>=1. ? 0. : pdf->xf(pid,x,t);
}
extern "C" double boundary_eval(int pid,double x,double q,double Q,double M,
                                int n,const double* nodes,const double* weights) {
  if(q>=Q)return 0.;
  const double cf=4./3., pi=std::acos(-1.), beta=23./(12.*pi);
  const double lambda2=91.188*91.188*std::exp(-1./(beta*.130));
  const double lo=std::log(q*q),hi=std::log(Q*Q),half=.5*(hi-lo),mid=.5*(hi+lo);
  double result=0.;
  for(int i=0;i<n;++i){
    double t=std::exp(mid+half*nodes[i]),r=t/(M*M);
    double Z=1.-.5*r*(std::sqrt(1.+4./r)-1.);
    double den=pdf->xf(pid,x,t);
    if(!(den>0.))return std::numeric_limits<double>::quiet_NaN();
    double inner=0.,lower=std::max(Z,x);
    // The densities vanish at x/z>=1. Integrate that strip analytically,
    // leaving a smooth supported PDF convolution for the fixed quadrature.
    if(lower>Z){
      auto primitive=[&](double z){return cf*(-2.*std::log(1.-z)-z-.5*z*z);};
      inner-=primitive(lower)-primitive(Z);
    }
    for(int j=0;j<n;++j){
      double u=.5*(nodes[j]+1.),z=1.-(1.-lower)*u,y=x/z;
      double fq=y<1.?pdf->xf(pid,y,t):0.;
      double fg=y<1.?pdf->xf(21,y,t):0.;
      double diagonal=cf*(1.+z*z)/(1.-z)*(fq/den-1.);
      double gluon=.5*(z*z+(1.-z)*(1.-z))*fg/den;
      inner+=.5*weights[j]*(1.-lower)*(diagonal+gluon);
    }
    double rate=(cf*Z*(1.-Z)+inner)/(2.*pi*beta*std::log(t/lambda2));
    result+=half*weights[i]*rate;
  }
  return result;
}
extern "C" double boundary_benchmark(int cases,const int* pid,const double* x,
  const double* q,const double* Q,const double* M,int n,const double* nodes,
  const double* weights,int repetitions,double* checksum) {
  auto start=std::chrono::steady_clock::now();
  double sum=0.;
  for(int i=0;i<repetitions;++i){
    int k=i%cases;
    sum+=boundary_eval(pid[k],x[k],q[k],Q[k],M[k],n,nodes,weights);
  }
  *checksum=sum;
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
