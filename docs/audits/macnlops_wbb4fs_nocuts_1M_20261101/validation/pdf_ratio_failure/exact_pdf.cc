#include "Pythia8/Pythia.h"
#include "LHAPDF/LHAPDF.h"
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){
 LHAPDF::setVerbosity(0);auto d=LHAPDF::mkPDF("NNPDF40_nlo_as_01180_nf_4",0);
 Pythia8::Pythia p;p.readString("PDF:pSet = LHAPDF6:NNPDF40_nlo_as_01180_nf_4/0");auto n=p.getPDFPtr(2212);if(!n||!n->isSetup())return 2;
 const double xb=.022904539932296634,xr=.058242641107197977,t=1.7028105905461708,q2=2283.313521912757;
 std::cout<<std::setprecision(17);
 for(double q:{std::sqrt(2.)*1.2,t,1.71,1.72,1.75,std::sqrt(q2)})for(auto point:{std::make_pair(4,xb),std::make_pair(21,xr)})
 std::cout<<"EXACT "<<q<<" "<<point.first<<" "<<point.second<<" "<<d->xfxQ(point.first,point.second,q)<<" "<<n->xf(point.first,point.second,q*q)<<"\n";
 double lo=1.7,hi=1.75;for(int i=0;i<60;++i){double m=(lo+hi)/2;if(d->xfxQ(4,xb,m)>0)hi=m;else lo=m;}
 std::cout<<"ROOT "<<lo<<" "<<hi<<" "<<d->xfxQ(4,xb,lo)<<" "<<d->xfxQ(4,xb,hi)<<"\n";
 std::cout<<"INFO "<<d->qMin()<<" "<<d->qMax()<<"\n";delete d;
}
