#include "Pythia8/Pythia.h"
#include "LHAPDF/LHAPDF.h"
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){
 LHAPDF::setVerbosity(0);auto direct=LHAPDF::mkPDF("NNPDF40_nlo_as_01180_nf_4",0);
 Pythia8::Pythia p;p.readString("PDF:pSet = LHAPDF6:NNPDF40_nlo_as_01180_nf_4/0");
 auto native=p.getPDFPtr(2212);
 if(!native || !native->isSetup())return 2;
 const double x=2.*148.879509559928124/13000.;
 std::cout<<std::setprecision(17)<<"PDFSCAN x Q id LHAPDFraw LHAPDFfrozen Pythia_xf Pythia_val Pythia_sea\n";
 for(double q:{.5,1.,1.2,1.5,1.65,std::sqrt(2.)*1.2,1.7,1.75,1.8,2.,2.5,3.,4.,5.,10.,50.})for(int id:{4,-4,21,3,-3})
 std::cout<<"PDFSCAN "<<x<<" "<<q<<" "<<id<<" "<<direct->xfxQ(id,x,q)<<" "<<direct->xfxQ(id,x,std::max(1.65,q))<<" "<<native->xf(id,x,q*q)<<" "<<native->xfVal(id,x,q*q)<<" "<<native->xfSea(id,x,q*q)<<"\n";
 delete direct;
}
