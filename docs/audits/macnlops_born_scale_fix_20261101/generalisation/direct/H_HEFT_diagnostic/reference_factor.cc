#include <LHAPDF/LHAPDF.h>
#include <memory>
#include <cmath>
#include <limits>
namespace { std::unique_ptr<LHAPDF::PDF> pdf; }
extern "C" int init_born_scale_pdf() {
 try { pdf.reset(LHAPDF::mkPDF("NNPDF40_nlo_as_01180",0));return 0; }
 catch(...) {return 1;}
}
extern "C" int born_scale_factors(long n,const int* ids,const double* x,
 const double* muBorn,const double* muReal,const double* alphaK,
 double* values) {
 if(!pdf)return 1;
 try {
 for(long i=0;i<n;++i) {
   double* out=values+6*i;
   for(int j=0;j<6;++j)out[j]=std::numeric_limits<double>::quiet_NaN();
   if(!(muBorn[i]>0.&&muReal[i]>0.&&alphaK[i]>0.))continue;
   const double aB=pdf->alphasQ(muBorn[i]);
   const double aR=pdf->alphasQ(muReal[i]);
   double lumB=1.,lumR=1.;
   for(int side=0;side<2;++side) {
      const int j=2*i+side;
      lumB*=pdf->xfxQ(ids[j],x[j],muBorn[i]);
      lumR*=pdf->xfxQ(ids[j],x[j],muReal[i]);
   }
   if(!(lumB>0.&&lumR>0.))return 2;
   out[0]=aB*aB/(alphaK[i]*alphaK[i]);
   out[1]=lumB/lumR;
   out[2]=out[0]*out[1];
   out[3]=aB;out[4]=aR;out[5]=aR/alphaK[i]-1.;
 }
 return 0;
 }catch(...){return 3;}
}
