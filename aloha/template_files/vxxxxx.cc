#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;
void vxxxxx(double p[4],double vmass,int nhel,int nsv, complex<double> vc[6]){
  double hel,hel0,pt,pt2,pp,pzpt,emp,sqh;
  int nsvahl;
  sqh = sqrt(0.5);
  hel = double(nhel);
  nsvahl = nsv*std::abs(hel);
  pt2 = (p[1]*p[1])+(p[2]*p[2]);
  pp = min(p[0],sqrt(pt2+(p[3]*p[3])));
  pt =min(pp,sqrt(pt2));
  vc[0] = complex<double>(p[0]*nsv,p[3]*nsv);
  vc[1] = complex<double>(p[1]*nsv,p[2]*nsv);
  if (vmass != 0.0){
    hel0 = 1.0-std::abs(hel);
    if( pp == 0.0 ){ 
      vc[2] = complex<double>(0.0,0.0);
      vc[3] = complex<double>(-hel*sqh,0.0);
      vc[4] = complex<double>(0.0,nsvahl*sqh);
      vc[5] = complex<double>(hel0,0.0);
    }
    else{
      emp = p[0]/(vmass*pp);
      vc[2] = complex<double>(hel0*pp/vmass,0.0);
      vc[5] = complex<double>(hel0*p[3]*emp+hel*pt/pp*sqh,0.0); 
      if ( pt != 0.0){
        pzpt = p[3]/(pp*pt)*sqh*hel; 
        vc[3] = complex<double>(hel0*p[1]*emp-p[1]*pzpt,-nsvahl*p[2]/pt*sqh);
        vc[4] = complex<double>(hel0*p[2]*emp - p[2]*pzpt,nsvahl*p[1]/pt*sqh);
      }
      else{
        vc[3] = complex<double>(-hel*sqh,0.0);
        vc[4] = complex<double>(0.0,nsvahl*Sgn(sqh,p[3])); 
      }
    }
  }
  else{
    pp = p[0];
    pt = sqrt((p[1]*p[1])+(p[2]*p[2]));
    vc[2] = complex<double>(0.0,0.0);
    vc[5] = complex<double>(hel*pt/pp*sqh,0.0);
    if (pt != 0.0) {
      pzpt = p[3]/(pp*pt)*sqh*hel;
      vc[3] = complex<double>(-p[1]*pzpt,-nsv*p[2]/pt*sqh);
      vc[4] = complex<double>(-p[2]*pzpt,nsv*p[1]/pt*sqh);
    }
    else { 
      vc[3] = complex<double>(-hel*sqh,0.0);
      vc[4] = complex<double>(0.0,nsv*Sgn(sqh,p[3]));
    }
  }
  return;
}
