#include <complex>
#include <cmath> 
#include "aloha_aux_functions.h"
using namespace std;
void ixxxxx(double p[4],double fmass,int nhel, int nsf, complex<double> fi[6]){ 
  complex<double> chi[2];
  double sf[2],sfomega[2],omega[2],pp,pp3,sqp0p3,sqm[2];
  int ip,im,nh;
  fi[0] = complex<double> (-p[0]*nsf,-p[3]*nsf);
  fi[1] = complex<double> (-p[1]*nsf,-p[2]*nsf);
  nh = nhel*nsf;
  if (fmass != 0.0) { 
    pp = min(p[0],pow((pow(p[1],2)+pow(p[2],2)+pow(p[3],2)),0.5));
    if (pp == 0.0){ 
      sqm[0] = pow(abs(fmass),0.5); 
      sqm[1] = Sgn(sqm[0],fmass);
      ip = (1+nh)/2;
      im = (1-nh)/2;
      fi[2]= ip*sqm[ip];
      fi[3] = im*nsf*sqm[ip];
      fi[4] = ip*nsf*sqm[im];
      fi[5] = im *sqm[im]; 
    }
    else{
      sf[0] = (1+nsf+(1-nsf)*nh)*0.5;
      sf[1] = (1+nsf-(1-nsf)*nh)*0.5;
      omega[0] = pow(p[0]+pp,0.5);
      omega[1] = fmass/omega[0];
      ip = (1+nh)/2;
      im = (1-nh)/2;
      sfomega[0] = sf[0]*omega[ip];
      sfomega[1] = sf[1]*omega[im];
      pp3 = max(pp+p[3], 0.0);
      chi[0] = complex<double>(pow(pp3*0.5/pp,0.5),0);
      if (pp3 == 0.0){
        chi[1] = complex<double> (-nh,0);
      }
      else{
        chi[1] = complex<double> (nh*p[1],p[2])/pow(2.0*pp*pp3,0.5);
      }
      fi[2] = sfomega[0]*chi[im];
      fi[3] = sfomega[0]*chi[ip];
      fi[4] = sfomega[1]*chi[im];
      fi[5] = sfomega[1]*chi[ip];
    }
  }
  else{ 
    if (p[1] == 0.0 and p[2] == 0.0 and p[3]<0.0){ 
      sqp0p3 = 0.0;
    }
    else{
    sqp0p3 = pow(max(p[0]+p[3],0.0),0.5)*nsf;}
    chi[0] = complex<double>(sqp0p3,0.0);
    if (sqp0p3 ==0.0){
      chi[1] = complex<double>(-nhel*pow(2.0*p[0],0.5),0.0);}
    else{
      chi[1] = complex<double>(nh*p[1],p[2])/sqp0p3;}
    if (nh == 1) {
      fi[2] = complex<double>(0.0,0.0);
      fi[3] = complex<double>(0.0,0.0);
      fi[4] = chi[0];
      fi[5] = chi[1];}
    else {
      fi[2] = chi[1];
      fi[3] = chi[0];
      fi[4] = complex<double>(0.0,0.0);
      fi[5] = complex<double>(0.0,0.0);
    }
  }
  return;} 
