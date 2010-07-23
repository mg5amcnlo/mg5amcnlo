#include <complex>
#include "need.h"
using namespace std;
void vxxxxx(double p[4],double vmass,int nhel,int nsv, complex<double> vc[6]){
	double hel,hel0,pt,pt2,pp,pzpt,emp,sqh;
	int nsvahl;
	
	sqh = pow(0.5,0.5);
	hel = double(nhel);
	nsvahl = nsv*Abs(hel);
	pt2 = pow(p[1],2)+pow(p[2],2);
	pp = Min(p[0],pow(pt2+pow(p[3],2),0.5));
	pt =Min(pp,pow(pt2,0.5));

	vc[4] = complex<double>(p[0]*nsv,p[3]*nsv);
	vc[5] = complex<double>(p[1]*nsv,p[2]*nsv);

	if (vmass != 0.0){
		hel0 = 1.0-Abs(hel);
	
		if( pp == 0.0 ){ 
			vc[0] = complex<double>(0.0,0.0);
			vc[1] = complex<double>(-hel*sqh,0.0);
			vc[2] = complex<double>(0.0,nsvahl*sqh);
			vc[3] = complex<double>(hel0,0.0);
		}
		else{
			emp = p[0]/(vmass*pp);
			vc[0] = complex<double>(hel0*pp/vmass,0.0);
			vc[3] = complex<double>(hel0*p[3]*emp+hel*pt/pp*sqh,0.0); 
			if ( pt != 0.0){
				pzpt = p[3]/(pp*pt)*sqh*hel; 
				vc[1] = complex<double>(hel0*p[1]*emp-p[1]*pzpt,-nsvahl*p[2]/pt*sqh);
				vc[2] = complex<double>(hel0*p[2]*emp - p[2]*pzpt,nsvahl*p[1]/pt*sqh);
			}
			else{
				vc[1] = complex<double>(-hel*sqh,0.0);
				vc[2] = complex<double>(0.0,nsvahl*Sgn(sqh,p[3])); 
			}

		}
	}
	else{
		
	pp = p[0];
	pt = pow(pow(p[1],2)+pow(p[2],2),0.5);
	vc[0] = complex<double>(0.0,0.0);
	vc[3] = complex<double>(hel*pt/pp*sqh,0.0);
	if (pt != 0.0) {
		pzpt = p[3]/(pp*pt)*sqh*hel;
		vc[1] = complex<double>(-p[1]*pzpt,-nsv*p[2]/pt*sqh);
		vc[2] = complex<double>(-p[2]*pzpt,nsv*p[1]/pt*sqh);
		}
	else { 
		vc[1] = complex<double>(-hel*sqh,0.0);
		vc[2] = complex<double>(0.0,nsv*Sgn(sqh,p[3]));
		}
	}
	return;
	}
