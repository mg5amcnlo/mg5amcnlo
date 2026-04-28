#include <cmath>
using namespace std;

double Sgn(double a,double b){ 
  return (b<0)?-abs(a):abs(a);
}


std::ostream& operator<<(std::ostream& os, const ALOHAOBJ& obj) {
    os << "p = [ ";
    for (int i = 0; i < 4; ++i) os << obj.p[i] <<  " " ;
    os << " ]\n";
    os << "W = [ ";
    for (int i = 0; i < 4; ++i) os << obj.W[i] << " " ;
    os << " ]\n";
    return os;
}
#include <complex>
#include <cmath> 
#include "aloha_aux_functions.h"
using namespace std;
void ixxxxx(double p[4], double fmass, int nhel, int nsf, int flv, ALOHAOBJ &fi)
{
  complex<double> chi[2];
  double sf[2], sfomega[2], omega[2], pp, pp3, sqp0p3, sqm[2];
  int ip, im, nh;

  fi.flv_index = flv;
  for (int i =0; i<4;i++){
    fi.p[i] = -p[i]*nsf;
  }
  //fi[0] = complex<double> (-p[0] * nsf, -p[3] * nsf);
  //fi[1] = complex<double> (-p[1] * nsf, -p[2] * nsf);
  nh = nhel * nsf;
  if (fmass != 0.0)
  {
    pp = min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    if (pp == 0.0)
    {
      sqm[0] = sqrt(std::abs(fmass));
      sqm[1] = Sgn(sqm[0], fmass);
      ip = (1 + nh)/2;
      im = (1 - nh)/2;
      fi.W[0] = ip * sqm[ip];
      fi.W[1] = im * nsf * sqm[ip];
      fi.W[2] = ip * nsf * sqm[im];
      fi.W[3] = im * sqm[im];
    }
    else
    {
      sf[0] = (1 + nsf + (1 - nsf) * nh) * 0.5;
      sf[1] = (1 + nsf - (1 - nsf) * nh) * 0.5;
      omega[0] = sqrt(p[0] + pp);
      omega[1] = fmass/omega[0];
      ip = (1 + nh)/2;
      im = (1 - nh)/2;
      sfomega[0] = sf[0] * omega[ip];
      sfomega[1] = sf[1] * omega[im];
      pp3 = max(pp + p[3], 0.0);
      chi[0] = complex<double> (sqrt(pp3 * 0.5/pp), 0);
      if (pp3 == 0.0)
      {
        chi[1] = complex<double> (-nh, 0);
      }
      else
      {
        chi[1] = complex<double> (nh * p[1], p[2])/sqrt(2.0 * pp * pp3);
      }
      fi.W[0] = sfomega[0] * chi[im];
      fi.W[1] = sfomega[0] * chi[ip];
      fi.W[2] = sfomega[1] * chi[im];
      fi.W[3] = sfomega[1] * chi[ip];
    }
  }
  else
  {
    if (p[1] == 0.0 and p[2] == 0.0 and p[3] < 0.0)
    {
      sqp0p3 = 0.0;
    }
    else
    {
      sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
    }
    chi[0] = complex<double> (sqp0p3, 0.0);
    if (sqp0p3 == 0.0)
    {
      chi[1] = complex<double> (-nhel * sqrt(2.0 * p[0]), 0.0);
    }
    else
    {
      chi[1] = complex<double> (nh * p[1], p[2])/sqp0p3;
    }
    if (nh == 1)
    {
      fi.W[0] = complex<double> (0.0, 0.0);
      fi.W[1] = complex<double> (0.0, 0.0);
      fi.W[2] = chi[0];
      fi.W[3] = chi[1];
    }
    else
    {
      fi.W[0] = chi[1];
      fi.W[1] = chi[0];
      fi.W[2] = complex<double> (0.0, 0.0);
      fi.W[3] = complex<double> (0.0, 0.0);
    }
  }
  return;
}


#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;

void oxxxxx(double p[4], double fmass, int nhel, int nsf, int flv, ALOHAOBJ &fo)
{
  complex<double> chi[2];
  double sf[2], sfomeg[2], omega[2], pp, pp3, sqp0p3, sqm[2];
  int nh, ip, im;

  fo.flv_index = flv;
  for(int i =0; i<4;i++){
    fo.p[i] = p[i]*nsf;
  }

  nh = nhel * nsf;
  if (fmass != 0.000)
  {
    pp = min(p[0], sqrt((p[1] * p[1]) + (p[2] * p[2]) + (p[3] * p[3])));
    if (pp == 0.000)
    {
      sqm[0] = sqrt(std::abs(fmass));
      sqm[1] = Sgn(sqm[0], fmass);
      ip = -((1 - nh)/2) * nhel;
      im = (1 + nh)/2 * nhel;
      fo.W[0] = im * sqm[std::abs(ip)];
      fo.W[1] = ip * nsf * sqm[std::abs(ip)];
      fo.W[2] = im * nsf * sqm[std::abs(im)];
      fo.W[3] = ip * sqm[std::abs(im)];
    }
    else
    {
      // pp = min(p[0],sqrt((p[1]*p[1])+(p[2]*p[2])+(p[3]*p[3])));
      sf[0] = double(1 + nsf + (1 - nsf) * nh) * 0.5;
      sf[1] = double(1 + nsf - (1 - nsf) * nh) * 0.5;
      omega[0] = sqrt(p[0] + pp);
      omega[1] = fmass/omega[0];
      ip = (1 + nh)/2;
      im = (1 - nh)/2;
      sfomeg[0] = sf[0] * omega[ip];
      sfomeg[1] = sf[1] * omega[im];
      pp3 = max(pp + p[3], 0.00);
      chi[0] = complex<double> (sqrt(pp3 * 0.5/pp), 0.00);
      if (pp3 == 0.00)
      {
        chi[1] = complex<double> (-nh, 0.00);
      }
      else
      {
        chi[1] = complex<double> (nh * p[1], -p[2])/sqrt(2.0 * pp * pp3);
      }
      fo.W[0] = sfomeg[1] * chi[im];
      fo.W[1] = sfomeg[1] * chi[ip];
      fo.W[2] = sfomeg[0] * chi[im];
      fo.W[3] = sfomeg[0] * chi[ip];
    }
  }
  else
  {
    if((p[1] == 0.00) and (p[2] == 0.00) and (p[3] < 0.00))
    {
      sqp0p3 = 0.00;
    }
    else
    {
      sqp0p3 = sqrt(max(p[0] + p[3], 0.00)) * nsf;
    }
    chi[0] = complex<double> (sqp0p3, 0.00);
    if(sqp0p3 == 0.000)
    {
      chi[1] = complex<double> (-nhel, 0.00) * sqrt(2.0 * p[0]);
    }
    else
    {
      chi[1] = complex<double> (nh * p[1], -p[2])/sqp0p3;
    }
    if(nh == 1)
    {
      fo.W[0] = chi[0];
      fo.W[1] = chi[1];
      fo.W[2] = complex<double> (0.00, 0.00);
      fo.W[3] = complex<double> (0.00, 0.00);
    }
    else
    {
      fo.W[0] = complex<double> (0.00, 0.00);
      fo.W[1] = complex<double> (0.00, 0.00);
      fo.W[2] = chi[1];
      fo.W[3] = chi[0];
    }
  }
  return;
}

#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;
void sxxxxx(double p[4],int nss, ALOHAOBJ &sc){
  sc.W[0] = complex<double>(1.00,0.00);
  
for ( int i =0; i<4;i++){
    sc.p[i] = nss*p[i];    
}
  return;
}
#include <complex>
#include <cmath> 
#include "aloha_aux_functions.h"
using namespace std;
#include <iostream>
#include <cstdlib>

//void txxxxx(double p[4], double tmass, int nhel, int nst, complex<double> tc[18])
//{
//	complex<double> ft[6][4], ep[4], em[4], e0[4];
//	double pt, pt2, pp, pzpt, emp, sqh, sqs;
//	int i, j;
//	
//	sqh = sqrt( 0.5);
//	sqs = sqrt( 0.5/3);
//	
//	pt2 = p[1]*p[1] + p[2]*p[2];
//	pp = min( p[0], sqrt( pt2+p[3]*p[3]) );
//	pt = min( pp, sqrt( pt2) );
//	
//	ft[4][0] = complex<double>( p[0]*nst, p[3]*nst );
//	ft[5][0] = complex<double>( p[1]*nst, p[2]*nst );
//	
//	// construct eps+
//	if( nhel >= 0 )
//	{
//		if( pp == 0 )
//		{
//			ep[0] = complex<double>( 0, 0 );
//			ep[1] = complex<double>( -sqh, 0 );
//			ep[2] = complex<double>( 0, nst*sqh );
//			ep[3] = complex<double>( 0, 0 );
//		}
//		else
//		{
//			ep[0] = complex<double>( 0, 0 );
//			ep[3] = complex<double>( pt/pp*sqh, 0 );
//			
//			if( pt != 0 )
//			{
//				pzpt = p[3]/(pp*pt)*sqh;
//				ep[1] = complex<double>( -p[1]*pzpt, -nst*p[2]/pt*sqh );
//				ep[2] = complex<double>( -p[2]*pzpt, nst*p[1]/pt*sqh );
//			}
//			else
//			{
//				ep[1] = complex<double>( -sqh, 0 );
//				ep[2] = complex<double>( 0, nst*Sgn(sqh,p[3]) );
//			}
//		}
//		
//	}
//	
//	// construct eps-
//	if( nhel <= 0 )
//	{
//		if( pp == 0 )
//		{
//			em[0] = complex<double>( 0, 0 );
//			em[1] = complex<double>( sqh, 0 );
//			em[2] = complex<double>( 0, nst*sqh );
//			em[3] = complex<double>( 0, 0 );
//		}
//		else
//		{
//			em[0] = complex<double>( 0, 0 );
//			em[3] = complex<double>( -pt/pp*sqh, 0 );
//			
//			if( pt != 0 )
//			{
//				pzpt = -p[3]/(pp*pt)*sqh;
//				em[1] = complex<double>( -p[1]*pzpt, -nst*p[2]/pt*sqh );
//				em[2] = complex<double>( -p[2]*pzpt,  nst*p[1]/pt*sqh );
//			}
//			else
//			{
//				em[1] = complex<double>( sqh, 0 );
//				em[2] = complex<double>( 0, nst*Sgn(sqh,p[3]) );
//			}
//		}
//	}
//	
//	// construct eps0
//	if( std::labs(nhel) <= 1 )
//	{
//		if( pp == 0 )
//		{
//			e0[0] = complex<double>( 0, 0 );
//			e0[1] = complex<double>( 0, 0 );
//			e0[2] = complex<double>( 0, 0 );
//			e0[3] = complex<double>( 1, 0 );
//		}
//		else
//		{
//			emp = p[0]/(tmass*pp);
//			e0[0] = complex<double>( pp/tmass, 0 );
//			e0[3] = complex<double>( p[3]*emp, 0 );
//			
//			if( pt != 0 )
//			{
//				e0[1] = complex<double>( p[1]*emp, 0 );
//				e0[2] = complex<double>( p[2]*emp, 0 );
//			}
//			else
//			{
//				e0[1] = complex<double>( 0, 0 );
//				e0[2] = complex<double>( 0, 0 );
//			}
//		}
//	}
//	
//	if( nhel == 2 )
//	{
//		for( j=0; j<4; j++ )
//		{
//			for( i=0; i<4; i++ ) ft[i][j] = ep[i]*ep[j];
//		}
//	}
//	else if( nhel == -2 )
//	{
//		for( j=0; j<4; j++ )
//		{
//			for( i=0; i<4; i++ ) ft[i][j] = em[i]*em[j];
//		}
//	}
//	else if( tmass == 0 )
//	{
//		for( j=0; j<4; j++ )
//		{
//			for( i=0; i<4; i++ ) ft[i][j] = 0;
//		}
//	}
//	else if( tmass != 0 )
//	{
//		if( nhel == 1 )
//		{
//			for( j=0; j<4; j++ )
//			{
//				for( i=0; i<4; i++ ) ft[i][j] = sqh*( ep[i]*e0[j] + e0[i]*ep[j] );
//			}
//		}
//		else if( nhel == 0 )
//		{
//			for( j=0; j<4; j++ )
//			{
//				for( i=0; i<4; i++ ) ft[i][j] = sqs*( ep[i]*em[j] + em[i]*ep[j]
//					+ 2.0*e0[i]*e0[j] );
//			}
//		}
//		else if( nhel == -1 )
//		{
//			for( j=0; j<4; j++ )
//			{
//				for( i=0; i<4; i++ ) ft[i][j] = sqh*( em[i]*e0[j] + e0[i]*em[j] );
//			}
//		}
//		else
//		{
//			std::cerr << "Invalid helicity in txxxxx.\n";
//			std::exit(1);
//		}
//	}
//	
//	tc[0] = ft[4][0];
//	tc[1] = ft[5][0];
//	
//	for( j=0; j<4; j++ )
//	{
//		for( i=0; i<4; i++ ) tc[j*4+i+2] = ft[j][i];
//	}
//}
#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;
void vxxxxx(double p[4],double vmass,int nhel,int nsv, ALOHAOBJ &vc){
  double hel,hel0,pt,pt2,pp,pzpt,emp,sqh;
  int nsvahl;
  sqh = sqrt(0.5);
  hel = double(nhel);
  nsvahl = nsv*std::abs(hel);
  pt2 = (p[1]*p[1])+(p[2]*p[2]);
  pp = min(p[0],sqrt(pt2+(p[3]*p[3])));
  pt =min(pp,sqrt(pt2));

  for(int i = 0 ; i<4;i++){
      vc.p[i] = nsv * p[i];
  }

  if (vmass != 0.0){
    hel0 = 1.0-std::abs(hel);
    if( pp == 0.0 ){ 
      vc.W[0] = complex<double>(0.0,0.0);
      vc.W[1] = complex<double>(-hel*sqh,0.0);
      vc.W[2] = complex<double>(0.0,nsvahl*sqh);
      vc.W[3] = complex<double>(hel0,0.0);
    }
    else{
      emp = p[0]/(vmass*pp);
      vc.W[0] = complex<double>(hel0*pp/vmass,0.0);
      vc.W[3] = complex<double>(hel0*p[3]*emp+hel*pt/pp*sqh,0.0); 
      if ( pt != 0.0){
        pzpt = p[3]/(pp*pt)*sqh*hel; 
        vc.W[1] = complex<double>(hel0*p[1]*emp-p[1]*pzpt,-nsvahl*p[2]/pt*sqh);
        vc.W[2] = complex<double>(hel0*p[2]*emp - p[2]*pzpt,nsvahl*p[1]/pt*sqh);
      }
      else{
        vc.W[1] = complex<double>(-hel*sqh,0.0);
        vc.W[2] = complex<double>(0.0,nsvahl*Sgn(sqh,p[3])); 
      }
    }
  }
  else{
    pp = p[0];
    pt = sqrt((p[1]*p[1])+(p[2]*p[2]));
    vc.W[0] = complex<double>(0.0,0.0);
    vc.W[3] = complex<double>(hel*pt/pp*sqh,0.0);
    if (pt != 0.0) {
      pzpt = p[3]/(pp*pt)*sqh*hel;
      vc.W[1] = complex<double>(-p[1]*pzpt,-nsv*p[2]/pt*sqh);
      vc.W[2] = complex<double>(-p[2]*pzpt,nsv*p[1]/pt*sqh);
    }
    else { 
      vc.W[1] = complex<double>(-hel*sqh,0.0);
      vc.W[2] = complex<double>(0.0,nsv*Sgn(sqh,p[3]));
    }
  }
  return;
}
