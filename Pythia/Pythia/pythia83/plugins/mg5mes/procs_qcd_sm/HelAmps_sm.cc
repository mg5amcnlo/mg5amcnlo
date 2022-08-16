//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "HelAmps_sm.h"
#include "Complex.h"
#include <cmath> 
#include <iostream> 
#include <cstdlib> 
using namespace std; 

namespace Pythia8_sm 
{

void sxxxxx(double p[4], int nss, Complex<double> sc[3])
{
  sc[2] = Complex<double> (1.00, 0.00); 
  sc[0] = Complex<double> (p[0] * nss, p[3] * nss); 
  sc[1] = Complex<double> (p[1] * nss, p[2] * nss); 
  return; 
}

void vxxxxx(double p[4], double vmass, int nhel, int nsv, Complex<double> vc[6])
{
  double hel, hel0, pt, pt2, pp, pzpt, emp, sqh; 
  int nsvahl; 
  sqh = sqrt(0.5); 
  hel = double(nhel); 
  nsvahl = nsv * std::abs(hel); 
  pt2 = (p[1] * p[1]) + (p[2] * p[2]); 
  pp = min(p[0], sqrt(pt2 + (p[3] * p[3]))); 
  pt = min(pp, sqrt(pt2)); 
  vc[0] = Complex<double> (p[0] * nsv, p[3] * nsv); 
  vc[1] = Complex<double> (p[1] * nsv, p[2] * nsv); 
  if (vmass != 0.0)
  {
    hel0 = 1.0 - std::abs(hel); 
    if(pp == 0.0)
    {
      vc[2] = Complex<double> (0.0, 0.0); 
      vc[3] = Complex<double> (-hel * sqh, 0.0); 
      vc[4] = Complex<double> (0.0, nsvahl * sqh); 
      vc[5] = Complex<double> (hel0, 0.0); 
    }
    else
    {
      emp = p[0]/(vmass * pp); 
      vc[2] = Complex<double> (hel0 * pp/vmass, 0.0); 
      vc[5] = Complex<double> (hel0 * p[3] * emp + hel * pt/pp * sqh, 0.0); 
      if (pt != 0.0)
      {
        pzpt = p[3]/(pp * pt) * sqh * hel; 
        vc[3] = Complex<double> (hel0 * p[1] * emp - p[1] * pzpt, -nsvahl *
            p[2]/pt * sqh);
        vc[4] = Complex<double> (hel0 * p[2] * emp - p[2] * pzpt, nsvahl *
            p[1]/pt * sqh);
      }
      else
      {
        vc[3] = Complex<double> (-hel * sqh, 0.0); 
        vc[4] = Complex<double> (0.0, nsvahl * Sgn(sqh, p[3])); 
      }
    }
  }
  else
  {
    pp = p[0]; 
    pt = sqrt((p[1] * p[1]) + (p[2] * p[2])); 
    vc[2] = Complex<double> (0.0, 0.0); 
    vc[5] = Complex<double> (hel * pt/pp * sqh, 0.0); 
    if (pt != 0.0)
    {
      pzpt = p[3]/(pp * pt) * sqh * hel; 
      vc[3] = Complex<double> (-p[1] * pzpt, -nsv * p[2]/pt * sqh); 
      vc[4] = Complex<double> (-p[2] * pzpt, nsv * p[1]/pt * sqh); 
    }
    else
    {
      vc[3] = Complex<double> (-hel * sqh, 0.0); 
      vc[4] = Complex<double> (0.0, nsv * Sgn(sqh, p[3])); 
    }
  }
  return; 
}

void ixxxxx(double p[4], double fmass, int nhel, int nsf, Complex<double> fi[6])
{
  Complex<double> chi[2]; 
  double sf[2], sfomega[2], omega[2], pp, pp3, sqp0p3, sqm[2]; 
  int ip, im, nh; 
  fi[0] = Complex<double> (-p[0] * nsf, -p[3] * nsf); 
  fi[1] = Complex<double> (-p[1] * nsf, -p[2] * nsf); 
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
      fi[2] = ip * sqm[ip]; 
      fi[3] = im * nsf * sqm[ip]; 
      fi[4] = ip * nsf * sqm[im]; 
      fi[5] = im * sqm[im]; 
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
      chi[0] = Complex<double> (sqrt(pp3 * 0.5/pp), 0); 
      if (pp3 == 0.0)
      {
        chi[1] = Complex<double> (-nh, 0); 
      }
      else
      {
        chi[1] = Complex<double> (nh * p[1], p[2])/sqrt(2.0 * pp * pp3); 
      }
      fi[2] = sfomega[0] * chi[im]; 
      fi[3] = sfomega[0] * chi[ip]; 
      fi[4] = sfomega[1] * chi[im]; 
      fi[5] = sfomega[1] * chi[ip]; 
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
    chi[0] = Complex<double> (sqp0p3, 0.0); 
    if (sqp0p3 == 0.0)
    {
      chi[1] = Complex<double> (-nhel * sqrt(2.0 * p[0]), 0.0); 
    }
    else
    {
      chi[1] = Complex<double> (nh * p[1], p[2])/sqp0p3; 
    }
    if (nh == 1)
    {
      fi[2] = Complex<double> (0.0, 0.0); 
      fi[3] = Complex<double> (0.0, 0.0); 
      fi[4] = chi[0]; 
      fi[5] = chi[1]; 
    }
    else
    {
      fi[2] = chi[1]; 
      fi[3] = chi[0]; 
      fi[4] = Complex<double> (0.0, 0.0); 
      fi[5] = Complex<double> (0.0, 0.0); 
    }
  }
  return; 
}


void txxxxx(double p[4], double tmass, int nhel, int nst, Complex<double>
    tc[18])
{
  Complex<double> ft[6][4], ep[4], em[4], e0[4]; 
  double pt, pt2, pp, pzpt, emp, sqh, sqs; 
  int i, j; 

  sqh = sqrt(0.5); 
  sqs = sqrt(0.5/3); 

  pt2 = p[1] * p[1] + p[2] * p[2]; 
  pp = min(p[0], sqrt(pt2 + p[3] * p[3])); 
  pt = min(pp, sqrt(pt2)); 

  ft[4][0] = Complex<double> (p[0] * nst, p[3] * nst); 
  ft[5][0] = Complex<double> (p[1] * nst, p[2] * nst); 

  // construct eps+
  if(nhel >= 0)
  {
    if(pp == 0)
    {
      ep[0] = Complex<double> (0, 0); 
      ep[1] = Complex<double> (-sqh, 0); 
      ep[2] = Complex<double> (0, nst * sqh); 
      ep[3] = Complex<double> (0, 0); 
    }
    else
    {
      ep[0] = Complex<double> (0, 0); 
      ep[3] = Complex<double> (pt/pp * sqh, 0); 

      if(pt != 0)
      {
        pzpt = p[3]/(pp * pt) * sqh; 
        ep[1] = Complex<double> (-p[1] * pzpt, -nst * p[2]/pt * sqh); 
        ep[2] = Complex<double> (-p[2] * pzpt, nst * p[1]/pt * sqh); 
      }
      else
      {
        ep[1] = Complex<double> (-sqh, 0); 
        ep[2] = Complex<double> (0, nst * Sgn(sqh, p[3])); 
      }
    }

  }

  // construct eps-
  if(nhel <= 0)
  {
    if(pp == 0)
    {
      em[0] = Complex<double> (0, 0); 
      em[1] = Complex<double> (sqh, 0); 
      em[2] = Complex<double> (0, nst * sqh); 
      em[3] = Complex<double> (0, 0); 
    }
    else
    {
      em[0] = Complex<double> (0, 0); 
      em[3] = Complex<double> (-pt/pp * sqh, 0); 

      if(pt != 0)
      {
        pzpt = -p[3]/(pp * pt) * sqh; 
        em[1] = Complex<double> (-p[1] * pzpt, -nst * p[2]/pt * sqh); 
        em[2] = Complex<double> (-p[2] * pzpt, nst * p[1]/pt * sqh); 
      }
      else
      {
        em[1] = Complex<double> (sqh, 0); 
        em[2] = Complex<double> (0, nst * Sgn(sqh, p[3])); 
      }
    }
  }

  // construct eps0
  if(std::labs(nhel) <= 1)
  {
    if(pp == 0)
    {
      e0[0] = Complex<double> (0, 0); 
      e0[1] = Complex<double> (0, 0); 
      e0[2] = Complex<double> (0, 0); 
      e0[3] = Complex<double> (1, 0); 
    }
    else
    {
      emp = p[0]/(tmass * pp); 
      e0[0] = Complex<double> (pp/tmass, 0); 
      e0[3] = Complex<double> (p[3] * emp, 0); 

      if(pt != 0)
      {
        e0[1] = Complex<double> (p[1] * emp, 0); 
        e0[2] = Complex<double> (p[2] * emp, 0); 
      }
      else
      {
        e0[1] = Complex<double> (0, 0); 
        e0[2] = Complex<double> (0, 0); 
      }
    }
  }

  if(nhel == 2)
  {
    for(j = 0; j < 4; j++ )
    {
      for(i = 0; i < 4; i++ )
        ft[i][j] = ep[i] * ep[j]; 
    }
  }
  else if(nhel == -2)
  {
    for(j = 0; j < 4; j++ )
    {
      for(i = 0; i < 4; i++ )
        ft[i][j] = em[i] * em[j]; 
    }
  }
  else if(tmass == 0)
  {
    for(j = 0; j < 4; j++ )
    {
      for(i = 0; i < 4; i++ )
        ft[i][j] = 0; 
    }
  }
  else if(tmass != 0)
  {
    if(nhel == 1)
    {
      for(j = 0; j < 4; j++ )
      {
        for(i = 0; i < 4; i++ )
          ft[i][j] = sqh * (ep[i] * e0[j] + e0[i] * ep[j]); 
      }
    }
    else if(nhel == 0)
    {
      for(j = 0; j < 4; j++ )
      {
        for(i = 0; i < 4; i++ )
          ft[i][j] = sqs * (ep[i] * em[j] + em[i] * ep[j]
         + 2.0 * e0[i] * e0[j]); 
      }
    }
    else if(nhel == -1)
    {
      for(j = 0; j < 4; j++ )
      {
        for(i = 0; i < 4; i++ )
          ft[i][j] = sqh * (em[i] * e0[j] + e0[i] * em[j]); 
      }
    }
    else
    {
      std::cerr <<  "Invalid helicity in txxxxx.\n"; 
      std::exit(1); 
    }
  }

  tc[0] = ft[4][0]; 
  tc[1] = ft[5][0]; 

  for(j = 0; j < 4; j++ )
  {
    for(i = 0; i < 4; i++ )
      tc[j * 4 + i + 2] = ft[j][i]; 
  }
}


double Sgn(double a, double b)
{
  return (b < 0)? - abs(a):abs(a); 
}

void oxxxxx(double p[4], double fmass, int nhel, int nsf, Complex<double> fo[6])
{
  Complex<double> chi[2]; 
  double sf[2], sfomeg[2], omega[2], pp, pp3, sqp0p3, sqm[2]; 
  int nh, ip, im; 
  fo[0] = Complex<double> (p[0] * nsf, p[3] * nsf); 
  fo[1] = Complex<double> (p[1] * nsf, p[2] * nsf); 
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
      fo[2] = im * sqm[std::abs(ip)]; 
      fo[3] = ip * nsf * sqm[std::abs(ip)]; 
      fo[4] = im * nsf * sqm[std::abs(im)]; 
      fo[5] = ip * sqm[std::abs(im)]; 
    }
    else
    {
      pp = min(p[0], sqrt((p[1] * p[1]) + (p[2] * p[2]) + (p[3] * p[3]))); 
      sf[0] = double(1 + nsf + (1 - nsf) * nh) * 0.5; 
      sf[1] = double(1 + nsf - (1 - nsf) * nh) * 0.5; 
      omega[0] = sqrt(p[0] + pp); 
      omega[1] = fmass/omega[0]; 
      ip = (1 + nh)/2; 
      im = (1 - nh)/2; 
      sfomeg[0] = sf[0] * omega[ip]; 
      sfomeg[1] = sf[1] * omega[im]; 
      pp3 = max(pp + p[3], 0.00); 
      chi[0] = Complex<double> (sqrt(pp3 * 0.5/pp), 0.00); 
      if (pp3 == 0.00)
      {
        chi[1] = Complex<double> (-nh, 0.00); 
      }
      else
      {
        chi[1] = Complex<double> (nh * p[1], -p[2])/sqrt(2.0 * pp * pp3); 
      }
      fo[2] = sfomeg[1] * chi[im]; 
      fo[3] = sfomeg[1] * chi[ip]; 
      fo[4] = sfomeg[0] * chi[im]; 
      fo[5] = sfomeg[0] * chi[ip]; 
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
    chi[0] = Complex<double> (sqp0p3, 0.00); 
    if(sqp0p3 == 0.000)
    {
      chi[1] = Complex<double> (-nhel, 0.00) * sqrt(2.0 * p[0]); 
    }
    else
    {
      chi[1] = Complex<double> (nh * p[1], -p[2])/sqp0p3; 
    }
    if(nh == 1)
    {
      fo[2] = chi[0]; 
      fo[3] = chi[1]; 
      fo[4] = Complex<double> (0.00, 0.00); 
      fo[5] = Complex<double> (0.00, 0.00); 
    }
    else
    {
      fo[2] = Complex<double> (0.00, 0.00); 
      fo[3] = Complex<double> (0.00, 0.00); 
      fo[4] = chi[1]; 
      fo[5] = chi[0]; 
    }
  }
  return; 
}

void FFS2_0(Complex<double> F1[], Complex<double> F2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP1; 
  Complex<double> TMP0; 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  vertex = COUP * S3[2] * (-cI * (TMP0) + cI * (TMP1)); 
}


void SSS1_3(Complex<double> S1[], Complex<double> S2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  Complex<double> denom; 
  S3[0] = +S1[0] + S2[0]; 
  S3[1] = +S1[1] + S2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * S2[2] * S1[2]; 
}


void FFV3_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * (-2. * cI) * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + (+1./2. * (M1 * (+2. * (F2[4] * (-1./2.) * (V3[2] + V3[5]))
      - F2[5] * (V3[3] + cI * (V3[4])))) + F2[3] * (P1[0] * (V3[3] + cI *
      (V3[4])) + (P1[1] * (-1.) * (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI *
      (V3[2] + V3[5])) + P1[3] * (V3[3] + cI * (V3[4])))))));
  F1[3] = denom * (-2. * cI) * (F2[2] * (P1[0] * (V3[3] - cI * (V3[4])) +
      (P1[1] * (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) +
      P1[3] * (+cI * (V3[4]) - V3[3])))) + (+1./2. * (M1 * (F2[5] * (V3[5] -
      V3[2]) + 2. * (F2[4] * 1./2. * (+cI * (V3[4]) - V3[3])))) + F2[3] *
      (P1[0] * (-1.) * (V3[2] + V3[5]) + (P1[1] * (V3[3] + cI * (V3[4])) +
      (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[2] + V3[5]))))));
  F1[4] = denom * cI * (F2[4] * (P1[0] * (-1.) * (V3[2] + V3[5]) + (P1[1] *
      (V3[3] - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[2]
      + V3[5])))) + (F2[5] * (P1[0] * (-1.) * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[2] - V3[5]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) + P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * 2. * (V3[5] - V3[2]) + 2. *
      (F2[3] * (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * (-cI) * (F2[4] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (-1.) * (V3[2] + V3[5]) + (P1[2] * (+cI * (V3[2] + V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + (F2[5] * (P1[0] * (V3[2] - V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) + P1[3]
      * (V3[2] - V3[5])))) + M1 * (F2[2] * 2. * (+cI * (V3[4]) - V3[3]) + 2. *
      (F2[3] * (V3[2] + V3[5])))));
}


void VVVV2_3(Complex<double> V1[], Complex<double> V2[], Complex<double> V4[],
    Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP27; 
  Complex<double> TMP9; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP27 = (P3[0] * V4[2] - P3[1] * V4[3] - P3[2] * V4[4] - P3[3] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (OM3 * P3[0] * (-2. * cI * (TMP12 * TMP27) + cI * (TMP17 *
      TMP20 + TMP9 * TMP21)) + (-cI * (V2[2] * TMP20 + V1[2] * TMP21) + 2. * cI
      * (TMP12 * V4[2])));
  V3[3] = denom * (OM3 * P3[1] * (-2. * cI * (TMP12 * TMP27) + cI * (TMP17 *
      TMP20 + TMP9 * TMP21)) + (-cI * (V2[3] * TMP20 + V1[3] * TMP21) + 2. * cI
      * (TMP12 * V4[3])));
  V3[4] = denom * (OM3 * P3[2] * (-2. * cI * (TMP12 * TMP27) + cI * (TMP17 *
      TMP20 + TMP9 * TMP21)) + (-cI * (V2[4] * TMP20 + V1[4] * TMP21) + 2. * cI
      * (TMP12 * V4[4])));
  V3[5] = denom * (OM3 * P3[3] * (-2. * cI * (TMP12 * TMP27) + cI * (TMP17 *
      TMP20 + TMP9 * TMP21)) + (-cI * (V2[5] * TMP20 + V1[5] * TMP21) + 2. * cI
      * (TMP12 * V4[5])));
}


void FFV1_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * (V3[2] - V3[5]) + F1[5] * (+cI *
      (V3[4]) - V3[3]))));
  F2[3] = denom * (-cI) * (F1[2] * (P2[0] * (-1.) * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) - P2[3] *
      (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[5] - V3[2]) + (P2[1] *
      (V3[3] - cI * (V3[4])) + (P2[2] * (V3[4] + cI * (V3[3])) + P2[3] * (V3[5]
      - V3[2])))) + M2 * (F1[4] * (V3[3] + cI * (V3[4])) - F1[5] * (V3[2] +
      V3[5]))));
  F2[4] = denom * (-cI) * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] * (V3[3] +
      cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5] -
      V3[2])))) + (F1[5] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] * (-1.) *
      (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) + P2[3] * (V3[3] - cI
      * (V3[4]))))) + M2 * (F1[2] * (-1.) * (V3[2] + V3[5]) + F1[3] * (+cI *
      (V3[4]) - V3[3]))));
  F2[5] = denom * cI * (F1[4] * (P2[0] * (-1.) * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[2] - V3[5]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[5] * (P2[0] * (V3[2] + V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) - P2[3] * (V3[2] + V3[5])))) + M2 * (F1[2] * (V3[3] + cI *
      (V3[4])) + F1[3] * (V3[2] - V3[5]))));
}


void VVVV4_3(Complex<double> V1[], Complex<double> V2[], Complex<double> V4[],
    Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P3[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP27; 
  Complex<double> TMP9; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP27 = (P3[0] * V4[2] - P3[1] * V4[3] - P3[2] * V4[4] - P3[3] * V4[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (OM3 * P3[0] * (-cI * (TMP12 * TMP27) + cI * (TMP9 * TMP21))
      + (-cI * (V1[2] * TMP21) + cI * (TMP12 * V4[2])));
  V3[3] = denom * (OM3 * P3[1] * (-cI * (TMP12 * TMP27) + cI * (TMP9 * TMP21))
      + (-cI * (V1[3] * TMP21) + cI * (TMP12 * V4[3])));
  V3[4] = denom * (OM3 * P3[2] * (-cI * (TMP12 * TMP27) + cI * (TMP9 * TMP21))
      + (-cI * (V1[4] * TMP21) + cI * (TMP12 * V4[4])));
  V3[5] = denom * (OM3 * P3[3] * (-cI * (TMP12 * TMP27) + cI * (TMP9 * TMP21))
      + (-cI * (V1[5] * TMP21) + cI * (TMP12 * V4[5])));
}


void VSS1_2(Complex<double> V1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> S2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  double P3[4]; 
  Complex<double> denom; 
  Complex<double> TMP9; 
  Complex<double> TMP8; 
  P3[0] = S3[0].real(); 
  P3[1] = S3[1].real(); 
  P3[2] = S3[1].imag(); 
  P3[3] = S3[0].imag(); 
  S2[0] = +V1[0] + S3[0]; 
  S2[1] = +V1[1] + S3[1]; 
  P2[0] = -S2[0].real(); 
  P2[1] = -S2[1].real(); 
  P2[2] = -S2[1].imag(); 
  P2[3] = -S2[0].imag(); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S2[2] = denom * S3[2] * (-cI * (TMP9) + cI * (TMP8)); 
}


void VVV1_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P1[4]; 
  double P2[4]; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  Complex<double> TMP14; 
  Complex<double> TMP18; 
  Complex<double> TMP9; 
  Complex<double> TMP13; 
  Complex<double> TMP8; 
  P1[0] = V1[0].real(); 
  P1[1] = V1[1].real(); 
  P1[2] = V1[1].imag(); 
  P1[3] = V1[0].imag(); 
  P2[0] = V2[0].real(); 
  P2[1] = V2[1].real(); 
  P2[2] = V2[1].imag(); 
  P2[3] = V2[0].imag(); 
  P3[0] = V3[0].real(); 
  P3[1] = V3[1].real(); 
  P3[2] = V3[1].imag(); 
  P3[3] = V3[0].imag(); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * (TMP12 * (-cI * (TMP14) + cI * (TMP15)) + (TMP16 * (-cI *
      (TMP17) + cI * (TMP13)) + TMP18 * (-cI * (TMP8) + cI * (TMP9))));
}


void VVSS1_1(Complex<double> V2[], Complex<double> S3[], Complex<double> S4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  double OM1; 
  Complex<double> TMP13; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + S3[0] + S4[0]; 
  V1[1] = +V2[1] + S3[1] + S4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * S3[2] * S4[2] * (-cI * (V2[2]) + cI * (P1[0] * OM1 * TMP13)); 
  V1[3] = denom * S3[2] * S4[2] * (-cI * (V2[3]) + cI * (P1[1] * OM1 * TMP13)); 
  V1[4] = denom * S3[2] * S4[2] * (-cI * (V2[4]) + cI * (P1[2] * OM1 * TMP13)); 
  V1[5] = denom * S3[2] * S4[2] * (-cI * (V2[5]) + cI * (P1[3] * OM1 * TMP13)); 
}


void FFV4_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP6; 
  Complex<double> TMP4; 
  TMP4 = (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
      F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])));
  TMP6 = (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
      F1[5] * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] + V3[5])));
  vertex = COUP * (-1.) * (+cI * (TMP4) + 2. * cI * (TMP6)); 
}


void VVVV3_2(Complex<double> V1[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P2[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP15; 
  Complex<double> TMP26; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (OM2 * P2[0] * (-cI * (TMP8 * TMP26) + cI * (TMP15 * TMP20))
      + (-cI * (V3[2] * TMP20) + cI * (V1[2] * TMP26)));
  V2[3] = denom * (OM2 * P2[1] * (-cI * (TMP8 * TMP26) + cI * (TMP15 * TMP20))
      + (-cI * (V3[3] * TMP20) + cI * (V1[3] * TMP26)));
  V2[4] = denom * (OM2 * P2[2] * (-cI * (TMP8 * TMP26) + cI * (TMP15 * TMP20))
      + (-cI * (V3[4] * TMP20) + cI * (V1[4] * TMP26)));
  V2[5] = denom * (OM2 * P2[3] * (-cI * (TMP8 * TMP26) + cI * (TMP15 * TMP20))
      + (-cI * (V3[5] * TMP20) + cI * (V1[5] * TMP26)));
}


void FFS1_0(Complex<double> F1[], Complex<double> F2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP0; 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  vertex = COUP * - cI * TMP0 * S3[2]; 
}

void FFS1_3_0(Complex<double> F1[], Complex<double> F2[], Complex<double> S3[],
    Complex<double> COUP1, Complex<double> COUP2, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> tmp; 
  FFS1_0(F1, F2, S3, COUP1, vertex); 
  FFS3_0(F1, F2, S3, COUP2, tmp); 
  vertex = vertex + tmp; 
}

void VSS1P0_1(Complex<double> S2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  double P2[4]; 
  double P3[4]; 
  Complex<double> denom; 
  P2[0] = S2[0].real(); 
  P2[1] = S2[1].real(); 
  P2[2] = S2[1].imag(); 
  P2[3] = S2[0].imag(); 
  P3[0] = S3[0].real(); 
  P3[1] = S3[1].real(); 
  P3[2] = S3[1].imag(); 
  P3[3] = S3[0].imag(); 
  V1[0] = +S2[0] + S3[0]; 
  V1[1] = +S2[1] + S3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * S2[2] * S3[2] * (-cI * (P2[0]) + cI * (P3[0])); 
  V1[3] = denom * S2[2] * S3[2] * (-cI * (P2[1]) + cI * (P3[1])); 
  V1[4] = denom * S2[2] * S3[2] * (-cI * (P2[2]) + cI * (P3[2])); 
  V1[5] = denom * S2[2] * S3[2] * (-cI * (P2[3]) + cI * (P3[3])); 
}


void VVV1P0_2(Complex<double> V1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P1[4]; 
  double P2[4]; 
  double P3[4]; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  Complex<double> TMP14; 
  Complex<double> TMP9; 
  Complex<double> TMP8; 
  P1[0] = V1[0].real(); 
  P1[1] = V1[1].real(); 
  P1[2] = V1[1].imag(); 
  P1[3] = V1[0].imag(); 
  P3[0] = V3[0].real(); 
  P3[1] = V3[1].real(); 
  P3[2] = V3[1].imag(); 
  P3[3] = V3[0].imag(); 
  V2[0] = +V1[0] + V3[0]; 
  V2[1] = +V1[1] + V3[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (TMP16 * (-cI * (P3[0]) + cI * (P1[0])) + (V1[2] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[2] * (-cI * (TMP8) + cI * (TMP9))));
  V2[3] = denom * (TMP16 * (-cI * (P3[1]) + cI * (P1[1])) + (V1[3] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[3] * (-cI * (TMP8) + cI * (TMP9))));
  V2[4] = denom * (TMP16 * (-cI * (P3[2]) + cI * (P1[2])) + (V1[4] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[4] * (-cI * (TMP8) + cI * (TMP9))));
  V2[5] = denom * (TMP16 * (-cI * (P3[3]) + cI * (P1[3])) + (V1[5] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[5] * (-cI * (TMP8) + cI * (TMP9))));
}


void VVVV1_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> V4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP21; 
  Complex<double> TMP18; 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  vertex = COUP * (-cI * (TMP18 * TMP20) + cI * (TMP16 * TMP21)); 
}


void FFV2_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI *
      (V3[4])));
  F1[3] = denom * - cI * M1 * (F2[4] * (+cI * (V3[4]) - V3[3]) + F2[5] * (V3[5]
      - V3[2]));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * (-cI) * (F2[4] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (-1.) * (V3[2] + V3[5]) + (P1[2] * (+cI * (V3[2] + V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + F2[5] * (P1[0] * (V3[2] - V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) + P1[3]
      * (V3[2] - V3[5])))));
}

void FFV2_3_1(Complex<double> F2[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  int i; 
  Complex<double> Ftmp[6]; 
  FFV2_1(F2, V3, COUP1, M1, W1, F1); 
  FFV3_1(F2, V3, COUP2, M1, W1, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F1[i] = F1[i] + Ftmp[i]; 
    i++; 
  }
}
void FFV2_4_1(Complex<double> F2[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  int i; 
  Complex<double> Ftmp[6]; 
  FFV2_1(F2, V3, COUP1, M1, W1, F1); 
  FFV4_1(F2, V3, COUP2, M1, W1, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F1[i] = F1[i] + Ftmp[i]; 
    i++; 
  }
}
void FFV2_5_1(Complex<double> F2[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  int i; 
  Complex<double> Ftmp[6]; 
  FFV2_1(F2, V3, COUP1, M1, W1, F1); 
  FFV5_1(F2, V3, COUP2, M1, W1, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F1[i] = F1[i] + Ftmp[i]; 
    i++; 
  }
}

void VVVV4P0_2(Complex<double> V1[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> TMP16; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (-cI * (TMP16 * V4[2]) + cI * (V1[2] * TMP26)); 
  V2[3] = denom * (-cI * (TMP16 * V4[3]) + cI * (V1[3] * TMP26)); 
  V2[4] = denom * (-cI * (TMP16 * V4[4]) + cI * (V1[4] * TMP26)); 
  V2[5] = denom * (-cI * (TMP16 * V4[5]) + cI * (V1[5] * TMP26)); 
}


void FFV5_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP6; 
  Complex<double> TMP4; 
  TMP4 = (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
      F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])));
  TMP6 = (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
      F1[5] * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] + V3[5])));
  vertex = COUP * (-1.) * (+cI * (TMP4) + 4. * cI * (TMP6)); 
}


void FFS4_0(Complex<double> F1[], Complex<double> F2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP1; 
  Complex<double> TMP0; 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  vertex = COUP * - S3[2] * (+cI * (TMP0 + TMP1)); 
}


void FFV3_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * 2. * (V3[5] - V3[2]) + 2. * (F1[5]
      * (V3[3] - cI * (V3[4]))))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))) + M2 * (F1[4] * 2. * (V3[3] + cI *
      (V3[4])) - 2. * (F1[5] * (V3[2] + V3[5])))));
  F2[4] = denom * 2. * cI * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] * (V3[3]
      + cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5] -
      V3[2])))) + (+1./2. * (M2 * (F1[3] * (V3[3] - cI * (V3[4])) + 2. * (F1[2]
      * 1./2. * (V3[2] + V3[5])))) + F1[5] * (P2[0] * (V3[3] - cI * (V3[4])) +
      (P2[1] * (-1.) * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] - cI * (V3[4])))))));
  F2[5] = denom * 2. * cI * (F1[4] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[2]) + cI * (V3[5])) - P2[3] *
      (V3[3] + cI * (V3[4]))))) + (+1./2. * (M2 * (F1[3] * (V3[2] - V3[5]) + 2.
      * (F1[2] * 1./2. * (V3[3] + cI * (V3[4]))))) + F1[5] * (P2[0] * (-1.) *
      (V3[2] + V3[5]) + (P2[1] * (V3[3] - cI * (V3[4])) + (P2[2] * (V3[4] + cI
      * (V3[3])) + P2[3] * (V3[2] + V3[5]))))));
}


void FFV1_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] * (V3[3] - cI
      * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5] - V3[2]))))
      + (F2[3] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] * (-1.) * (V3[2] +
      V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (V3[3] + cI *
      (V3[4]))))) + M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI *
      (V3[4])))));
  F1[3] = denom * (-cI) * (F2[2] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] - V3[5]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + (F2[3] * (P1[0] * (V3[2] + V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) - P1[3]
      * (V3[2] + V3[5])))) + M1 * (F2[4] * (+cI * (V3[4]) - V3[3]) + F2[5] *
      (V3[5] - V3[2]))));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + (F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * (V3[5] - V3[2]) + F2[3] *
      (V3[3] + cI * (V3[4])))));
  F1[5] = denom * cI * (F2[4] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (F2[5] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + M1 * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] +
      V3[5]))));
}


void VVS1_2(Complex<double> V1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + S3[0]; 
  V2[1] = +V1[1] + S3[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * S3[2] * (-cI * (V1[2]) + cI * (P2[0] * TMP8 * OM2)); 
  V2[3] = denom * S3[2] * (-cI * (V1[3]) + cI * (P2[1] * TMP8 * OM2)); 
  V2[4] = denom * S3[2] * (-cI * (V1[4]) + cI * (P2[2] * TMP8 * OM2)); 
  V2[5] = denom * S3[2] * (-cI * (V1[5]) + cI * (P2[3] * TMP8 * OM2)); 
}


void FFS2_2(Complex<double> F1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + S3[0]; 
  F2[1] = +F1[1] + S3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * - cI * S3[2] * (F1[4] * (P2[0] - P2[3]) + (F1[5] * (+cI *
      (P2[2]) - P2[1]) - F1[2] * M2));
  F2[3] = denom * cI * S3[2] * (F1[4] * (P2[1] + cI * (P2[2])) + (F1[5] * (-1.)
      * (P2[0] + P2[3]) + F1[3] * M2));
  F2[4] = denom * - cI * S3[2] * (F1[2] * (-1.) * (P2[0] + P2[3]) + (F1[3] *
      (+cI * (P2[2]) - P2[1]) + F1[4] * M2));
  F2[5] = denom * cI * S3[2] * (F1[2] * (P2[1] + cI * (P2[2])) + (F1[3] *
      (P2[0] - P2[3]) - F1[5] * M2));
}


void VVSS1_2(Complex<double> V1[], Complex<double> S3[], Complex<double> S4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + S3[0] + S4[0]; 
  V2[1] = +V1[1] + S3[1] + S4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * S3[2] * S4[2] * (-cI * (V1[2]) + cI * (P2[0] * TMP8 * OM2)); 
  V2[3] = denom * S3[2] * S4[2] * (-cI * (V1[3]) + cI * (P2[1] * TMP8 * OM2)); 
  V2[4] = denom * S3[2] * S4[2] * (-cI * (V1[4]) + cI * (P2[2] * TMP8 * OM2)); 
  V2[5] = denom * S3[2] * S4[2] * (-cI * (V1[5]) + cI * (P2[3] * TMP8 * OM2)); 
}


void VVSS1P0_1(Complex<double> V2[], Complex<double> S3[], Complex<double>
    S4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  V1[0] = +V2[0] + S3[0] + S4[0]; 
  V1[1] = +V2[1] + S3[1] + S4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * - cI * V2[2] * S4[2] * S3[2]; 
  V1[3] = denom * - cI * V2[3] * S4[2] * S3[2]; 
  V1[4] = denom * - cI * V2[4] * S4[2] * S3[2]; 
  V1[5] = denom * - cI * V2[5] * S4[2] * S3[2]; 
}


void SSS1_1(Complex<double> S2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> S1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  S1[0] = +S2[0] + S3[0]; 
  S1[1] = +S2[1] + S3[1]; 
  P1[0] = -S1[0].real(); 
  P1[1] = -S1[1].real(); 
  P1[2] = -S1[1].imag(); 
  P1[3] = -S1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S1[2] = denom * cI * S3[2] * S2[2]; 
}


void SSSS1_4(Complex<double> S1[], Complex<double> S2[], Complex<double> S3[],
    Complex<double> COUP, double M4, double W4, Complex<double> S4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P4[4]; 
  S4[0] = +S1[0] + S2[0] + S3[0]; 
  S4[1] = +S1[1] + S2[1] + S3[1]; 
  P4[0] = -S4[0].real(); 
  P4[1] = -S4[1].real(); 
  P4[2] = -S4[1].imag(); 
  P4[3] = -S4[0].imag(); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S4[2] = denom * cI * S3[2] * S2[2] * S1[2]; 
}


void VVVV1P0_3(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V4[], Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI * (V2[2] * TMP20) + cI * (V1[2] * TMP21)); 
  V3[3] = denom * (-cI * (V2[3] * TMP20) + cI * (V1[3] * TMP21)); 
  V3[4] = denom * (-cI * (V2[4] * TMP20) + cI * (V1[4] * TMP21)); 
  V3[5] = denom * (-cI * (V2[5] * TMP20) + cI * (V1[5] * TMP21)); 
}


void VVVV4_1(Complex<double> V2[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP14; 
  double OM1; 
  Complex<double> TMP13; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (OM1 * P1[0] * (-cI * (TMP13 * TMP26) + cI * (TMP14 * TMP21))
      + (-cI * (V3[2] * TMP21) + cI * (V2[2] * TMP26)));
  V1[3] = denom * (OM1 * P1[1] * (-cI * (TMP13 * TMP26) + cI * (TMP14 * TMP21))
      + (-cI * (V3[3] * TMP21) + cI * (V2[3] * TMP26)));
  V1[4] = denom * (OM1 * P1[2] * (-cI * (TMP13 * TMP26) + cI * (TMP14 * TMP21))
      + (-cI * (V3[4] * TMP21) + cI * (V2[4] * TMP26)));
  V1[5] = denom * (OM1 * P1[3] * (-cI * (TMP13 * TMP26) + cI * (TMP14 * TMP21))
      + (-cI * (V3[5] * TMP21) + cI * (V2[5] * TMP26)));
}


void VSS1_0(Complex<double> V1[], Complex<double> S2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  double P3[4]; 
  Complex<double> TMP9; 
  Complex<double> TMP8; 
  P2[0] = S2[0].real(); 
  P2[1] = S2[1].real(); 
  P2[2] = S2[1].imag(); 
  P2[3] = S2[0].imag(); 
  P3[0] = S3[0].real(); 
  P3[1] = S3[1].real(); 
  P3[2] = S3[1].imag(); 
  P3[3] = S3[0].imag(); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  vertex = COUP * S2[2] * S3[2] * (-cI * (TMP8) + cI * (TMP9)); 
}


void VVV1_2(Complex<double> V1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P1[4]; 
  Complex<double> TMP10; 
  double P2[4]; 
  Complex<double> TMP9; 
  double P3[4]; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  Complex<double> TMP14; 
  double OM2; 
  Complex<double> TMP19; 
  Complex<double> TMP8; 
  P1[0] = V1[0].real(); 
  P1[1] = V1[1].real(); 
  P1[2] = V1[1].imag(); 
  P1[3] = V1[0].imag(); 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  P3[0] = V3[0].real(); 
  P3[1] = V3[1].real(); 
  P3[2] = V3[1].imag(); 
  P3[3] = V3[0].imag(); 
  V2[0] = +V1[0] + V3[0]; 
  V2[1] = +V1[1] + V3[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP19 = (P2[0] * P3[0] - P2[1] * P3[1] - P2[2] * P3[2] - P2[3] * P3[3]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP10 = (P1[0] * P2[0] - P1[1] * P2[1] - P1[2] * P2[2] - P1[3] * P2[3]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (OM2 * P2[0] * (TMP16 * (-cI * (TMP10) + cI * (TMP19)) + (-cI
      * (TMP9 * TMP15) + cI * (TMP8 * TMP14))) + (TMP16 * (-cI * (P3[0]) + cI *
      (P1[0])) + (V1[2] * (-cI * (TMP14) + cI * (TMP15)) + V3[2] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V2[3] = denom * (OM2 * P2[1] * (TMP16 * (-cI * (TMP10) + cI * (TMP19)) + (-cI
      * (TMP9 * TMP15) + cI * (TMP8 * TMP14))) + (TMP16 * (-cI * (P3[1]) + cI *
      (P1[1])) + (V1[3] * (-cI * (TMP14) + cI * (TMP15)) + V3[3] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V2[4] = denom * (OM2 * P2[2] * (TMP16 * (-cI * (TMP10) + cI * (TMP19)) + (-cI
      * (TMP9 * TMP15) + cI * (TMP8 * TMP14))) + (TMP16 * (-cI * (P3[2]) + cI *
      (P1[2])) + (V1[4] * (-cI * (TMP14) + cI * (TMP15)) + V3[4] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V2[5] = denom * (OM2 * P2[3] * (TMP16 * (-cI * (TMP10) + cI * (TMP19)) + (-cI
      * (TMP9 * TMP15) + cI * (TMP8 * TMP14))) + (TMP16 * (-cI * (P3[3]) + cI *
      (P1[3])) + (V1[5] * (-cI * (TMP14) + cI * (TMP15)) + V3[5] * (-cI *
      (TMP8) + cI * (TMP9)))));
}


void VVVV3P0_4(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V3[], Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> denom; 
  double P4[4]; 
  Complex<double> TMP18; 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (-cI * (V1[2] * TMP18) + cI * (V3[2] * TMP12)); 
  V4[3] = denom * (-cI * (V1[3] * TMP18) + cI * (V3[3] * TMP12)); 
  V4[4] = denom * (-cI * (V1[4] * TMP18) + cI * (V3[4] * TMP12)); 
  V4[5] = denom * (-cI * (V1[5] * TMP18) + cI * (V3[5] * TMP12)); 
}


void FFS3_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> TMP1; 
  double P3[4]; 
  S3[0] = +F1[0] + F2[0]; 
  S3[1] = +F1[1] + F2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * TMP1; 
}


void VVVV1P0_4(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V3[], Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP16; 
  Complex<double> denom; 
  double P4[4]; 
  Complex<double> TMP18; 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (-cI * (V1[2] * TMP18) + cI * (V2[2] * TMP16)); 
  V4[3] = denom * (-cI * (V1[3] * TMP18) + cI * (V2[3] * TMP16)); 
  V4[4] = denom * (-cI * (V1[4] * TMP18) + cI * (V2[4] * TMP16)); 
  V4[5] = denom * (-cI * (V1[5] * TMP18) + cI * (V2[5] * TMP16)); 
}


void VVVV4P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (-cI * (V3[2] * TMP21) + cI * (V2[2] * TMP26)); 
  V1[3] = denom * (-cI * (V3[3] * TMP21) + cI * (V2[3] * TMP26)); 
  V1[4] = denom * (-cI * (V3[4] * TMP21) + cI * (V2[4] * TMP26)); 
  V1[5] = denom * (-cI * (V3[5] * TMP21) + cI * (V2[5] * TMP26)); 
}


void VVVV3_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> V4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP20; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * (-cI * (TMP18 * TMP20) + cI * (TMP12 * TMP26)); 
}


void VVVV4_4(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP16; 
  double OM4; 
  Complex<double> denom; 
  Complex<double> TMP28; 
  double P4[4]; 
  Complex<double> TMP25; 
  OM4 = 0.; 
  if (M4 != 0.)
    OM4 = 1./(M4 * M4); 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP25 = (V2[2] * P4[0] - V2[3] * P4[1] - V2[4] * P4[2] - V2[5] * P4[3]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP28 = (V3[2] * P4[0] - V3[3] * P4[1] - V3[4] * P4[2] - V3[5] * P4[3]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (OM4 * P4[0] * (-cI * (TMP12 * TMP28) + cI * (TMP16 * TMP25))
      + (-cI * (V2[2] * TMP16) + cI * (V3[2] * TMP12)));
  V4[3] = denom * (OM4 * P4[1] * (-cI * (TMP12 * TMP28) + cI * (TMP16 * TMP25))
      + (-cI * (V2[3] * TMP16) + cI * (V3[3] * TMP12)));
  V4[4] = denom * (OM4 * P4[2] * (-cI * (TMP12 * TMP28) + cI * (TMP16 * TMP25))
      + (-cI * (V2[4] * TMP16) + cI * (V3[4] * TMP12)));
  V4[5] = denom * (OM4 * P4[3] * (-cI * (TMP12 * TMP28) + cI * (TMP16 * TMP25))
      + (-cI * (V2[5] * TMP16) + cI * (V3[5] * TMP12)));
}


void FFS4_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> TMP1; 
  Complex<double> TMP0; 
  double P3[4]; 
  S3[0] = +F1[0] + F2[0]; 
  S3[1] = +F1[1] + F2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * (+cI * (TMP0 + TMP1)); 
}


void FFS1_1(Complex<double> F2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + S3[0]; 
  F1[1] = +F2[1] + S3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * F2[2] * M1 * S3[2]; 
  F1[3] = denom * cI * F2[3] * M1 * S3[2]; 
  F1[4] = denom * cI * S3[2] * (F2[2] * (P1[3] - P1[0]) + F2[3] * (P1[1] + cI *
      (P1[2])));
  F1[5] = denom * - cI * S3[2] * (F2[2] * (+cI * (P1[2]) - P1[1]) + F2[3] *
      (P1[0] + P1[3]));
}

void FFS1_3_1(Complex<double> F2[], Complex<double> S3[], Complex<double>
    COUP1, Complex<double> COUP2, double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  int i; 
  Complex<double> Ftmp[6]; 
  FFS1_1(F2, S3, COUP1, M1, W1, F1); 
  FFS3_1(F2, S3, COUP2, M1, W1, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F1[i] = F1[i] + Ftmp[i]; 
    i++; 
  }
}

void VVVV3P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (-cI * (TMP18 * V4[2]) + cI * (V2[2] * TMP26)); 
  V1[3] = denom * (-cI * (TMP18 * V4[3]) + cI * (V2[3] * TMP26)); 
  V1[4] = denom * (-cI * (TMP18 * V4[4]) + cI * (V2[4] * TMP26)); 
  V1[5] = denom * (-cI * (TMP18 * V4[5]) + cI * (V2[5] * TMP26)); 
}


void FFV1P0_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * (V3[2] - V3[5]) + F1[5] * (+cI *
      (V3[4]) - V3[3]))));
  F2[3] = denom * (-cI) * (F1[2] * (P2[0] * (-1.) * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) - P2[3] *
      (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[5] - V3[2]) + (P2[1] *
      (V3[3] - cI * (V3[4])) + (P2[2] * (V3[4] + cI * (V3[3])) + P2[3] * (V3[5]
      - V3[2])))) + M2 * (F1[4] * (V3[3] + cI * (V3[4])) - F1[5] * (V3[2] +
      V3[5]))));
  F2[4] = denom * (-cI) * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] * (V3[3] +
      cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5] -
      V3[2])))) + (F1[5] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] * (-1.) *
      (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) + P2[3] * (V3[3] - cI
      * (V3[4]))))) + M2 * (F1[2] * (-1.) * (V3[2] + V3[5]) + F1[3] * (+cI *
      (V3[4]) - V3[3]))));
  F2[5] = denom * cI * (F1[4] * (P2[0] * (-1.) * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[2] - V3[5]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[5] * (P2[0] * (V3[2] + V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) - P2[3] * (V3[2] + V3[5])))) + M2 * (F1[2] * (V3[3] + cI *
      (V3[4])) + F1[3] * (V3[2] - V3[5]))));
}


void VVVV1_2(Complex<double> V1[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P2[4]; 
  Complex<double> TMP23; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  double OM2; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP23 = (P2[0] * V4[2] - P2[1] * V4[3] - P2[2] * V4[4] - P2[3] * V4[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (OM2 * P2[0] * (-cI * (TMP16 * TMP23) + cI * (TMP15 * TMP20))
      + (-cI * (V3[2] * TMP20) + cI * (TMP16 * V4[2])));
  V2[3] = denom * (OM2 * P2[1] * (-cI * (TMP16 * TMP23) + cI * (TMP15 * TMP20))
      + (-cI * (V3[3] * TMP20) + cI * (TMP16 * V4[3])));
  V2[4] = denom * (OM2 * P2[2] * (-cI * (TMP16 * TMP23) + cI * (TMP15 * TMP20))
      + (-cI * (V3[4] * TMP20) + cI * (TMP16 * V4[4])));
  V2[5] = denom * (OM2 * P2[3] * (-cI * (TMP16 * TMP23) + cI * (TMP15 * TMP20))
      + (-cI * (V3[5] * TMP20) + cI * (TMP16 * V4[5])));
}


void FFV2_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP5; 
  double P3[4]; 
  double OM3; 
  Complex<double> denom; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP5 = (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
      F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])));
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI) * (F2[4] * F1[2] + F2[5] * F1[3] - P3[0] * OM3 * TMP5); 
  V3[3] = denom * (-cI) * (-F2[5] * F1[2] - F2[4] * F1[3] - P3[1] * OM3 *
      TMP5);
  V3[4] = denom * (-cI) * (-cI * (F2[5] * F1[2]) + cI * (F2[4] * F1[3]) - P3[2]
      * OM3 * TMP5);
  V3[5] = denom * (-cI) * (F2[5] * F1[3] - F2[4] * F1[2] - P3[3] * OM3 * TMP5); 
}

void FFV2_3_3(Complex<double> F1[], Complex<double> F2[], Complex<double>
    COUP1, Complex<double> COUP2, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  double OM3; 
  int i; 
  Complex<double> denom; 
  Complex<double> Vtmp[6]; 
  FFV2_3(F1, F2, COUP1, M3, W3, V3); 
  FFV3_3(F1, F2, COUP2, M3, W3, Vtmp); 
  i = 2; 
  while (i < 6)
  {
    V3[i] = V3[i] + Vtmp[i]; 
    i++; 
  }
}
void FFV2_4_3(Complex<double> F1[], Complex<double> F2[], Complex<double>
    COUP1, Complex<double> COUP2, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  double OM3; 
  int i; 
  Complex<double> denom; 
  Complex<double> Vtmp[6]; 
  FFV2_3(F1, F2, COUP1, M3, W3, V3); 
  FFV4_3(F1, F2, COUP2, M3, W3, Vtmp); 
  i = 2; 
  while (i < 6)
  {
    V3[i] = V3[i] + Vtmp[i]; 
    i++; 
  }
}
void FFV2_5_3(Complex<double> F1[], Complex<double> F2[], Complex<double>
    COUP1, Complex<double> COUP2, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  double OM3; 
  int i; 
  Complex<double> denom; 
  Complex<double> Vtmp[6]; 
  FFV2_3(F1, F2, COUP1, M3, W3, V3); 
  FFV5_3(F1, F2, COUP2, M3, W3, Vtmp); 
  i = 2; 
  while (i < 6)
  {
    V3[i] = V3[i] + Vtmp[i]; 
    i++; 
  }
}

void FFS2_1(Complex<double> F2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + S3[0]; 
  F1[1] = +F2[1] + S3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * S3[2] * (F2[4] * (P1[0] + P1[3]) + (F2[5] * (P1[1] + cI
      * (P1[2])) + F2[2] * M1));
  F1[3] = denom * - cI * S3[2] * (F2[4] * (+cI * (P1[2]) - P1[1]) + (F2[5] *
      (P1[3] - P1[0]) - F2[3] * M1));
  F1[4] = denom * cI * S3[2] * (F2[2] * (P1[3] - P1[0]) + (F2[3] * (P1[1] + cI
      * (P1[2])) - F2[4] * M1));
  F1[5] = denom * - cI * S3[2] * (F2[2] * (+cI * (P1[2]) - P1[1]) + (F2[3] *
      (P1[0] + P1[3]) + F2[5] * M1));
}


void VVVV4P0_4(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V3[], Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP16; 
  Complex<double> denom; 
  double P4[4]; 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (-cI * (V2[2] * TMP16) + cI * (V3[2] * TMP12)); 
  V4[3] = denom * (-cI * (V2[3] * TMP16) + cI * (V3[3] * TMP12)); 
  V4[4] = denom * (-cI * (V2[4] * TMP16) + cI * (V3[4] * TMP12)); 
  V4[5] = denom * (-cI * (V2[5] * TMP16) + cI * (V3[5] * TMP12)); 
}


void FFV5_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * 4. * (V3[2] - V3[5]) + 4. * (F1[5]
      * (+cI * (V3[4]) - V3[3])))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))) + M2 * (F1[4] * (-4.) * (V3[3] + cI
      * (V3[4])) + 4. * (F1[5] * (V3[2] + V3[5])))));
  F2[4] = denom * (-4. * cI) * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] *
      (V3[3] + cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5]
      - V3[2])))) + (+1./4. * (M2 * (F1[3] * (+cI * (V3[4]) - V3[3]) + 4. *
      (F1[2] * (-1./4.) * (V3[2] + V3[5])))) + F1[5] * (P2[0] * (V3[3] - cI *
      (V3[4])) + (P2[1] * (-1.) * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] +
      V3[5])) + P2[3] * (V3[3] - cI * (V3[4])))))));
  F2[5] = denom * (-4. * cI) * (F1[4] * (P2[0] * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[2]) + cI * (V3[5])) -
      P2[3] * (V3[3] + cI * (V3[4]))))) + (+1./4. * (M2 * (F1[3] * (V3[5] -
      V3[2]) + 4. * (F1[2] * (-1./4.) * (V3[3] + cI * (V3[4]))))) + F1[5] *
      (P2[0] * (-1.) * (V3[2] + V3[5]) + (P2[1] * (V3[3] - cI * (V3[4])) +
      (P2[2] * (V3[4] + cI * (V3[3])) + P2[3] * (V3[2] + V3[5]))))));
}


void SSS1_2(Complex<double> S1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> S2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  S2[0] = +S1[0] + S3[0]; 
  S2[1] = +S1[1] + S3[1]; 
  P2[0] = -S2[0].real(); 
  P2[1] = -S2[1].real(); 
  P2[2] = -S2[1].imag(); 
  P2[3] = -S2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S2[2] = denom * cI * S3[2] * S1[2]; 
}


void SSSS1_1(Complex<double> S2[], Complex<double> S3[], Complex<double> S4[],
    Complex<double> COUP, double M1, double W1, Complex<double> S1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P1[4]; 
  S1[0] = +S2[0] + S3[0] + S4[0]; 
  S1[1] = +S2[1] + S3[1] + S4[1]; 
  P1[0] = -S1[0].real(); 
  P1[1] = -S1[1].real(); 
  P1[2] = -S1[1].imag(); 
  P1[3] = -S1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S1[2] = denom * cI * S4[2] * S3[2] * S2[2]; 
}


void VVVV2_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> V4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP21; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * (-1.) * (-2. * cI * (TMP12 * TMP26) + cI * (TMP18 * TMP20 +
      TMP16 * TMP21));
}


void FFV1_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P3[4]; 
  double OM3; 
  Complex<double> TMP3; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP3 = (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
      (F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])) +
      (F1[4] * (F2[2] * (P3[0] - P3[3]) - F2[3] * (P3[1] + cI * (P3[2]))) +
      F1[5] * (F2[2] * (+cI * (P3[2]) - P3[1]) + F2[3] * (P3[0] + P3[3])))));
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI) * (F2[4] * F1[2] + F2[5] * F1[3] + F2[2] * F1[4] +
      F2[3] * F1[5] - P3[0] * OM3 * TMP3);
  V3[3] = denom * (-cI) * (F2[3] * F1[4] + F2[2] * F1[5] - F2[5] * F1[2] -
      F2[4] * F1[3] - P3[1] * OM3 * TMP3);
  V3[4] = denom * (-cI) * (-cI * (F2[5] * F1[2] + F2[2] * F1[5]) + cI * (F2[4]
      * F1[3] + F2[3] * F1[4]) - P3[2] * OM3 * TMP3);
  V3[5] = denom * (-cI) * (F2[5] * F1[3] + F2[2] * F1[4] - F2[4] * F1[2] -
      F2[3] * F1[5] - P3[3] * OM3 * TMP3);
}


void SSSS1_0(Complex<double> S1[], Complex<double> S2[], Complex<double> S3[],
    Complex<double> S4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  vertex = COUP * - cI * S4[2] * S3[2] * S2[2] * S1[2]; 
}


void VSS1_3(Complex<double> V1[], Complex<double> S2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  double P3[4]; 
  Complex<double> denom; 
  Complex<double> TMP9; 
  Complex<double> TMP8; 
  P2[0] = S2[0].real(); 
  P2[1] = S2[1].real(); 
  P2[2] = S2[1].imag(); 
  P2[3] = S2[0].imag(); 
  S3[0] = +V1[0] + S2[0]; 
  S3[1] = +V1[1] + S2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * S2[2] * (-cI * (TMP9) + cI * (TMP8)); 
}


void VVVV2P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (-1.) * (-2. * cI * (V2[2] * TMP26) + cI * (TMP18 * V4[2] +
      V3[2] * TMP21));
  V1[3] = denom * (-1.) * (-2. * cI * (V2[3] * TMP26) + cI * (TMP18 * V4[3] +
      V3[3] * TMP21));
  V1[4] = denom * (-1.) * (-2. * cI * (V2[4] * TMP26) + cI * (TMP18 * V4[4] +
      V3[4] * TMP21));
  V1[5] = denom * (-1.) * (-2. * cI * (V2[5] * TMP26) + cI * (TMP18 * V4[5] +
      V3[5] * TMP21));
}


void VVV1_1(Complex<double> V2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> TMP11; 
  double P1[4]; 
  Complex<double> TMP10; 
  double P2[4]; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP15; 
  Complex<double> TMP14; 
  double OM1; 
  Complex<double> TMP13; 
  Complex<double> TMP18; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  P2[0] = V2[0].real(); 
  P2[1] = V2[1].real(); 
  P2[2] = V2[1].imag(); 
  P2[3] = V2[0].imag(); 
  P3[0] = V3[0].real(); 
  P3[1] = V3[1].real(); 
  P3[2] = V3[1].imag(); 
  P3[3] = V3[0].imag(); 
  V1[0] = +V2[0] + V3[0]; 
  V1[1] = +V2[1] + V3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP11 = (P1[0] * P3[0] - P1[1] * P3[1] - P1[2] * P3[2] - P1[3] * P3[3]); 
  TMP10 = (P1[0] * P2[0] - P1[1] * P2[1] - P1[2] * P2[2] - P1[3] * P2[3]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (OM1 * P1[0] * (TMP18 * (-cI * (TMP11) + cI * (TMP10)) + (-cI
      * (TMP13 * TMP15) + cI * (TMP14 * TMP17))) + (TMP18 * (-cI * (P2[0]) + cI
      * (P3[0])) + (V2[2] * (-cI * (TMP14) + cI * (TMP15)) + V3[2] * (-cI *
      (TMP17) + cI * (TMP13)))));
  V1[3] = denom * (OM1 * P1[1] * (TMP18 * (-cI * (TMP11) + cI * (TMP10)) + (-cI
      * (TMP13 * TMP15) + cI * (TMP14 * TMP17))) + (TMP18 * (-cI * (P2[1]) + cI
      * (P3[1])) + (V2[3] * (-cI * (TMP14) + cI * (TMP15)) + V3[3] * (-cI *
      (TMP17) + cI * (TMP13)))));
  V1[4] = denom * (OM1 * P1[2] * (TMP18 * (-cI * (TMP11) + cI * (TMP10)) + (-cI
      * (TMP13 * TMP15) + cI * (TMP14 * TMP17))) + (TMP18 * (-cI * (P2[2]) + cI
      * (P3[2])) + (V2[4] * (-cI * (TMP14) + cI * (TMP15)) + V3[4] * (-cI *
      (TMP17) + cI * (TMP13)))));
  V1[5] = denom * (OM1 * P1[3] * (TMP18 * (-cI * (TMP11) + cI * (TMP10)) + (-cI
      * (TMP13 * TMP15) + cI * (TMP14 * TMP17))) + (TMP18 * (-cI * (P2[3]) + cI
      * (P3[3])) + (V2[5] * (-cI * (TMP14) + cI * (TMP15)) + V3[5] * (-cI *
      (TMP17) + cI * (TMP13)))));
}


void VVSS1_0(Complex<double> V1[], Complex<double> V2[], Complex<double> S3[],
    Complex<double> S4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * - cI * TMP12 * S4[2] * S3[2]; 
}


void FFS3_0(Complex<double> F1[], Complex<double> F2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP1; 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  vertex = COUP * - cI * TMP1 * S3[2]; 
}


void FFV4_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP5; 
  Complex<double> TMP7; 
  double P3[4]; 
  double OM3; 
  Complex<double> denom; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP5 = (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
      F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])));
  TMP7 = (F1[4] * (F2[2] * (P3[0] - P3[3]) - F2[3] * (P3[1] + cI * (P3[2]))) +
      F1[5] * (F2[2] * (+cI * (P3[2]) - P3[1]) + F2[3] * (P3[0] + P3[3])));
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-2. * cI) * (OM3 * - 1./2. * P3[0] * (TMP5 + 2. * (TMP7)) +
      (+1./2. * (F2[4] * F1[2] + F2[5] * F1[3]) + F2[2] * F1[4] + F2[3] *
      F1[5]));
  V3[3] = denom * (-2. * cI) * (OM3 * - 1./2. * P3[1] * (TMP5 + 2. * (TMP7)) +
      (-1./2. * (F2[5] * F1[2] + F2[4] * F1[3]) + F2[3] * F1[4] + F2[2] *
      F1[5]));
  V3[4] = denom * 2. * cI * (OM3 * 1./2. * P3[2] * (TMP5 + 2. * (TMP7)) +
      (+1./2. * cI * (F2[5] * F1[2]) - 1./2. * cI * (F2[4] * F1[3]) - cI *
      (F2[3] * F1[4]) + cI * (F2[2] * F1[5])));
  V3[5] = denom * 2. * cI * (OM3 * 1./2. * P3[3] * (TMP5 + 2. * (TMP7)) +
      (+1./2. * (F2[4] * F1[2]) - 1./2. * (F2[5] * F1[3]) - F2[2] * F1[4] +
      F2[3] * F1[5]));
}


void FFV5P0_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * 4. * cI * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] * (V3[3]
      - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5] -
      V3[2])))) + (+1./4. * (M1 * (F2[5] * (V3[3] + cI * (V3[4])) + 4. * (F2[4]
      * 1./4. * (V3[2] + V3[5])))) + F2[3] * (P1[0] * (V3[3] + cI * (V3[4])) +
      (P1[1] * (-1.) * (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] +
      V3[5])) + P1[3] * (V3[3] + cI * (V3[4])))))));
  F1[3] = denom * 4. * cI * (F2[2] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (+1./4. * (M1 * (F2[5] * (V3[2] - V3[5]) + 4. *
      (F2[4] * 1./4. * (V3[3] - cI * (V3[4]))))) + F2[3] * (P1[0] * (-1.) *
      (V3[2] + V3[5]) + (P1[1] * (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI
      * (V3[3])) + P1[3] * (V3[2] + V3[5]))))));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + (F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * 4. * (V3[5] - V3[2]) + 4. *
      (F2[3] * (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * cI * (F2[4] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (F2[5] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + M1 * (F2[2] * 4. * (+cI * (V3[4]) - V3[3]) + 4. * (F2[3] *
      (V3[2] + V3[5])))));
}


void VVVV1P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP18; 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (-cI * (TMP18 * V4[2]) + cI * (V3[2] * TMP21)); 
  V1[3] = denom * (-cI * (TMP18 * V4[3]) + cI * (V3[3] * TMP21)); 
  V1[4] = denom * (-cI * (TMP18 * V4[4]) + cI * (V3[4] * TMP21)); 
  V1[5] = denom * (-cI * (TMP18 * V4[5]) + cI * (V3[5] * TMP21)); 
}


void VVVV5_4(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP16; 
  double OM4; 
  Complex<double> denom; 
  Complex<double> TMP28; 
  double P4[4]; 
  Complex<double> TMP24; 
  Complex<double> TMP25; 
  Complex<double> TMP18; 
  OM4 = 0.; 
  if (M4 != 0.)
    OM4 = 1./(M4 * M4); 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP24 = (V1[2] * P4[0] - V1[3] * P4[1] - V1[4] * P4[2] - V1[5] * P4[3]); 
  TMP25 = (V2[2] * P4[0] - V2[3] * P4[1] - V2[4] * P4[2] - V2[5] * P4[3]); 
  TMP28 = (V3[2] * P4[0] - V3[3] * P4[1] - V3[4] * P4[2] - V3[5] * P4[3]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * 1./2. * (OM4 * - P4[0] * (-2. * cI * (TMP18 * TMP24) + cI *
      (TMP16 * TMP25 + TMP12 * TMP28)) + (-2. * cI * (V1[2] * TMP18) + cI *
      (V2[2] * TMP16 + V3[2] * TMP12)));
  V4[3] = denom * 1./2. * (OM4 * - P4[1] * (-2. * cI * (TMP18 * TMP24) + cI *
      (TMP16 * TMP25 + TMP12 * TMP28)) + (-2. * cI * (V1[3] * TMP18) + cI *
      (V2[3] * TMP16 + V3[3] * TMP12)));
  V4[4] = denom * 1./2. * (OM4 * - P4[2] * (-2. * cI * (TMP18 * TMP24) + cI *
      (TMP16 * TMP25 + TMP12 * TMP28)) + (-2. * cI * (V1[4] * TMP18) + cI *
      (V2[4] * TMP16 + V3[4] * TMP12)));
  V4[5] = denom * 1./2. * (OM4 * - P4[3] * (-2. * cI * (TMP18 * TMP24) + cI *
      (TMP16 * TMP25 + TMP12 * TMP28)) + (-2. * cI * (V1[5] * TMP18) + cI *
      (V2[5] * TMP16 + V3[5] * TMP12)));
}


void VVV1P0_3(Complex<double> V1[], Complex<double> V2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P1[4]; 
  double P2[4]; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> denom; 
  Complex<double> TMP9; 
  Complex<double> TMP13; 
  Complex<double> TMP8; 
  P1[0] = V1[0].real(); 
  P1[1] = V1[1].real(); 
  P1[2] = V1[1].imag(); 
  P1[3] = V1[0].imag(); 
  P2[0] = V2[0].real(); 
  P2[1] = V2[1].real(); 
  P2[2] = V2[1].imag(); 
  P2[3] = V2[0].imag(); 
  V3[0] = +V1[0] + V2[0]; 
  V3[1] = +V1[1] + V2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (TMP12 * (-cI * (P1[0]) + cI * (P2[0])) + (V1[2] * (-cI *
      (TMP17) + cI * (TMP13)) + V2[2] * (-cI * (TMP8) + cI * (TMP9))));
  V3[3] = denom * (TMP12 * (-cI * (P1[1]) + cI * (P2[1])) + (V1[3] * (-cI *
      (TMP17) + cI * (TMP13)) + V2[3] * (-cI * (TMP8) + cI * (TMP9))));
  V3[4] = denom * (TMP12 * (-cI * (P1[2]) + cI * (P2[2])) + (V1[4] * (-cI *
      (TMP17) + cI * (TMP13)) + V2[4] * (-cI * (TMP8) + cI * (TMP9))));
  V3[5] = denom * (TMP12 * (-cI * (P1[3]) + cI * (P2[3])) + (V1[5] * (-cI *
      (TMP17) + cI * (TMP13)) + V2[5] * (-cI * (TMP8) + cI * (TMP9))));
}


void SSSS1_2(Complex<double> S1[], Complex<double> S3[], Complex<double> S4[],
    Complex<double> COUP, double M2, double W2, Complex<double> S2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  S2[0] = +S1[0] + S3[0] + S4[0]; 
  S2[1] = +S1[1] + S3[1] + S4[1]; 
  P2[0] = -S2[0].real(); 
  P2[1] = -S2[1].real(); 
  P2[2] = -S2[1].imag(); 
  P2[3] = -S2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S2[2] = denom * cI * S4[2] * S3[2] * S1[2]; 
}


void FFS3_1(Complex<double> F2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + S3[0]; 
  F1[1] = +F2[1] + S3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * - cI * S3[2] * (F2[4] * (P1[0] + P1[3]) + F2[5] * (P1[1] + cI
      * (P1[2])));
  F1[3] = denom * cI * S3[2] * (F2[4] * (+cI * (P1[2]) - P1[1]) + F2[5] *
      (P1[3] - P1[0]));
  F1[4] = denom * cI * F2[4] * M1 * S3[2]; 
  F1[5] = denom * cI * F2[5] * M1 * S3[2]; 
}


void FFV2_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP4; 
  TMP4 = (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
      F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])));
  vertex = COUP * - cI * TMP4; 
}

void FFV2_3_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP1, Complex<double> COUP2, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> tmp; 
  FFV2_0(F1, F2, V3, COUP1, vertex); 
  FFV3_0(F1, F2, V3, COUP2, tmp); 
  vertex = vertex + tmp; 
}
void FFV2_4_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP1, Complex<double> COUP2, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> tmp; 
  FFV2_0(F1, F2, V3, COUP1, vertex); 
  FFV4_0(F1, F2, V3, COUP2, tmp); 
  vertex = vertex + tmp; 
}
void FFV2_5_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP1, Complex<double> COUP2, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> tmp; 
  FFV2_0(F1, F2, V3, COUP1, vertex); 
  FFV5_0(F1, F2, V3, COUP2, tmp); 
  vertex = vertex + tmp; 
}

void VVVV4P0_3(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V4[], Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P3[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI * (V1[2] * TMP21) + cI * (TMP12 * V4[2])); 
  V3[3] = denom * (-cI * (V1[3] * TMP21) + cI * (TMP12 * V4[3])); 
  V3[4] = denom * (-cI * (V1[4] * TMP21) + cI * (TMP12 * V4[4])); 
  V3[5] = denom * (-cI * (V1[5] * TMP21) + cI * (TMP12 * V4[5])); 
}


void FFV5_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * 4. * cI * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] * (V3[3]
      - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5] -
      V3[2])))) + (+1./4. * (M1 * (F2[5] * (V3[3] + cI * (V3[4])) + 4. * (F2[4]
      * 1./4. * (V3[2] + V3[5])))) + F2[3] * (P1[0] * (V3[3] + cI * (V3[4])) +
      (P1[1] * (-1.) * (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] +
      V3[5])) + P1[3] * (V3[3] + cI * (V3[4])))))));
  F1[3] = denom * 4. * cI * (F2[2] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (+1./4. * (M1 * (F2[5] * (V3[2] - V3[5]) + 4. *
      (F2[4] * 1./4. * (V3[3] - cI * (V3[4]))))) + F2[3] * (P1[0] * (-1.) *
      (V3[2] + V3[5]) + (P1[1] * (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI
      * (V3[3])) + P1[3] * (V3[2] + V3[5]))))));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + (F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * 4. * (V3[5] - V3[2]) + 4. *
      (F2[3] * (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * cI * (F2[4] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (F2[5] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + M1 * (F2[2] * 4. * (+cI * (V3[4]) - V3[3]) + 4. * (F2[3] *
      (V3[2] + V3[5])))));
}


void FFS4_1(Complex<double> F2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + S3[0]; 
  F1[1] = +F2[1] + S3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * - cI * S3[2] * (F2[4] * (P1[0] + P1[3]) + (F2[5] * (P1[1] +
      cI * (P1[2])) - F2[2] * M1));
  F1[3] = denom * cI * S3[2] * (F2[4] * (+cI * (P1[2]) - P1[1]) + (F2[5] *
      (P1[3] - P1[0]) + F2[3] * M1));
  F1[4] = denom * cI * S3[2] * (F2[2] * (P1[3] - P1[0]) + (F2[3] * (P1[1] + cI
      * (P1[2])) + F2[4] * M1));
  F1[5] = denom * - cI * S3[2] * (F2[2] * (+cI * (P1[2]) - P1[1]) + (F2[3] *
      (P1[0] + P1[3]) - F2[5] * M1));
}


void FFV3P0_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * 2. * (V3[5] - V3[2]) + 2. * (F1[5]
      * (V3[3] - cI * (V3[4]))))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))) + M2 * (F1[4] * 2. * (V3[3] + cI *
      (V3[4])) - 2. * (F1[5] * (V3[2] + V3[5])))));
  F2[4] = denom * 2. * cI * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] * (V3[3]
      + cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5] -
      V3[2])))) + (+1./2. * (M2 * (F1[3] * (V3[3] - cI * (V3[4])) + 2. * (F1[2]
      * 1./2. * (V3[2] + V3[5])))) + F1[5] * (P2[0] * (V3[3] - cI * (V3[4])) +
      (P2[1] * (-1.) * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] - cI * (V3[4])))))));
  F2[5] = denom * 2. * cI * (F1[4] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[2]) + cI * (V3[5])) - P2[3] *
      (V3[3] + cI * (V3[4]))))) + (+1./2. * (M2 * (F1[3] * (V3[2] - V3[5]) + 2.
      * (F1[2] * 1./2. * (V3[3] + cI * (V3[4]))))) + F1[5] * (P2[0] * (-1.) *
      (V3[2] + V3[5]) + (P2[1] * (V3[3] - cI * (V3[4])) + (P2[2] * (V3[4] + cI
      * (V3[3])) + P2[3] * (V3[2] + V3[5]))))));
}


void FFV1P0_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  Complex<double> denom; 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI) * (F2[4] * F1[2] + F2[5] * F1[3] + F2[2] * F1[4] +
      F2[3] * F1[5]);
  V3[3] = denom * (-cI) * (F2[3] * F1[4] + F2[2] * F1[5] - F2[5] * F1[2] -
      F2[4] * F1[3]);
  V3[4] = denom * (-cI) * (-cI * (F2[5] * F1[2] + F2[2] * F1[5]) + cI * (F2[4]
      * F1[3] + F2[3] * F1[4]));
  V3[5] = denom * (-cI) * (F2[5] * F1[3] + F2[2] * F1[4] - F2[4] * F1[2] -
      F2[3] * F1[5]);
}


void VVVV3P0_3(Complex<double> V1[], Complex<double> V2[], Complex<double>
    V4[], Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> denom; 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-cI * (V2[2] * TMP20) + cI * (TMP12 * V4[2])); 
  V3[3] = denom * (-cI * (V2[3] * TMP20) + cI * (TMP12 * V4[3])); 
  V3[4] = denom * (-cI * (V2[4] * TMP20) + cI * (TMP12 * V4[4])); 
  V3[5] = denom * (-cI * (V2[5] * TMP20) + cI * (TMP12 * V4[5])); 
}


void VVS1_1(Complex<double> V2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  double OM1; 
  Complex<double> TMP13; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + S3[0]; 
  V1[1] = +V2[1] + S3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * S3[2] * (-cI * (V2[2]) + cI * (P1[0] * OM1 * TMP13)); 
  V1[3] = denom * S3[2] * (-cI * (V2[3]) + cI * (P1[1] * OM1 * TMP13)); 
  V1[4] = denom * S3[2] * (-cI * (V2[4]) + cI * (P1[2] * OM1 * TMP13)); 
  V1[5] = denom * S3[2] * (-cI * (V2[5]) + cI * (P1[3] * OM1 * TMP13)); 
}


void FFS2_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> TMP1; 
  Complex<double> TMP0; 
  double P3[4]; 
  S3[0] = +F1[0] + F2[0]; 
  S3[1] = +F1[1] + F2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP1 = (F2[4] * F1[4] + F2[5] * F1[5]); 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * (-cI * (TMP1) + cI * (TMP0)); 
}


void VVVV5P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * 1./2. * (-2. * cI * (TMP18 * V4[2]) + cI * (V3[2] * TMP21 +
      V2[2] * TMP26));
  V1[3] = denom * 1./2. * (-2. * cI * (TMP18 * V4[3]) + cI * (V3[3] * TMP21 +
      V2[3] * TMP26));
  V1[4] = denom * 1./2. * (-2. * cI * (TMP18 * V4[4]) + cI * (V3[4] * TMP21 +
      V2[4] * TMP26));
  V1[5] = denom * 1./2. * (-2. * cI * (TMP18 * V4[5]) + cI * (V3[5] * TMP21 +
      V2[5] * TMP26));
}


void FFS3P0_2(Complex<double> F1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + S3[0]; 
  F2[1] = +F1[1] + S3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * S3[2] * (F1[4] * (P2[0] - P2[3]) + F1[5] * (+cI *
      (P2[2]) - P2[1]));
  F2[3] = denom * - cI * S3[2] * (F1[4] * (P2[1] + cI * (P2[2])) - F1[5] *
      (P2[0] + P2[3]));
  F2[4] = denom * cI * F1[4] * M2 * S3[2]; 
  F2[5] = denom * cI * F1[5] * M2 * S3[2]; 
}


void SSS1_0(Complex<double> S1[], Complex<double> S2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  vertex = COUP * - cI * S3[2] * S2[2] * S1[2]; 
}


void FFV3_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP6; 
  Complex<double> TMP4; 
  TMP4 = (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
      F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])));
  TMP6 = (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
      F1[5] * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] + V3[5])));
  vertex = COUP * (-cI * (TMP4) + 2. * cI * (TMP6)); 
}


void VVVV5_2(Complex<double> V1[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P2[4]; 
  Complex<double> TMP23; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  Complex<double> TMP26; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP23 = (P2[0] * V4[2] - P2[1] * V4[3] - P2[2] * V4[4] - P2[3] * V4[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * 1./2. * (OM2 * - P2[0] * (-2. * cI * (TMP15 * TMP20) + cI *
      (TMP16 * TMP23 + TMP8 * TMP26)) + (-2. * cI * (V3[2] * TMP20) + cI *
      (TMP16 * V4[2] + V1[2] * TMP26)));
  V2[3] = denom * 1./2. * (OM2 * - P2[1] * (-2. * cI * (TMP15 * TMP20) + cI *
      (TMP16 * TMP23 + TMP8 * TMP26)) + (-2. * cI * (V3[3] * TMP20) + cI *
      (TMP16 * V4[3] + V1[3] * TMP26)));
  V2[4] = denom * 1./2. * (OM2 * - P2[2] * (-2. * cI * (TMP15 * TMP20) + cI *
      (TMP16 * TMP23 + TMP8 * TMP26)) + (-2. * cI * (V3[4] * TMP20) + cI *
      (TMP16 * V4[4] + V1[4] * TMP26)));
  V2[5] = denom * 1./2. * (OM2 * - P2[3] * (-2. * cI * (TMP15 * TMP20) + cI *
      (TMP16 * TMP23 + TMP8 * TMP26)) + (-2. * cI * (V3[5] * TMP20) + cI *
      (TMP16 * V4[5] + V1[5] * TMP26)));
}


void VVVV2_2(Complex<double> V1[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P2[4]; 
  Complex<double> TMP23; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP15; 
  Complex<double> TMP26; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP23 = (P2[0] * V4[2] - P2[1] * V4[3] - P2[2] * V4[4] - P2[3] * V4[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (OM2 * P2[0] * (-2. * cI * (TMP8 * TMP26) + cI * (TMP15 *
      TMP20 + TMP16 * TMP23)) + (-cI * (V3[2] * TMP20 + TMP16 * V4[2]) + 2. *
      cI * (V1[2] * TMP26)));
  V2[3] = denom * (OM2 * P2[1] * (-2. * cI * (TMP8 * TMP26) + cI * (TMP15 *
      TMP20 + TMP16 * TMP23)) + (-cI * (V3[3] * TMP20 + TMP16 * V4[3]) + 2. *
      cI * (V1[3] * TMP26)));
  V2[4] = denom * (OM2 * P2[2] * (-2. * cI * (TMP8 * TMP26) + cI * (TMP15 *
      TMP20 + TMP16 * TMP23)) + (-cI * (V3[4] * TMP20 + TMP16 * V4[4]) + 2. *
      cI * (V1[4] * TMP26)));
  V2[5] = denom * (OM2 * P2[3] * (-2. * cI * (TMP8 * TMP26) + cI * (TMP15 *
      TMP20 + TMP16 * TMP23)) + (-cI * (V3[5] * TMP20 + TMP16 * V4[5]) + 2. *
      cI * (V1[5] * TMP26)));
}


void VVVV4_2(Complex<double> V1[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> TMP23; 
  Complex<double> TMP16; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  double OM2; 
  Complex<double> TMP8; 
  OM2 = 0.; 
  if (M2 != 0.)
    OM2 = 1./(M2 * M2); 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP23 = (P2[0] * V4[2] - P2[1] * V4[3] - P2[2] * V4[4] - P2[3] * V4[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (OM2 * P2[0] * (-cI * (TMP8 * TMP26) + cI * (TMP16 * TMP23))
      + (-cI * (TMP16 * V4[2]) + cI * (V1[2] * TMP26)));
  V2[3] = denom * (OM2 * P2[1] * (-cI * (TMP8 * TMP26) + cI * (TMP16 * TMP23))
      + (-cI * (TMP16 * V4[3]) + cI * (V1[3] * TMP26)));
  V2[4] = denom * (OM2 * P2[2] * (-cI * (TMP8 * TMP26) + cI * (TMP16 * TMP23))
      + (-cI * (TMP16 * V4[4]) + cI * (V1[4] * TMP26)));
  V2[5] = denom * (OM2 * P2[3] * (-cI * (TMP8 * TMP26) + cI * (TMP16 * TMP23))
      + (-cI * (TMP16 * V4[5]) + cI * (V1[5] * TMP26)));
}


void VSS1_1(Complex<double> S2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP11; 
  double P1[4]; 
  Complex<double> TMP10; 
  double P2[4]; 
  double P3[4]; 
  Complex<double> denom; 
  double OM1; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  P2[0] = S2[0].real(); 
  P2[1] = S2[1].real(); 
  P2[2] = S2[1].imag(); 
  P2[3] = S2[0].imag(); 
  P3[0] = S3[0].real(); 
  P3[1] = S3[1].real(); 
  P3[2] = S3[1].imag(); 
  P3[3] = S3[0].imag(); 
  V1[0] = +S2[0] + S3[0]; 
  V1[1] = +S2[1] + S3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP11 = (P1[0] * P3[0] - P1[1] * P3[1] - P1[2] * P3[2] - P1[3] * P3[3]); 
  TMP10 = (P1[0] * P2[0] - P1[1] * P2[1] - P1[2] * P2[2] - P1[3] * P2[3]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * S2[2] * S3[2] * (OM1 * P1[0] * (-cI * (TMP11) + cI * (TMP10))
      + (-cI * (P2[0]) + cI * (P3[0])));
  V1[3] = denom * S2[2] * S3[2] * (OM1 * P1[1] * (-cI * (TMP11) + cI * (TMP10))
      + (-cI * (P2[1]) + cI * (P3[1])));
  V1[4] = denom * S2[2] * S3[2] * (OM1 * P1[2] * (-cI * (TMP11) + cI * (TMP10))
      + (-cI * (P2[2]) + cI * (P3[2])));
  V1[5] = denom * S2[2] * S3[2] * (OM1 * P1[3] * (-cI * (TMP11) + cI * (TMP10))
      + (-cI * (P2[3]) + cI * (P3[3])));
}


void VVV1_3(Complex<double> V1[], Complex<double> V2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP11; 
  double P1[4]; 
  double P2[4]; 
  Complex<double> TMP19; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP9; 
  Complex<double> TMP13; 
  Complex<double> TMP8; 
  P1[0] = V1[0].real(); 
  P1[1] = V1[1].real(); 
  P1[2] = V1[1].imag(); 
  P1[3] = V1[0].imag(); 
  P2[0] = V2[0].real(); 
  P2[1] = V2[1].real(); 
  P2[2] = V2[1].imag(); 
  P2[3] = V2[0].imag(); 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0]; 
  V3[1] = +V1[1] + V2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP19 = (P2[0] * P3[0] - P2[1] * P3[1] - P2[2] * P3[2] - P2[3] * P3[3]); 
  TMP8 = (P2[0] * V1[2] - P2[1] * V1[3] - P2[2] * V1[4] - P2[3] * V1[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP11 = (P1[0] * P3[0] - P1[1] * P3[1] - P1[2] * P3[2] - P1[3] * P3[3]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (OM3 * P3[0] * (TMP12 * (-cI * (TMP19) + cI * (TMP11)) + (-cI
      * (TMP9 * TMP13) + cI * (TMP8 * TMP17))) + (TMP12 * (-cI * (P1[0]) + cI *
      (P2[0])) + (V1[2] * (-cI * (TMP17) + cI * (TMP13)) + V2[2] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V3[3] = denom * (OM3 * P3[1] * (TMP12 * (-cI * (TMP19) + cI * (TMP11)) + (-cI
      * (TMP9 * TMP13) + cI * (TMP8 * TMP17))) + (TMP12 * (-cI * (P1[1]) + cI *
      (P2[1])) + (V1[3] * (-cI * (TMP17) + cI * (TMP13)) + V2[3] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V3[4] = denom * (OM3 * P3[2] * (TMP12 * (-cI * (TMP19) + cI * (TMP11)) + (-cI
      * (TMP9 * TMP13) + cI * (TMP8 * TMP17))) + (TMP12 * (-cI * (P1[2]) + cI *
      (P2[2])) + (V1[4] * (-cI * (TMP17) + cI * (TMP13)) + V2[4] * (-cI *
      (TMP8) + cI * (TMP9)))));
  V3[5] = denom * (OM3 * P3[3] * (TMP12 * (-cI * (TMP19) + cI * (TMP11)) + (-cI
      * (TMP9 * TMP13) + cI * (TMP8 * TMP17))) + (TMP12 * (-cI * (P1[3]) + cI *
      (P2[3])) + (V1[5] * (-cI * (TMP17) + cI * (TMP13)) + V2[5] * (-cI *
      (TMP8) + cI * (TMP9)))));
}


void FFS3_2(Complex<double> F1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + S3[0]; 
  F2[1] = +F1[1] + S3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * S3[2] * (F1[4] * (P2[0] - P2[3]) + F1[5] * (+cI *
      (P2[2]) - P2[1]));
  F2[3] = denom * - cI * S3[2] * (F1[4] * (P2[1] + cI * (P2[2])) - F1[5] *
      (P2[0] + P2[3]));
  F2[4] = denom * cI * F1[4] * M2 * S3[2]; 
  F2[5] = denom * cI * F1[5] * M2 * S3[2]; 
}


void FFV4_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * 2. * cI * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] * (V3[3]
      - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5] -
      V3[2])))) + (+1./2. * (M1 * (F2[5] * (V3[3] + cI * (V3[4])) + 2. * (F2[4]
      * 1./2. * (V3[2] + V3[5])))) + F2[3] * (P1[0] * (V3[3] + cI * (V3[4])) +
      (P1[1] * (-1.) * (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] +
      V3[5])) + P1[3] * (V3[3] + cI * (V3[4])))))));
  F1[3] = denom * 2. * cI * (F2[2] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (+1./2. * (M1 * (F2[5] * (V3[2] - V3[5]) + 2. *
      (F2[4] * 1./2. * (V3[3] - cI * (V3[4]))))) + F2[3] * (P1[0] * (-1.) *
      (V3[2] + V3[5]) + (P1[1] * (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI
      * (V3[3])) + P1[3] * (V3[2] + V3[5]))))));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + (F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * 2. * (V3[5] - V3[2]) + 2. *
      (F2[3] * (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * cI * (F2[4] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (F2[5] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + M1 * (F2[2] * 2. * (+cI * (V3[4]) - V3[3]) + 2. * (F2[3] *
      (V3[2] + V3[5])))));
}


void VVVV1_4(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP16; 
  double OM4; 
  Complex<double> denom; 
  double P4[4]; 
  Complex<double> TMP24; 
  Complex<double> TMP25; 
  Complex<double> TMP18; 
  OM4 = 0.; 
  if (M4 != 0.)
    OM4 = 1./(M4 * M4); 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP24 = (V1[2] * P4[0] - V1[3] * P4[1] - V1[4] * P4[2] - V1[5] * P4[3]); 
  TMP25 = (V2[2] * P4[0] - V2[3] * P4[1] - V2[4] * P4[2] - V2[5] * P4[3]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (OM4 * P4[0] * (-cI * (TMP16 * TMP25) + cI * (TMP18 * TMP24))
      + (-cI * (V1[2] * TMP18) + cI * (V2[2] * TMP16)));
  V4[3] = denom * (OM4 * P4[1] * (-cI * (TMP16 * TMP25) + cI * (TMP18 * TMP24))
      + (-cI * (V1[3] * TMP18) + cI * (V2[3] * TMP16)));
  V4[4] = denom * (OM4 * P4[2] * (-cI * (TMP16 * TMP25) + cI * (TMP18 * TMP24))
      + (-cI * (V1[4] * TMP18) + cI * (V2[4] * TMP16)));
  V4[5] = denom * (OM4 * P4[3] * (-cI * (TMP16 * TMP25) + cI * (TMP18 * TMP24))
      + (-cI * (V1[5] * TMP18) + cI * (V2[5] * TMP16)));
}


void FFV2P0_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI *
      (V3[4])));
  F1[3] = denom * - cI * M1 * (F2[4] * (+cI * (V3[4]) - V3[3]) + F2[5] * (V3[5]
      - V3[2]));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * (-cI) * (F2[4] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (-1.) * (V3[2] + V3[5]) + (P1[2] * (+cI * (V3[2] + V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + F2[5] * (P1[0] * (V3[2] - V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) + P1[3]
      * (V3[2] - V3[5])))));
}


void FFV5_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP5; 
  Complex<double> TMP7; 
  double P3[4]; 
  double OM3; 
  Complex<double> denom; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP5 = (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
      F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])));
  TMP7 = (F1[4] * (F2[2] * (P3[0] - P3[3]) - F2[3] * (P3[1] + cI * (P3[2]))) +
      F1[5] * (F2[2] * (+cI * (P3[2]) - P3[1]) + F2[3] * (P3[0] + P3[3])));
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (-4. * cI) * (OM3 * - 1./4. * P3[0] * (TMP5 + 4. * (TMP7)) +
      (+1./4. * (F2[4] * F1[2] + F2[5] * F1[3]) + F2[2] * F1[4] + F2[3] *
      F1[5]));
  V3[3] = denom * (-4. * cI) * (OM3 * - 1./4. * P3[1] * (TMP5 + 4. * (TMP7)) +
      (-1./4. * (F2[5] * F1[2] + F2[4] * F1[3]) + F2[3] * F1[4] + F2[2] *
      F1[5]));
  V3[4] = denom * 4. * cI * (OM3 * 1./4. * P3[2] * (TMP5 + 4. * (TMP7)) +
      (+1./4. * cI * (F2[5] * F1[2]) - 1./4. * cI * (F2[4] * F1[3]) - cI *
      (F2[3] * F1[4]) + cI * (F2[2] * F1[5])));
  V3[5] = denom * 4. * cI * (OM3 * 1./4. * P3[3] * (TMP5 + 4. * (TMP7)) +
      (+1./4. * (F2[4] * F1[2]) - 1./4. * (F2[5] * F1[3]) - F2[2] * F1[4] +
      F2[3] * F1[5]));
}


void VVVV3_3(Complex<double> V1[], Complex<double> V2[], Complex<double> V4[],
    Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP27; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP27 = (P3[0] * V4[2] - P3[1] * V4[3] - P3[2] * V4[4] - P3[3] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (OM3 * P3[0] * (-cI * (TMP12 * TMP27) + cI * (TMP17 * TMP20))
      + (-cI * (V2[2] * TMP20) + cI * (TMP12 * V4[2])));
  V3[3] = denom * (OM3 * P3[1] * (-cI * (TMP12 * TMP27) + cI * (TMP17 * TMP20))
      + (-cI * (V2[3] * TMP20) + cI * (TMP12 * V4[3])));
  V3[4] = denom * (OM3 * P3[2] * (-cI * (TMP12 * TMP27) + cI * (TMP17 * TMP20))
      + (-cI * (V2[4] * TMP20) + cI * (TMP12 * V4[4])));
  V3[5] = denom * (OM3 * P3[3] * (-cI * (TMP12 * TMP27) + cI * (TMP17 * TMP20))
      + (-cI * (V2[5] * TMP20) + cI * (TMP12 * V4[5])));
}


void FFV3P0_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * (-2. * cI) * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + (+1./2. * (M1 * (+2. * (F2[4] * (-1./2.) * (V3[2] + V3[5]))
      - F2[5] * (V3[3] + cI * (V3[4])))) + F2[3] * (P1[0] * (V3[3] + cI *
      (V3[4])) + (P1[1] * (-1.) * (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI *
      (V3[2] + V3[5])) + P1[3] * (V3[3] + cI * (V3[4])))))));
  F1[3] = denom * (-2. * cI) * (F2[2] * (P1[0] * (V3[3] - cI * (V3[4])) +
      (P1[1] * (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) +
      P1[3] * (+cI * (V3[4]) - V3[3])))) + (+1./2. * (M1 * (F2[5] * (V3[5] -
      V3[2]) + 2. * (F2[4] * 1./2. * (+cI * (V3[4]) - V3[3])))) + F2[3] *
      (P1[0] * (-1.) * (V3[2] + V3[5]) + (P1[1] * (V3[3] + cI * (V3[4])) +
      (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[2] + V3[5]))))));
  F1[4] = denom * cI * (F2[4] * (P1[0] * (-1.) * (V3[2] + V3[5]) + (P1[1] *
      (V3[3] - cI * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[2]
      + V3[5])))) + (F2[5] * (P1[0] * (-1.) * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[2] - V3[5]) + (P1[2] * (-cI * (V3[5]) + cI * (V3[2])) + P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * 2. * (V3[5] - V3[2]) + 2. *
      (F2[3] * (V3[3] + cI * (V3[4]))))));
  F1[5] = denom * (-cI) * (F2[4] * (P1[0] * (V3[3] - cI * (V3[4])) + (P1[1] *
      (-1.) * (V3[2] + V3[5]) + (P1[2] * (+cI * (V3[2] + V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + (F2[5] * (P1[0] * (V3[2] - V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) + P1[3]
      * (V3[2] - V3[5])))) + M1 * (F2[2] * 2. * (+cI * (V3[4]) - V3[3]) + 2. *
      (F2[3] * (V3[2] + V3[5])))));
}


void VVV1P0_1(Complex<double> V2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  double P1[4]; 
  double P2[4]; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP15; 
  Complex<double> TMP14; 
  Complex<double> TMP13; 
  Complex<double> TMP18; 
  P2[0] = V2[0].real(); 
  P2[1] = V2[1].real(); 
  P2[2] = V2[1].imag(); 
  P2[3] = V2[0].imag(); 
  P3[0] = V3[0].real(); 
  P3[1] = V3[1].real(); 
  P3[2] = V3[1].imag(); 
  P3[3] = V3[0].imag(); 
  V1[0] = +V2[0] + V3[0]; 
  V1[1] = +V2[1] + V3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP15 = (P2[0] * V3[2] - P2[1] * V3[3] - P2[2] * V3[4] - P2[3] * V3[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (TMP18 * (-cI * (P2[0]) + cI * (P3[0])) + (V2[2] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[2] * (-cI * (TMP17) + cI * (TMP13))));
  V1[3] = denom * (TMP18 * (-cI * (P2[1]) + cI * (P3[1])) + (V2[3] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[3] * (-cI * (TMP17) + cI * (TMP13))));
  V1[4] = denom * (TMP18 * (-cI * (P2[2]) + cI * (P3[2])) + (V2[4] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[4] * (-cI * (TMP17) + cI * (TMP13))));
  V1[5] = denom * (TMP18 * (-cI * (P2[3]) + cI * (P3[3])) + (V2[5] * (-cI *
      (TMP14) + cI * (TMP15)) + V3[5] * (-cI * (TMP17) + cI * (TMP13))));
}


void FFS1_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> TMP0; 
  double P3[4]; 
  S3[0] = +F1[0] + F2[0]; 
  S3[1] = +F1[1] + F2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP0 = (F2[2] * F1[2] + F2[3] * F1[3]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * TMP0; 
}

void FFS1_3_3(Complex<double> F1[], Complex<double> F2[], Complex<double>
    COUP1, Complex<double> COUP2, double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> denom; 
  Complex<double> Stmp[3]; 
  double P3[4]; 
  int i; 
  FFS1_3(F1, F2, COUP1, M3, W3, S3); 
  FFS3_3(F1, F2, COUP2, M3, W3, Stmp); 
  i = 2; 
  while (i < 3)
  {
    S3[i] = S3[i] + Stmp[i]; 
    i++; 
  }
}

void FFV3_3(Complex<double> F1[], Complex<double> F2[], Complex<double> COUP,
    double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP5; 
  Complex<double> TMP7; 
  double P3[4]; 
  double OM3; 
  Complex<double> denom; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +F1[0] + F2[0]; 
  V3[1] = +F1[1] + F2[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP5 = (F1[2] * (F2[4] * (P3[0] + P3[3]) + F2[5] * (P3[1] + cI * (P3[2]))) +
      F1[3] * (F2[4] * (P3[1] - cI * (P3[2])) + F2[5] * (P3[0] - P3[3])));
  TMP7 = (F1[4] * (F2[2] * (P3[0] - P3[3]) - F2[3] * (P3[1] + cI * (P3[2]))) +
      F1[5] * (F2[2] * (+cI * (P3[2]) - P3[1]) + F2[3] * (P3[0] + P3[3])));
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * 2. * cI * (OM3 * 1./2. * P3[0] * (TMP5 - 2. * (TMP7)) +
      (-1./2. * (F2[4] * F1[2] + F2[5] * F1[3]) + F2[2] * F1[4] + F2[3] *
      F1[5]));
  V3[3] = denom * 2. * cI * (OM3 * 1./2. * P3[1] * (TMP5 - 2. * (TMP7)) +
      (+1./2. * (F2[5] * F1[2] + F2[4] * F1[3]) + F2[3] * F1[4] + F2[2] *
      F1[5]));
  V3[4] = denom * (-2. * cI) * (OM3 * 1./2. * P3[2] * (+2. * (TMP7) - TMP5) +
      (-1./2. * cI * (F2[5] * F1[2]) + 1./2. * cI * (F2[4] * F1[3]) - cI *
      (F2[3] * F1[4]) + cI * (F2[2] * F1[5])));
  V3[5] = denom * (-2. * cI) * (OM3 * 1./2. * P3[3] * (+2. * (TMP7) - TMP5) +
      (-1./2. * (F2[4] * F1[2]) + 1./2. * (F2[5] * F1[3]) - F2[2] * F1[4] +
      F2[3] * F1[5]));
}


void VVVV1_1(Complex<double> V2[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP22; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP14; 
  double OM1; 
  Complex<double> TMP18; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP22 = (P1[0] * V4[2] - P1[1] * V4[3] - P1[2] * V4[4] - P1[3] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (OM1 * P1[0] * (-cI * (TMP14 * TMP21) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[2]) + cI * (V3[2] * TMP21)));
  V1[3] = denom * (OM1 * P1[1] * (-cI * (TMP14 * TMP21) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[3]) + cI * (V3[3] * TMP21)));
  V1[4] = denom * (OM1 * P1[2] * (-cI * (TMP14 * TMP21) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[4]) + cI * (V3[4] * TMP21)));
  V1[5] = denom * (OM1 * P1[3] * (-cI * (TMP14 * TMP21) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[5]) + cI * (V3[5] * TMP21)));
}


void FFV2_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))));
  F2[4] = denom * - cI * M2 * (F1[2] * (-1.) * (V3[2] + V3[5]) + F1[3] * (+cI *
      (V3[4]) - V3[3]));
  F2[5] = denom * cI * M2 * (F1[2] * (V3[3] + cI * (V3[4])) + F1[3] * (V3[2] -
      V3[5]));
}

void FFV2_3_2(Complex<double> F1[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> Ftmp[6]; 
  double P2[4]; 
  Complex<double> denom; 
  int i; 
  FFV2_2(F1, V3, COUP1, M2, W2, F2); 
  FFV3_2(F1, V3, COUP2, M2, W2, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F2[i] = F2[i] + Ftmp[i]; 
    i++; 
  }
}
void FFV2_4_2(Complex<double> F1[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> Ftmp[6]; 
  double P2[4]; 
  Complex<double> denom; 
  int i; 
  FFV2_2(F1, V3, COUP1, M2, W2, F2); 
  FFV4_2(F1, V3, COUP2, M2, W2, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F2[i] = F2[i] + Ftmp[i]; 
    i++; 
  }
}
void FFV2_5_2(Complex<double> F1[], Complex<double> V3[], Complex<double>
    COUP1, Complex<double> COUP2, double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> Ftmp[6]; 
  double P2[4]; 
  Complex<double> denom; 
  int i; 
  FFV2_2(F1, V3, COUP1, M2, W2, F2); 
  FFV5_2(F1, V3, COUP2, M2, W2, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F2[i] = F2[i] + Ftmp[i]; 
    i++; 
  }
}

void VVVV3_4(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double OM4; 
  Complex<double> denom; 
  Complex<double> TMP28; 
  double P4[4]; 
  Complex<double> TMP24; 
  Complex<double> TMP18; 
  OM4 = 0.; 
  if (M4 != 0.)
    OM4 = 1./(M4 * M4); 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP24 = (V1[2] * P4[0] - V1[3] * P4[1] - V1[4] * P4[2] - V1[5] * P4[3]); 
  TMP28 = (V3[2] * P4[0] - V3[3] * P4[1] - V3[4] * P4[2] - V3[5] * P4[3]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (OM4 * P4[0] * (-cI * (TMP12 * TMP28) + cI * (TMP18 * TMP24))
      + (-cI * (V1[2] * TMP18) + cI * (V3[2] * TMP12)));
  V4[3] = denom * (OM4 * P4[1] * (-cI * (TMP12 * TMP28) + cI * (TMP18 * TMP24))
      + (-cI * (V1[3] * TMP18) + cI * (V3[3] * TMP12)));
  V4[4] = denom * (OM4 * P4[2] * (-cI * (TMP12 * TMP28) + cI * (TMP18 * TMP24))
      + (-cI * (V1[4] * TMP18) + cI * (V3[4] * TMP12)));
  V4[5] = denom * (OM4 * P4[3] * (-cI * (TMP12 * TMP28) + cI * (TMP18 * TMP24))
      + (-cI * (V1[5] * TMP18) + cI * (V3[5] * TMP12)));
}


void VVVV5_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> V4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> TMP21; 
  Complex<double> TMP26; 
  Complex<double> TMP18; 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * 1./2. * (-2. * cI * (TMP18 * TMP20) + cI * (TMP16 * TMP21 +
      TMP12 * TMP26));
}


void SSSS1_3(Complex<double> S1[], Complex<double> S2[], Complex<double> S4[],
    Complex<double> COUP, double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P3[4]; 
  Complex<double> denom; 
  S3[0] = +S1[0] + S2[0] + S4[0]; 
  S3[1] = +S1[1] + S2[1] + S4[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * S4[2] * S2[2] * S1[2]; 
}


void VVVV2_1(Complex<double> V2[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP22; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP14; 
  double OM1; 
  Complex<double> TMP13; 
  Complex<double> TMP18; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP22 = (P1[0] * V4[2] - P1[1] * V4[3] - P1[2] * V4[4] - P1[3] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (OM1 * P1[0] * (-2. * cI * (TMP13 * TMP26) + cI * (TMP18 *
      TMP22 + TMP14 * TMP21)) + (-cI * (TMP18 * V4[2] + V3[2] * TMP21) + 2. *
      cI * (V2[2] * TMP26)));
  V1[3] = denom * (OM1 * P1[1] * (-2. * cI * (TMP13 * TMP26) + cI * (TMP18 *
      TMP22 + TMP14 * TMP21)) + (-cI * (TMP18 * V4[3] + V3[3] * TMP21) + 2. *
      cI * (V2[3] * TMP26)));
  V1[4] = denom * (OM1 * P1[2] * (-2. * cI * (TMP13 * TMP26) + cI * (TMP18 *
      TMP22 + TMP14 * TMP21)) + (-cI * (TMP18 * V4[4] + V3[4] * TMP21) + 2. *
      cI * (V2[4] * TMP26)));
  V1[5] = denom * (OM1 * P1[3] * (-2. * cI * (TMP13 * TMP26) + cI * (TMP18 *
      TMP22 + TMP14 * TMP21)) + (-cI * (TMP18 * V4[5] + V3[5] * TMP21) + 2. *
      cI * (V2[5] * TMP26)));
}


void FFV1_0(Complex<double> F1[], Complex<double> F2[], Complex<double> V3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP2; 
  TMP2 = (F1[2] * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) +
      (F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])) +
      (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) +
      F1[5] * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] + V3[5])))));
  vertex = COUP * - cI * TMP2; 
}


void VVS1_3(Complex<double> V1[], Complex<double> V2[], Complex<double> COUP,
    double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P3[4]; 
  Complex<double> denom; 
  S3[0] = +V1[0] + V2[0]; 
  S3[1] = +V1[1] + V2[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * TMP12; 
}


void VVVV1P0_2(Complex<double> V1[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> denom; 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (-cI * (V3[2] * TMP20) + cI * (TMP16 * V4[2])); 
  V2[3] = denom * (-cI * (V3[3] * TMP20) + cI * (TMP16 * V4[3])); 
  V2[4] = denom * (-cI * (V3[4] * TMP20) + cI * (TMP16 * V4[4])); 
  V2[5] = denom * (-cI * (V3[5] * TMP20) + cI * (TMP16 * V4[5])); 
}


void VVVV2P0_2(Complex<double> V1[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP16; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (-1.) * (-2. * cI * (V1[2] * TMP26) + cI * (V3[2] * TMP20 +
      TMP16 * V4[2]));
  V2[3] = denom * (-1.) * (-2. * cI * (V1[3] * TMP26) + cI * (V3[3] * TMP20 +
      TMP16 * V4[3]));
  V2[4] = denom * (-1.) * (-2. * cI * (V1[4] * TMP26) + cI * (V3[4] * TMP20 +
      TMP16 * V4[4]));
  V2[5] = denom * (-1.) * (-2. * cI * (V1[5] * TMP26) + cI * (V3[5] * TMP20 +
      TMP16 * V4[5]));
}


void VVSS1_3(Complex<double> V1[], Complex<double> V2[], Complex<double> S4[],
    Complex<double> COUP, double M3, double W3, Complex<double> S3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  double P3[4]; 
  Complex<double> denom; 
  S3[0] = +V1[0] + V2[0] + S4[0]; 
  S3[1] = +V1[1] + V2[1] + S4[1]; 
  P3[0] = -S3[0].real(); 
  P3[1] = -S3[1].real(); 
  P3[2] = -S3[1].imag(); 
  P3[3] = -S3[0].imag(); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S3[2] = denom * cI * TMP12 * S4[2]; 
}


void VVSS1P0_2(Complex<double> V1[], Complex<double> S3[], Complex<double>
    S4[], Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  V2[0] = +V1[0] + S3[0] + S4[0]; 
  V2[1] = +V1[1] + S3[1] + S4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * - cI * V1[2] * S4[2] * S3[2]; 
  V2[3] = denom * - cI * V1[3] * S4[2] * S3[2]; 
  V2[4] = denom * - cI * V1[4] * S4[2] * S3[2]; 
  V2[5] = denom * - cI * V1[5] * S4[2] * S3[2]; 
}


void FFV4_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * 2. * (V3[2] - V3[5]) + 2. * (F1[5]
      * (+cI * (V3[4]) - V3[3])))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))) + M2 * (F1[4] * (-2.) * (V3[3] + cI
      * (V3[4])) + 2. * (F1[5] * (V3[2] + V3[5])))));
  F2[4] = denom * (-2. * cI) * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] *
      (V3[3] + cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5]
      - V3[2])))) + (+1./2. * (M2 * (F1[3] * (+cI * (V3[4]) - V3[3]) + 2. *
      (F1[2] * (-1./2.) * (V3[2] + V3[5])))) + F1[5] * (P2[0] * (V3[3] - cI *
      (V3[4])) + (P2[1] * (-1.) * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] +
      V3[5])) + P2[3] * (V3[3] - cI * (V3[4])))))));
  F2[5] = denom * (-2. * cI) * (F1[4] * (P2[0] * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[2]) + cI * (V3[5])) -
      P2[3] * (V3[3] + cI * (V3[4]))))) + (+1./2. * (M2 * (F1[3] * (V3[5] -
      V3[2]) + 2. * (F1[2] * (-1./2.) * (V3[3] + cI * (V3[4]))))) + F1[5] *
      (P2[0] * (-1.) * (V3[2] + V3[5]) + (P2[1] * (V3[3] - cI * (V3[4])) +
      (P2[2] * (V3[4] + cI * (V3[3])) + P2[3] * (V3[2] + V3[5]))))));
}


void FFV5P0_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + (F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))) + M2 * (F1[4] * 4. * (V3[2] - V3[5]) + 4. * (F1[5]
      * (+cI * (V3[4]) - V3[3])))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + (F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))) + M2 * (F1[4] * (-4.) * (V3[3] + cI
      * (V3[4])) + 4. * (F1[5] * (V3[2] + V3[5])))));
  F2[4] = denom * (-4. * cI) * (F1[4] * (P2[0] * (V3[5] - V3[2]) + (P2[1] *
      (V3[3] + cI * (V3[4])) + (P2[2] * (V3[4] - cI * (V3[3])) + P2[3] * (V3[5]
      - V3[2])))) + (+1./4. * (M2 * (F1[3] * (+cI * (V3[4]) - V3[3]) + 4. *
      (F1[2] * (-1./4.) * (V3[2] + V3[5])))) + F1[5] * (P2[0] * (V3[3] - cI *
      (V3[4])) + (P2[1] * (-1.) * (V3[2] + V3[5]) + (P2[2] * (+cI * (V3[2] +
      V3[5])) + P2[3] * (V3[3] - cI * (V3[4])))))));
  F2[5] = denom * (-4. * cI) * (F1[4] * (P2[0] * (V3[3] + cI * (V3[4])) +
      (P2[1] * (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[2]) + cI * (V3[5])) -
      P2[3] * (V3[3] + cI * (V3[4]))))) + (+1./4. * (M2 * (F1[3] * (V3[5] -
      V3[2]) + 4. * (F1[2] * (-1./4.) * (V3[3] + cI * (V3[4]))))) + F1[5] *
      (P2[0] * (-1.) * (V3[2] + V3[5]) + (P2[1] * (V3[3] - cI * (V3[4])) +
      (P2[2] * (V3[4] + cI * (V3[3])) + P2[3] * (V3[2] + V3[5]))))));
}


void FFV2P0_2(Complex<double> F1[], Complex<double> V3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + V3[0]; 
  F2[1] = +F1[1] + V3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * (F1[2] * (P2[0] * (V3[2] + V3[5]) + (P2[1] * (-1.) *
      (V3[3] + cI * (V3[4])) + (P2[2] * (+cI * (V3[3]) - V3[4]) - P2[3] *
      (V3[2] + V3[5])))) + F1[3] * (P2[0] * (V3[3] - cI * (V3[4])) + (P2[1] *
      (V3[5] - V3[2]) + (P2[2] * (-cI * (V3[5]) + cI * (V3[2])) + P2[3] * (+cI
      * (V3[4]) - V3[3])))));
  F2[3] = denom * cI * (F1[2] * (P2[0] * (V3[3] + cI * (V3[4])) + (P2[1] *
      (-1.) * (V3[2] + V3[5]) + (P2[2] * (-1.) * (+cI * (V3[2] + V3[5])) +
      P2[3] * (V3[3] + cI * (V3[4]))))) + F1[3] * (P2[0] * (V3[2] - V3[5]) +
      (P2[1] * (+cI * (V3[4]) - V3[3]) + (P2[2] * (-1.) * (V3[4] + cI *
      (V3[3])) + P2[3] * (V3[2] - V3[5])))));
  F2[4] = denom * - cI * M2 * (F1[2] * (-1.) * (V3[2] + V3[5]) + F1[3] * (+cI *
      (V3[4]) - V3[3]));
  F2[5] = denom * cI * M2 * (F1[2] * (V3[3] + cI * (V3[4])) + F1[3] * (V3[2] -
      V3[5]));
}


void VVVV4_0(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> V4[], Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP16; 
  Complex<double> TMP21; 
  Complex<double> TMP26; 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * (-cI * (TMP16 * TMP21) + cI * (TMP12 * TMP26)); 
}


void VVS1P0_1(Complex<double> V2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  V1[0] = +V2[0] + S3[0]; 
  V1[1] = +V2[1] + S3[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * - cI * V2[2] * S3[2]; 
  V1[3] = denom * - cI * V2[3] * S3[2]; 
  V1[4] = denom * - cI * V2[4] * S3[2]; 
  V1[5] = denom * - cI * V2[5] * S3[2]; 
}


void FFS1P0_1(Complex<double> F2[], Complex<double> S3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + S3[0]; 
  F1[1] = +F2[1] + S3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * F2[2] * M1 * S3[2]; 
  F1[3] = denom * cI * F2[3] * M1 * S3[2]; 
  F1[4] = denom * cI * S3[2] * (F2[2] * (P1[3] - P1[0]) + F2[3] * (P1[1] + cI *
      (P1[2])));
  F1[5] = denom * - cI * S3[2] * (F2[2] * (+cI * (P1[2]) - P1[1]) + F2[3] *
      (P1[0] + P1[3]));
}


void VVSS1_4(Complex<double> V1[], Complex<double> V2[], Complex<double> S3[],
    Complex<double> COUP, double M4, double W4, Complex<double> S4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> denom; 
  double P4[4]; 
  S4[0] = +V1[0] + V2[0] + S3[0]; 
  S4[1] = +V1[1] + V2[1] + S3[1]; 
  P4[0] = -S4[0].real(); 
  P4[1] = -S4[1].real(); 
  P4[2] = -S4[1].imag(); 
  P4[3] = -S4[0].imag(); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  S4[2] = denom * cI * TMP12 * S3[2]; 
}


void VVVV5_3(Complex<double> V1[], Complex<double> V2[], Complex<double> V4[],
    Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP27; 
  Complex<double> TMP9; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP27 = (P3[0] * V4[2] - P3[1] * V4[3] - P3[2] * V4[4] - P3[3] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * 1./2. * (OM3 * - P3[0] * (-2. * cI * (TMP17 * TMP20) + cI *
      (TMP9 * TMP21 + TMP12 * TMP27)) + (-2. * cI * (V2[2] * TMP20) + cI *
      (V1[2] * TMP21 + TMP12 * V4[2])));
  V3[3] = denom * 1./2. * (OM3 * - P3[1] * (-2. * cI * (TMP17 * TMP20) + cI *
      (TMP9 * TMP21 + TMP12 * TMP27)) + (-2. * cI * (V2[3] * TMP20) + cI *
      (V1[3] * TMP21 + TMP12 * V4[3])));
  V3[4] = denom * 1./2. * (OM3 * - P3[2] * (-2. * cI * (TMP17 * TMP20) + cI *
      (TMP9 * TMP21 + TMP12 * TMP27)) + (-2. * cI * (V2[4] * TMP20) + cI *
      (V1[4] * TMP21 + TMP12 * V4[4])));
  V3[5] = denom * 1./2. * (OM3 * - P3[3] * (-2. * cI * (TMP17 * TMP20) + cI *
      (TMP9 * TMP21 + TMP12 * TMP27)) + (-2. * cI * (V2[5] * TMP20) + cI *
      (V1[5] * TMP21 + TMP12 * V4[5])));
}


void VVVV3_1(Complex<double> V2[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP22; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  double OM1; 
  Complex<double> TMP13; 
  Complex<double> TMP18; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP22 = (P1[0] * V4[2] - P1[1] * V4[3] - P1[2] * V4[4] - P1[3] * V4[5]); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * (OM1 * P1[0] * (-cI * (TMP13 * TMP26) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[2]) + cI * (V2[2] * TMP26)));
  V1[3] = denom * (OM1 * P1[1] * (-cI * (TMP13 * TMP26) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[3]) + cI * (V2[3] * TMP26)));
  V1[4] = denom * (OM1 * P1[2] * (-cI * (TMP13 * TMP26) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[4]) + cI * (V2[4] * TMP26)));
  V1[5] = denom * (OM1 * P1[3] * (-cI * (TMP13 * TMP26) + cI * (TMP18 * TMP22))
      + (-cI * (TMP18 * V4[5]) + cI * (V2[5] * TMP26)));
}


void FFS4_2(Complex<double> F1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + S3[0]; 
  F2[1] = +F1[1] + S3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * S3[2] * (F1[4] * (P2[0] - P2[3]) + (F1[5] * (+cI *
      (P2[2]) - P2[1]) + F1[2] * M2));
  F2[3] = denom * - cI * S3[2] * (F1[4] * (P2[1] + cI * (P2[2])) + (F1[5] *
      (-1.) * (P2[0] + P2[3]) - F1[3] * M2));
  F2[4] = denom * - cI * S3[2] * (F1[2] * (-1.) * (P2[0] + P2[3]) + (F1[3] *
      (+cI * (P2[2]) - P2[1]) - F1[4] * M2));
  F2[5] = denom * cI * S3[2] * (F1[2] * (P2[1] + cI * (P2[2])) + (F1[3] *
      (P2[0] - P2[3]) + F1[5] * M2));
}


void FFS1_2(Complex<double> F1[], Complex<double> S3[], Complex<double> COUP,
    double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> denom; 
  F2[0] = +F1[0] + S3[0]; 
  F2[1] = +F1[1] + S3[1]; 
  P2[0] = -F2[0].real(); 
  P2[1] = -F2[1].real(); 
  P2[2] = -F2[1].imag(); 
  P2[3] = -F2[0].imag(); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F2[2] = denom * cI * F1[2] * M2 * S3[2]; 
  F2[3] = denom * cI * F1[3] * M2 * S3[2]; 
  F2[4] = denom * - cI * S3[2] * (F1[2] * (-1.) * (P2[0] + P2[3]) + F1[3] *
      (+cI * (P2[2]) - P2[1]));
  F2[5] = denom * cI * S3[2] * (F1[2] * (P2[1] + cI * (P2[2])) + F1[3] * (P2[0]
      - P2[3]));
}

void FFS1_3_2(Complex<double> F1[], Complex<double> S3[], Complex<double>
    COUP1, Complex<double> COUP2, double M2, double W2, Complex<double> F2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> Ftmp[6]; 
  double P2[4]; 
  Complex<double> denom; 
  int i; 
  FFS1_2(F1, S3, COUP1, M2, W2, F2); 
  FFS3_2(F1, S3, COUP2, M2, W2, Ftmp); 
  i = 2; 
  while (i < 6)
  {
    F2[i] = F2[i] + Ftmp[i]; 
    i++; 
  }
}

void VVVV2_4(Complex<double> V1[], Complex<double> V2[], Complex<double> V3[],
    Complex<double> COUP, double M4, double W4, Complex<double> V4[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  Complex<double> TMP16; 
  double OM4; 
  Complex<double> denom; 
  Complex<double> TMP28; 
  double P4[4]; 
  Complex<double> TMP24; 
  Complex<double> TMP25; 
  Complex<double> TMP18; 
  OM4 = 0.; 
  if (M4 != 0.)
    OM4 = 1./(M4 * M4); 
  V4[0] = +V1[0] + V2[0] + V3[0]; 
  V4[1] = +V1[1] + V2[1] + V3[1]; 
  P4[0] = -V4[0].real(); 
  P4[1] = -V4[1].real(); 
  P4[2] = -V4[1].imag(); 
  P4[3] = -V4[0].imag(); 
  TMP24 = (V1[2] * P4[0] - V1[3] * P4[1] - V1[4] * P4[2] - V1[5] * P4[3]); 
  TMP25 = (V2[2] * P4[0] - V2[3] * P4[1] - V2[4] * P4[2] - V2[5] * P4[3]); 
  TMP28 = (V3[2] * P4[0] - V3[3] * P4[1] - V3[4] * P4[2] - V3[5] * P4[3]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP16 = (V3[2] * V1[2] - V3[3] * V1[3] - V3[4] * V1[4] - V3[5] * V1[5]); 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  Complex<double> tmp = ((P4[0] * P4[0]) - (P4[1] * P4[1]) - (P4[2] * P4[2]) -
      (P4[3] * P4[3]) - M4 * (M4 - cI * W4));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V4[2] = denom * (OM4 * P4[0] * (-2. * cI * (TMP12 * TMP28) + cI * (TMP18 *
      TMP24 + TMP16 * TMP25)) + (-cI * (V1[2] * TMP18 + V2[2] * TMP16) + 2. *
      cI * (V3[2] * TMP12)));
  V4[3] = denom * (OM4 * P4[1] * (-2. * cI * (TMP12 * TMP28) + cI * (TMP18 *
      TMP24 + TMP16 * TMP25)) + (-cI * (V1[3] * TMP18 + V2[3] * TMP16) + 2. *
      cI * (V3[3] * TMP12)));
  V4[4] = denom * (OM4 * P4[2] * (-2. * cI * (TMP12 * TMP28) + cI * (TMP18 *
      TMP24 + TMP16 * TMP25)) + (-cI * (V1[4] * TMP18 + V2[4] * TMP16) + 2. *
      cI * (V3[4] * TMP12)));
  V4[5] = denom * (OM4 * P4[3] * (-2. * cI * (TMP12 * TMP28) + cI * (TMP18 *
      TMP24 + TMP16 * TMP25)) + (-cI * (V1[5] * TMP18 + V2[5] * TMP16) + 2. *
      cI * (V3[5] * TMP12)));
}


void VVVV3P0_2(Complex<double> V1[], Complex<double> V3[], Complex<double>
    V4[], Complex<double> COUP, double M2, double W2, Complex<double> V2[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P2[4]; 
  Complex<double> TMP20; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  V2[0] = +V1[0] + V3[0] + V4[0]; 
  V2[1] = +V1[1] + V3[1] + V4[1]; 
  P2[0] = -V2[0].real(); 
  P2[1] = -V2[1].real(); 
  P2[2] = -V2[1].imag(); 
  P2[3] = -V2[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  Complex<double> tmp = ((P2[0] * P2[0]) - (P2[1] * P2[1]) - (P2[2] * P2[2]) -
      (P2[3] * P2[3]) - M2 * (M2 - cI * W2));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V2[2] = denom * (-cI * (V3[2] * TMP20) + cI * (V1[2] * TMP26)); 
  V2[3] = denom * (-cI * (V3[3] * TMP20) + cI * (V1[3] * TMP26)); 
  V2[4] = denom * (-cI * (V3[4] * TMP20) + cI * (V1[4] * TMP26)); 
  V2[5] = denom * (-cI * (V3[5] * TMP20) + cI * (V1[5] * TMP26)); 
}


void VVS1_0(Complex<double> V1[], Complex<double> V2[], Complex<double> S3[],
    Complex<double> COUP, Complex<double> & vertex)
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP12; 
  TMP12 = (V1[2] * V2[2] - V1[3] * V2[3] - V1[4] * V2[4] - V1[5] * V2[5]); 
  vertex = COUP * - cI * TMP12 * S3[2]; 
}


void VVVV5_1(Complex<double> V2[], Complex<double> V3[], Complex<double> V4[],
    Complex<double> COUP, double M1, double W1, Complex<double> V1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> TMP22; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  Complex<double> TMP26; 
  Complex<double> TMP14; 
  double OM1; 
  Complex<double> TMP13; 
  Complex<double> TMP18; 
  OM1 = 0.; 
  if (M1 != 0.)
    OM1 = 1./(M1 * M1); 
  V1[0] = +V2[0] + V3[0] + V4[0]; 
  V1[1] = +V2[1] + V3[1] + V4[1]; 
  P1[0] = -V1[0].real(); 
  P1[1] = -V1[1].real(); 
  P1[2] = -V1[1].imag(); 
  P1[3] = -V1[0].imag(); 
  TMP26 = (V3[2] * V4[2] - V3[3] * V4[3] - V3[4] * V4[4] - V3[5] * V4[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  TMP22 = (P1[0] * V4[2] - P1[1] * V4[3] - P1[2] * V4[4] - P1[3] * V4[5]); 
  TMP18 = (V3[2] * V2[2] - V3[3] * V2[3] - V3[4] * V2[4] - V3[5] * V2[5]); 
  TMP14 = (P1[0] * V3[2] - P1[1] * V3[3] - P1[2] * V3[4] - P1[3] * V3[5]); 
  TMP13 = (P1[0] * V2[2] - P1[1] * V2[3] - P1[2] * V2[4] - P1[3] * V2[5]); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V1[2] = denom * 1./2. * (OM1 * - P1[0] * (-2. * cI * (TMP18 * TMP22) + cI *
      (TMP14 * TMP21 + TMP13 * TMP26)) + (-2. * cI * (TMP18 * V4[2]) + cI *
      (V3[2] * TMP21 + V2[2] * TMP26)));
  V1[3] = denom * 1./2. * (OM1 * - P1[1] * (-2. * cI * (TMP18 * TMP22) + cI *
      (TMP14 * TMP21 + TMP13 * TMP26)) + (-2. * cI * (TMP18 * V4[3]) + cI *
      (V3[3] * TMP21 + V2[3] * TMP26)));
  V1[4] = denom * 1./2. * (OM1 * - P1[2] * (-2. * cI * (TMP18 * TMP22) + cI *
      (TMP14 * TMP21 + TMP13 * TMP26)) + (-2. * cI * (TMP18 * V4[4]) + cI *
      (V3[4] * TMP21 + V2[4] * TMP26)));
  V1[5] = denom * 1./2. * (OM1 * - P1[3] * (-2. * cI * (TMP18 * TMP22) + cI *
      (TMP14 * TMP21 + TMP13 * TMP26)) + (-2. * cI * (TMP18 * V4[5]) + cI *
      (V3[5] * TMP21 + V2[5] * TMP26)));
}


void FFV1P0_1(Complex<double> F2[], Complex<double> V3[], Complex<double> COUP,
    double M1, double W1, Complex<double> F1[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  double P1[4]; 
  Complex<double> denom; 
  F1[0] = +F2[0] + V3[0]; 
  F1[1] = +F2[1] + V3[1]; 
  P1[0] = -F1[0].real(); 
  P1[1] = -F1[1].real(); 
  P1[2] = -F1[1].imag(); 
  P1[3] = -F1[0].imag(); 
  Complex<double> tmp = ((P1[0] * P1[0]) - (P1[1] * P1[1]) - (P1[2] * P1[2]) -
      (P1[3] * P1[3]) - M1 * (M1 - cI * W1));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  F1[2] = denom * cI * (F2[2] * (P1[0] * (V3[5] - V3[2]) + (P1[1] * (V3[3] - cI
      * (V3[4])) + (P1[2] * (V3[4] + cI * (V3[3])) + P1[3] * (V3[5] - V3[2]))))
      + (F2[3] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] * (-1.) * (V3[2] +
      V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (V3[3] + cI *
      (V3[4]))))) + M1 * (F2[4] * (V3[2] + V3[5]) + F2[5] * (V3[3] + cI *
      (V3[4])))));
  F1[3] = denom * (-cI) * (F2[2] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] - V3[5]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) + P1[3] *
      (V3[3] - cI * (V3[4]))))) + (F2[3] * (P1[0] * (V3[2] + V3[5]) + (P1[1] *
      (-1.) * (V3[3] + cI * (V3[4])) + (P1[2] * (+cI * (V3[3]) - V3[4]) - P1[3]
      * (V3[2] + V3[5])))) + M1 * (F2[4] * (+cI * (V3[4]) - V3[3]) + F2[5] *
      (V3[5] - V3[2]))));
  F1[4] = denom * (-cI) * (F2[4] * (P1[0] * (V3[2] + V3[5]) + (P1[1] * (+cI *
      (V3[4]) - V3[3]) + (P1[2] * (-1.) * (V3[4] + cI * (V3[3])) - P1[3] *
      (V3[2] + V3[5])))) + (F2[5] * (P1[0] * (V3[3] + cI * (V3[4])) + (P1[1] *
      (V3[5] - V3[2]) + (P1[2] * (-cI * (V3[2]) + cI * (V3[5])) - P1[3] *
      (V3[3] + cI * (V3[4]))))) + M1 * (F2[2] * (V3[5] - V3[2]) + F2[3] *
      (V3[3] + cI * (V3[4])))));
  F1[5] = denom * cI * (F2[4] * (P1[0] * (+cI * (V3[4]) - V3[3]) + (P1[1] *
      (V3[2] + V3[5]) + (P1[2] * (-1.) * (+cI * (V3[2] + V3[5])) + P1[3] * (+cI
      * (V3[4]) - V3[3])))) + (F2[5] * (P1[0] * (V3[5] - V3[2]) + (P1[1] *
      (V3[3] + cI * (V3[4])) + (P1[2] * (V3[4] - cI * (V3[3])) + P1[3] * (V3[5]
      - V3[2])))) + M1 * (F2[2] * (+cI * (V3[4]) - V3[3]) + F2[3] * (V3[2] +
      V3[5]))));
}


void VVVV1_3(Complex<double> V1[], Complex<double> V2[], Complex<double> V4[],
    Complex<double> COUP, double M3, double W3, Complex<double> V3[])
{
  static Complex<double> cI = Complex<double> (0., 1.); 
  Complex<double> TMP17; 
  double P3[4]; 
  Complex<double> TMP20; 
  Complex<double> TMP21; 
  Complex<double> denom; 
  double OM3; 
  Complex<double> TMP9; 
  OM3 = 0.; 
  if (M3 != 0.)
    OM3 = 1./(M3 * M3); 
  V3[0] = +V1[0] + V2[0] + V4[0]; 
  V3[1] = +V1[1] + V2[1] + V4[1]; 
  P3[0] = -V3[0].real(); 
  P3[1] = -V3[1].real(); 
  P3[2] = -V3[1].imag(); 
  P3[3] = -V3[0].imag(); 
  TMP20 = (V1[2] * V4[2] - V1[3] * V4[3] - V1[4] * V4[4] - V1[5] * V4[5]); 
  TMP17 = (P3[0] * V2[2] - P3[1] * V2[3] - P3[2] * V2[4] - P3[3] * V2[5]); 
  TMP9 = (P3[0] * V1[2] - P3[1] * V1[3] - P3[2] * V1[4] - P3[3] * V1[5]); 
  TMP21 = (V2[2] * V4[2] - V2[3] * V4[3] - V2[4] * V4[4] - V2[5] * V4[5]); 
  Complex<double> tmp = ((P3[0] * P3[0]) - (P3[1] * P3[1]) - (P3[2] * P3[2]) -
      (P3[3] * P3[3]) - M3 * (M3 - cI * W3));
  denom = COUP/abs(tmp) * tmp.real()/abs(tmp.real()); 
  V3[2] = denom * (OM3 * P3[0] * (-cI * (TMP9 * TMP21) + cI * (TMP17 * TMP20))
      + (-cI * (V2[2] * TMP20) + cI * (V1[2] * TMP21)));
  V3[3] = denom * (OM3 * P3[1] * (-cI * (TMP9 * TMP21) + cI * (TMP17 * TMP20))
      + (-cI * (V2[3] * TMP20) + cI * (V1[3] * TMP21)));
  V3[4] = denom * (OM3 * P3[2] * (-cI * (TMP9 * TMP21) + cI * (TMP17 * TMP20))
      + (-cI * (V2[4] * TMP20) + cI * (V1[4] * TMP21)));
  V3[5] = denom * (OM3 * P3[3] * (-cI * (TMP9 * TMP21) + cI * (TMP17 * TMP20))
      + (-cI * (V2[5] * TMP20) + cI * (V1[5] * TMP21)));
}


}  // end namespace $(namespace)s_sm

