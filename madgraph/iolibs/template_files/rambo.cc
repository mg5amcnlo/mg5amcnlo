#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <stdlib.h>

#include "rambo.h"

using namespace std;

double Random::ranmar(){
/*     -----------------
 * universal random number generator proposed by marsaglia and zaman
 * in report fsu-scri-87-50
 * in this version rvec is a double precision variable. */
  double uni = ranu[iranmr] - ranu[jranmr];
  if(uni < 0) uni = uni + 1;
  ranu[iranmr] = uni;
  iranmr = iranmr - 1;
  jranmr = jranmr - 1;
  if(iranmr == 0) iranmr = 97;
  if(jranmr == 0) jranmr = 97;
  ranc = ranc - rancd;
  if(ranc < 0) ranc = ranc + rancm;
  uni = uni - ranc;
  if(uni < 0) uni = uni + 1;
  return uni;
}
 
void Random::rmarin(int ij, int kl){
/*     -----------------
 * initializing routine for ranmar, must be called before generating
 * any pseudorandom numbers with ranmar. the input values should be in
 * the ranges 0<=ij<=31328 ; 0<=kl<=30081 */
/* this shows correspondence between the simplified input seeds ij, kl
 * and the original marsaglia-zaman seeds i,j,k,l.
 * to get the standard values in the marsaglia-zaman paper (i=12,j=34
 * k=56,l=78) put ij=1802, kl=9373 */
  int i = ij/177 % 177 + 2;
  int j = ij % 177 + 2;
  int k = (kl/169) % 178 + 1;
  int l = kl % 169;
  for (int ii = 1; ii < 98; ii++){
    double s =  0;
    double t = .5;
    for (int jj = 1; jj < 25; jj++){
      int m = ((i*j % 179)*k) % 179;
      i = j;
      j = k;
      k = m;
      l = (53*l+1) % 169;
      if((l*m) % 64 >= 32) s = s + t;
      t = .5*t;
    }
    ranu[ii] = s;
  }
  ranc  =   362436. / 16777216.;
  rancd =  7654321. / 16777216.;
  rancm = 16777213. / 16777216.;
  iranmr = 97;
  jranmr = 33;
}

double rn(int idummy){
  static Random rand;
  double ran;
  static int init = 1;
  // Prevent unused variable warning
  if(false) idummy=idummy;
  if (init==1){
    init=0;
    rand.rmarin(1802,9373);
  }
  
  while(true){
    ran = rand.ranmar();
    if (ran>1e-16) break;
  }
  return ran;
}

vector<double*> get_momenta(int ninitial, double energy, 
			    vector<double> masses, double& wgt)
{
//---- auxiliary function to change convention between MadGraph5_aMC@NLO and rambo
//---- four momenta. 	  
  int nexternal = masses.size();
  int nfinal = nexternal - ninitial;
  double e2=pow(energy, 2);
  double m1 = masses[0];

  if (ninitial == 1){
// Momenta for the incoming particle
    vector<double*> p(1, new double[4]);
    p[0][0] = m1;
    p[0][1] = 0.;
    p[0][2] = 0.;
    p[0][3] = 0.;

    vector<double> finalmasses(++masses.begin(), masses.end());
    vector<double*> p_rambo = rambo(m1, finalmasses, wgt);
    p.insert(++p.begin(), p_rambo.begin(), p_rambo.end());

    return p;
  }

  else if (ninitial != 2){
    cout << "Rambo needs 1 or 2 incoming particles" << endl;
    exit(-1);
  }
  
  if(nfinal == 1)
    energy = m1;
        
  double m2 = masses[1];

  double mom = sqrt((pow(e2,2) - 2*e2*pow(m1,2) + pow(m1,4) - 2*e2*pow(m2,2) -
		     2*pow(m1,2)*pow(m2,2) + pow(m2,4)) / (4*e2));
  double energy1 = sqrt(pow(mom,2)+pow(m1,2));
  double energy2 = sqrt(pow(mom,2)+pow(m2,2));
// Set momenta for incoming particles
  vector<double*> p(1, new double[4]);
  p[0][0] = energy1;
  p[0][1] = 0;
  p[0][2] = 0;
  p[0][3] = mom;
  p.push_back(new double[4]);
  p[1][0] = energy2;
  p[1][1] = 0;
  p[1][2] = 0;
  p[1][3] = -mom;
  
  if (nfinal == 1){
    p.push_back(new double[4]);
    p[2][0] = energy;
    wgt = 1;
    return p;
  }
  vector<double> finalmasses(++(++masses.begin()), masses.end());
  vector<double*> p_rambo = rambo(energy, finalmasses, wgt);
  p.insert(++(++p.begin()), p_rambo.begin(), p_rambo.end());
  return p;
}


vector<double*> rambo(double et, vector<double>& xm, double& wt){
/**********************************************************************
 *                       rambo                                         *
 *    ra(ndom)  m(omenta)  b(eautifully)  o(rganized)                  *
 *                                                                     *
 *    a democratic multi-particle phase space generator                *
 *    authors:  s.d. ellis,  r. kleiss,  w.j. stirling                 *
 *    this is version 1.0 -  written by r. kleiss                      *
 *    -- adjusted by hans kuijf, weights are logarithmic (20-08-90)    *
 *                                                                     *
 *    n  = number of particles                                         *
 *    et = total centre-of-mass energy                                 *
 *    xm = particle masses ( dim=nexternal-nincoming )                 *
 *    p  = particle momenta ( dim=(4,nexternal-nincoming) )            *
 *    wt = weight of the event                                         *
 ***********************************************************************/
  int n = xm.size();
  vector<double*> q, p;
  vector<double> z(n), r(4), b(3), p2(n), xm2(n), e(n), v(n);
  static vector<int> iwarn(5, 0);
  static double acc = 1e-14;
  static int itmax = 6,ibegin = 0;
  static double twopi=8.*atan(1.);
  static double po2log=log(twopi/4.);

  for(int i=0; i < n; i++){
    q.push_back(new double[4]);
    p.push_back(new double[4]);
  }
// initialization step: factorials for the phase space weight
  if(ibegin==0){
    ibegin=1;
    z[1]=po2log;
    for(int k=2;k < n;k++)
      z[k]=z[k-1]+po2log-2.*log(double(k-1));
    for(int k=2;k < n;k++)
      z[k]=(z[k]-log(double(k)));
  }
// check on the number of particles
  if(n<1 || n>101){
    cout << "Too few or many particles: " << n << endl;
    exit(-1);
  }
// check whether total energy is sufficient; count nonzero masses
  double xmt=0.;
  int nm=0;
  for(int i=0; i<n; i++){
    if(xm[i]!=0.) nm=nm+1;
    xmt=xmt+abs(xm[i]);
  }
  if (xmt>et){
    cout << "Too low energy: " << et << " needed " << xmt << endl;
    exit(-1);
  }
// the parameter values are now accepted

// generate n massless momenta in infinite phase space
  for(int i=0; i<n;i++){
    double r1=rn(1);
    double c=2.*r1-1.;
    double s=sqrt(1.-c*c);
    double f=twopi*rn(2);
    r1=rn(3);
    double r2=rn(4);
    q[i][0]=-log(r1*r2);
    q[i][3]=q[i][0]*c;
    q[i][2]=q[i][0]*s*cos(f);
    q[i][1]=q[i][0]*s*sin(f);
  }
// calculate the parameters of the conformal transformation
  for (int i=0;i < 4;i++)
    r[i]=0.;
  for(int i=0;i < n;i++){
    for (int k=0; k<4; k++)
      r[k]=r[k]+q[i][k];
  }
  double rmas=sqrt(pow(r[0],2)-pow(r[3],2)-pow(r[2],2)-pow(r[1],2));
  for(int k=1;k < 4; k++)
    b[k-1]=-r[k]/rmas;
  double g=r[0]/rmas;
  double a=1./(1.+g);
  double x=et/rmas;

// transform the q's conformally into the p's
  for(int i=0; i< n;i++){
    double bq=b[0]*q[i][1]+b[1]*q[i][2]+b[2]*q[i][3];
    for (int k=1;k<4;k++)
      p[i][k]=x*(q[i][k]+b[k-1]*(q[i][0]+a*bq));
    p[i][0]=x*(g*q[i][0]+bq);
  }

// calculate weight and possible warnings
  wt=po2log;
  if(n!=2) wt=(2.*n-4.)*log(et)+z[n-1];
  if(wt<-180.){
    if(iwarn[0]<=5) cout << "Too small wt, risk for underflow: " << wt << endl;
    iwarn[0]=iwarn[0]+1;
  }
  if(wt> 174.){
    if(iwarn[1]<=5) cout << "Too large wt, risk for overflow: " << wt << endl;
    iwarn[1]=iwarn[1]+1;
  }

// return for weighted massless momenta
  if(nm==0){
// return log of weight
    return p;
  }

// massive particles: rescale the momenta by a factor x
  double xmax=sqrt(1.-pow(xmt/et, 2));
  for(int i=0;i < n; i++){
    xm2[i]=pow(xm[i],2);
    p2[i]=pow(p[i][0],2);
  }
  int iter=0;
  x=xmax;
  double accu=et*acc;
  while(true){
    double f0=-et;
    double g0=0.;
    double x2=x*x;
    for(int i=0; i < n; i++){
      e[i]=sqrt(xm2[i]+x2*p2[i]);
      f0=f0+e[i];
      g0=g0+p2[i]/e[i];
    }
    if(abs(f0)<=accu) break;
    iter=iter+1;
    if(iter>itmax){
      cout << "Too many iterations without desired accuracy: " << itmax << endl;
      break;
    }
    x=x-f0/(x*g0);
  }
  for(int i=0;i < n;i++){
    v[i]=x*p[i][0];
    for(int k=1;k < 4; k++)
      p[i][k]=x*p[i][k];
    p[i][0]=e[i];
  }

// calculate the mass-effect weight factor
  double wt2=1.;
  double wt3=0.;
  for(int i=0;i < n; i++){
    wt2=wt2*v[i]/e[i];
    wt3=wt3+pow(v[i],2)/e[i];
  }
  double wtm=(2.*n-3.)*log(x)+log(wt2/wt3*et);

// return for  weighted massive momenta
  wt=wt+wtm;
  if(wt<-180.){
    if(iwarn[2]<=5) cout << "Too small wt, risk for underflow: " << wt << endl;
    iwarn[2]=iwarn[2]+1;
  }
  if(wt> 174.){
    if(iwarn[3]<=5)  cout << "Too large wt, risk for overflow: " << wt << endl;
    iwarn[3]=iwarn[3]+1;
  }
// return log of weight
  return p;
}


