#include <iostream> 
#include <cmath>
#include <complex>
using namespace std;
 
#include "cavh_olo.h"

int main(){
  
  complex<double> rslt[3];
  
  double rmu = 1.0;
  OLO_SCALE( &rmu );

  complex<double> p1  ( 0.1,0.0);
  complex<double> p2  ( 0.1,0.0);
  complex<double> p3  ( 3.0,0.0);
  complex<double> p4  ( 4.0,0.0);
  complex<double> p12 ( 5.0,0.0);
  complex<double> p23 (-6.0,0.0);
  complex<double> m1  ( 0.1, 0.00);
  complex<double> m2  ( 0.0, 0.00);
  complex<double> m3  ( 0.2, 0.00);
  complex<double> m4  ( 0.1,-0.02);

  OLO_D0cc( &rslt ,&p1,&p2,&p3,&p4,&p12,&p23 ,&m1,&m2,&m3,&m4 );

  cout.precision(16);
  cout<<"eps^( 0):"<<rslt[0]<<endl;
  cout<<"eps^(-1):"<<rslt[1]<<endl;
  cout<<"eps^(-2):"<<rslt[2]<<endl;

  complex<double> bn[9][3];
  int rank = 4;
  OLO_BNcc( &bn ,&rank ,&p3,&m1,&m3 );

  cout.precision(16);
  cout<<"B0 eps^( 0):"<<bn[0][0]<<endl;
  cout<<"B0 eps^(-1):"<<bn[0][1]<<endl;
  cout<<"B0 eps^(-2):"<<bn[0][2]<<endl;
  cout<<"B1 eps^( 0):"<<bn[1][0]<<endl;
  cout<<"B1 eps^(-1):"<<bn[1][1]<<endl;
  cout<<"B1 eps^(-2):"<<bn[1][2]<<endl;
  cout<<"B00 eps^( 0):"<<bn[2][0]<<endl;
  cout<<"B00 eps^(-1):"<<bn[2][1]<<endl;
  cout<<"B00 eps^(-2):"<<bn[2][2]<<endl;
  cout<<"B11 eps^( 0):"<<bn[3][0]<<endl;
  cout<<"B11 eps^(-1):"<<bn[3][1]<<endl;
  cout<<"B11 eps^(-2):"<<bn[3][2]<<endl;
  cout<<"B1111 eps^( 0):"<<bn[8][0]<<endl;
  cout<<"B1111 eps^(-1):"<<bn[8][1]<<endl;
  cout<<"B1111 eps^(-2):"<<bn[8][2]<<endl;

  return 0;
}
