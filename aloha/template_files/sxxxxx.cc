#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;
void sxxxxx(double p[4],int nss, complex<double> sc[3]){
  sc[2] = complex<double>(1.00,0.00);
  sc[0] = complex<double>(p[0]*nss,p[3]*nss);
  sc[1] = complex<double>(p[1]*nss,p[2]*nss);
  return;
}
