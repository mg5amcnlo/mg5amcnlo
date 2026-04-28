#ifndef aloha_aux_functions_guard
#define aloha_aux_functions_guard
#include <iostream>
double Sgn(double e,double f);

struct ALOHAOBJ{
     double p[4];
     std::complex<double> W[4];
     int flv_index =1;

     public:
        //ALOHAOBJ(double p[4], std::complex<double> W[4], int flav):p(p), W(W), flav(flav){};
        inline ALOHAOBJ() {};
};
//ALOHAOBJ::ALOHAOBJ() {}

struct ALOHAOBJ2D{
     double p[4];
     std::complex<double> W[16];
     int flv_index =1;

     public:
        //ALOHAOBJ(double p[4], std::complex<double> W[4], int flav):p(p), W(W), flav(flav){};
        inline ALOHAOBJ2D() {};
};
//ALOHAOBJ2D::ALOHAOBJ2D() {}

#endif
#ifndef i_guard
#define i_guard
#include <complex>

void ixxxxx(double p[4],double fmass,int nhel,int nsf, int flv, ALOHAOBJ &fi);
#endif
#ifndef o_guard
#define o_guard
#include <complex>
void oxxxxx(double p[4],double fmass,int nhel,int nsf, int flv,  ALOHAOBJ &fo);
#endif
#ifndef s_guard
#define s_guard
#include <complex>
void sxxxxx(double p[4],int nss, ALOHAOBJ &sc);
#endif
#ifndef t_guard
#define t_guard
#include <complex>
void txxxxx(double p[4],double tmass,int nhel,int nst,ALOHAOBJ2D fi[18]);
#endif
#ifndef v_guard
#define v_guard
#include <complex>
void vxxxxx(double p[4],double vmass, int nhel,int nsv, ALOHAOBJ &v);
#endif
