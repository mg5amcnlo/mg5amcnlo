#ifndef aloha_aux_functions_guard
#define aloha_aux_functions_guard
#include <iostream>
#include <complex>
double Sgn(double e,double f);

struct ALOHAOBJ{
     double p[4];
     std::complex<double> W[5];
     int flv_index =1;

     public:
        //ALOHAOBJ(double p[4], std::complex<double> W[4], int flav):p(p), W(W), flav(flav){};
        inline ALOHAOBJ() {
            for (int i = 0; i < 4; ++i) p[i] = 0.;
            for (int i = 0; i < 5; ++i) W[i] = std::complex<double>(0., 0.);
        };
};
//ALOHAOBJ::ALOHAOBJ() {}

struct ALOHAOBJ2D{
     double p[4];
     std::complex<double> W[16];
     int flv_index =1;

     public:
        //ALOHAOBJ(double p[4], std::complex<double> W[4], int flav):p(p), W(W), flav(flav){};
        inline ALOHAOBJ2D() {
            for (int i = 0; i < 4; ++i) p[i] = 0.;
            for (int i = 0; i < 16; ++i) W[i] = std::complex<double>(0., 0.);
        };
};
//ALOHAOBJ2D::ALOHAOBJ2D() {}
void define_gauge_dir(const std::complex<double> q[5], double n[5]);
void multiply_propagator_factor(const ALOHAOBJ &win, double m, ALOHAOBJ &wout);

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
 inline std::complex<double> theta_functionr(double cond, double valtrue, double valfalse) noexcept {
     // Heaviside with Θ(0) = 1, matching Fortran's .ge. 0d0
     return (cond >= 0.0) ? std::complex<double>(valtrue) : std::complex<double>(valfalse);
 }

 inline std::complex<double> theta_functionr(std::complex<double> cond, double valtrue, double valfalse) noexcept {
     // If the condition is carried as complex but is physically real, use the real part.
     // Optional: assert imag==0 if that should never happen.
     // If you truly need magnitude semantics, use std::abs(cond) instead of cond.real().
     return theta_functionr(cond.real(), valtrue, valfalse);
 }

#endif
