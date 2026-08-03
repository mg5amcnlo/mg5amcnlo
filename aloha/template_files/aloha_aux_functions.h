#ifndef aloha_aux_functions_guard
#define aloha_aux_functions_guard
#include <complex>
double Sgn(double e,double f);
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
