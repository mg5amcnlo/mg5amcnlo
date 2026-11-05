// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Floating point exception traps. A runtime switch with process-wide effect,
// kept apart from the compile-time math of constexpr_math.h, where it had
// ended up after the backend split.

#ifndef MG_FPE_H
#define MG_FPE_H 1

#include <cfenv> // for feenableexcept and FE_XXX
#include <iostream>

namespace madmatrix
{
  // Enable FPE traps (see #701, #733, #831 - except on MacOS where feenableexcept is not defined #730)
  // [NB1: Fortran default is -ffpe-trap=none, i.e. FPE traps are not enabled, https://gcc.gnu.org/onlinedocs/gfortran/Debugging-Options.html]
  // [NB2: Fortran default is -ffpe-summary=invalid,zero,overflow,underflow,denormal, i.e. warn at the end on STOP]
  inline void
  fpeEnable()
  {
    static bool first = true; // FIXME: quick and dirty hack to do this only once (can be removed when separate C++/CUDA builds are implemented)
    if( !first ) return;
    first = false;
#ifndef __APPLE__ // on MacOS feenableexcept is not defined #730
    constexpr bool enableFPE = true; // this is hardcoded and no longer controlled by getenv( "CUDACPP_RUNTIME_ENABLEFPE" )
    if( enableFPE )
    {
      std::cout << "INFO: The following Floating Point Exceptions will cause SIGFPE program aborts: FE_DIVBYZERO, FE_INVALID, FE_OVERFLOW" << std::endl;
      feenableexcept( FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW ); // new strategy #831 (do not enable FE_UNDERFLOW)
    }
#else
    //std::cout << "INFO: Keep default SIGFPE settings because feenableexcept is not available on MacOS" << std::endl;
#endif
  }
}

#endif // MG_FPE_H
