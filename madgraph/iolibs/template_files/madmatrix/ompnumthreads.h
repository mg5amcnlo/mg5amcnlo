// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef OMPNUMTHREADS_H
#define OMPNUMTHREADS_H 1

#ifdef _OPENMP

#include <omp.h>

#include <iostream>

// The OMP_NUM_THREADS environment variable is used to control OMP multi-threading
// By default, all available $(nproc) threads are used if OMP_NUM_THREADS is not set:
// if ompnumthreadsNotSetMeansOneThread is called, only one thread is used instead
inline void
ompnumthreadsNotSetMeansOneThread( int debuglevel ) // quiet(-1), info(0), debug(1)
{
  // Set OMP_NUM_THREADS equal to 1 if it is not yet set
  char* ompnthr = getenv( "OMP_NUM_THREADS" );
  if( debuglevel == 1 )
  {
    std::cout << "DEBUG: entering ompnumthreadsNotSetMeansOneThread" << std::endl;
    std::cout << "DEBUG: omp_get_num_threads() = "
              << omp_get_num_threads() << std::endl; // always == 1 here!
    std::cout << "DEBUG: omp_get_max_threads() = "
              << omp_get_max_threads() << std::endl;
    std::cout << "DEBUG: ${OMP_NUM_THREADS}    = '"
              << ( ompnthr == 0 ? "[not set]" : ompnthr ) << "'" << std::endl;
  }
  if( ompnthr == NULL ||
      std::string( ompnthr ).find_first_not_of( "0123456789" ) != std::string::npos ||
      atol( ompnthr ) == 0 )
  {
    if( ompnthr != NULL )
      std::cout << "(ompnumthreadsNotSetMeansOneThread) "
                << "WARNING! OMP_NUM_THREADS is invalid: will use only 1 thread" << std::endl;
    else if( debuglevel >= 0 )
      std::cout << "(ompnumthreadsNotSetMeansOneThread) "
                << "DEBUG: OMP_NUM_THREADS is not set: will use only 1 thread" << std::endl;
    omp_set_num_threads( 1 ); // https://stackoverflow.com/a/22816325
    if( debuglevel == 1 )
    {
      std::cout << "DEBUG: omp_get_num_threads() = "
                << omp_get_num_threads() << std::endl; // always == 1 here!
      std::cout << "DEBUG: omp_get_max_threads() = "
                << omp_get_max_threads() << std::endl;
    }
  }
  else if( debuglevel >= 0 )
    std::cout << "(ompnumthreadsNotSetMeansOneThread) "
              << "DEBUG: OMP_NUM_THREADS = " << ompnthr << std::endl;
  if( debuglevel >= 0 )
    std::cout << "(ompnumthreadsNotSetMeansOneThread) "
              << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;
  if( debuglevel == 1 )
    std::cout << "DEBUG: exiting ompnumthreadsNotSetMeansOneThread" << std::endl;
}
#endif

#endif // OMPNUMTHREADS_H
