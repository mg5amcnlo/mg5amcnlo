// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Spin-density matrices of the decaying particles of a process, the quantity
// MadSpin contracts production and decay with. C++ backends (scalar, simd) only.
//
// The library of each subprocess exports these next to the UMAMI functions. They
// take a handle of umami_initialize, whose parameters (param card,
// umami_set_parameter) they are evaluated with, and which keeps their helicity
// filters and beam polarisation: the instances of a library are independent.
// Momenta are event major: momenta[count][npar][4],
// (E, px, py, pz) per particle, in the leg order of the process. alpha_s and
// flavor are per event, flavor being the index of the flavor combination
// (ProcessData::flavorPDGs) of that event.
//
// The conventions are those of the Fortran standalone GET_DENSITY, so that
// MadSpin reads the two the same way:
//   - pos[n_changing] are the (1-based) legs whose helicity is left open, and
//     allow_hel[n_comb][n_changing] the n_comb helicity combinations they take;
//   - the other legs are summed over the rows of the helicity table whose open
//     legs carry the first combination, allow_hel[0];
//   - inter[count][n_comb*(n_comb+1)/2] (complex, as re/im pairs) holds
//     rho(I,J) = sum_spectators JAMP_J^dagger C JAMP_I / iden for I <= J, I
//     outer, iden being the spin/color average and identical-particle factor
//     of the process (helcolDenominators), and the beam polarisation set by
//     madmatrix_set_beampol folded into the spectator sum.
// A helicity row whose amplitudes all vanish over the first evaluations of a
// given (flavor, open legs) is then skipped, exactly as the Fortran SMATRIX
// filters its helicities.

#ifndef MADMATRIX_DENSITY_H
#define MADMATRIX_DENSITY_H 1

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Returns 0, or non-zero (nothing written) for an invalid request.
  int madmatrix_density( void* handle,
                         size_t count,
                         const double* momenta,
                         const double* alpha_s,
                         const unsigned int* flavor,
                         int n_changing,
                         const int* pos,
                         int n_comb,
                         const int* allow_hel,
                         double* inter );

  // madmatrix_density for momenta given in the rest frame of a boost:
  // boosts[count][4] is, per event, the lab momentum of the frame its momenta are
  // in (a MadSpin decay is generated at rest, and boosted along its parent).
  int madmatrix_density_boosted( void* handle,
                                 size_t count,
                                 const double* momenta,
                                 const double* boosts,
                                 const double* alpha_s,
                                 const unsigned int* flavor,
                                 int n_changing,
                                 const int* pos,
                                 int n_comb,
                                 const int* allow_hel,
                                 double* inter );

  // |M|^2 summed over helicities and colors, averaged as the Fortran SMATRIX:
  // divided by iden, times the identical-particle factor of the flavor and with
  // the beam polarisation. Returns 0, or non-zero for an invalid request.
  int madmatrix_matrix_element( void* handle,
                                size_t count,
                                const double* momenta,
                                const double* alpha_s,
                                const unsigned int* flavor,
                                double* me );

  // Beam polarisation of the two incoming legs for the calls on handle. Same
  // convention as madevent and the Fortran density: 1 (or anything with
  // |pol| <= 1) is unpolarised, |pol| grows to 2 for a fully polarised beam and
  // its sign gives the favoured helicity. Ignored for a 1 -> N process.
  void madmatrix_set_beampol( void* handle, double pol1, double pol2 );

  // Process constants: npar, npari, ncomb, nmaxflavor, ncolor, iden (in this order).
  void madmatrix_process_info( int* info );

#ifdef __cplusplus
}
#endif

#endif // MADMATRIX_DENSITY_H
