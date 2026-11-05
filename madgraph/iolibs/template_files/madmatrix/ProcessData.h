// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Process-specific compile-time constants, generated once per subprocess.
// This is the single source of truth for these values: CPPProcess (P1-generated)
// and the backend-owned driver code (backend/{cpu,simd,gpu}/) both read from here
// instead of duplicating the literals.

#ifndef PROCESSDATA_H
#define PROCESSDATA_H 1

namespace ProcessData
{
  constexpr int np4 = 4; // dimensions of a 4-momentum (E,px,py,pz)
  constexpr int nw6 = %(nwave)d; // dimensions of each wavefunction (HELAS KEK 91-11)
  constexpr int npari = %(nincoming)d;
  constexpr int nparf = %(noutcoming)d;
  constexpr int npar = npari + nparf;
  constexpr int ncomb = %(nbhel)d; // #helicity combinations
  constexpr int ndiagrams = %(ndiagrams)d;
  constexpr int ncolor = %(ncolor)d;
  // Squared split orders: the amplitudes fall into nampso amplitude orders, the jamps
  // carry one vector per order (njampso long in total) and the color sum pairs them into
  // nsqampso squared orders. 1, ncolor and 1 unless a '^2' constraint splits the amplitudes.
  constexpr int nampso = %(nampso)d;
  constexpr int njampso = ncolor * nampso;
  constexpr int nsqampso = %(nsqampso)d;
  constexpr int nmaxflavor = %(nmaxflavor)d;
  constexpr int nwf = %(nwf)d; // #wavefunctions = #external (npar) + #internal (see #644)
  constexpr int nproc = %(nproc)d; // 2 if this process has a mirror process, else 1
  constexpr int proc_id = %(proc_id)d;
  // spin/color/identical-particle denominators. NB: assumes nprocesses == 1 (#272 and #343),
  // although den_factors has one entry per matrix element: see MadGraph7 issue #168
  constexpr int helcolDenominators[1] = { %(den_factors)s };

  // SM independent parameters/couplings/flavor-couplings used by this process
  // (see #823: nIPC/nIPD/nIPF may vary per P1, unlike nicoup which is model-wide)
  constexpr int nIPD = %(nipd)d;
  constexpr int nIPC = %(nipc)d;
  constexpr int nIPF = %(nipf)d;
  constexpr int nDPF = %(ndpf)d;

  // Helicities for the process [NB do keep 'static' for this constexpr array, see issue #283]
  // *** NB There is no automatic check yet that these are in the same order as Fortran! #569 ***
%(thel_lines)s

  // Host-side flavor table: single source of truth for PDG ids (used by both
  // CPPProcess's constructor copy into cFlavors and CPPProcess::flavorPDG).
%(tflavors_lines)s
}

// Process identification for test/debug tooling. Must stay #define (not
// constexpr): used for macro token-pasting (test suite names) and
// stringification, e.g. TEST( XTESTID( MG_EPOCH_PROCESS_ID ), ... ).
#define MG_EPOCH_PROCESS_ID %(processid_uppercase)s
#define MG_EPOCH_REFERENCE_FILE_NAME "../../test/ref/dump_CPUTest.%(processid)s.txt"

#endif // PROCESSDATA_H
