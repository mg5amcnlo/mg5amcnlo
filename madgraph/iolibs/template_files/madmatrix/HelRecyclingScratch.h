// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// --hel_recycling (CPU only): the per-thread memory of the recycled build.
// The recycled tables of a page (every amplitude of every helicity row, the
// wavefunction slots) are MBs for a big process. Every subprocess library of
// an output links this common library, so the area lives here, once per
// thread for all of them: a run over many recycled subprocesses holds the
// biggest of their tables once per thread, not every one of them.
//==========================================================================
#ifndef HELRECYCLINGSCRATCH_H
#define HELRECYCLINGSCRATCH_H 1

#include <cstddef>

namespace madmatrix
{
  // `bytes` of this thread's scratch area, aligned to 64 bytes. Its first
  // `zeroed` bytes are zero whenever `owner` (an address that names the
  // caller, one per library) is not the owner of the previous call or the area
  // had to grow: what a caller leaves there is only kept while no other caller
  // uses the area of this thread.
  void* hr_thread_scratch( std::size_t bytes, std::size_t zeroed, const void* owner );
}

#endif // HELRECYCLINGSCRATCH_H
