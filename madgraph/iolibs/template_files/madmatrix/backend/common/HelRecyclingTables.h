// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// --hel_recycling (CPU only): the tabulated amplitude stage of the recycled
// build. A fragment of namespace madmatrix, not a standalone header: it is
// #included inside the namespace, after nParity and the process constants
// (njampso, ndiagrams), by SigmaKin.cc and by HelRecyclingChunks.h.
//
// A P1N group of the recycled block is the vertex evaluated once with one leg
// left off, then, for each helicity row that shares it (a member), the dot
// product with that row's own wavefunction of the peeled leg and the fold of
// the result into the row's tables. Written out member by member that is most
// of the recycled code -- 90% of the 1.29M statements of g g > t t~ g g g --
// and code that size is fetched from memory on every event page. The
// recycler (madgraph/madevent/hel_recycle.py, CppDialect._p1n_table) writes
// the members and their folds as static tables instead, which hr_p1n_fold
// reads in order. The arithmetic is the one the straight-line code does, in
// the same order.
//==========================================================================

// One member of a P1N group: the wavefunction slot of its peeled leg, its
// helicity row in the recycled tables, and how many fold entries follow.
struct HrMember
{
  unsigned int slot;
  unsigned short row;
  unsigned short nfold;
};

// Add one amplitude of row `row` into that row's tables, entry by entry. An
// entry is 32 bits: kind (2: jampAll_sv, the second table tabAll_sv, numAll_sv,
// a store into tabAll_sv), how it adds (2: +=, -=, += real *, += complex *),
// its coefficient in c[] as (re, im) pairs (8), the index in the row (20).
// The second table, of `ntab` entries per row, is jampTmpAll_sv (the shared
// sub-expressions, fold mode) or ampAll_sv (the amplitudes, rows mode).
// Returns where the next amplitude's entries start.
inline const unsigned int*
hr_fold_amp( const cxtype_amp_sv& amp,
             const int row,
             const unsigned int* f,
             const int nfold,
             const double* c,
             const int ntab,
             cxtype_amp_sv* jampAll_sv,
             cxtype_amp_sv* tabAll_sv,
             fptype_amp_sv* numAll_sv,
             const int iParity,
             const bool storeChannelWeights )
{
  const size_t slot = (size_t)row * nParity + iParity;
  cxtype_amp_sv* jamp = jampAll_sv + slot * njampso;
  cxtype_amp_sv* tab = tabAll_sv + slot * ntab;
  for( int k = 0; k < nfold; k++ )
  {
    const unsigned int e = f[k];
    const unsigned int index = e & 0xfffffu;
    const unsigned int kind = e >> 30;
    if( kind == 3 )
    {
      tab[index] = amp;
      continue;
    }
    if( kind == 2 )
    {
      if( storeChannelWeights ) numAll_sv[slot * ndiagrams + index] += cxabs2( amp );
      continue;
    }
    cxtype_amp_sv& target = ( kind == 0 ? jamp[index] : tab[index] );
    const double* cc = c + 2 * ( ( e >> 20 ) & 0xffu );
    switch( ( e >> 28 ) & 0x3u )
    {
      case 0: target += amp; break;
      case 1: target -= amp; break;
      case 2: target += cc[0] * amp; break;
      default: target += cxtype( cc[0], cc[1] ) * amp; break;
    }
  }
  return f + nfold;
}

// The members of one P1N group: p1n holds the vertex with the peeled leg (of
// NCOMP components) left off; each member is its dot product with the member's
// own wavefunction of that leg.
// A fermion peel makes the flavour test here (see CppDialect._p1n_group).
template<class W_ACCESS, int NCOMP, bool FERMION>
inline void
hr_p1n_fold( const ALOHAOBJ& p1n,
             const ALOHAOBJ* aloha_obj,
             const HrMember* m,
             const int nm,
             const unsigned int* f,
             const double* c,
             const int ntab,
             cxtype_amp_sv* jampAll_sv,
             cxtype_amp_sv* tabAll_sv,
             fptype_amp_sv* numAll_sv,
             const int iParity,
             const bool storeChannelWeights )
{
  const cxtype_sv* _pt = W_ACCESS::kernelAccessConst( p1n.w );
  for( int i = 0; i < nm; i++ )
  {
    const ALOHAOBJ& w = aloha_obj[m[i].slot];
    const cxtype_sv* _pw = W_ACCESS::kernelAccessConst( w.w );
    cxtype_amp_sv amp;
    if constexpr( NCOMP == 1 )
      amp = _pt[0] * _pw[0];
    else
      amp = _pt[0] * _pw[0] + _pt[1] * _pw[1] + _pt[2] * _pw[2] + _pt[3] * _pw[3];
    if constexpr( FERMION )
    {
      if( p1n.flv_index != w.flv_index || p1n.flv_index == -1 ) amp = cxzero_sv();
    }
    f = hr_fold_amp( amp, m[i].row, f, m[i].nfold, c, ntab, jampAll_sv, tabAll_sv, numAll_sv, iParity, storeChannelWeights );
  }
}
