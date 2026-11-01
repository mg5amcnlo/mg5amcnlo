// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
//==========================================================================
// The per-page setup the generated diagram calls read: the couplings of the
// page (dependent and fixed), its momenta, the flavor coupling views and the
// scalar iflavor. A fragment, not a header: #include'd inside the iParity loop of calculate_jamps and, for
// --hel_recycling, of calculate_all_jamps, which evaluate the same calls.
//==========================================================================
constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings
// nxcoup can be 0 (no alpha_s-dependent coupling, and only flavor couplings, e.g.
// u u~ > u u~ QED^2==4 with flavor grouping): a zero-sized array is only a compiler
// extension (and refused in GPU device code), so size them at least 1, as cIPD/cIPC
const fptype* allCOUPs[nxcoup > 0 ? nxcoup : 1];
for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
  allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
  allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
// C++ kernels take input/output buffers with momenta/MEs for one specific event (the first in the current event page)
const fptype_momenta* momenta = M_ACCESS::ieventAccessRecordConst( allmomenta, ievt0 );
const fptype* COUPs[nxcoup > 0 ? nxcoup : 1];
for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
  COUPs[idcoup] = CD_ACCESS::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 ); // dependent couplings, vary event-by-event
for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
  COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
// Create an array of views over the Flavor Couplings
FLV_COUPLING_ARRAY<nIPF, nMF> flvCOUPs{ cIPF_partner1, cIPF_partner2, cIPF_value };

// Dependent (event-by-event, running-alphas) flavor couplings (Step 3): the per-flavor
// values are NOT baked in (they run per event). Gather the current values of the
// underlying dependent couplings for this event page into an AOSOA buffer dpf_value
// (one nx2*neppC SIMD record per (coupling,flavor) slot, matching CD_ACCESS), then build
// an ordinary value-based view over it. The flavor index is constant across a SIMD lane
// (guaranteed by the phase-space integrator), so each lane gets its own running value
// while sharing the same flavor selection. This is the direct analogue of Fortran's
// FLV_xx%VAL(k)%P => GC_yyy(J). The vertex routines are instantiated with CDPF_ACCESS
// (= CD_ACCESS here) so get_coupling_def reads dpf_value with the right per-flavor stride.
constexpr int ndpfbuf = ( nDPF > 0 ? nDPF * nMF * CDPF_ACCESS::flv_stride : 1 );
// cppAlign is only defined for SIMD
alignas( mgOnGpu::cppAlign ) fptype dpf_value[ndpfbuf]{};
for( int idpf = 0; idpf < nDPF; idpf++ )
  for( int imf = 0; imf < nMF; imf++ )
  {
    const int idc = cDPF_idcoup[idpf * nMF + imf];
    if( idc >= 0 )
      CDPF_ACCESS::kernelAccess( dpf_value + ( idpf * nMF + imf ) * CDPF_ACCESS::flv_stride ) =
        CD_ACCESS::kernelAccessConst( COUPs[idc] );
  }
FLV_COUPLING_ARRAY<nDPF, nMF, CDPF_ACCESS::flv_stride> flvCOUPs_dep{ cDPF_partner1, cDPF_partner2, dpf_value };

// Scalar iflavor for the current event page (constant across the SIMD vector)
const unsigned int* iflavor_rec = F_ACCESS::ieventAccessRecordConst( iflavorVec, ievt0 );
const uint_sv iflavor_sv = F_ACCESS::kernelAccessConst( iflavor_rec );
const unsigned int iflavor = reinterpret_cast<const unsigned int*>( &iflavor_sv )[0];
