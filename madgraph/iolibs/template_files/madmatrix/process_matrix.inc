! Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MG5aMC CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
! Further modified by: J. Teig, A. Valassi (2021-2024).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================

      // *** COLOR CHOICE BELOW ***

      // Store the leading color flows for choice of color
#ifndef MGONGPUCPP_GPUIMPL
      if( jamp2_sv ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor; icol++ )
          jamp2_sv[ncolor * iParity + icol] += cxabs2( jamp_sv[icol] ); // may underflow #831
      }
#else /* clang-format off */
      assert( iParity == 0 ); // sanity check for J2_ACCESS
      using J2_ACCESS = DeviceAccessJamp2;
      if( colAllJamp2s ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor; icol++ )
          // NB: atomicAdd is needed after moving to cuda streams with one helicity per stream!
          atomicAdd( &J2_ACCESS::kernelAccessIcol( colAllJamp2s, icol ), cxabs2( jamp_sv[icol] ) );
      }
#endif /* clang-format on */

      // *** PREPARE OUTPUT JAMPS ***
#ifdef MGONGPUCPP_GPUIMPL
      //printf( "calculate_jamps: dcNGoodHel=%%d\n", dcNGoodHel );
      // In CUDA, copy the local jamp to the output global-memory jamp
      constexpr int ihel0 = 0; // the allJamps buffer already points to a specific helicity _within a super-buffer for dcNGoodHel helicities_
      using J_ACCESS = DeviceAccessJamp;
      for( int icol = 0; icol < ncolor; icol++ )
        J_ACCESS::kernelAccessIcolIhelNhel( allJamps, icol, ihel0, dcNGoodHel ) = jamp_sv[icol];
#else
      // In C++, copy the local jamp to the output array passed as function argument
      for( int icol = 0; icol < ncolor; icol++ )
        allJamp_sv[iParity * ncolor + icol] = jamp_sv[icol];
#endif
    }
    // END LOOP ON IPARITY

    mgDebug( 1, __FUNCTION__ );
    return;
  }
