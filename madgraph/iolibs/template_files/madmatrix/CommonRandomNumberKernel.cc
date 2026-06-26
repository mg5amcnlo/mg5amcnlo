// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#include "CommonRandomNumbers.h"
#include "GpuAbstraction.h"
#include "MemoryBuffers.h"
#include "RandomNumberKernels.h"

#include <cassert>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  CommonRandomNumberKernel::CommonRandomNumberKernel( BufferRndNumMomenta& rnarray )
    : RandomNumberKernelBase( rnarray )
    , m_seed( 20211220 )
  {
    if( m_rnarray.isOnDevice() )
      throw std::runtime_error( "CommonRandomNumberKernel on host with a device random number array" );
  }

  //--------------------------------------------------------------------------

  void CommonRandomNumberKernel::generateRnarray()
  {
    std::vector<double> rnd = CommonRandomNumbers::generate<double>( m_rnarray.size(), m_seed ); // NB: generate as double (HARDCODED)
    std::copy( rnd.begin(), rnd.end(), m_rnarray.data() );                                       // NB: copy may imply a double-to-float conversion
  }

  //--------------------------------------------------------------------------
}
