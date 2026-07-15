// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef CROSSSECTIONKERNELS_H
#define CROSSSECTIONKERNELS_H 1

#include "mgOnGpuConfig.h"

#include "EventStatistics.h"
#include "MemoryBuffers.h"

//============================================================================

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  // Helper function for Bridge.h: must be compiled without fast math
  // Iterate through all output MEs and replace any NaN/abnormal ones by sqrt(-1)
  void flagAbnormalMEs( fptype* hstMEs, unsigned int nevt );

  //--------------------------------------------------------------------------

  // A base class encapsulating the calculation of event statistics on a CPU host or on a GPU device
  class CrossSectionKernelBase //: virtual public ICrossSectionKernel
  {
  protected:

    // Constructor from existing input and output buffers
    CrossSectionKernelBase( const BufferWeights& samplingWeights,       // input: sampling weights
                            const BufferMatrixElements& matrixElements, // input: matrix elements
                            EventStatistics& stats )                    // output: event statistics
      : m_samplingWeights( samplingWeights )
      , m_matrixElements( matrixElements )
      , m_stats( stats )
      , m_iter( 0 )
    {
      // NB: do not initialise EventStatistics (you may be asked to update an existing result)
    }

  public:

    // Destructor
    virtual ~CrossSectionKernelBase() {}

    // Update event statistics
    virtual void updateEventStatistics( const bool debug = false ) = 0;

    // Is this a host or device kernel?
    virtual bool isOnDevice() const = 0;

  protected:

    // The buffer for the sampling weights
    const BufferWeights& m_samplingWeights;

    // The buffer for the output matrix elements
    const BufferMatrixElements& m_matrixElements;

    // The event statistics
    EventStatistics& m_stats;

    // The number of iterations processed so far
    size_t m_iter;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating the calculation of event statistics on a CPU host
  class CrossSectionKernelHost final : public CrossSectionKernelBase, public NumberOfEvents
  {
  public:

    // Constructor from existing input and output buffers
    CrossSectionKernelHost( const BufferWeights& samplingWeights,       // input: sampling weights
                            const BufferMatrixElements& matrixElements, // input: matrix elements
                            EventStatistics& stats,                     // output: event statistics
                            const size_t nevt );

    // Destructor
    virtual ~CrossSectionKernelHost() {}

    // Update event statistics
    void updateEventStatistics( const bool debug = false ) override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return false; }
  };

  //--------------------------------------------------------------------------

  /*
#ifdef MGONGPUCPP_GPUIMPL
  // A class encapsulating the calculation of event statistics on a GPU device
  class CrossSectionKernelDevice : public CrossSectionKernelBase, public NumberOfEvents
  {
  public:

    // Constructor from existing input and output buffers
    CrossSectionKernelDevice( const BufferWeights& samplingWeights,       // input: sampling weights
                              const BufferMatrixElements& matrixElements, // input: matrix elements
                              EventStatistics& stats,                     // output: event statistics
                              const size_t gpublocks,
                              const size_t gputhreads );

    // Destructor
    virtual ~CrossSectionKernelDevice(){}

    // Reset gpublocks and gputhreads
    void setGrid( const size_t gpublocks, const size_t gputhreads );

    // Update event statistics
    void updateEventStatistics( const bool debug=false ) override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return true; }

  private:

    // The number of blocks in the GPU grid
    size_t m_gpublocks;

    // The number of threads in the GPU grid
    size_t m_gputhreads;

  };
#endif
  */

  //--------------------------------------------------------------------------
}
#endif // CROSSSECTIONKERNELS_H
