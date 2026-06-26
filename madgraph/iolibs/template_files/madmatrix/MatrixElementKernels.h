// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro, J. Teig, A. Thete, A. Valassi, Z. Wettersten (2022-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MATRIXELEMENTKERNELS_H
#define MATRIXELEMENTKERNELS_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "GpuAbstraction.h"
#include "MemoryBuffers.h"

#include <map>
#include <memory>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  // A base class encapsulating matrix element calculations on a CPU host or on a GPU device
  class MatrixElementKernelBase //: virtual public IMatrixElementKernel
  {
  protected:

    // Constructor from existing input and output buffers
    MatrixElementKernelBase( const BufferMomenta& momenta,         // input: momenta
                             const BufferGs& gs,                   // input: gs for alphaS
                             const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                             const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                             const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                             const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                             BufferMatrixElements& matrixElements, // output: matrix elements
                             BufferSelectedHelicity& selhel,       // output: helicity selection
                             BufferSelectedColor& selcol);          // output: color selection

  public:

    // Destructor
    virtual ~MatrixElementKernelBase();

    // Compute good helicities (returns nGoodHel, the number of good helicity combinations out of ncomb)
    virtual int computeGoodHelicities() = 0;

    // Compute matrix elements
    virtual void computeMatrixElements( const bool useChannelIds ) = 0;

    // Is this a host or device kernel?
    virtual bool isOnDevice() const = 0;

    // Dump signalling FPEs (#831 and #837)
    static void dumpSignallingFPEs();

#ifdef MGONGPU_CHANNELID_DEBUG
    // Add a MEK identifier for the channelId debug printout
    void setTagForNevtProcessedByChannel( const std::string& tag ) { m_tag = tag; }

  protected:
    // Update number of events processed by channel
    void updateNevtProcessedByChannel( const unsigned int* pHstChannelIds, const size_t nevt );

    // Dump number of events processed by channel
    void dumpNevtProcessedByChannel();
#endif

  protected:

    // The buffer for the input momenta
    const BufferMomenta& m_momenta;

    // The buffer for the gs to calculate the alphaS values
    const BufferGs& m_gs;

    // The buffer for the flavor indices for the flavor combination
    const BufferIflavorVec& m_iflavorVec;

    // The buffer for the random numbers for helicity selection
    const BufferRndNumHelicity& m_rndhel;

    // The buffer for the random numbers for color selection
    const BufferRndNumColor& m_rndcol;

    // The buffer for the channel ids for single-diagram enhancement
    const BufferChannelIds& m_channelIds;

    // The buffer for the output matrix elements
    BufferMatrixElements& m_matrixElements;

    // The buffer for the output helicity selection
    BufferSelectedHelicity& m_selhel;

    // The buffer for the output color selection
    BufferSelectedColor& m_selcol;

#ifdef MGONGPU_CHANNELID_DEBUG
    // The events-per-channel counter for debugging
    std::map<size_t, size_t> m_nevtProcessedByChannel;

    // The tag for events-per-channel debugging
    std::string m_tag;
#endif
  };

  //--------------------------------------------------------------------------

#ifndef MGONGPUCPP_GPUIMPL
  // A class encapsulating matrix element calculations on a CPU host
  class MatrixElementKernelHost final : public MatrixElementKernelBase, public NumberOfEvents
  {
  public:

    // Constructor from existing input and output buffers
    MatrixElementKernelHost( const BufferMomenta& momenta,         // input: momenta
                             const BufferGs& gs,                   // input: gs for alphaS
                             const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                             const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                             const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                             const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                             BufferMatrixElements& matrixElements, // output: matrix elements
                             BufferSelectedHelicity& selhel,       // output: helicity selection
                             BufferSelectedColor& selcol,          // output: color selection
                             const size_t nevt);

    // Destructor
    virtual ~MatrixElementKernelHost();

    // Compute good helicities (returns nGoodHel, the number of good helicity combinations out of ncomb)
    int computeGoodHelicities() override final;

    // Compute matrix elements
    void computeMatrixElements( const bool useChannelIds ) override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return false; }

  private:

    // Does this host system support the SIMD used in the matrix element calculation?
    // [NB: this is private, SIMD vectorization in mg5amc C++ code is currently only used in the ME calculations below MatrixElementKernelHost!]
    static bool hostSupportsSIMD( const bool verbose = false ); // ZW: default verbose false

  private:

    // The buffer for the event-by-event couplings that depends on alphas QCD
    HostBufferCouplings m_couplings;

    // The buffer for the event-by-event numerators of multichannel factors
    HostBufferNumerators m_numerators;

    // The buffer for the event-by-event denominators of multichannel factors
    HostBufferDenominators m_denominators;
  };
#endif

  //--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
  // A class encapsulating matrix element calculations on a GPU device
  class MatrixElementKernelDevice : public MatrixElementKernelBase, public NumberOfEvents
  {
  public:

    // Constructor from existing input and output buffers
    MatrixElementKernelDevice( const BufferMomenta& momenta,         // input: momenta
                               const BufferGs& gs,                   // input: gs for alphaS
                               const BufferIflavorVec& iflavorVec,   // input: flavor indices for the flavor combination
                               const BufferRndNumHelicity& rndhel,   // input: random numbers for helicity selection
                               const BufferRndNumColor& rndcol,      // input: random numbers for color selection
                               const BufferChannelIds& channelIds,   // input: channel ids for single-diagram enhancement
                               BufferMatrixElements& matrixElements, // output: matrix elements
                               BufferSelectedHelicity& selhel,       // output: helicity selection
                               BufferSelectedColor& selcol,          // output: color selection
                               const size_t gpublocks,
                               const size_t gputhreads );

    // Destructor
    virtual ~MatrixElementKernelDevice();

    // Reset gpublocks and gputhreads
    void setGrid( const int gpublocks, const int gputhreads );

    // Compute good helicities (returns nGoodHel, the number of good helicity combinations out of ncomb)
    int computeGoodHelicities() override final;

    // Compute matrix elements
    void computeMatrixElements( const bool useChannelIds ) override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return true; }

  private:

    // The buffer for the event-by-event couplings that depends on alphas QCD
    DeviceBufferCouplings m_couplings;

    // The super-buffer of nGoodHel ME buffers (dynamically allocated because nGoodHel is determined at runtime)
    std::unique_ptr<DeviceBufferSimple> m_pHelMEs;

    // The super-buffer of nGoodHel jamp buffers (dynamically allocated because nGoodHel is determined at runtime)
    std::unique_ptr<DeviceBufferSimple> m_pHelJamps;

    // The super-buffer of nGoodHel numerator buffers (dynamically allocated because nGoodHel is determined at runtime)
    std::unique_ptr<DeviceBufferSimple> m_pHelNumerators;

    // The super-buffer of nGoodHel denominator buffers (dynamically allocated because nGoodHel is determined at runtime)
    std::unique_ptr<DeviceBufferSimple> m_pHelDenominators;

    // The super-buffer of ncolor jamp2 buffers
    DeviceBufferSimple m_colJamp2s;

#ifdef MGONGPU_CHANNELID_DEBUG
    // The **host** buffer for the channelId array
    // FIXME? MEKD should accept a host buffer as an argument instead of a device buffer, so that a second copy can be avoided?
    PinnedHostBufferChannelIds m_hstChannelIds;
#endif

#ifndef MGONGPU_HAS_NO_BLAS
    // Decide at runtime whether to use BLAS for color sums
    bool m_blasColorSum;

    // Decide at runtime whether TF32TENSOR math should be used in cuBLAS
    bool m_blasTf32Tensor;

    // The super-buffer of nGoodHel cuBLAS/hipBLAS temporary buffers
    std::unique_ptr<DeviceBufferSimple2> m_pHelBlasTmp;

    // The cuBLAS/hipBLAS handle (a single one for all good helicities)
    gpuBlasHandle_t m_blasHandle;
#endif

    // The array of GPU streams (one for each good helicity)
    gpuStream_t m_helStreams[CPPProcess::ncomb]; // reserve ncomb streams (but only nGoodHel <= ncomb will be used)

    // The number of blocks in the GPU grid
    size_t m_gpublocks;

    // The number of threads in the GPU grid
    size_t m_gputhreads;
  };
#endif

  //--------------------------------------------------------------------------
}
#endif // MATRIXELEMENTKERNELS_H
