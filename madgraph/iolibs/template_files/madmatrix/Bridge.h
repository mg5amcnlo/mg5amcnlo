// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: S. Roiser (Nov 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro, S. Roiser, J. Teig, A. Thete, A. Valassi, Z. Wettersten (2021-2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef BRIDGE_H
#define BRIDGE_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"           // for CPPProcess
#include "CrossSectionKernels.h"  // for flagAbnormalMEs
#include "MatrixElementKernels.h" // for MatrixElementKernelHost, MatrixElementKernelDevice
#include "MemoryAccessMomenta.h"  // for MemoryAccessMomenta::neppM
#include "MemoryBuffers.h"        // for HostBufferMomenta, DeviceBufferMomenta etc

//#ifdef __HIPCC__
//#include <experimental/filesystem> // see
//https://rocm.docs.amd.com/en/docs-5.4.3/CHANGELOG.html#id79 #else #include
//<filesystem> // bypass this completely to ease portability on LUMI #803 #endif

#include <sys/stat.h> // bypass std::filesystem #803

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------
  /**
 * A base class for a class whose pointer is passed between Fortran and C++.
 * This is not really necessary, but it allows minimal type checks on all such
 * pointers.
 */
  struct CppObjectInFortran
  {
    CppObjectInFortran() {}
    virtual ~CppObjectInFortran() {}
  };

  //--------------------------------------------------------------------------
  /**
 * A templated class for calling the CUDA/C++ matrix element calculations of the
 * event generation workflow. The FORTRANFPTYPE template parameter indicates the
 * precision of the Fortran momenta from MadEvent (float or double). The
 * precision of the matrix element calculation is hardcoded in the fptype
 * typedef in CUDA/C++.
 *
 * The Fortran momenta passed in are in the form of
 *   DOUBLE PRECISION P_MULTI(0:3, NEXTERNAL, VECSIZE_USED)
 * where the dimensions are <np4F(#momenta)>, <nparF(#particles)>,
 * <nevtF(#events)>. In memory, this is stored in a way that C reads as an array
 * P_MULTI[nevtF][nparF][np4F]. The CUDA/C++ momenta are stored as an
 * array[npagM][npar][np4][neppM] with nevt=npagM*neppM. The Bridge is
 * configured to store nevt==nevtF events in CUDA/C++. It also checks that
 * Fortran and C++ parameters match, nparF==npar and np4F==np4.
 *
 * The cpu/gpu sequences take FORTRANFPTYPE* (not fptype*) momenta/MEs.
 * This allows mixing double in MadEvent Fortran with float in CUDA/C++
 * sigmaKin. In the fcheck_sa.f test, Fortran uses double while CUDA/C++ may use
 * double or float. In the check_sa "--bridge" test, everything is implemented
 * in fptype (double or float).
 */
  template<typename FORTRANFPTYPE>
  class Bridge final : public CppObjectInFortran
  {
  public:
    /**
   * Constructor
   *
   * @param nevtF (VECSIZE_USED, vector.inc) number of events in Fortran array
   * loops (VECSIZE_USED <= VECSIZE_MEMMAX)
   * @param nparF (NEXTERNAL, nexternal.inc) number of external particles in
   * Fortran arrays (KEPT FOR SANITY CHECKS ONLY)
   * @param np4F number of momenta components, usually 4, in Fortran arrays
   * (KEPT FOR SANITY CHECKS ONLY)
   */
    Bridge( unsigned int nevtF, unsigned int nparF, unsigned int np4F );

    /**
   * Destructor
   */
    virtual ~Bridge() {}

    // Delete copy/move constructors and assignment operators
    Bridge( const Bridge& ) = delete;
    Bridge( Bridge&& ) = delete;
    Bridge& operator=( const Bridge& ) = delete;
    Bridge& operator=( Bridge&& ) = delete;

#ifdef MGONGPUCPP_GPUIMPL
    /**
   * Set the gpublocks and gputhreads for the gpusequence - throws if evnt !=
   * gpublocks*gputhreads (this is needed for BridgeKernel tests rather than for
   * actual production use in Fortran)
   *
   * @param gpublocks number of gpublocks
   * @param gputhreads number of gputhreads
   */
    void set_gpugrid( const int gpublocks, const int gputhreads );

    /**
   * Sequence to be executed for the Cuda matrix element calculation
   *
   * @param momenta the pointer to the input 4-momenta
   * @param gs the pointer to the input Gs (running QCD coupling constant
   * alphas)
   * @param iflavorVec the index of the flavor combination
   * @param rndhel the pointer to the input random numbers for helicity
   * selection
   * @param rndcol the pointer to the input random numbers for color selection
   * @param channelIds the Feynman diagram to enhance in multi-channel mode if 1
   * to n
   * @param mes the pointer to the output matrix elements
   * @param selhel the pointer to the output selected helicities
   * @param selcol the pointer to the output selected colors
   * @param goodHelOnly quit after computing good helicities?
   */
    void gpu_sequence( const FORTRANFPTYPE* momenta, const FORTRANFPTYPE* gs, const unsigned int* iflavorVec, const FORTRANFPTYPE* rndhel, const FORTRANFPTYPE* rndcol, const unsigned int* channelIds, FORTRANFPTYPE* mes, int* selhel, int* selcol, const bool goodHelOnly = false );
#else
    /**
   * Sequence to be executed for the vectorized CPU matrix element calculation
   *
   * @param momenta the pointer to the input 4-momenta
   * @param gs the pointer to the input Gs (running QCD coupling constant
   * alphas)
   * @param iflavorVec the index of the flavor combination
   * @param rndhel the pointer to the input random numbers for helicity
   * selection
   * @param rndcol the pointer to the input random numbers for color selection
   * @param channelIds the Feynman diagram to enhance in multi-channel mode if 1
   * to n
   * @param mes the pointer to the output matrix elements
   * @param selhel the pointer to the output selected helicities
   * @param selcol the pointer to the output selected colors
   * @param goodHelOnly quit after computing good helicities?
   */
    void cpu_sequence( const FORTRANFPTYPE* momenta, const FORTRANFPTYPE* gs, const unsigned int* iflavorVec, const FORTRANFPTYPE* rndhel, const FORTRANFPTYPE* rndcol, const unsigned int* channelIds, FORTRANFPTYPE* mes, int* selhel, int* selcol, const bool goodHelOnly = false );
#endif

    // Return the number of good helicities (-1 initially when they have not yet
    // been calculated)
    int nGoodHel() const { return m_nGoodHel; }

    // Return the total number of helicities (expose cudacpp ncomb in the Bridge
    // interface to Fortran)
    constexpr int nTotHel() const { return CPPProcess::ncomb; }

  private:
    unsigned int m_nevt; // number of events
    int m_nGoodHel;      // the number of good helicities (-1 initially when they have
                         // not yet been calculated)

#ifdef MGONGPUCPP_GPUIMPL
    int m_gputhreads; // number of gpu threads (default set from number of
                      // events, can be modified)
    int m_gpublocks;  // number of gpu blocks (default set from number of events,
                      // can be modified)
    DeviceBuffer<FORTRANFPTYPE, sizePerEventMomenta> m_devMomentaF;
    DeviceBufferMomenta m_devMomentaC;
    DeviceBufferGs m_devGs;
    DeviceBufferIflavorVec m_devIflavorVec;
    DeviceBufferRndNumHelicity m_devRndHel;
    DeviceBufferRndNumColor m_devRndCol;
    DeviceBufferMatrixElements m_devMEs;
    DeviceBufferSelectedHelicity m_devSelHel;
    DeviceBufferSelectedColor m_devSelCol;
    DeviceBufferChannelIds m_devChannelIds;
    PinnedHostBufferIflavorVec m_hstIflavorVec;
    PinnedHostBufferGs m_hstGs;
    PinnedHostBufferRndNumHelicity m_hstRndHel;
    PinnedHostBufferRndNumColor m_hstRndCol;
    PinnedHostBufferMatrixElements m_hstMEs;
    PinnedHostBufferSelectedHelicity m_hstSelHel;
    PinnedHostBufferSelectedColor m_hstSelCol;
    PinnedHostBufferChannelIds m_hstChannelIds;
    std::unique_ptr<MatrixElementKernelDevice> m_pmek;
    // static constexpr int s_gputhreadsmin = 16; // minimum number of gpu threads
    // (TEST VALUE FOR MADEVENT)
    static constexpr int s_gputhreadsmin =
      32; // minimum number of gpu threads (DEFAULT)
#else
    HostBufferMomenta m_hstMomentaC;
    HostBufferGs m_hstGs;
    HostBufferIflavorVec m_hstIflavorVec;
    HostBufferRndNumHelicity m_hstRndHel;
    HostBufferRndNumColor m_hstRndCol;
    HostBufferMatrixElements m_hstMEs;
    HostBufferSelectedHelicity m_hstSelHel;
    HostBufferSelectedColor m_hstSelCol;
    HostBufferChannelIds m_hstChannelIds;
    std::unique_ptr<MatrixElementKernelHost> m_pmek;
#endif
  };

  //--------------------------------------------------------------------------
  //
  // Forward declare transposition methods
  //

#ifdef MGONGPUCPP_GPUIMPL

  template<typename Tin, typename Tout>
  __global__ void dev_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt );

#endif // MGONGPUCPP_GPUIMPL

  template<typename Tin, typename Tout>
  void hst_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt );

  template<typename Tin, typename Tout>
  void hst_transposeMomentaC2F( const Tin* in, Tout* out, const unsigned int nevt );

  //--------------------------------------------------------------------------
  //
  // Implementations of member functions of class Bridge
  //

  template<typename FORTRANFPTYPE>
  Bridge<FORTRANFPTYPE>::Bridge( unsigned int nevtF, unsigned int nparF, unsigned int np4F)
    : m_nevt( nevtF ), m_nGoodHel( -1 )
#ifdef MGONGPUCPP_GPUIMPL
    , m_gputhreads( 256 )                  // default number of gpu threads
    , m_gpublocks( m_nevt / m_gputhreads ) // this ensures m_nevt <= m_gpublocks*m_gputhreads
    , m_devMomentaF( m_nevt )
    , m_devMomentaC( m_nevt )
    , m_devIflavorVec( m_nevt )
    , m_devGs( m_nevt )
    , m_devRndHel( m_nevt )
    , m_devRndCol( m_nevt )
    , m_devMEs( m_nevt )
    , m_devSelHel( m_nevt )
    , m_devSelCol( m_nevt )
    , m_devChannelIds( m_nevt )
#else
    , m_hstMomentaC( m_nevt )
#endif
    , m_hstGs( m_nevt )
    , m_hstIflavorVec( m_nevt )
    , m_hstRndHel( m_nevt )
    , m_hstRndCol( m_nevt )
    , m_hstMEs( m_nevt )
    , m_hstSelHel( m_nevt )
    , m_hstSelCol( m_nevt )
    , m_hstChannelIds( m_nevt )
    , m_pmek( nullptr )
  {
    if( nparF != CPPProcess::npar )
      throw std::runtime_error( "Bridge constructor: npar mismatch" );
    if( np4F != CPPProcess::np4 )
      throw std::runtime_error( "Bridge constructor: np4 mismatch" );
#ifdef MGONGPUCPP_GPUIMPL
    if( ( m_nevt < s_gputhreadsmin ) || ( m_nevt % s_gputhreadsmin != 0 ) )
      throw std::runtime_error(
        "Bridge constructor: nevt should be a multiple of " +
        std::to_string( s_gputhreadsmin ) );
    while( m_nevt != m_gpublocks * m_gputhreads )
    {
      m_gputhreads /= 2;
      if( m_gputhreads < s_gputhreadsmin )
        throw std::logic_error(
          "Bridge constructor: FIXME! cannot choose gputhreads" ); // this
                                                                   // should
                                                                   // never
                                                                   // happen!
      m_gpublocks = m_nevt / m_gputhreads;
    }
#ifdef MGONGPUCPP_VERBOSE
    std::cout << "WARNING! Instantiate device Bridge (nevt=" << m_nevt
              << ", gpublocks=" << m_gpublocks << ", gputhreads=" << m_gputhreads
              << ", gpublocks*gputhreads=" << m_gpublocks * m_gputhreads << ")"
              << std::endl;
#endif
    m_pmek.reset( new MatrixElementKernelDevice(
      m_devMomentaC, m_devGs, m_devIflavorVec, m_devRndHel, m_devRndCol, m_devChannelIds, m_devMEs, m_devSelHel, m_devSelCol, m_gpublocks, m_gputhreads) );
#else
#ifdef MGONGPUCPP_VERBOSE
    std::cout << "WARNING! Instantiate host Bridge (nevt=" << m_nevt << ")"
              << std::endl;
#endif
    m_pmek.reset( new MatrixElementKernelHost(
      m_hstMomentaC, m_hstGs, m_hstIflavorVec, m_hstRndHel, m_hstRndCol, m_hstChannelIds, m_hstMEs, m_hstSelHel, m_hstSelCol, m_nevt ) );
#endif // MGONGPUCPP_GPUIMPL
    // Create a process object, read param card and set parameters
    // FIXME: the process instance can happily go out of scope because it is only
    // needed to read parameters?
    // FIXME: the CPPProcess should really be a singleton? what if fbridgecreate
    // is called from several Fortran threads?
    CPPProcess process( /*verbose=*/false );
    std::string paramCard =
      "../Cards/param_card.dat"; // ZW: change default param_card.dat location
                                 // to one dir down
    /*
#ifdef __HIPCC__
  if( !std::experimental::filesystem::exists( paramCard ) ) paramCard = "../" +
paramCard; #else if( !std::filesystem::exists( paramCard ) ) paramCard = "../" +
paramCard; #endif
  */
    // struct stat dummybuffer; // bypass std::filesystem #803
    // if( !( stat( paramCard.c_str(), &dummyBuffer ) == 0 ) ) paramCard = "../" +
    // paramCard; //
    auto fileExists = []( std::string& fileName )
    {
      struct stat buffer;
      return stat( fileName.c_str(), &buffer ) == 0;
    };
    size_t paramCardCheck = 2; // ZW: check for paramCard up to 2 directories up
    for( size_t k = 0; k < paramCardCheck; ++k )
    {
      if( fileExists( paramCard ) ) break; // bypass std::filesystem #803
      paramCard = "../" + paramCard;
    }
    process.initProc( paramCard );
  }

#ifdef MGONGPUCPP_GPUIMPL
  template<typename FORTRANFPTYPE>
  void Bridge<FORTRANFPTYPE>::set_gpugrid( const int gpublocks,
                                           const int gputhreads )
  {
    if( m_nevt != gpublocks * gputhreads )
      throw std::runtime_error(
        "Bridge: gpublocks*gputhreads must equal m_nevt in set_gpugrid" );
    m_gpublocks = gpublocks;
    m_gputhreads = gputhreads;
#ifdef MGONGPUCPP_VERBOSE
    std::cout << "WARNING! Set grid in Bridge (nevt=" << m_nevt
              << ", gpublocks=" << m_gpublocks << ", gputhreads=" << m_gputhreads
              << ", gpublocks*gputhreads=" << m_gpublocks * m_gputhreads << ")"
              << std::endl;
#endif
    m_pmek->setGrid( m_gpublocks, m_gputhreads );
  }
#endif

#ifdef MGONGPUCPP_GPUIMPL
  template<typename FORTRANFPTYPE>
  void Bridge<FORTRANFPTYPE>::gpu_sequence( const FORTRANFPTYPE* momenta,
                                            const FORTRANFPTYPE* gs,
                                            const unsigned int* iflavorVec,
                                            const FORTRANFPTYPE* rndhel,
                                            const FORTRANFPTYPE* rndcol,
                                            const unsigned int* channelIds,
                                            FORTRANFPTYPE* mes,
                                            int* selhel,
                                            int* selcol,
                                            const bool goodHelOnly )
  {
    constexpr int neppM = MemoryAccessMomenta::neppM;
    if constexpr( neppM == 1 && std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      gpuMemcpy( m_devMomentaC.data(), momenta, m_devMomentaC.bytes(), gpuMemcpyHostToDevice );
    }
    else
    {
      gpuMemcpy( m_devMomentaF.data(), momenta, m_devMomentaF.bytes(), gpuMemcpyHostToDevice );
      const int thrPerEvt =
        CPPProcess::npar *
        CPPProcess::np4; // AV: transpose alg does 1 element per thread (NOT 1
                         // event per thread)
      // const int thrPerEvt = 1; // AV: try new alg with 1 event per thread...
      // this seems slower
      gpuLaunchKernel( dev_transposeMomentaF2C, m_gpublocks * thrPerEvt, m_gputhreads, m_devMomentaF.data(), m_devMomentaC.data(), m_nevt );
    }
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( m_hstGs.data(), gs, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndHel.data(), rndhel, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndCol.data(), rndcol, m_nevt * sizeof( FORTRANFPTYPE ) );
    }
    else
    {
      std::copy( gs, gs + m_nevt, m_hstGs.data() );
      std::copy( rndhel, rndhel + m_nevt, m_hstRndHel.data() );
      std::copy( rndcol, rndcol + m_nevt, m_hstRndCol.data() );
    }
    const bool useChannelIds = ( channelIds != nullptr ) && ( !goodHelOnly );
    if( useChannelIds )
      memcpy( m_hstChannelIds.data(), channelIds, m_nevt * sizeof( unsigned int ) );
    // else ... // no need to initialize m_hstChannel: it is allocated with
    // gpuMallocHost and NOT initialized in PinnedHostBufferBase, but it is NOT
    // used later on
    // initialise iflavorVec
    memcpy( m_hstIflavorVec.data(), iflavorVec, m_nevt * sizeof( unsigned int ) );
    copyDeviceFromHost( m_devGs, m_hstGs );
    copyDeviceFromHost( m_devRndHel, m_hstRndHel );
    copyDeviceFromHost( m_devRndCol, m_hstRndCol );
    if( useChannelIds ) copyDeviceFromHost( m_devChannelIds, m_hstChannelIds );
    copyDeviceFromHost( m_devIflavorVec, m_hstIflavorVec );
    if( m_nGoodHel < 0 )
    {
      m_nGoodHel = m_pmek->computeGoodHelicities();
      if( m_nGoodHel < 0 )
        throw std::runtime_error(
          "Bridge gpu_sequence: computeGoodHelicities returned nGoodHel<0" );
    }
    if( goodHelOnly ) return;
    m_pmek->computeMatrixElements( useChannelIds );
    copyHostFromDevice( m_hstMEs, m_devMEs );
#ifdef MGONGPUCPP_VERBOSE
    flagAbnormalMEs( m_hstMEs.data(), m_nevt );
#endif
    copyHostFromDevice( m_hstSelHel, m_devSelHel );
    copyHostFromDevice( m_hstSelCol, m_devSelCol );
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( mes, m_hstMEs.data(), m_hstMEs.bytes() );
      memcpy( selhel, m_hstSelHel.data(), m_hstSelHel.bytes() );
      memcpy( selcol, m_hstSelCol.data(), m_hstSelCol.bytes() );
    }
    else
    {
      std::copy( m_hstMEs.data(), m_hstMEs.data() + m_nevt, mes );
      std::copy( m_hstSelHel.data(), m_hstSelHel.data() + m_nevt, selhel );
      std::copy( m_hstSelCol.data(), m_hstSelCol.data() + m_nevt, selcol );
    }
  }
#endif

#ifndef MGONGPUCPP_GPUIMPL
  template<typename FORTRANFPTYPE>
  void Bridge<FORTRANFPTYPE>::cpu_sequence( const FORTRANFPTYPE* momenta,
                                            const FORTRANFPTYPE* gs,
                                            const unsigned int* iflavorVec,
                                            const FORTRANFPTYPE* rndhel,
                                            const FORTRANFPTYPE* rndcol,
                                            const unsigned int* channelIds,
                                            FORTRANFPTYPE* mes,
                                            int* selhel,
                                            int* selcol,
                                            const bool goodHelOnly )
  {
    hst_transposeMomentaF2C( momenta, m_hstMomentaC.data(), m_nevt );
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( m_hstGs.data(), gs, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndHel.data(), rndhel, m_nevt * sizeof( FORTRANFPTYPE ) );
      memcpy( m_hstRndCol.data(), rndcol, m_nevt * sizeof( FORTRANFPTYPE ) );
    }
    else
    {
      std::copy( gs, gs + m_nevt, m_hstGs.data() );
      std::copy( rndhel, rndhel + m_nevt, m_hstRndHel.data() );
      std::copy( rndcol, rndcol + m_nevt, m_hstRndCol.data() );
    }
    const bool useChannelIds = ( channelIds != nullptr ) && ( !goodHelOnly );
    if( useChannelIds )
      memcpy( m_hstChannelIds.data(), channelIds, m_nevt * sizeof( unsigned int ) );
    // else ... // no need to initialize m_hstChannel: it is allocated and default
    // initialized in HostBufferBase (and it is not used later on anyway)
    // initialise iflavorVec
    memcpy( m_hstIflavorVec.data(), iflavorVec, m_nevt * sizeof( unsigned int ) );
    if( m_nGoodHel < 0 )
    {
      m_nGoodHel = m_pmek->computeGoodHelicities();
      if( m_nGoodHel < 0 )
        throw std::runtime_error(
          "Bridge cpu_sequence: computeGoodHelicities returned nGoodHel<0" );
    }
    if( goodHelOnly ) return;
    m_pmek->computeMatrixElements( useChannelIds );
#ifdef MGONGPUCPP_VERBOSE
    flagAbnormalMEs( m_hstMEs.data(), m_nevt );
#endif
    if constexpr( std::is_same_v<FORTRANFPTYPE, fptype> )
    {
      memcpy( mes, m_hstMEs.data(), m_hstMEs.bytes() );
      memcpy( selhel, m_hstSelHel.data(), m_hstSelHel.bytes() );
      memcpy( selcol, m_hstSelCol.data(), m_hstSelCol.bytes() );
    }
    else
    {
      std::copy( m_hstMEs.data(), m_hstMEs.data() + m_nevt, mes );
      std::copy( m_hstSelHel.data(), m_hstSelHel.data() + m_nevt, selhel );
      std::copy( m_hstSelCol.data(), m_hstSelCol.data() + m_nevt, selcol );
    }
  }
#endif

  //--------------------------------------------------------------------------
  //
  // Implementations of transposition methods
  // - FORTRAN arrays: P_MULTI(0:3, NEXTERNAL, VECSIZE_USED) ==>
  // p_multi[nevtF][nparF][np4F] in C++ (AOS)
  // - C++ array: momenta[npagM][npar][np4][neppM] with nevt=npagM*neppM (AOSOA)
  //

#ifdef MGONGPUCPP_GPUIMPL
  template<typename Tin, typename Tout>
  __global__ void dev_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool oldImplementation = true; // default: use old implementation
    if constexpr( oldImplementation )
    {
      // SR initial implementation
      constexpr int part = CPPProcess::npar;
      constexpr int mome = CPPProcess::np4;
      constexpr int strd = MemoryAccessMomenta::neppM;
      int pos = blockDim.x * blockIdx.x + threadIdx.x;
      int arrlen = nevt * part * mome;
      if( pos < arrlen )
      {
        int page_i = pos / ( strd * mome * part );
        int rest_1 = pos % ( strd * mome * part );
        int part_i = rest_1 / ( strd * mome );
        int rest_2 = rest_1 % ( strd * mome );
        int mome_i = rest_2 / strd;
        int strd_i = rest_2 % strd;
        int inpos = ( page_i * strd + strd_i ) // event number
            * ( part * mome )                  // event size (pos of event)
          + part_i * mome                      // particle inside event
          + mome_i;                            // momentum inside particle
        out[pos] = in[inpos];                  // F2C (Fortran to C)
      }
    }
    else
    {
      // AV attempt another implementation with 1 event per thread: this seems
      // slower... F-style: AOS[nevtF][nparF][np4F] C-style:
      // AOSOA[npagM][npar][np4][neppM] with nevt=npagM*neppM
      constexpr int npar = CPPProcess::npar;
      constexpr int np4 = CPPProcess::np4;
      constexpr int neppM = MemoryAccessMomenta::neppM;
      assert( nevt % neppM ==
              0 ); // number of events is not a multiple of neppM???
      int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      int ipagM = ievt / neppM;
      int ieppM = ievt % neppM;
      for( int ip4 = 0; ip4 < np4; ip4++ )
        for( int ipar = 0; ipar < npar; ipar++ )
        {
          int cpos = ipagM * npar * np4 * neppM + ipar * np4 * neppM +
            ip4 * neppM + ieppM;
          int fpos = ievt * npar * np4 + ipar * np4 + ip4;
          out[cpos] = in[fpos]; // F2C (Fortran to C)
        }
    }
  }
#endif

  template<typename Tin, typename Tout, bool F2C>
  void hst_transposeMomenta( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool oldImplementation = false; // default: use new implementation
    if constexpr( oldImplementation )
    {
      // SR initial implementation
      constexpr unsigned int part = CPPProcess::npar;
      constexpr unsigned int mome = CPPProcess::np4;
      constexpr unsigned int strd = MemoryAccessMomenta::neppM;
      unsigned int arrlen = nevt * part * mome;
      for( unsigned int pos = 0; pos < arrlen; ++pos )
      {
        unsigned int page_i = pos / ( strd * mome * part );
        unsigned int rest_1 = pos % ( strd * mome * part );
        unsigned int part_i = rest_1 / ( strd * mome );
        unsigned int rest_2 = rest_1 % ( strd * mome );
        unsigned int mome_i = rest_2 / strd;
        unsigned int strd_i = rest_2 % strd;
        unsigned int inpos = ( page_i * strd + strd_i ) // event number
            * ( part * mome )                           // event size (pos of event)
          + part_i * mome                               // particle inside event
          + mome_i;                                     // momentum inside particle
        if constexpr( F2C )                             // needs c++17 and cuda >=11.2 (#333)
          out[pos] = in[inpos];                         // F2C (Fortran to C)
        else
          out[inpos] = in[pos]; // C2F (C to Fortran)
      }
    }
    else
    {
      // AV attempt another implementation: this is slightly faster (better c++
      // pipelining?) [NB! this is not a transposition, it is an AOS to AOSOA
      // conversion: if neppM=1, a memcpy is enough] F-style:
      // AOS[nevtF][nparF][np4F] C-style: AOSOA[npagM][npar][np4][neppM] with
      // nevt=npagM*neppM
      constexpr unsigned int npar = CPPProcess::npar;
      constexpr unsigned int np4 = CPPProcess::np4;
      constexpr unsigned int neppM = MemoryAccessMomenta::neppM;
      if constexpr( neppM == 1 && std::is_same_v<Tin, Tout> )
      {
        memcpy( out, in, nevt * npar * np4 * sizeof( Tin ) );
      }
      else
      {
        const unsigned int npagM = nevt / neppM;
        assert( nevt % neppM ==
                0 ); // number of events is not a multiple of neppM???
        for( unsigned int ipagM = 0; ipagM < npagM; ipagM++ )
          for( unsigned int ip4 = 0; ip4 < np4; ip4++ )
            for( unsigned int ipar = 0; ipar < npar; ipar++ )
              for( unsigned int ieppM = 0; ieppM < neppM; ieppM++ )
              {
                unsigned int ievt = ipagM * neppM + ieppM;
                unsigned int cpos = ipagM * npar * np4 * neppM +
                  ipar * np4 * neppM + ip4 * neppM + ieppM;
                unsigned int fpos = ievt * npar * np4 + ipar * np4 + ip4;
                if constexpr( F2C )
                  out[cpos] = in[fpos]; // F2C (Fortran to C)
                else
                  out[fpos] = in[cpos]; // C2F (C to Fortran)
              }
      }
    }
  }

  template<typename Tin, typename Tout>
  void hst_transposeMomentaF2C( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool F2C = true;
    hst_transposeMomenta<Tin, Tout, F2C>( in, out, nevt );
  }

  template<typename Tin, typename Tout>
  void hst_transposeMomentaC2F( const Tin* in, Tout* out, const unsigned int nevt )
  {
    constexpr bool F2C = false;
    hst_transposeMomenta<Tin, Tout, F2C>( in, out, nevt );
  }

  //--------------------------------------------------------------------------
} // namespace mg5amcGpu
#endif // BRIDGE_H
