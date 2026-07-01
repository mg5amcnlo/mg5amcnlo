// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: O. Mattelaer (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2026).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
//
// Standalone script for MadGraph7 standalone mode.
// Generates phase-space points with RAMBO and evaluates the matrix element
// through the UMAMI interface (umami.h).
//
// Two run modes:
//   * matrix (default): evaluates one phase-space point (generated with the
//                       classic standalone RAMBO, so identical to the one of
//                       the Fortran/C++ standalone 'check' drivers at the
//                       same energy) and prints it together with the matrix
//                       element of every flavor combination.
//   * perf            : runs nblocks*nthreads*niter events on a single flavor
//                       and prints performance counters.
//
//==========================================================================

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "GpuAbstraction.h"
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"
#include "epoch_process_id.h"
#include "read_slha.h"
#include "timermap.h"
#include "umami.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#define STRINGIFY( s ) #s
#define XSTRINGIFY( s ) STRINGIFY( s )
#define SEP79 79

namespace
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  // Fixed physics inputs
  fptype kEnergy = 1500.;                  // Ecms = 1.5 TeV and changed for the matrix mode to 1TeV
  constexpr unsigned long long kSeed = 20200805ULL;  // reproducible RAMBO seed

  // Matrix-mode always runs 8 events on a single flavor index.
  constexpr unsigned int kMatrixBlocks = 1;
  constexpr unsigned int kMatrixThreads = 8;

  // Power of GeV of the matrix-element output; depends only on the number of external legs.
  constexpr int kMEGeVExponent = -( 2 * CPPProcess::npar - 8 );

  bool is_number( const char* s )
  {
    const char* t = s;
    while( *t != '\0' && isdigit( *t ) ) ++t;
    return (int)strlen( s ) == t - s;
  }

  // Accepts plain decimal numbers such as "1000" or "1000.0" (used for the
  // optional energy argument of matrix mode).
  bool is_float( const char* s )
  {
    int ndots = 0;
    const char* t = s;
    while( *t != '\0' && ( isdigit( *t ) || ( *t == '.' && ndots++ == 0 ) ) ) ++t;
    return t != s && (int)strlen( s ) == t - s;
  }

  enum Mode { MODE_MATRIX, MODE_PERF };

  enum RamboType { RAMBO_MASSIVE, RAMBO_MASSLESS };

  int usage( const char* argv0, int ret = 1 )
  {
    std::cout
      << "Usage:\n"
      << "  " << argv0 << " [matrix] [-v|--verbose] [<energy>]\n"
      << "  " << argv0 << " perf [-v|--verbose] [-f|--flavor <int>] [--rambo-massless]"
      << " [<#blocksPerGrid> <#threadsPerBlock>] <#iterations>\n"
      << "  " << argv0 << " -p [opts]   (legacy alias for `perf`)\n"
      << "\n"
      << "Subcommands:\n"
      << "  matrix (default)  Evaluate one phase-space point (classic standalone\n"
      << "                    RAMBO, identical to the Fortran/C++ 'check' drivers,\n"
      << "                    Ecms = <energy>, default 1000 GeV) and print it with\n"
      << "                    the matrix element for each flavor combination.\n"
      << "                    With -v also prints backend/fptype/hardcodePARAM header.\n"
      << "  perf              Run #blocks*#threads events over #iterations iterations\n"
      << "                    on a single flavor index, then print performance counters.\n"
      << "                    Always prints inputs + backend/fptype header.\n"
      << "                    With -v also dumps every event's phase-space point and ME.\n"
      << "\n"
      << "perf-mode defaults if positional args are omitted:\n"
      << "  #blocksPerGrid = 64, #threadsPerBlock = 256, #iterations = 1.\n";
    return ret;
  }

  // AOSOA -> UMAMI SoA single-event helper. Layout reminder:
  //   AOSOA: aosoa[i_page * npar*4*neppM + ipar*4*neppM + ip4*neppM + i_vector]
  //   UMAMI: soa[ip4 * npar*nevt + ipar*nevt + ievt]
  __host__ __device__ inline void
  aosoa_to_umami_one( const fptype* aosoa,
                      double* soa,
                      std::size_t ievt,
                      std::size_t nevt )
  {
    constexpr int npar = CPPProcess::npar;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      for( int ip4 = 0; ip4 < 4; ++ip4 )
      {
        soa[(std::size_t)ip4 * npar * nevt + (std::size_t)ipar * nevt + ievt] =
          (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, ip4, ipar );
      }
    }
  }

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  aosoa_to_umami_kernel( const fptype* aosoa,
                         double* soa,
                         std::size_t nevt )
  {
    std::size_t ievt = blockDim.x * blockIdx.x + threadIdx.x;
    if( ievt >= nevt ) return;
    aosoa_to_umami_one( aosoa, soa, ievt, nevt );
  }
#endif

  const char* backend_label()
  {
#ifdef __CUDACC__
    return "CUDA";
#elif defined( __HIPCC__ )
    return "HIP";
#else
    return "CPP";
#endif
  }

  const char* fp_label()
  {
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    return "MIXED";
#elif defined MGONGPU_FPTYPE_DOUBLE
    return "DOUBLE";
#elif defined MGONGPU_FPTYPE_FLOAT
    return "FLOAT";
#else
    return "UNKNOWN";
#endif
  }

  void print_run_header( std::ostream& os )
  {
    os << "Process                     = " << XSTRINGIFY( MG_EPOCH_PROCESS_ID ) << "_" << backend_label()
#ifdef MGONGPU_HARDCODE_PARAM
       << " [hardcodePARAM=1]" << std::endl
#else
       << " [hardcodePARAM=0]" << std::endl
#endif
       << "FP precision                = " << fp_label() << std::endl
       << "Random number generation    = COMMON RANDOM HOST" << std::endl;
  }

  void print_momenta_table( std::ostream& os, const fptype* aosoa, unsigned int ievt )
  {
    constexpr int npar = CPPProcess::npar;
    os << std::string( SEP79, '-' ) << std::endl
       << " n        E             px             py              pz" << std::endl;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      double E  = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 0, ipar );
      double px = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 1, ipar );
      double py = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 2, ipar );
      double pz = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 3, ipar );
      os << std::scientific << std::setprecision( 7 )
         << std::setw( 2 ) << ipar + 1
         << std::setw( 16 ) << E
         << std::setw( 16 ) << px
         << std::setw( 16 ) << py
         << std::setw( 16 ) << pz
         << std::endl
         << std::defaultfloat;
    }
    os << std::string( SEP79, '-' ) << std::endl;
  }

  // Run sigmaKin via UMAMI for `nevt` events and copy back the MEs.
  // Both the momenta (UMAMI SoA layout) and the per-event flavor buffer must be set
  // by the caller. On GPU the buffers are device pointers and `hstMEs` receives the
  // host-side copy; on CPU `umamiMEs` is the output buffer.
  bool run_umami(
    UmamiHandle handle,
    unsigned int nevt,
    mgOnGpu::TimerMap& timermap,
    double& wavetime,
#ifdef MGONGPUCPP_GPUIMPL
    const DeviceBufferBase<double>& devUmamiMomenta,
    const DeviceBufferBase<unsigned int>& devFlv,
    DeviceBufferBase<double>& devUmamiMEs,
    std::vector<double>& hstMEs
#else
    const std::vector<double>& umamiMomenta,
    const std::vector<unsigned int>& flvVec,
    std::vector<double>& umamiMEs
#endif
  )
  {
    constexpr unsigned int UmamiInKeyNum = 2;
    timermap.start( "3a SigmaKin" );
    UmamiInputKey in_keys[UmamiInKeyNum] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX };
    UmamiOutputKey out_keys[1] = { UMAMI_OUT_MATRIX_ELEMENT };
#ifdef MGONGPUCPP_GPUIMPL
    const void* inputs[UmamiInKeyNum] = { devUmamiMomenta.data(), devFlv.data() };
    void* outputs[1] = { devUmamiMEs.data() };
#else
    const void* inputs[UmamiInKeyNum] = { umamiMomenta.data(), flvVec.data() };
    void* outputs[1] = { umamiMEs.data() };
#endif
    UmamiStatus st = umami_matrix_element(
      handle, nevt, nevt, 0, UmamiInKeyNum, in_keys, inputs, 1, out_keys, outputs );
    wavetime += timermap.stop();
    if( st != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << std::endl;
      return false;
    }

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "3b CpDTHmes" );
    gpuMemcpy( hstMEs.data(), devUmamiMEs.data(), nevt * sizeof( double ), gpuMemcpyDeviceToHost );
    wavetime += timermap.stop();
#endif
    return true;
  }

  // --------------------------------------------------------------------------
  // Classic MadGraph standalone RAMBO (RANMAR generator seeded with the fixed
  // values 1802/9373), reproducing the exact phase-space point used by the
  // Fortran and C++ standalone 'check' drivers at the same energy, so that
  // matrix-mode output can be compared line by line across backends.
  // Host-side only; ported from madgraph/iolibs/template_files/rambo.cc.
  // --------------------------------------------------------------------------
  namespace classic_rambo
  {
    struct Random
    {
      double ranu[98];
      double ranc, rancd, rancm;
      int iranmr, jranmr;

      // universal random number generator proposed by Marsaglia and Zaman
      double ranmar()
      {
        double uni = ranu[iranmr] - ranu[jranmr];
        if( uni < 0 ) uni = uni + 1;
        ranu[iranmr] = uni;
        iranmr = iranmr - 1;
        jranmr = jranmr - 1;
        if( iranmr == 0 ) iranmr = 97;
        if( jranmr == 0 ) jranmr = 97;
        ranc = ranc - rancd;
        if( ranc < 0 ) ranc = ranc + rancm;
        uni = uni - ranc;
        if( uni < 0 ) uni = uni + 1;
        return uni;
      }

      void rmarin( int ij, int kl )
      {
        int i = ij / 177 % 177 + 2;
        int j = ij % 177 + 2;
        int k = ( kl / 169 ) % 178 + 1;
        int l = kl % 169;
        for( int ii = 1; ii < 98; ii++ )
        {
          double s = 0;
          double t = .5;
          for( int jj = 1; jj < 25; jj++ )
          {
            int m = ( ( i * j % 179 ) * k ) % 179;
            i = j;
            j = k;
            k = m;
            l = ( 53 * l + 1 ) % 169;
            if( ( l * m ) % 64 >= 32 ) s = s + t;
            t = .5 * t;
          }
          ranu[ii] = s;
        }
        ranc = 362436. / 16777216.;
        rancd = 7654321. / 16777216.;
        rancm = 16777213. / 16777216.;
        iranmr = 97;
        jranmr = 33;
      }
    };

    inline double rn()
    {
      static Random rand;
      static bool init = true;
      if( init )
      {
        init = false;
        rand.rmarin( 1802, 9373 );
      }
      double ran;
      while( true )
      {
        ran = rand.ranmar();
        if( ran > 1e-16 ) break;
      }
      return ran;
    }

    // RAMBO: democratic multi-particle phase space generator (S.D. Ellis,
    // R. Kleiss, W.J. Stirling); weights are logarithmic.
    inline std::vector<std::vector<double>>
    rambo( double et, const std::vector<double>& xm, double& wt )
    {
      const int n = (int)xm.size();
      std::vector<std::vector<double>> q( n, std::vector<double>( 4 ) );
      std::vector<std::vector<double>> p( n, std::vector<double>( 4 ) );
      std::vector<double> z( n ), r( 4 ), b( 3 ), p2( n ), xm2( n ), e( n ), v( n );
      const double acc = 1e-14;
      const int itmax = 6;
      const double twopi = 8. * atan( 1. );
      const double po2log = log( twopi / 4. );

      // factorials for the phase-space weight
      z[1] = po2log;
      for( int k = 2; k < n; k++ ) z[k] = z[k - 1] + po2log - 2. * log( double( k - 1 ) );
      for( int k = 2; k < n; k++ ) z[k] = z[k] - log( double( k ) );

      if( n < 1 || n > 101 )
      {
        std::cout << "Too few or many particles: " << n << std::endl;
        exit( -1 );
      }
      double xmt = 0.;
      int nm = 0;
      for( int i = 0; i < n; i++ )
      {
        if( xm[i] != 0. ) nm = nm + 1;
        xmt = xmt + std::abs( xm[i] );
      }
      if( xmt > et )
      {
        std::cout << "Too low energy: " << et << " needed " << xmt << std::endl;
        exit( -1 );
      }

      // generate n massless momenta in infinite phase space
      for( int i = 0; i < n; i++ )
      {
        double r1 = rn();
        double c = 2. * r1 - 1.;
        double s = sqrt( 1. - c * c );
        double f = twopi * rn();
        r1 = rn();
        double r2 = rn();
        q[i][0] = -log( r1 * r2 );
        q[i][3] = q[i][0] * c;
        q[i][2] = q[i][0] * s * cos( f );
        q[i][1] = q[i][0] * s * sin( f );
      }

      // parameters of the conformal transformation
      for( int k = 0; k < 4; k++ ) r[k] = 0.;
      for( int i = 0; i < n; i++ )
        for( int k = 0; k < 4; k++ ) r[k] = r[k] + q[i][k];
      double rmas = sqrt( pow( r[0], 2 ) - pow( r[3], 2 ) - pow( r[2], 2 ) - pow( r[1], 2 ) );
      for( int k = 1; k < 4; k++ ) b[k - 1] = -r[k] / rmas;
      double g = r[0] / rmas;
      double a = 1. / ( 1. + g );
      double x = et / rmas;

      // transform the q's conformally into the p's
      for( int i = 0; i < n; i++ )
      {
        double bq = b[0] * q[i][1] + b[1] * q[i][2] + b[2] * q[i][3];
        for( int k = 1; k < 4; k++ ) p[i][k] = x * ( q[i][k] + b[k - 1] * ( q[i][0] + a * bq ) );
        p[i][0] = x * ( g * q[i][0] + bq );
      }

      wt = po2log;
      if( n != 2 ) wt = ( 2. * n - 4. ) * log( et ) + z[n - 1];

      // massless case is done
      if( nm == 0 ) return p;

      // massive particles: rescale the momenta by a factor x
      double xmax = sqrt( 1. - pow( xmt / et, 2 ) );
      for( int i = 0; i < n; i++ )
      {
        xm2[i] = pow( xm[i], 2 );
        p2[i] = pow( p[i][0], 2 );
      }
      int iter = 0;
      x = xmax;
      double accu = et * acc;
      while( true )
      {
        double f0 = -et;
        double g0 = 0.;
        double x2 = x * x;
        for( int i = 0; i < n; i++ )
        {
          e[i] = sqrt( xm2[i] + x2 * p2[i] );
          f0 = f0 + e[i];
          g0 = g0 + p2[i] / e[i];
        }
        if( std::abs( f0 ) <= accu ) break;
        iter = iter + 1;
        if( iter > itmax )
        {
          std::cout << "Too many iterations without desired accuracy: " << itmax << std::endl;
          break;
        }
        x = x - f0 / ( x * g0 );
      }
      for( int i = 0; i < n; i++ )
      {
        v[i] = x * p[i][0];
        for( int k = 1; k < 4; k++ ) p[i][k] = x * p[i][k];
        p[i][0] = e[i];
      }

      double wt2 = 1.;
      double wt3 = 0.;
      for( int i = 0; i < n; i++ )
      {
        wt2 = wt2 * v[i] / e[i];
        wt3 = wt3 + pow( v[i], 2 ) / e[i];
      }
      double wtm = ( 2. * n - 3. ) * log( x ) + log( wt2 / wt3 * et );
      wt = wt + wtm;
      return p;
    }

    // Auxiliary function changing convention between MadGraph5_aMC@NLO and
    // RAMBO four-momenta (same as get_momenta in the standalone_cpp driver).
    inline std::vector<std::vector<double>>
    get_momenta( int ninitial, double energy, const std::vector<double>& masses, double& wgt )
    {
      const int nexternal = (int)masses.size();
      const int nfinal = nexternal - ninitial;
      const double e2 = pow( energy, 2 );
      const double m1 = masses[0];

      if( ninitial == 1 )
      {
        std::vector<std::vector<double>> p( 1, std::vector<double>( 4, 0. ) );
        p[0][0] = m1;
        std::vector<double> finalmasses( masses.begin() + 1, masses.end() );
        std::vector<std::vector<double>> p_rambo = rambo( m1, finalmasses, wgt );
        p.insert( p.end(), p_rambo.begin(), p_rambo.end() );
        return p;
      }

      if( ninitial != 2 )
      {
        std::cout << "Rambo needs 1 or 2 incoming particles" << std::endl;
        exit( -1 );
      }

      double etot = energy;
      if( nfinal == 1 ) etot = m1;
      const double m2 = masses[1];
      const double mom = sqrt( ( pow( e2, 2 ) - 2 * e2 * pow( m1, 2 ) + pow( m1, 4 ) - 2 * e2 * pow( m2, 2 ) - 2 * pow( m1, 2 ) * pow( m2, 2 ) + pow( m2, 4 ) ) / ( 4 * e2 ) );
      const double energy1 = sqrt( pow( mom, 2 ) + pow( m1, 2 ) );
      const double energy2 = sqrt( pow( mom, 2 ) + pow( m2, 2 ) );
      std::vector<std::vector<double>> p( 2, std::vector<double>( 4, 0. ) );
      p[0][0] = energy1;
      p[0][3] = mom;
      p[1][0] = energy2;
      p[1][3] = -mom;

      if( nfinal == 1 )
      {
        p.push_back( std::vector<double>( 4, 0. ) );
        p[2][0] = etot;
        wgt = 1;
        return p;
      }
      std::vector<double> finalmasses( masses.begin() + 2, masses.end() );
      std::vector<std::vector<double>> p_rambo = rambo( etot, finalmasses, wgt );
      p.insert( p.end(), p_rambo.begin(), p_rambo.end() );
      return p;
    }
  }

  // --------------------------------------------------------------------------
  // matrix mode: same PS point fed to every flavor combination, print event 0.
  // The point is generated with the classic standalone RAMBO so it is
  // identical to the one of the Fortran/C++ standalone 'check' drivers.
  // --------------------------------------------------------------------------
  int run_matrix_mode( bool verbose )
  {
    constexpr unsigned int nevt = kMatrixBlocks * kMatrixThreads;
    const unsigned int nFlavors = CPPProcess::nmaxflavor;

    mgOnGpu::TimerMap timermap;

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "00 GpuInit" );
    GpuRuntime gpuRuntime( false );

    PinnedHostBufferRndNumMomenta hstRndmom( nevt );
    PinnedHostBufferMomenta hstMomenta( nevt );
    PinnedHostBufferWeights hstWeights( nevt );
    DeviceBufferRndNumMomenta devRndmom( nevt );
    DeviceBufferMomenta devMomenta( nevt );
    DeviceBufferWeights devWeights( nevt );
    DeviceBufferBase<double> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    DeviceBufferBase<double> devUmamiMEs( nevt );
    DeviceBufferBase<unsigned int> devFlv( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<unsigned int> flvVec( nevt );
    std::vector<double> hstUmamiMEs( nevt );
#else
    HostBufferRndNumMomenta hstRndmom( nevt );
    HostBufferMomenta hstMomenta( nevt );
    HostBufferWeights hstWeights( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<double> umamiMEs( nevt );
    std::vector<unsigned int> flvVec( nevt );
#endif

    UmamiHandle umami_handle = nullptr;
    if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_initialize failed" << std::endl;
      return 2;
    }

    // Generate one shared phase-space point used by every flavor, with the
    // classic standalone RAMBO so it matches the Fortran/C++ 'check' drivers.
    CPPProcess process;
    process.initProc( "../../Cards/param_card.dat" );
    double rambowgt = 0.;

    // Retrieve masses
    int npar_meta = 0;
    if( umami_get_meta( UMAMI_META_PARTICLE_COUNT, &npar_meta ) != UMAMI_SUCCESS || npar_meta != CPPProcess::npar )
    {
      std::cerr << "ERROR! umami_get_meta(UMAMI_META_PARTICLE_COUNT) failed" << std::endl;
      umami_free( umami_handle );
      return 2;
    }
    std::vector<double> massesD( npar_meta );
    if( umami_get_meta( UMAMI_META_MASSES, massesD.data() ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_get_meta(UMAMI_META_MASSES) failed" << std::endl;
      umami_free( umami_handle );
      return 2;
    }
    const std::vector<fptype> masses( massesD.begin(), massesD.end() );

    std::vector<std::vector<double>> point =
      classic_rambo::get_momenta( CPPProcess::npari, (double)kEnergy, masses, rambowgt );

    // alpha_s from the param card so the couplings match the Fortran/C++
    // 'check' drivers (UMAMI otherwise falls back to a hardcoded g_s).
    SLHAReader slha( "../../Cards/param_card.dat", false );
    const double alphaS = slha.get_block_entry( "sminputs", 3, 1.180000e-01 );
    std::vector<double> alphasVec( nevt, alphaS );
#ifdef MGONGPUCPP_GPUIMPL
    DeviceBufferBase<double> devAlphaS( nevt );
    gpuMemcpy( devAlphaS.data(), alphasVec.data(), nevt * sizeof( double ), gpuMemcpyHostToDevice );
#endif

    // Always massive RAMBO
    std::unique_ptr<SamplingKernelBase> prsk(
      new RamboSamplingKernelHost( kEnergy, hstRndmom, masses, CPPProcess::npari, nevt, hstMomenta, hstWeights ) );
    prsk->getMomentaInitial();
    prsk->getMomentaFinal();

    // Fill the UMAMI SoA buffer with nevt copies of the same event:
    // soa[ip4 * npar*nevt + ipar*nevt + ievt]
    for( int ip4 = 0; ip4 < 4; ++ip4 )
      for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
          umamiMomenta[(std::size_t)ip4 * CPPProcess::npar * nevt + (std::size_t)ipar * nevt + ievt] = point[ipar][ip4];
#ifdef MGONGPUCPP_GPUIMPL
    gpuMemcpy( devUmamiMomenta.data(), umamiMomenta.data(), umamiMomenta.size() * sizeof( double ), gpuMemcpyHostToDevice );
    // Host only implementation now (copy)
    copyDeviceFromHost( devMomenta, hstMomenta );
    gpuLaunchKernel( aosoa_to_umami_kernel, kMatrixBlocks, kMatrixThreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
    checkGpu( gpuPeekAtLastError() );
#else
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
      aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif

    if( verbose )
    {
      std::cout << std::string( SEP79, '*' ) << std::endl;
      print_run_header( std::cout );
      std::cout << std::string( SEP79, '*' ) << std::endl;
    }

    std::cout << "Phase space point:" << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << " n        E             px             py              pz" << std::endl;
    for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
    {
      std::cout << std::scientific << std::setprecision( 7 )
                << std::setw( 2 ) << ipar + 1
                << std::setw( 16 ) << point[ipar][0]
                << std::setw( 16 ) << point[ipar][1]
                << std::setw( 16 ) << point[ipar][2]
                << std::setw( 16 ) << point[ipar][3]
                << std::endl
                << std::defaultfloat;
    }
    std::cout << std::string( SEP79, '-' ) << std::endl;

    for( unsigned int iflav = 0; iflav < nFlavors; ++iflav )
    {
      std::fill( flvVec.begin(), flvVec.end(), iflav );
#ifdef MGONGPUCPP_GPUIMPL
      gpuMemcpy( devFlv.data(), flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
#endif
      timermap.start( "3a SigmaKin" );
      UmamiInputKey in_keys[3] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_ALPHA_S };
      UmamiOutputKey out_keys[1] = { UMAMI_OUT_MATRIX_ELEMENT };
#ifdef MGONGPUCPP_GPUIMPL
      const void* inputs[3] = { devUmamiMomenta.data(), devFlv.data(), devAlphaS.data() };
      void* outputs[1] = { devUmamiMEs.data() };
#else
      const void* inputs[3] = { umamiMomenta.data(), flvVec.data(), alphasVec.data() };
      void* outputs[1] = { umamiMEs.data() };
#endif
      UmamiStatus st = umami_matrix_element(
        umami_handle, nevt, nevt, 0, 3, in_keys, inputs, 1, out_keys, outputs );
      timermap.stop();
      if( st != UMAMI_SUCCESS )
      {
        std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << std::endl;
        umami_free( umami_handle );
        return 3;
      }
#ifdef MGONGPUCPP_GPUIMPL
      gpuMemcpy( hstUmamiMEs.data(), devUmamiMEs.data(), nevt * sizeof( double ), gpuMemcpyDeviceToHost );
      const double* mes = hstUmamiMEs.data();
#else
      const double* mes = umamiMEs.data();
#endif

      std::cout << " PDG";
      for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
        std::cout << std::setw( 12 ) << CPPProcess::flavorPDG( iflav, ipar );
      std::cout << std::endl
                << " Matrix element = " << std::scientific << std::setprecision( 16 )
                << mes[0] << " GeV^" << kMEGeVExponent << std::endl
                << std::defaultfloat
                << std::string( SEP79, '-' ) << std::endl;
    }

    umami_free( umami_handle );
    return 0;
  }

  // --------------------------------------------------------------------------
  // perf mode: nblocks*nthreads events per iteration on a single flavor.
  // --------------------------------------------------------------------------
  int run_perf_mode( bool verbose,
                     unsigned int gpublocks,
                     unsigned int gputhreads,
                     unsigned int niter,
                     unsigned int flavorID,
                     RamboType ramboType )
  {
    const unsigned int nevt = gpublocks * gputhreads;

    mgOnGpu::TimerMap timermap;

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "00 GpuInit" );
    GpuRuntime gpuRuntime( false );

    PinnedHostBufferRndNumMomenta hstRndmom( nevt );
    PinnedHostBufferMomenta hstMomenta( nevt );
    PinnedHostBufferWeights hstWeights( nevt );
    DeviceBufferRndNumMomenta devRndmom( nevt );
    DeviceBufferMomenta devMomenta( nevt );
    DeviceBufferWeights devWeights( nevt );
    DeviceBufferBase<double> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    DeviceBufferBase<double> devUmamiMEs( nevt );
    DeviceBufferBase<unsigned int> devFlv( nevt );
    std::vector<unsigned int> flvVec( nevt, flavorID );
    std::vector<double> hstUmamiMEs( nevt );
    // perf-mode runs a single flavor, so the device-side flavor buffer is filled once.
    gpuMemcpy( devFlv.data(), flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
#else
    HostBufferRndNumMomenta hstRndmom( nevt );
    HostBufferMomenta hstMomenta( nevt );
    HostBufferWeights hstWeights( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<double> umamiMEs( nevt );
    std::vector<unsigned int> flvVec( nevt, flavorID );
#endif

    std::unique_ptr<RandomNumberKernelBase> prnk(
      new CommonRandomNumberKernel( hstRndmom ) );

    UmamiHandle umami_handle = nullptr;
    if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_initialize failed" << std::endl;
      return 2;
    }

    // Retrieve masses
    std::vector<fptype> masses;
    if( ramboType != RAMBO_MASSLESS)
    {
      int npar_meta = 0;
      if( umami_get_meta( UMAMI_META_PARTICLE_COUNT, &npar_meta ) != UMAMI_SUCCESS || npar_meta != CPPProcess::npar )
      {
        std::cerr << "ERROR! umami_get_meta(UMAMI_META_PARTICLE_COUNT) failed" << std::endl;
        umami_free( umami_handle );
        return 2;
      }
      std::vector<double> massesD( npar_meta );
      if( umami_get_meta( UMAMI_META_MASSES, massesD.data() ) != UMAMI_SUCCESS )
      {
        std::cerr << "ERROR! umami_get_meta(UMAMI_META_MASSES) failed" << std::endl;
        umami_free( umami_handle );
        return 2;
      }
      masses.assign( massesD.begin(), massesD.end() );
    }

    std::unique_ptr<SamplingKernelBase> prsk;
    if( ramboType != RAMBO_MASSLESS )
    {
      // Massive host only (copy) 
      prsk.reset( new RamboSamplingKernelHost( kEnergy, hstRndmom, masses, CPPProcess::npari, nevt, hstMomenta, hstWeights ) );
    }
    else
    {
#ifdef MGONGPUCPP_GPUIMPL
      prsk.reset( new MasslessRamboSamplingKernelDevice( kEnergy, devRndmom, devMomenta, devWeights, gpublocks, gputhreads ) );
#else
      prsk.reset( new MasslessRamboSamplingKernelHost( kEnergy, hstRndmom, hstMomenta, hstWeights, nevt ) );
#endif
    }

    std::unique_ptr<double[]> genrtimes( new double[niter] );
    std::unique_ptr<double[]> rambtimes( new double[niter] );
    std::unique_ptr<double[]> wavetimes( new double[niter] );

    unsigned int nevtABN = 0;
    unsigned int nevtZERO = 0;
    double sumME = 0.;
    double sumMEsq = 0.;
    double minME = std::numeric_limits<double>::infinity();
    double maxME = -std::numeric_limits<double>::infinity();
    unsigned int nevtALL = 0;

    for( unsigned int iiter = 0; iiter < niter; ++iiter )
    {
      double genrtime = 0;
      timermap.start( "1a GenSeed " );
      prnk->seedGenerator( kSeed + iiter );
      genrtime += timermap.stop();
      timermap.start( "1b GenRnGen" );
      prnk->generateRnarray();
      genrtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
      if( ramboType == RAMBO_MASSLESS )
      {
        timermap.start( "1c CpHTDrnd" );
        copyDeviceFromHost( devRndmom, hstRndmom );
        genrtime += timermap.stop();
      }
#endif

      double rambtime = 0;
      timermap.start( "2a RamboIni" );
      prsk->getMomentaInitial();
      rambtime += timermap.stop();
      timermap.start( "2b RamboFin" );
      prsk->getMomentaFinal();
      rambtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
      // Massive host only (copy)
      if( ramboType != RAMBO_MASSLESS )
      {
        timermap.start( "2c CpHTDmom" );
        copyDeviceFromHost( devMomenta, hstMomenta );
        rambtime += timermap.stop();
      }
#endif

      timermap.start( "2d Aosoa2U " );
#ifdef MGONGPUCPP_GPUIMPL
      gpuLaunchKernel( aosoa_to_umami_kernel, gpublocks, gputhreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
      checkGpu( gpuPeekAtLastError() );
#else
      for( std::size_t ievt = 0; ievt < nevt; ++ievt )
        aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif
      rambtime += timermap.stop();

      double wavetime = 0;
      if( !run_umami( umami_handle, nevt, timermap, wavetime,
#ifdef MGONGPUCPP_GPUIMPL
                      devUmamiMomenta, devFlv, devUmamiMEs, hstUmamiMEs
#else
                      umamiMomenta, flvVec, umamiMEs
#endif
                      ) )
      {
        umami_free( umami_handle );
        return 3;
      }

#ifdef MGONGPUCPP_GPUIMPL
      if( verbose )
      {
        timermap.start( "3c CpDTHmom" );
        copyHostFromDevice( hstMomenta, devMomenta );
        wavetime += timermap.stop();
      }
      const double* mes = hstUmamiMEs.data();
#else
      const double* mes = umamiMEs.data();
#endif

      timermap.start( "4@ UpdtStat" );
      for( unsigned int ievt = 0; ievt < nevt; ++ievt )
      {
        double me = mes[ievt];
        ++nevtALL;
        if( !std::isfinite( me ) )
          ++nevtABN;
        else if( me == 0. )
          ++nevtZERO;
        sumME += me;
        sumMEsq += me * me;
        if( me < minME ) minME = me;
        if( me > maxME ) maxME = me;
      }

      genrtimes[iiter] = genrtime;
      rambtimes[iiter] = rambtime;
      wavetimes[iiter] = wavetime;

      if( verbose )
      {
        std::cout << std::string( SEP79, '*' ) << std::endl
                  << "Iteration #" << iiter + 1 << " of " << niter << std::endl;
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
        {
          std::cout << "Event #" << ievt + 1 << std::endl;
          print_momenta_table( std::cout, hstMomenta.data(), ievt );
          std::cout << " Matrix element = " << std::scientific << std::setprecision( 16 )
                    << mes[ievt] << " GeV^" << kMEGeVExponent << std::endl
                    << std::defaultfloat
                    << std::string( SEP79, '-' ) << std::endl;
        }
      }
    }

    double sumgtim = 0, sumrtim = 0, sumwtim = 0;
    double minwtim = wavetimes[0], maxwtim = wavetimes[0];
    for( unsigned int i = 0; i < niter; ++i )
    {
      sumgtim += genrtimes[i];
      sumrtim += rambtimes[i];
      sumwtim += wavetimes[i];
      minwtim = std::min( minwtim, wavetimes[i] );
      maxwtim = std::max( maxwtim, wavetimes[i] );
    }
    double meanwtim = sumwtim / niter;

    unsigned int nevtGood = nevtALL - nevtABN;
    double meanME = ( nevtGood > 0 ) ? sumME / nevtGood : 0.;
    double varME = ( nevtGood > 0 ) ? sumMEsq / nevtGood - meanME * meanME : 0.;
    double stdME = ( varME > 0 ) ? std::sqrt( varME ) : 0.;

    std::cout << std::string( SEP79, '*' ) << std::endl;
    print_run_header( std::cout );
    std::cout << "NumBlocksPerGrid            = " << gpublocks << std::endl
              << "NumThreadsPerBlock          = " << gputhreads << std::endl
              << "NumIterations               = " << niter << std::endl
              << "FlavorIndex                 = " << flavorID << " / " << CPPProcess::nmaxflavor << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "NaN/abnormal MEs            = " << nevtABN << std::endl
              << "Zero MEs                    = " << nevtZERO << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "NumberOfEntries             = " << niter << std::endl
              << std::scientific
              << "TotalTime[Rnd+Rmb+ME] (123) = ( " << sumgtim + sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[Rambo+ME]    (23) = ( " << sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[RndNumGen]    (1) = ( " << sumgtim << " )  sec" << std::endl
              << "TotalTime[Rambo]        (2) = ( " << sumrtim << " )  sec" << std::endl
              << "TotalTime[MatrixElems]  (3) = ( " << sumwtim << " )  sec" << std::endl
              << "MeanTimeInMatrixElems       = ( " << meanwtim << " )  sec" << std::endl
              << "[Min,Max]TimeInMatrixElems  = [ " << minwtim << " ,  " << maxwtim << " ]  sec" << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "TotalEventsComputed         = " << nevtALL << std::endl
              << "EvtsPerSec[Rnd+Rmb+ME](123) = ( " << nevtALL / ( sumgtim + sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[Rmb+ME]     (23) = ( " << nevtALL / ( sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[MatrixElems] (3) = ( " << nevtALL / sumwtim << " )  sec^-1" << std::endl
              << std::defaultfloat
              << std::string( SEP79, '*' ) << std::endl
              << "MeanMatrixElemValue         = ( " << meanME << " +- " << stdME / std::sqrt( (double)std::max( 1u, nevtGood ) )
              << " )  GeV^" << kMEGeVExponent << std::endl
              << "[Min,Max]MatrixElemValue    = [ " << minME << " ,  " << maxME << " ]  GeV^" << kMEGeVExponent << std::endl
              << std::string( SEP79, '*' ) << std::endl;
    timermap.dump();
    std::cout << std::string( SEP79, '*' ) << std::endl;

    umami_free( umami_handle );
    return 0;
  }
}

int main( int argc, char** argv )
{

  Mode mode = MODE_MATRIX;
  RamboType ramboType = RAMBO_MASSIVE; // default
  bool ramboTypeSet = false;
  bool verbose = false;
  unsigned int flavorID = 0;
  unsigned int gpublocks = 64;
  unsigned int gputhreads = 256;
  unsigned int niter = 1;
  unsigned int numvec[3] = { 0, 0, 0 };
  int nnum = 0;

  // Optional leading subcommand (no leading dash).
  int firstArg = 1;
  if( firstArg < argc )
  {
    std::string a = argv[firstArg];
    if( a == "matrix" ) { mode = MODE_MATRIX; ++firstArg; }
    else if( a == "perf" ) { mode = MODE_PERF; ++firstArg; }
  }

  double energyArg = -1.;

  for( int argn = firstArg; argn < argc; ++argn )
  {
    std::string arg = argv[argn];
    if( arg == "--verbose" || arg == "-v" )
      verbose = true;
    else if( arg == "--performance" || arg == "-p" )
      mode = MODE_PERF; // legacy alias
    else if( ( arg == "--flavor" || arg == "-f" ) && argn + 1 < argc && is_number( argv[argn + 1] ) )
      flavorID = strtoul( argv[++argn], nullptr, 0 );
    else if( arg == "--rambo-massless"  )
    {
      std::string r = argv[++argn];
      ramboType = RAMBO_MASSLESS;
      ramboTypeSet = true;
    }
    else if( is_number( argv[argn] ) && nnum < 3 )
    {
      numvec[nnum++] = strtoul( argv[argn], nullptr, 0 );
      if( energyArg < 0 ) energyArg = atof( argv[argn] );
    }
    else if( is_float( argv[argn] ) && energyArg < 0 )
    {
      // decimal number: only meaningful as the matrix-mode energy
      energyArg = atof( argv[argn] );
    }
    else
      return usage( argv[0] );
  }
//ENERGY CHANGE FOR THE MATRIX MODE
// (default 1000 GeV as for the Fortran/C++ standalone 'check' drivers;
//  can be overridden with a single positional argument)
  if( mode == MODE_MATRIX ) kEnergy = ( energyArg > 0 ) ? energyArg : 1000.;

  if( mode == MODE_MATRIX )
  {
    if( ramboType == RAMBO_MASSLESS )
    {
      std::cerr << "ERROR: matrix mode only supports the classic RAMBO (-r c)." << std::endl;
      return usage( argv[0] );
    }
    if( nnum > 1 )
    {
      std::cerr << "WARNING: extra positional args are ignored in matrix mode "
                << "(dimensions are fixed at " << kMatrixBlocks << " " << kMatrixThreads << " 1)."
                << std::endl;
    }
    return run_matrix_mode( verbose );
  }

  // perf mode
  if( nnum == 3 )
  {
    gpublocks = numvec[0];
    gputhreads = numvec[1];
    niter = numvec[2];
  }
  else if( nnum == 1 )
  {
    niter = numvec[0];
  }
  else if( nnum != 0 )
  {
    return usage( argv[0] );
  }
  if( niter == 0 ) return usage( argv[0] );

  if( flavorID >= CPPProcess::nmaxflavor )
  {
    std::cerr << "ERROR: flavor index " << flavorID
              << " is out of range [0, " << CPPProcess::nmaxflavor << ")." << std::endl;
    return 1;
  }

  return run_perf_mode( verbose, gpublocks, gputhreads, niter, flavorID, ramboType );
}
