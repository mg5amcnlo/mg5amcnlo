// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MadGraph7 CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: O. Mattelaer (Nov 2020) for the MadGraph7 CUDACPP plugin.
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
#ifdef MGONGPUCPP_GPUIMPL
#include "GpuAbstraction.h"
#include "GpuRuntime.h"
#endif
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "rambo.h" // classic RAMBO of the matrix mode
#include "RandomNumberKernels.h"
#include "ProcessData.h"
#include "read_slha.h"
#include "timermap.h"
#include "umami.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define STRINGIFY( s ) #s
#define XSTRINGIFY( s ) STRINGIFY( s )
#define SEP79 79

namespace
{
  using namespace madmatrix;

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

  // One external-particle list per LHE event, each particle stored as (E, px, py, pz).
  using LheEvent = std::array<std::array<double, 4>, CPPProcess::npar>;

  bool read_lhe_events( const std::string& path, std::vector<LheEvent>& events )
  {
    constexpr int npar = CPPProcess::npar;
    std::ifstream in( path );
    if( !in )
    {
      std::cerr << "ERROR! cannot open LHE file '" << path << "'" << std::endl;
      return false;
    }
    std::string line;
    while( std::getline( in, line ) )
    {
      if( line.find( "<event>" ) == std::string::npos ) continue;
      if( !std::getline( in, line ) ) break;
      std::istringstream hdr( line );
      int nptcl = 0;
      hdr >> nptcl;
      if( nptcl != npar )
      {
        std::cerr << "ERROR! LHE event has " << nptcl << " particles, expected " << npar << std::endl;
        return false;
      }
      // particle lines: pdg status mother1 mother2 color1 color2 px py pz E m lifetime spin
      LheEvent ev;
      int ipar = 0;
      while( ipar < npar && std::getline( in, line ) )
      {
        if( line.empty() ) continue;
        std::istringstream ls( line );
        long pdg;
        int status, m1, m2, c1, c2;
        double px, py, pz, E;
        if( !( ls >> pdg >> status >> m1 >> m2 >> c1 >> c2 >> px >> py >> pz >> E ) )
        {
          std::cerr << "ERROR! malformed LHE particle line: " << line << std::endl;
          return false;
        }
        ev[ipar] = { E, px, py, pz };
        ++ipar;
      }
      if( ipar != npar )
      {
        std::cerr << "ERROR! truncated LHE event (got " << ipar << " of " << npar << " particles)" << std::endl;
        return false;
      }
      events.push_back( ev );
    }
    if( events.empty() )
    {
      std::cerr << "ERROR! no events found in '" << path << "'" << std::endl;
      return false;
    }
    return true;
  }

  // Raw binary momenta, as written by --dump-momenta: double precision, (E, px, py, pz)
  // for each leg of each event, with no header. Used by 'check precision' to feed the
  // very same phase-space points to builds of different floating point precision.
  bool read_binary_momenta( const std::string& path, std::vector<LheEvent>& events )
  {
    static_assert( sizeof( LheEvent ) == 4 * CPPProcess::npar * sizeof( double ), "LheEvent must be contiguous doubles" );
    std::ifstream in( path, std::ios::binary );
    if( !in )
    {
      std::cerr << "ERROR! cannot open momenta file '" << path << "'" << std::endl;
      return false;
    }
    LheEvent ev;
    while( in.read( reinterpret_cast<char*>( ev.data() ), sizeof( LheEvent ) ) )
      events.push_back( ev );
    if( events.empty() )
    {
      std::cerr << "ERROR! no events found in '" << path << "'" << std::endl;
      return false;
    }
    return true;
  }

  // Raw binary per-event values (--alphas <file>, --flavors, --rnd-hel, --rnd-col, --hel-index): one
  // value of type T per event, with no header, as array.tofile writes them in Python.
  template<typename T>
  bool read_binary_values( const std::string& path, const char* what, std::vector<T>& values )
  {
    std::ifstream in( path, std::ios::binary );
    if( !in )
    {
      std::cerr << "ERROR! cannot open " << what << " file '" << path << "'" << std::endl;
      return false;
    }
    T value;
    while( in.read( reinterpret_cast<char*>( &value ), sizeof( T ) ) )
      values.push_back( value );
    if( values.empty() || in.gcount() != 0 )
    {
      std::cerr << "ERROR! " << what << " file '" << path << "' is empty or not a whole number of "
                << sizeof( T ) << "-byte values" << std::endl;
      return false;
    }
    return true;
  }

  // Perf-mode options that add inputs or outputs to the umami call, or change how it runs
  struct PerfIO
  {
    std::string alphas;                                     // --alphas: a value, or a file of doubles
    std::string flavorsFile;                                // --flavors: a file of 32-bit unsigned ints
    std::string rndHelFile;                                 // --rnd-hel: a file of doubles
    std::string rndColFile;                                 // --rnd-col: a file of doubles
    std::string helIndexFile;                               // --hel-index: a file of 32-bit ints
    std::string dumpHelFile;                                // --dump-hel: write the helicity indices
    std::string dumpColFile;                                // --dump-col: write the color indices
    std::vector<std::pair<std::string, double>> parameters; // --param <name>=<value>
    bool nonblockingStream = false;                         // --nonblocking-stream (CUDA/HIP)
  };

  int usage( const char* argv0, int ret = 1 )
  {
    std::cout
      << "Usage:\n"
      << "  " << argv0 << " [matrix] [-v|--verbose] [<energy>]\n"
      << "  " << argv0 << " perf [-v|--verbose] [-f|--flavor <int>] [--rambo-massless]"
      << " [-e|--events <file.lhe>] [--momenta <file>] [--dump-momenta <file>] [--dump-me <file>]"
      << " [--energy <GeV>] [--alphas <value|file>] [--flavors <file>] [--rnd-hel <file>]"
      << " [--rnd-col <file>] [--dump-hel <file>] [--dump-col <file>] [--param <name>=<value>]"
      << " [--nonblocking-stream] [<#blocksPerGrid> <#threadsPerBlock>] <#iterations>\n"
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
      << "Options:\n"
      << "  -e|--events <file.lhe>  (perf only) Read the external momenta from an LHE\n"
      << "                    file instead of generating them with RAMBO. The events are\n"
      << "                    processed in batches of #blocks*#threads; #iterations is\n"
      << "                    ignored (derived from the number of events in the file).\n"
      << "  --momenta <file>  (perf only) Same, from a raw binary file of doubles\n"
      << "                    (E,px,py,pz per leg per event) as written by --dump-momenta.\n"
      << "  --dump-momenta <file>  (perf only) Write the momenta of every event, raw doubles.\n"
      << "  --dump-me <file>  (perf only) Write the matrix element of every event, raw doubles.\n"
      << "  --energy <GeV>    (perf only) Ecms for RAMBO (default 1500 GeV).\n"
      << "\n"
      << "Per-event inputs and outputs (perf only). A <file> holds one raw value per event\n"
      << "(no header): doubles, or 32-bit ints for the flavor and the indices. An input file\n"
      << "needs a value for every event of the run, in the order of the momenta.\n"
      << "  --alphas <value|file>  alpha_s of every event (default: the param card).\n"
      << "  --flavors <file>  Flavor index of every event (instead of -f).\n"
      << "  --rnd-hel <file>  Random numbers of the helicity choice (umami otherwise uses 0.5\n"
      << "                    and returns the helicity sum for an interference |M|^2).\n"
      << "  --rnd-col <file>  Random numbers of the color choice (otherwise 0.5).\n"
      << "  --hel-index <file>  Helicity index [0,ncomb) of every event: the |M|^2 of that one\n"
      << "                    helicity instead of the helicity sum (UMAMI_IN_HELICITY_INDEX).\n"
      << "  --dump-hel <file>  Write the chosen helicity index of every event (-1: none).\n"
      << "  --dump-col <file>  Write the chosen color index of every event (-1: none).\n"
      << "  --param <name>=<value>  umami_set_parameter after umami_initialize (repeatable).\n"
      << "  --nonblocking-stream  (CUDA/HIP) Give umami a non-blocking stream, which does not\n"
      << "                    synchronise with the default stream.\n"
      << "\n"
      << "Both modes read the parameters, alpha_s included, from ../../Cards/param_card.dat.\n"
      << "perf times as MatrixElems (3) only the umami_matrix_element call (UMAMI input/output\n"
      << "handling included); device-to-host copies are listed separately in the timer map.\n"
      << "\n"
      << "perf-mode defaults if positional args are omitted:\n"
      << "  #blocksPerGrid = 64, #threadsPerBlock = 256, #iterations = 1.\n";
    return ret;
  }

  // AOSOA -> UMAMI SoA single-event helper. Layout reminder:
  //   AOSOA: aosoa[i_page * npar*4*neppM + ipar*4*neppM + ip4*neppM + i_vector]
  //   UMAMI: soa[ip4 * npar*nevt + ipar*nevt + ievt]
  __host__ __device__ inline void
  aosoa_to_umami_one( const fptype_momenta* aosoa,
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
  aosoa_to_umami_kernel( const fptype_momenta* aosoa,
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

  void print_momenta_table( std::ostream& os, const fptype_momenta* aosoa, unsigned int ievt )
  {
    auto constexpr prec = std::numeric_limits<fptype_momenta>::digits10;
    constexpr int npar = CPPProcess::npar;
    os << std::string( SEP79, '-' ) << std::endl
       << "   n    E           	 	 px             	  py              	   pz" << std::endl;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      double E  = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 0, ipar );
      double px = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 1, ipar );
      double py = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 2, ipar );
      double pz = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 3, ipar );
      os << std::scientific << std::setprecision( prec )
         << std::setw( 4 ) << ipar + 1
         << std::setw( prec + 10 ) << E
         << std::setw( prec + 10 ) << px
         << std::setw( prec + 10 ) << py
         << std::setw( prec + 10 ) << pz
         << std::endl
         << std::defaultfloat;
    }
    os << std::string( SEP79, '-' ) << std::endl;
  }

  // alpha_s of the param card, so the couplings match the Fortran/C++ 'check'
  // drivers (UMAMI otherwise falls back to a hardcoded g_s, alpha_s = 0.118).
  double param_card_alpha_s()
  {
    SLHAReader slha( "../../Cards/param_card.dat", false );
    return slha.get_block_entry( "sminputs", 3, 1.180000e-01 );
  }

  // Run sigmaKin via UMAMI for `nevt` events. The input and output buffers (the
  // momenta in the UMAMI SoA layout, the per-event flavor and alpha_s, ...) are set
  // by the caller, who also copies the outputs back: on GPU they are device pointers.
  bool run_umami(
    UmamiHandle handle,
    unsigned int nevt,
    mgOnGpu::TimerMap& timermap,
    double& wavetime,
    const std::vector<UmamiInputKey>& in_keys,
    const std::vector<const void*>& inputs,
    const std::vector<UmamiOutputKey>& out_keys,
    const std::vector<void*>& outputs )
  {
    timermap.start( "3a UmamiCall" );
    UmamiStatus st = umami_matrix_element(
      handle, nevt, nevt, 0, in_keys.size(), in_keys.data(), inputs.data(), out_keys.size(), out_keys.data(), outputs.data() );
#ifdef MGONGPUCPP_GPUIMPL
    // UMAMI only enqueues the kernels; wait for them so the ME time is the computation, not the launch.
    checkGpu( gpuDeviceSynchronize() );
#endif
    wavetime += timermap.stop();
    if( st != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << std::endl;
      return false;
    }
    return true;
  }

  // --------------------------------------------------------------------------
  // Classic MadGraph standalone RAMBO (rambo.h, RANMAR generator seeded with the
  // fixed values 1802/9373), reproducing the exact phase-space point used by the
  // Fortran and C++ standalone 'check' drivers at the same energy, so that
  // matrix-mode output can be compared line by line across backends. Its own
  // generator: the shared one (rambo::rn) feeds RamboSamplingKernelHost.
  // Host-side only.
  // --------------------------------------------------------------------------
  std::vector<std::vector<double>>
  classic_rambo_momenta( int ninitial, double energy, const std::vector<double>& masses, double& wgt )
  {
    static rambo::Random random;
    static bool seeded = false;
    if( !seeded )
    {
      random.rmarin( 1802, 9373 );
      seeded = true;
    }
    auto draw = []( int )
    {
      double r = random.ranmar();
      while( r <= 1e-16 ) r = random.ranmar();
      return r;
    };
    return rambo::get_momenta( ninitial, energy, masses, wgt, draw );
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

#ifdef MGONGPUCPP_GPUIMPL
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
    const std::vector<fptype> masses( massesD.begin(), massesD.end() ); // RamboSamplingKernelHost takes vector<fptype>

    // NB: feed the double-precision masses to the classic RAMBO, which works in
    // double throughout: 'masses' is fptype and would not convert at FPTYPE=f.
    std::vector<std::vector<double>> point =
      classic_rambo_momenta( CPPProcess::npari, (double)kEnergy, massesD, rambowgt );

    std::vector<double> alphasVec( nevt, param_card_alpha_s() );
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
                     RamboType ramboType,
                     const std::string& lheFile = "",
                     const std::string& momentaFile = "",
                     const std::string& dumpMomentaFile = "",
                     const std::string& dumpMEFile = "",
                     const PerfIO& io = PerfIO() )
  {
    const unsigned int nevt = gpublocks * gputhreads;

    // LHE (or raw binary momenta) instead of generating. Processed in batches of
    // nevt and niter is derived from the number of events read.
    std::vector<LheEvent> lheEvents;
    if( !lheFile.empty() )
    {
      if( !read_lhe_events( lheFile, lheEvents ) ) return 2;
      niter = (unsigned int)( ( lheEvents.size() + nevt - 1 ) / nevt );
      std::cout << "Reading events from LHE file = " << lheFile
                << " (" << lheEvents.size() << " events, " << niter
                << " batches of " << nevt << ")" << std::endl;
    }
    else if( !momentaFile.empty() )
    {
      if( !read_binary_momenta( momentaFile, lheEvents ) ) return 2;
      niter = (unsigned int)( ( lheEvents.size() + nevt - 1 ) / nevt );
      std::cout << "Reading events from momenta file = " << momentaFile
                << " (" << lheEvents.size() << " events, " << niter
                << " batches of " << nevt << ")" << std::endl;
    }
    const bool eventsFromFile = !lheEvents.empty();

    std::ofstream dumpMomenta, dumpMEs, dumpHel, dumpCol;
    for( auto [file, path, what] : { std::make_tuple( &dumpMomenta, &dumpMomentaFile, "momenta" ),
                                     std::make_tuple( &dumpMEs, &dumpMEFile, "matrix element" ),
                                     std::make_tuple( &dumpHel, &io.dumpHelFile, "helicity" ),
                                     std::make_tuple( &dumpCol, &io.dumpColFile, "color" ) } )
    {
      if( path->empty() ) continue;
      file->open( *path, std::ios::binary );
      if( !*file )
      {
        std::cerr << "ERROR! cannot write " << what << " file '" << *path << "'" << std::endl;
        return 2;
      }
    }

    // Per-event inputs from files: one value for each event of the run, in the order
    // of the momenta (the padding events of the last batch repeat its last real event)
    double alphasValue = param_card_alpha_s();
    std::vector<double> alphasIn, rndHelIn, rndColIn;
    std::vector<unsigned int> flavorsIn;
    std::vector<int> helIndexIn;
    if( is_float( io.alphas.c_str() ) )
      alphasValue = atof( io.alphas.c_str() );
    else if( !io.alphas.empty() && !read_binary_values( io.alphas, "alpha_s", alphasIn ) )
      return 2;
    if( !io.flavorsFile.empty() && !read_binary_values( io.flavorsFile, "flavor", flavorsIn ) ) return 2;
    if( !io.rndHelFile.empty() && !read_binary_values( io.rndHelFile, "helicity random number", rndHelIn ) ) return 2;
    if( !io.rndColFile.empty() && !read_binary_values( io.rndColFile, "color random number", rndColIn ) ) return 2;
    if( !io.helIndexFile.empty() && !read_binary_values( io.helIndexFile, "helicity index", helIndexIn ) ) return 2;
    const std::size_t nevtRun = eventsFromFile ? lheEvents.size() : (std::size_t)niter * nevt;
    for( auto [size, what] : { std::make_pair( alphasIn.size(), "alpha_s" ),
                               std::make_pair( flavorsIn.size(), "flavor" ),
                               std::make_pair( rndHelIn.size(), "helicity random number" ),
                               std::make_pair( rndColIn.size(), "color random number" ),
                               std::make_pair( helIndexIn.size(), "helicity index" ) } )
    {
      if( size != 0 && size < nevtRun )
      {
        std::cerr << "ERROR! the " << what << " file has " << size << " values for " << nevtRun << " events" << std::endl;
        return 2;
      }
    }
    for( unsigned int iflav : flavorsIn )
    {
      if( iflav >= CPPProcess::nmaxflavor )
      {
        std::cerr << "ERROR! flavor index " << iflav << " in the flavor file is out of range [0, "
                  << CPPProcess::nmaxflavor << ")" << std::endl;
        return 2;
      }
    }

    mgOnGpu::TimerMap timermap;

    // The per-event inputs and outputs of one batch (host side)
    std::vector<unsigned int> flvVec( nevt, flavorID );
    std::vector<double> alphasVec( nevt, alphasValue );
    std::vector<double> rndHelVec( nevt, 0.5 ), rndColVec( nevt, 0.5 );
    std::vector<int> helVec( nevt ), colVec( nevt );
    std::vector<int> helInVec( nevt, 0 );
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
    DeviceBufferBase<double> devAlphaS( nevt );
    DeviceBufferBase<double> devRndHel( nevt ), devRndCol( nevt );
    DeviceBufferBase<int> devHel( nevt ), devCol( nevt );
    DeviceBufferBase<int> devHelIn( nevt );
    std::vector<double> hstUmamiMEs( nevt );
    // the inputs that are the same for every event are copied to the device once
    gpuMemcpy( devFlv.data(), flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
    gpuMemcpy( devAlphaS.data(), alphasVec.data(), nevt * sizeof( double ), gpuMemcpyHostToDevice );
    void* umamiMomentaPtr = devUmamiMomenta.data();
    void* umamiMEsPtr = devUmamiMEs.data();
    unsigned int* flvPtr = devFlv.data();
    double* alphasPtr = devAlphaS.data();
    double* rndHelPtr = devRndHel.data();
    double* rndColPtr = devRndCol.data();
    int* helPtr = devHel.data();
    int* colPtr = devCol.data();
    int* helInPtr = devHelIn.data();
    gpuStream_t stream = nullptr; // nullptr: the default stream
    if( io.nonblockingStream ) gpuStreamCreateNonBlocking( &stream );
#else
    HostBufferRndNumMomenta hstRndmom( nevt );
    HostBufferMomenta hstMomenta( nevt );
    HostBufferWeights hstWeights( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<double> umamiMEs( nevt );
    void* umamiMomentaPtr = umamiMomenta.data();
    void* umamiMEsPtr = umamiMEs.data();
    unsigned int* flvPtr = flvVec.data();
    double* alphasPtr = alphasVec.data();
    double* rndHelPtr = rndHelVec.data();
    double* rndColPtr = rndColVec.data();
    int* helPtr = helVec.data();
    int* colPtr = colVec.data();
    int* helInPtr = helInVec.data();
#endif

    // The keys and buffers of the umami call, the same for every batch
    std::vector<UmamiInputKey> in_keys = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX, UMAMI_IN_ALPHA_S };
    std::vector<const void*> inputs = { umamiMomentaPtr, flvPtr, alphasPtr };
    std::vector<UmamiOutputKey> out_keys = { UMAMI_OUT_MATRIX_ELEMENT };
    std::vector<void*> outputs = { umamiMEsPtr };
    if( !rndHelIn.empty() )
    {
      in_keys.push_back( UMAMI_IN_RANDOM_HELICITY );
      inputs.push_back( rndHelPtr );
    }
    if( !rndColIn.empty() )
    {
      in_keys.push_back( UMAMI_IN_RANDOM_COLOR );
      inputs.push_back( rndColPtr );
    }
    if( !helIndexIn.empty() )
    {
      in_keys.push_back( UMAMI_IN_HELICITY_INDEX );
      inputs.push_back( helInPtr );
    }
    if( dumpHel.is_open() )
    {
      out_keys.push_back( UMAMI_OUT_HELICITY_INDEX );
      outputs.push_back( helPtr );
    }
    if( dumpCol.is_open() )
    {
      out_keys.push_back( UMAMI_OUT_COLOR_INDEX );
      outputs.push_back( colPtr );
    }
#ifdef MGONGPUCPP_GPUIMPL
    if( stream != nullptr )
    {
      out_keys.push_back( UMAMI_OUT_GPU_STREAM );
      outputs.push_back( stream );
    }
#endif

    std::unique_ptr<RandomNumberKernelBase> prnk(
      new CommonRandomNumberKernel( hstRndmom ) );

    UmamiHandle umami_handle = nullptr;
    if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_initialize failed" << std::endl;
      return 2;
    }
    for( const auto& [name, value] : io.parameters )
    {
      if( umami_set_parameter( umami_handle, name.c_str(), value, 0. ) != UMAMI_SUCCESS )
      {
        std::cerr << "ERROR! umami_set_parameter( " << name << ", " << value << " ) failed" << std::endl;
        umami_free( umami_handle );
        return 2;
      }
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
    double minME = std::numeric_limits<double>::max();
    double maxME = std::numeric_limits<double>::lowest();
    unsigned int nevtALL = 0;

    for( unsigned int iiter = 0; iiter < niter; ++iiter )
    {
      double genrtime = 0;
      double rambtime = 0;
      unsigned int nreal = nevt; // number of real (non-padding) events in this batch
      if( !eventsFromFile )
      {
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
          timermap.stop();
        }
#endif

        timermap.start( "2a RamboIni" );
        prsk->getMomentaInitial();
        rambtime += timermap.stop();
        timermap.start( "2b RamboFin" );
        prsk->getMomentaFinal();
#ifdef MGONGPUCPP_GPUIMPL
        // Massless RAMBO runs on the device
        checkGpu( gpuDeviceSynchronize() );
#endif
        rambtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
        // Massive host only (copy)
        if( ramboType != RAMBO_MASSLESS )
        {
          timermap.start( "2c CpHTDmom" );
          copyDeviceFromHost( devMomenta, hstMomenta );
          timermap.stop();
        }
#endif
      }
      else
      {
        // Fill this batch from the LHE events (AOSOA layout, (E,px,py,pz) per leg).
        // padded by repeating its last real event so the SIMD page is valid
        // only the nreal real events are counted below.
        timermap.start( "2e ReadLHE " );
        const std::size_t base = (std::size_t)iiter * nevt;
        nreal = (unsigned int)std::min<std::size_t>( nevt, lheEvents.size() - base );
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
        {
          const std::size_t src = base + std::min<std::size_t>( ievt, (std::size_t)nreal - 1 );
          const LheEvent& ev = lheEvents[src];
          for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
            for( int ip4 = 0; ip4 < 4; ++ip4 )
              MemoryAccessMomenta::ieventAccessIp4Ipar( hstMomenta.data(), ievt, ip4, ipar ) = (fptype_momenta)ev[ipar][ip4];
        }
        rambtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
        timermap.start( "2c CpHTDmom" );
        copyDeviceFromHost( devMomenta, hstMomenta );
        timermap.stop();
#endif
      }

      timermap.start( "2d Aosoa2U " );
#ifdef MGONGPUCPP_GPUIMPL
      gpuLaunchKernel( aosoa_to_umami_kernel, gpublocks, gputhreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
      checkGpu( gpuPeekAtLastError() );
      checkGpu( gpuDeviceSynchronize() );
#else
      for( std::size_t ievt = 0; ievt < nevt; ++ievt )
        aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif
      rambtime += timermap.stop();

      // The per-event inputs of this batch read from files
      const std::size_t base = (std::size_t)iiter * nevt;
      for( auto [batch, values] : { std::make_pair( &alphasVec, &alphasIn ),
                                    std::make_pair( &rndHelVec, &rndHelIn ),
                                    std::make_pair( &rndColVec, &rndColIn ) } )
      {
        if( values->empty() ) continue;
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
          ( *batch )[ievt] = ( *values )[base + std::min( ievt, nreal - 1 )];
      }
      if( !flavorsIn.empty() )
      {
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
          flvVec[ievt] = flavorsIn[base + std::min( ievt, nreal - 1 )];
      }
      if( !helIndexIn.empty() )
      {
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
          helInVec[ievt] = helIndexIn[base + std::min( ievt, nreal - 1 )];
      }
#ifdef MGONGPUCPP_GPUIMPL
      if( !alphasIn.empty() ) gpuMemcpy( alphasPtr, alphasVec.data(), nevt * sizeof( double ), gpuMemcpyHostToDevice );
      if( !rndHelIn.empty() ) gpuMemcpy( rndHelPtr, rndHelVec.data(), nevt * sizeof( double ), gpuMemcpyHostToDevice );
      if( !rndColIn.empty() ) gpuMemcpy( rndColPtr, rndColVec.data(), nevt * sizeof( double ), gpuMemcpyHostToDevice );
      if( !flavorsIn.empty() ) gpuMemcpy( flvPtr, flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
      if( !helIndexIn.empty() ) gpuMemcpy( helInPtr, helInVec.data(), nevt * sizeof( int ), gpuMemcpyHostToDevice );
      // A non-blocking stream does not wait for the inputs copied and transposed on the
      // default stream: finish them first, as any caller on such a stream must
      if( stream != nullptr ) checkGpu( gpuDeviceSynchronize() );
#endif

      double wavetime = 0;
      if( !run_umami( umami_handle, nevt, timermap, wavetime, in_keys, inputs, out_keys, outputs ) )
      {
        umami_free( umami_handle );
        return 3;
      }

#ifdef MGONGPUCPP_GPUIMPL
      timermap.start( "3b CpDTHmes" );
      // nor do the copies below wait for the work on a non-blocking stream
      if( stream != nullptr ) gpuStreamSynchronize( stream );
      gpuMemcpy( hstUmamiMEs.data(), umamiMEsPtr, nevt * sizeof( double ), gpuMemcpyDeviceToHost );
      if( dumpHel.is_open() ) gpuMemcpy( helVec.data(), helPtr, nevt * sizeof( int ), gpuMemcpyDeviceToHost );
      if( dumpCol.is_open() ) gpuMemcpy( colVec.data(), colPtr, nevt * sizeof( int ), gpuMemcpyDeviceToHost );
      // Device-to-host copies are reported by the timer map, not as ME time.
      timermap.stop();
      if( verbose || dumpMomenta.is_open() )
      {
        timermap.start( "3c CpDTHmom" );
        copyHostFromDevice( hstMomenta, devMomenta );
        timermap.stop();
      }
      const double* mes = hstUmamiMEs.data();
#else
      const double* mes = umamiMEs.data();
#endif

      if( dumpMomenta.is_open() )
      {
        for( unsigned int ievt = 0; ievt < nreal; ++ievt )
          for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
            for( int ip4 = 0; ip4 < 4; ++ip4 )
            {
              const double p = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( hstMomenta.data(), ievt, ip4, ipar );
              dumpMomenta.write( reinterpret_cast<const char*>( &p ), sizeof( double ) );
            }
      }
      if( dumpMEs.is_open() )
        dumpMEs.write( reinterpret_cast<const char*>( mes ), (std::streamsize)nreal * sizeof( double ) );
      if( dumpHel.is_open() )
        dumpHel.write( reinterpret_cast<const char*>( helVec.data() ), (std::streamsize)nreal * sizeof( int ) );
      if( dumpCol.is_open() )
        dumpCol.write( reinterpret_cast<const char*>( colVec.data() ), (std::streamsize)nreal * sizeof( int ) );

      timermap.start( "4@ UpdtStat" );
      for( unsigned int ievt = 0; ievt < nreal; ++ievt )
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
        for( unsigned int ievt = 0; ievt < nreal; ++ievt )
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
              << "FlavorIndex                 = " << ( flavorsIn.empty() ? std::to_string( flavorID ) : "per event" )
              << " / " << CPPProcess::nmaxflavor << std::endl
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
#ifdef MGONGPUCPP_GPUIMPL
    if( stream != nullptr ) gpuStreamDestroy( stream );
#endif
    for( std::ofstream* file : { &dumpMomenta, &dumpMEs, &dumpHel, &dumpCol } )
    {
      if( file->is_open() && !file->flush() )
      {
        std::cerr << "ERROR! failed to write the dump files" << std::endl;
        return 2;
      }
    }
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
  std::string lheFile; // -e/--events: read momenta from this LHE file (perf mode only)
  std::string momentaFile;     // --momenta: read momenta from this raw binary file (perf mode only)
  std::string dumpMomentaFile; // --dump-momenta: write the momenta of every event (perf mode only)
  std::string dumpMEFile;      // --dump-me: write the matrix element of every event (perf mode only)
  double perfEnergy = -1.;     // --energy: Ecms for the perf-mode RAMBO
  PerfIO io;                   // the per-event inputs and outputs of perf mode (--alphas, ...)

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
    else if( arg == "--rambo-massless" )
    {
      ramboType = RAMBO_MASSLESS;
      ramboTypeSet = true;
    }
    else if( ( arg == "--events" || arg == "-e" ) && argn + 1 < argc )
    {
      lheFile = argv[++argn];
      mode = MODE_PERF; // reading events from file only makes sense in perf mode
    }
    else if( arg == "--momenta" && argn + 1 < argc )
    {
      momentaFile = argv[++argn];
      mode = MODE_PERF;
    }
    else if( arg == "--dump-momenta" && argn + 1 < argc )
    {
      dumpMomentaFile = argv[++argn];
      mode = MODE_PERF;
    }
    else if( arg == "--dump-me" && argn + 1 < argc )
    {
      dumpMEFile = argv[++argn];
      mode = MODE_PERF;
    }
    else if( arg == "--energy" && argn + 1 < argc && is_float( argv[argn + 1] ) )
      perfEnergy = atof( argv[++argn] );
    else if( ( arg == "--alphas" || arg == "--flavors" || arg == "--rnd-hel" || arg == "--rnd-col" ||
               arg == "--hel-index" || arg == "--dump-hel" || arg == "--dump-col" ) && argn + 1 < argc )
    {
      std::string& value = ( arg == "--alphas" ? io.alphas : arg == "--flavors" ? io.flavorsFile :
                             arg == "--rnd-hel" ? io.rndHelFile : arg == "--rnd-col" ? io.rndColFile :
                             arg == "--hel-index" ? io.helIndexFile :
                             arg == "--dump-hel" ? io.dumpHelFile : io.dumpColFile );
      value = argv[++argn];
      mode = MODE_PERF;
    }
    else if( arg == "--param" && argn + 1 < argc && strchr( argv[argn + 1], '=' ) != nullptr )
    {
      const std::string param = argv[++argn];
      const std::size_t eq = param.find( '=' );
      io.parameters.emplace_back( param.substr( 0, eq ), atof( param.c_str() + eq + 1 ) );
      mode = MODE_PERF;
    }
    else if( arg == "--nonblocking-stream" )
    {
      io.nonblockingStream = true;
      mode = MODE_PERF;
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
  if( niter == 0 && lheFile.empty() && momentaFile.empty() ) return usage( argv[0] ); // niter is derived from the file in LHE mode
  if( !lheFile.empty() && !momentaFile.empty() )
  {
    std::cerr << "ERROR: --events and --momenta are mutually exclusive." << std::endl;
    return usage( argv[0] );
  }
  if( perfEnergy > 0 ) kEnergy = perfEnergy;
#ifndef MGONGPUCPP_GPUIMPL
  if( io.nonblockingStream )
  {
    std::cerr << "ERROR: --nonblocking-stream needs a CUDA or HIP build." << std::endl;
    return 1;
  }
#endif

  if( flavorID >= CPPProcess::nmaxflavor )
  {
    std::cerr << "ERROR: flavor index " << flavorID
              << " is out of range [0, " << CPPProcess::nmaxflavor << ")." << std::endl;
    return 1;
  }

  return run_perf_mode( verbose, gpublocks, gputhreads, niter, flavorID, ramboType, lheFile,
                        momentaFile, dumpMomentaFile, dumpMEFile, io );
}
