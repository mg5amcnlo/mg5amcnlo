// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: T. Heimel (Nov 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro (2026).
// Integrated with the MadGraph7 project in Feb 2026.

#include "umami.h"

#include "ProcessData.h"
#include "CPPProcess.h" // needed to construct/initProc the process object (umami_initialize)
#include "SigmaKin.h" // sigmaKin_getGoodHel/setGoodHel
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "rambo.h" // RAMBO points to choose the good helicities on (umami_initialize)

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <utility>

using namespace madmatrix;

namespace
{

  // The per-diagram channel weight handed to madspace as amp2. A subprocess whose
  // matrix element is identically zero -- an FCNC channel with every Wilson
  // coefficient at zero, say -- has all numerators at zero, so their sum, the
  // denominator, is zero too. Dividing would give 0/0 = nan for every diagram, and
  // that nan becomes the amp2 madspace builds its channel weights from, turning a
  // channel that should simply contribute nothing into a non-finite event weight
  // that aborts the whole integration.
  //
  // The denominator is a sum of |amp|^2 and so never negative: adding the smallest
  // normal double floors it away from zero (0/tiny is 0, no channel preferred) while
  // leaving every denominator a real amplitude produces bit-for-bit unchanged. This
  // must NOT be written as a test for zero -- this is compiled with -ffast-math, and
  // its -ffinite-math-only lets the compiler assume the quotient is finite and drop
  // such a guard as dead code (the trap behind #117 and #516).
  //
  // This is called from both the host and the device code paths, hence the
  // __host__ __device__ decoration in GPU builds. DBL_MIN is used rather than
  // std::numeric_limits<double>::min(), which is a host function and cannot be
  // called from device code.
#ifdef MGONGPUCPP_GPUIMPL
  __host__ __device__
#endif
    inline double
    channel_amp2( double numerator, double denominator )
  {
    return numerator / ( denominator + DBL_MIN );
  }


  void* initialize_impl(
    const fptype_momenta* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
    fptype_amp* color_jamps,
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    bool is_good_hel[ProcessData::ncomb];
    sigmaKin_getGoodHel(
      momenta, couplings, flavor_indices, matrix_elements, numerators, denominators,
      color_jamps,
      is_good_hel,
      count );
    sigmaKin_setGoodHel( is_good_hel );
    return nullptr;
  }

  void initialize(
    const fptype_momenta* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
    fptype_amp* color_jamps,
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    // static local initialization is called exactly once in a thread-safe way
    static void* dummy = initialize_impl( momenta, couplings, flavor_indices, matrix_elements,
                                          color_jamps,
                                          numerators,
                                          denominators,
                                          count );
  }

  __device__
    void
    transpose_momenta( const double* momenta_in, fptype_momenta* momenta_out, std::size_t i_event_in, std::size_t i_event_out, std::size_t stride )
  {
    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    std::size_t i_page = i_event_out / page_size;
    std::size_t i_vector = i_event_out % page_size;

    for( std::size_t i_part = 0; i_part < ProcessData::npar; ++i_part )
    {
      for( std::size_t i_mom = 0; i_mom < 4; ++i_mom )
      {
        momenta_out[i_page * ProcessData::npar * 4 * page_size +
                    i_part * 4 * page_size + i_mom * page_size + i_vector] = momenta_in[stride * ( ProcessData::npar * i_mom + i_part ) + i_event_in];
      }
    }
  }


  __global__ void copy_inputs(
    const double* momenta_in,
    const double* helicity_random_in,
    const double* color_random_in,
    const double* diagram_random_in,
    const double* alpha_s_in,
    const unsigned int* flavor_indices_in,
    const int* helicity_in,
    fptype_momenta* momenta,
    fptype* helicity_random,
    fptype* color_random,
    fptype* diagram_random,
    fptype* g_s,
    unsigned int* flavor_indices,
    int* helicity_given,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    // One thread per event of the rounded (padded) batch: sigmaKin runs on all
    // n_blocks * n_threads events, not only on the first count. The padding events
    // are copies of the first event. Left uninitialized, their flavor index is
    // whatever the fresh allocation holds, and cFlavors[iflavor] in the matrix
    // element then reads far out of bounds (illegal memory access on AMD GPUs).
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t i_in = i_event < count ? i_event : 0;

    transpose_momenta( &momenta_in[offset], momenta, i_in, i_event, stride );
    diagram_random[i_event] = diagram_random_in ? diagram_random_in[i_in + offset] : 0.5;
    helicity_random[i_event] = helicity_random_in ? helicity_random_in[i_in + offset] : 0.5;
    color_random[i_event] = color_random_in ? color_random_in[i_in + offset] : 0.5;
    g_s[i_event] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_in + offset] ) : 1.2177157847767195;
    flavor_indices[i_event] = flavor_indices_in ? flavor_indices_in[i_in + offset] : 0;
    if( helicity_in )
    {
      // an index out of [0,ncomb) cannot be reported from here (umami_matrix_element returns
      // before the kernels run): evaluate helicity 0 instead, never out of the helicity table,
      // and let copy_outputs flag the event (see umami_matrix_element)
      const int ihel = helicity_in[i_in + offset];
      helicity_given[i_event] = ( ihel >= 0 && ihel < ProcessData::ncomb ? ihel : 0 );
    }
  }

  __global__ void copy_outputs(
    fptype_amp* denominators,
    fptype_amp* numerators,
    fptype* matrix_elements,
    unsigned int* diagram_index,
    int* color_index,
    int* helicity_index,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out,
    const int* helicity_in,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_event >= count ) return;
    if( helicity_in )
    {
      const int ihel = helicity_in[i_event + offset];
      if( ihel < 0 || ihel >= ProcessData::ncomb )
      {
        // an invalid given helicity: no |M|^2 and no helicity, see umami_matrix_element
        if( m2_out ) m2_out[i_event + offset] = 0;
        if( helicity_out ) helicity_out[i_event + offset] = -1;
        return;
      }
    }

    if( m2_out ) m2_out[i_event + offset] = matrix_elements[i_event];
    if( amp2_out )
    {
      double denominator = denominators[i_event];
      for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
      {
        amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_event * ProcessData::ndiagrams + i_diag], denominator );
      }
    }
    if( diagram_out ) diagram_out[i_event + offset] = diagram_index[i_event] - 1;
    if( color_out ) color_out[i_event + offset] = color_index[i_event] - 1;
    if( helicity_out ) helicity_out[i_event + offset] = helicity_index[i_event] - 1;
  }


  struct InterfaceInstance
  {
    bool initialized = false;
    // parameter interference_helicity_summed (run card [generation]
    // interference_helicity = "summed"): an interference |M|^2 stays the helicity
    // sum even when the caller draws the helicity (see umami_matrix_element)
    bool summed_helicity = false;
  };

  std::vector<double> g_externalMasses;

  // The helicity filter (sigmaKin_getGoodHel) runs once per library and keeps a
  // helicity if its |M|^2 is not 0 for one of the first events. Run at the first
  // umami_matrix_element call, it depended on what that call held: a first batch of
  // special points, where some helicities happen to vanish (the forward direction of
  // u u~ > e+ e-, say), dropped them for the rest of the run, and every later |M|^2
  // was wrong. It runs instead here, at initialization, on RAMBO points: the matrix
  // element is evaluated there once (the filter itself scans every flavor). Where
  // RAMBO has no phase space (2 > 1, a decay below threshold), this does nothing and
  // the first call still chooses. As in the cpu and simd backends, but the inputs and
  // outputs of umami_matrix_element are device buffers here: the points are made on
  // the host and copied over, and everything runs on the default stream.
  // An incoming quark, gluon or photon comes out of a PDF (or of the photon flux of a
  // lepton), which sets the partonic centre-of-mass frame in motion along the beams.
  constexpr bool has_pdf_beams()
  {
    if( ProcessData::npari != 2 ) return false;
    for( int iflav = 0; iflav < ProcessData::nmaxflavor; ++iflav )
      for( int ipar = 0; ipar < ProcessData::npari; ++ipar )
      {
        const int pdg = ProcessData::flavorPDGs[iflav][ipar];
        const int apdg = pdg < 0 ? -pdg : pdg;
        if( ( apdg >= 1 && apdg <= 6 ) || apdg == 21 || apdg == 22 ) return true;
      }
    return false;
  }

  void choose_good_helicities( UmamiHandle handle )
  {
    constexpr int npar = ProcessData::npar;
    constexpr int npari = ProcessData::npari;
    constexpr std::size_t nevt = 16; // the events sigmaKin_getGoodHel looks at
    std::vector<double> masses( g_externalMasses.begin(), g_externalMasses.end() );
    double mfinal = 0, mtotal = 0;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      mtotal += masses[ipar];
      if( ipar >= npari ) mfinal += masses[ipar];
    }
    // rambo exits on a closed phase space: check it first
    if( npar - npari < 2 || ( npari == 1 && !( masses[0] > mfinal ) ) ) return;
    const double energy = std::max( 1500., 2 * mtotal ); // a decay is at rest instead
    // A private generator: the shared one (rambo::rn) makes the points of the check
    // drivers, which must not move.
    rambo::Random random;
    random.rmarin( 1802, 9373 );
    auto draw = [&random]( int )
    {
      double r = random.ranmar();
      while( r <= 1e-16 ) r = random.ranmar();
      return r;
    };
    std::vector<double> momenta( nevt * npar * 4 ); // momenta[nevt * ( npar * mu + ipar ) + ievt]
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
    {
      double weight = 0;
      std::vector<std::vector<double>> p = rambo::get_momenta( npari, energy, masses, weight, draw );
      // RAMBO points are in the partonic centre-of-mass frame, but with PDFs the matrix
      // element is called in frames boosted along the beams (madspace's lab frame), and
      // the helicity of a massive leg is not boost invariant: a helicity can vanish in
      // the centre-of-mass frame only. Every other point is therefore boosted along z,
      // with a random rapidity in [-2, 2], so that the choice covers both frames.
      if( has_pdf_beams() && ievt % 2 == 1 )
      {
        const double y = 4 * draw( 0 ) - 2, ch = std::cosh( y ), sh = std::sinh( y );
        for( std::vector<double>& q : p )
        {
          const double e = q[0], pz = q[3];
          q[0] = e * ch + pz * sh;
          q[3] = pz * ch + e * sh;
        }
      }
      for( int ipar = 0; ipar < npar; ++ipar )
      {
        // on shell, or rambo did not converge: leave the choice to the first call
        const double m2 = p[ipar][0] * p[ipar][0] - p[ipar][1] * p[ipar][1] - p[ipar][2] * p[ipar][2] - p[ipar][3] * p[ipar][3];
        if( std::abs( m2 - masses[ipar] * masses[ipar] ) > 1e-6 * energy * energy ) return;
        for( int mu = 0; mu < 4; ++mu ) momenta[nevt * ( npar * mu + ipar ) + ievt] = p[ipar][mu];
      }
    }
    double* device_momenta = nullptr;
    double* device_matrix_elements = nullptr;
    gpuMalloc( (void**)&device_momenta, momenta.size() * sizeof( double ) );
    gpuMalloc( (void**)&device_matrix_elements, nevt * sizeof( double ) );
    gpuMemcpy( device_momenta, momenta.data(), momenta.size() * sizeof( double ), gpuMemcpyHostToDevice );
    // no UMAMI_OUT_GPU_STREAM: the default stream, as the filter's own copies
    const UmamiInputKey input_keys[] = { UMAMI_IN_MOMENTA };
    const void* inputs[] = { device_momenta };
    const UmamiOutputKey output_keys[] = { UMAMI_OUT_MATRIX_ELEMENT };
    void* outputs[] = { device_matrix_elements };
    umami_matrix_element( handle, nevt, nevt, 0, 1, input_keys, inputs, 1, output_keys, outputs );
    // umami_matrix_element returns before its kernels end: wait before freeing
    gpuStreamSynchronize( nullptr );
    gpuFree( device_matrix_elements );
    gpuFree( device_momenta );
  }

}

extern "C"
{
  UmamiStatus umami_get_meta( UmamiMetaKey meta_key, void* result )
  {
    switch( meta_key )
    {
      case UMAMI_META_DEVICE:
      {
        UmamiDevice& device = *static_cast<UmamiDevice*>( result );
#ifdef __CUDACC__
        device = UMAMI_DEVICE_CUDA;
#elif defined( __HIPCC__ )
        device = UMAMI_DEVICE_HIP;
#endif
        break;
      }
      case UMAMI_META_PARTICLE_COUNT:
        *static_cast<int*>( result ) = ProcessData::npar;
        break;
      case UMAMI_META_DIAGRAM_COUNT:
        *static_cast<int*>( result ) = ProcessData::ndiagrams;
        break;
      case UMAMI_META_HELICITY_COUNT:
        *static_cast<int*>( result ) = ProcessData::ncomb;
        break;
      case UMAMI_META_COLOR_COUNT:
        return UMAMI_ERROR_UNSUPPORTED_META;
      case UMAMI_META_MASSES:
      {
        if( g_externalMasses.size() != (size_t)ProcessData::npar ) return UMAMI_ERROR_UNINITIALIZED_META;

        for( int ipar = 0; ipar < ProcessData::npar; ++ipar )
          static_cast<double*>( result )[ipar] = g_externalMasses[ipar];
        break;
      }
      default:
        return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_supported_inputs( bool const** supported, int* count )
  {
    // MOMENTA, ALPHA_S, FLAVOR_INDEX, RANDOM_COLOR, RANDOM_HELICITY, RANDOM_DIAGRAM,
    // HELICITY_INDEX, DIAGRAM_INDEX=true, CHANNEL_INDEX=false
    static const bool data[UMAMI_INPUT_KEY_COUNT] = { true, true, true, true, true, true, true, true };
    *supported = data;
    *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_required_inputs( bool const** required, int* count )
  {
    static const bool data[UMAMI_INPUT_KEY_COUNT] = { true }; // MOMENTA only
    *required = data;
    *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_supported_outputs( bool const** supported, int* count )
  {
    // MATRIX_ELEMENT, DIAGRAM_AMP2, COLOR_INDEX, HELICITY_INDEX, DIAGRAM_INDEX, GPU_STREAM
    static const bool data[UMAMI_OUTPUT_KEY_COUNT] = { true, true, true, true, true, true };
    *supported = data;
    *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_initialize( UmamiHandle* handle, char const* param_card_path )
  {
    CPPProcess process;
    process.initProc( param_card_path );

    const std::vector<fptype>& masses = process.getMasses();
    g_externalMasses.assign( masses.begin(), masses.end() );

    auto instance = new InterfaceInstance();
    *handle = instance;
    // once per library, like the filter (a thread-safe static initialization), on
    // the device current here, which initProc has just set the parameters of
    static const bool good_helicities = ( choose_good_helicities( instance ), true );
    (void)good_helicities;
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_set_parameter(
    UmamiHandle handle,
    char const* name,
    double parameter_real,
    [[maybe_unused]] double parameter_imag )
  {
    // The run card bw_cutoff (the $-excluded propagator window). It is shared
    // by all the instances of this library, like the model parameters read by
    // umami_initialize.
    if( std::string( name ) == "bwcutoff" )
    {
      setBwCutoff( parameter_real );
      return UMAMI_SUCCESS;
    }
    // Non-zero: the helicity choice of an interference |M|^2 keeps the helicity
    // sum as the |M|^2 (lower variance), and the chosen helicity is then not
    // meaningful (mg7 writes 9 in the LHE). Zero, the default: the madevent
    // convention, see umami_matrix_element. Per instance.
    if( std::string( name ) == "interference_helicity_summed" )
    {
      static_cast<InterfaceInstance*>( handle )->summed_helicity = parameter_real != 0;
      return UMAMI_SUCCESS;
    }
    // Parameters are process-wide (Parameters::getInstance()), not per-handle,
    // matching how umami_initialize itself never stores the CPPProcess it
    // constructs: initProc() only ever populates the Parameters singleton.
#ifdef MGONGPU_HARDCODE_PARAM
    // Hardcoded parameters are compile-time constants: there is nothing to set.
    return UMAMI_ERROR_NOT_IMPLEMENTED;
#else
    if( !Parameters::getInstance()->setParameterByName( name, parameter_real, parameter_imag ) )
      return UMAMI_ERROR_UNKNOWN_PARAMETER;
    // sigmaKin() and friends read cached copies of the independent parameters/couplings
    // (cIPD/cIPC/cIPF/bsmIndepParam), not the Parameters singleton itself, for
    // performance; refresh those caches now so the override actually takes effect.
    CPPProcess::refreshIndependentParams();
    // Keep g_externalMasses in sync with the (possibly updated) singleton.
    // Emit a one-time warning if a mass parameter was changed, since any phase-space
    // generator that cached masses from a prior umami_get_meta(UMAMI_META_MASSES,...) call
    // will need to re-query them.
    {
      std::vector<fptype> updatedMasses;
      CPPProcess::refreshExternalMasses( updatedMasses );
      bool changed = ( updatedMasses.size() != g_externalMasses.size() );
      for( size_t i = 0; !changed && i < updatedMasses.size(); ++i )
        changed = ( static_cast<double>( updatedMasses[i] ) != g_externalMasses[i] );
      if( changed )
      {
        g_externalMasses.assign( updatedMasses.begin(), updatedMasses.end() );
      }
    }
    return UMAMI_SUCCESS;
#endif
  }

  UmamiStatus umami_get_parameter(
    [[maybe_unused]] UmamiHandle handle,
    [[maybe_unused]] char const* name,
    [[maybe_unused]] double* parameter_real,
    [[maybe_unused]] double* parameter_imag )
  {
#ifdef MGONGPU_HARDCODE_PARAM
    return UMAMI_ERROR_NOT_IMPLEMENTED;
#else
    if( !Parameters::getInstance()->getParameterByName( name, parameter_real, parameter_imag ) )
      return UMAMI_ERROR_UNKNOWN_PARAMETER;
    return UMAMI_SUCCESS;
#endif
  }

  UmamiStatus umami_matrix_element(
    UmamiHandle handle,
    size_t count,
    size_t stride,
    size_t offset,
    size_t input_count,
    UmamiInputKey const* input_keys,
    void const* const* inputs,
    size_t output_count,
    UmamiOutputKey const* output_keys,
    void* const* outputs )
  {
    const double* momenta_in = nullptr;
    const double* alpha_s_in = nullptr;
    const unsigned int* flavor_indices_in = nullptr;
    const double* random_color_in = nullptr;
    const double* random_helicity_in = nullptr;
    const double* random_diagram_in = nullptr;
    const int* helicity_in = nullptr;
    [[maybe_unused]] const int* diagram_in = nullptr; // TODO: unused

    for( std::size_t i = 0; i < input_count; ++i )
    {
      const void* input = inputs[i];
      switch( input_keys[i] )
      {
        case UMAMI_IN_MOMENTA:
          momenta_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_ALPHA_S:
          alpha_s_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_FLAVOR_INDEX:
          flavor_indices_in = static_cast<const unsigned int*>( input );
          break;
        case UMAMI_IN_RANDOM_COLOR:
          random_color_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_RANDOM_HELICITY:
          random_helicity_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_RANDOM_DIAGRAM:
          random_diagram_in = static_cast<const double*>( input );
          break;
        case UMAMI_IN_HELICITY_INDEX:
          helicity_in = static_cast<const int*>( input );
          break;
        case UMAMI_IN_DIAGRAM_INDEX:
          diagram_in = static_cast<const int*>( input );
          break;
        default:
          return UMAMI_ERROR_UNSUPPORTED_INPUT;
      }
    }
    if( !momenta_in ) return UMAMI_ERROR_MISSING_INPUT;
    // UMAMI_IN_HELICITY_INDEX: evaluate each event for the one helicity it is given, an index
    // into the full helicity table [0,ncomb) (the UMAMI_OUT_HELICITY_INDEX convention), instead
    // of summing over the helicities. The |M|^2 is then the contribution of that helicity to
    // the helicity-summed (and averaged) |M|^2, so that the sum over all ncomb helicities gives
    // back the |M|^2 without this input; UMAMI_OUT_HELICITY_INDEX returns the given helicity,
    // the color and diagram are chosen for it, and UMAMI_IN_RANDOM_HELICITY is not used. The
    // indices live on the device and are only read by the kernels, so that an index out of
    // [0,ncomb) cannot fail the call as on the CPU: that event gets |M|^2 = 0 and helicity -1.

    gpuStream_t gpu_stream = nullptr;
    double* m2_out = nullptr;
    double* amp2_out = nullptr;
    int* diagram_out = nullptr;
    int* color_out = nullptr;
    int* helicity_out = nullptr;
    for( std::size_t i = 0; i < output_count; ++i )
    {
      void* output = outputs[i];
      switch( output_keys[i] )
      {
        case UMAMI_OUT_MATRIX_ELEMENT:
          m2_out = static_cast<double*>( output );
          break;
        case UMAMI_OUT_DIAGRAM_AMP2:
          amp2_out = static_cast<double*>( output );
          break;
        case UMAMI_OUT_COLOR_INDEX:
          color_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_HELICITY_INDEX:
          helicity_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_DIAGRAM_INDEX:
          diagram_out = static_cast<int*>( output );
          break;
        case UMAMI_OUT_GPU_STREAM:
          gpu_stream = static_cast<gpuStream_t>( output );
          break;
        default:
          return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
      }
    }

    std::size_t n_threads = 256;
    std::size_t n_blocks = ( count + n_threads - 1 ) / n_threads;
    std::size_t rounded_count = n_blocks * n_threads;

    fptype_momenta* momenta;
    fptype_amp* numerators;
    fptype_amp* denominators;
    fptype *couplings, *g_s, *helicity_random, *color_random, *diagram_random;
    fptype *matrix_elements, *ghel_matrix_elements;
    fptype_amp *color_jamps, *ghel_jamps;
    int *helicity_index, *color_index, *helicity_given;
    unsigned int *flavor_indices, *diagram_index;

    std::size_t n_coup = madmatrix::Parameters_dependentCouplings::ndcoup;
    std::array<std::pair<void**, std::size_t>, 17> ptrs_and_sizes = {{
        {reinterpret_cast<void**>(&momenta), rounded_count * ProcessData::npar * 4 * sizeof( fptype_momenta )},
        {reinterpret_cast<void**>(&couplings), rounded_count * n_coup * 2 * sizeof( fptype )},
        {reinterpret_cast<void**>(&g_s), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&flavor_indices), rounded_count * sizeof( unsigned int )},
        {reinterpret_cast<void**>(&helicity_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&color_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&matrix_elements), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_index), rounded_count * sizeof( unsigned int )},
        // The color flow is picked among ncolor_flow structures, which is more than
        // ncolor when the color sum runs on the DDM basis
        {reinterpret_cast<void**>(&color_jamps), rounded_count * CPPProcess::ncolor_flow * sizeof( fptype_amp )},
        // The numerators are accumulated in place over all helicities via atomicAdd (no helicity dimension),
        // and the denominators are derived from them, so neither buffer carries the ncomb factor anymore.
        {reinterpret_cast<void**>(&numerators), rounded_count * ProcessData::ndiagrams * sizeof( fptype_amp )},
        {reinterpret_cast<void**>(&denominators), rounded_count * sizeof( fptype_amp )},
        {reinterpret_cast<void**>(&helicity_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&color_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&helicity_given), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&ghel_matrix_elements), rounded_count * ProcessData::ncomb * sizeof( fptype )},
        // njampso = ncolor per amplitude split order (just ncolor without split orders)
        {reinterpret_cast<void**>(&ghel_jamps), rounded_count * ProcessData::ncomb * ProcessData::njampso * mgOnGpu::nx2 * sizeof( fptype_amp )},
    }};
    std::size_t total_size = 0;
    constexpr std::size_t MAX_SIZE = std::max( { sizeof( fptype ), sizeof( fptype_momenta ), sizeof( fptype ), sizeof( int ) } );
    for (auto [ptr, size] : ptrs_and_sizes) {
        std::size_t aligned_size = (size + MAX_SIZE - 1) / MAX_SIZE * MAX_SIZE;
        total_size += aligned_size;
    }
    uint8_t* buffer;
    // we can consider caching this between matrix element calls
    gpuMallocAsync( &buffer, total_size, gpu_stream );
    std::size_t buf_offset = 0;
    for (auto [ptr, size] : ptrs_and_sizes) {
        std::size_t aligned_size = (size + 7) / 8 * 8;
        *ptr = buffer + buf_offset;
        buf_offset += aligned_size;
    }

    copy_inputs<<<n_blocks, n_threads, 0, gpu_stream>>>(
      momenta_in,
      random_helicity_in,
      random_color_in,
      random_diagram_in,
      alpha_s_in,
      flavor_indices_in,
      helicity_in,
      momenta,
      helicity_random,
      color_random,
      diagram_random,
      g_s,
      flavor_indices,
      helicity_given,
      count,
      stride,
      offset );
    computeDependentCouplings<<<n_blocks, n_threads, 0, gpu_stream>>>( g_s, couplings );
    checkGpu( gpuPeekAtLastError() );

    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    if( !instance->initialized )
    {
      // the helicity filtering runs on the default stream: wait for the inputs copied on
      // gpu_stream, which need not synchronise with it (a non-blocking stream); its
      // synchronous copies back to the host finish it before the work below
      gpuStreamSynchronize( gpu_stream );
      // the jamp scratch of the helicity filtering (one helicity, njampso jamps, a few
      // events): ghel_jamps, whose size does not depend on ncolor_flow and nampso
      initialize(
        momenta, couplings, flavor_indices, matrix_elements, ghel_jamps, numerators, denominators, rounded_count );
      instance->initialized = true;
    }

    sigmaKin(
      momenta,
      couplings,
      flavor_indices,
      helicity_random,
      color_random,
      nullptr,
      diagram_random,
      matrix_elements,
      helicity_index,
      color_index,
      color_jamps,
      numerators,
      denominators,
      diagram_index,
      false,
      ghel_matrix_elements,
      ghel_jamps,
      nullptr,
      nullptr,
      &gpu_stream,
      true,
      n_blocks,
      n_threads,
      // a caller drawing the helicity (event generation) gets the madevent convention for
      // an interference |M|^2 (see add_and_select_hel), unless the instance was asked to
      // keep the helicity sum (interference_helicity_summed); without the random number
      // the |M|^2 stays the helicity sum (systematics re-evaluation, standalone checks)
      random_helicity_in != nullptr && !instance->summed_helicity,
      helicity_in ? helicity_given : nullptr );

    copy_outputs<<<n_blocks, n_threads, 0, gpu_stream>>>(
      denominators,
      numerators,
      matrix_elements,
      diagram_index,
      color_index,
      helicity_index,
      m2_out,
      amp2_out,
      diagram_out,
      color_out,
      helicity_out,
      helicity_in,
      count,
      stride,
      offset );
    checkGpu( gpuPeekAtLastError() );

    gpuFreeAsync( buffer, gpu_stream );
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_free( UmamiHandle handle )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    delete instance;
    return UMAMI_SUCCESS;
  }

  // A madmatrix extension, not part of UMAMI: the number of helicity combinations
  // kept by the good-helicity filter, which runs at the first umami_initialize (at
  // the first umami_matrix_element call where RAMBO has no phase space; 0 before
  // it). Like the filter, it is shared by every instance.
  int madmatrix_good_helicity_count()
  {
    return sigmaKin_getNGoodHel();
  }
}
