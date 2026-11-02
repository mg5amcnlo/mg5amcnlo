// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: T. Heimel (Nov 2025) for the MadGraph7 CUDACPP plugin.
// Further modified by: D. Massaro (2026).
// Integrated with the MadGraph7 project in Feb 2026.

#include "umami.h"

#include "ProcessData.h"
#include "CPPProcess.h" // needed to construct/initProc the process object (umami_initialize)
#include "SigmaKin.h" // sigmaKin_getGoodHel/setGoodHel
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
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    bool is_good_hel[ProcessData::ncomb];
    sigmaKin_getGoodHel(
      momenta, couplings, flavor_indices, matrix_elements, numerators, denominators,
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
    fptype_amp* numerators,
    fptype_amp* denominators,
    std::size_t count )
  {
    // static local initialization is called exactly once in a thread-safe way
    static void* dummy = initialize_impl( momenta, couplings, flavor_indices, matrix_elements,
                                          numerators,
                                          denominators,
                                          count );
  }

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
  // the first call still chooses.
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
    std::vector<double> matrix_elements( nevt );
    const UmamiInputKey input_keys[] = { UMAMI_IN_MOMENTA };
    const void* inputs[] = { momenta.data() };
    const UmamiOutputKey output_keys[] = { UMAMI_OUT_MATRIX_ELEMENT };
    void* outputs[] = { matrix_elements.data() };
    umami_matrix_element( handle, nevt, nevt, 0, 1, input_keys, inputs, 1, output_keys, outputs );
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
        device = UMAMI_DEVICE_CPU;
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
    static const bool data[UMAMI_OUTPUT_KEY_COUNT] = { true, true, true, true, true };
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
    // once per library, like the filter (a thread-safe static initialization)
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
    // the color and diagram are chosen for it, and UMAMI_IN_RANDOM_HELICITY is not used.
    if( helicity_in )
    {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        const int ihel = helicity_in[i_event + offset];
        if( ihel < 0 || ihel >= ProcessData::ncomb ) return UMAMI_ERROR;
      }
    }

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
        default:
          return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
      }
    }

    constexpr std::size_t vector_size = MemoryAccessMomentaBase::neppM;
    // The kernels run on one SIMD page of events at a time (neppM == neppV events,
    // a single event for the scalar backend), and on two pages at a time in mixed
    // precision (double amplitudes, float colour algebra): the batch is rounded up
    // to that, and no further. Every extra event is computed for nothing: a call
    // with one event used to evaluate two of them on the scalar backend.
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    constexpr std::size_t page_size2 = 2 * vector_size;
#else
    constexpr std::size_t page_size2 = vector_size;
#endif
    std::vector<std::size_t> permutation;
    std::size_t rounded_count;

    constexpr std::size_t flavor_count = ProcessData::nmaxflavor;
    HostBufferBase<unsigned int, false> flavor_indices( ((count + page_size2 - 1) / page_size2 + flavor_count) * page_size2 );
    bool sort_flavors = vector_size > 1 && flavor_count > 1 && flavor_indices_in;
    if ( sort_flavors ) 
    {
      permutation.resize(count);
      std::size_t voffset = 0;
      std::size_t vector_indices[flavor_count] = {};
      std::size_t vector_counts[flavor_count] = {};
      // determine permutation of inputs such that all entries in a SIMD vector
      // have the same flavor index
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        unsigned int flav = flavor_indices_in[i_event + offset];
        auto& vcount = vector_counts[flav];
        auto& vindex = vector_indices[flav];
        if ( vcount == 0 )
        {
          vindex = voffset * page_size2;
          voffset += 1;
        }
        permutation[i_event] = vindex + vcount;
        vcount = (vcount + 1) % page_size2;
      }
      rounded_count = voffset * page_size2;
    } else {
      rounded_count = ( count + page_size2 - 1 ) / page_size2 * page_size2;
    }

    HostBufferBase<fptype_momenta, false> momenta( rounded_count * ProcessData::npar * 4 );
    HostBufferBase<fptype, false> couplings( rounded_count * madmatrix::Parameters_dependentCouplings::ndcoup * 2 );
    HostBufferBase<fptype, false> g_s( rounded_count );
    HostBufferBase<fptype, false> helicity_random( rounded_count );
    HostBufferBase<fptype, false> color_random( rounded_count );
    HostBufferBase<fptype, false> diagram_random( rounded_count );
    HostBufferBase<fptype, false> matrix_elements( rounded_count );
    HostBufferBase<unsigned int, false> diagram_index( rounded_count );
    HostBufferBase<fptype_amp, false> numerators( rounded_count * ProcessData::ndiagrams );
    HostBufferBase<fptype_amp, false> denominators( rounded_count );
    HostBufferBase<int, false> helicity_index( rounded_count );
    HostBufferBase<int, false> color_index( rounded_count );
    HostBufferBase<int, false> helicity_given( rounded_count ); // as the other inputs, the padding lanes copy their source event
    // The input event of each lane. The padding lanes, which complete the last page
    // of a flavor, are copies of the first event of their page, as on GPUs. Left at
    // zero (momenta and g_s), they evaluated to nan, and the good-helicity filter of
    // the first call (a helicity is kept if its |M|^2 is not 0 for one of its first
    // events) took that for a non-zero value whenever a padding lane was among
    // them, as in a first call with a single event: every helicity was then kept
    // for the rest of the run, the vanishing ones of a q q~ initial state included.
    std::vector<std::size_t> source( rounded_count, count );
    for( std::size_t i_event = 0; i_event < count; ++i_event )
    {
      source[sort_flavors ? permutation[i_event] : i_event] = i_event;
    }
    for( std::size_t lane = 0; lane < rounded_count; ++lane )
    {
      // a page is opened by an event, so its first lane is never padding
      if( source[lane] == count ) source[lane] = source[lane - lane % page_size2];
    }
    for( std::size_t lane = 0; lane < rounded_count; ++lane )
    {
      const std::size_t i_event = source[lane];
      transpose_momenta( &momenta_in[offset], momenta.data(), i_event, lane, stride );
      helicity_random[lane] = random_helicity_in ? random_helicity_in[i_event + offset] : 0.5;
      color_random[lane] = random_color_in ? random_color_in[i_event + offset] : 0.5;
      diagram_random[lane] = random_diagram_in ? random_diagram_in[i_event + offset] : 0.5;
      g_s[lane] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_event + offset] ) : 1.2177157847767195;
      flavor_indices[lane] = flavor_indices_in ? flavor_indices_in[i_event + offset] : 0;
      if( helicity_in ) helicity_given[lane] = helicity_in[i_event + offset];
    }
    computeDependentCouplings( g_s.data(), couplings.data(), rounded_count );

    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    if( !instance->initialized )
    {
      initialize(
        momenta.data(),
        couplings.data(),
        flavor_indices.data(),
        matrix_elements.data(),
        numerators.data(),
        denominators.data(),
        rounded_count );
      instance->initialized = true;
    }

    sigmaKin(
      momenta.data(),
      couplings.data(),
      flavor_indices.data(),
      helicity_random.data(),
      color_random.data(),
      nullptr,
      diagram_random.data(),
      matrix_elements.data(),
      helicity_index.data(),
      color_index.data(),
      numerators.data(),
      denominators.data(),
      diagram_index.data(),
      false,
      rounded_count,
      // a caller drawing the helicity (event generation) gets the madevent convention for
      // an interference |M|^2, see select_helicity_signed, unless the instance was asked
      // to keep the helicity sum (interference_helicity_summed); without the random number
      // the |M|^2 stays the helicity sum (systematics re-evaluation, standalone checks)
      random_helicity_in != nullptr && !instance->summed_helicity,
      helicity_in ? helicity_given.data() : nullptr );

    if ( sort_flavors )
    {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        std::size_t i_sorted = permutation[i_event];
        std::size_t page_size = MemoryAccessMomentaBase::neppM;
        std::size_t i_page = i_sorted / page_size;
        std::size_t i_vector = i_sorted % page_size; // vector lane

        double denominator = denominators[i_sorted];
        if( m2_out != nullptr )
        {
          m2_out[i_event + offset] = matrix_elements[i_sorted];
        }
        if( amp2_out != nullptr )
        {
          for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_page * page_size * ProcessData::ndiagrams + i_diag * page_size + i_vector], denominator );
          }
        }
        if( diagram_out != nullptr )
        {
          diagram_out[i_event + offset] = diagram_index[i_sorted] - 1;
        }
        if( color_out != nullptr )
        {
          color_out[i_event + offset] = color_index[i_sorted] - 1;
        }
        if( helicity_out != nullptr )
        {
          helicity_out[i_event + offset] = helicity_index[i_sorted] - 1;
        }
      }
    } else {
      std::size_t page_size = MemoryAccessMomentaBase::neppM;
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        std::size_t i_page = i_event / page_size;
        std::size_t i_vector = i_event % page_size;

        double denominator = denominators[i_event];
        if( m2_out != nullptr )
        {
          m2_out[i_event + offset] = matrix_elements[i_event];
        }
        if( amp2_out != nullptr )
        {
          for( std::size_t i_diag = 0; i_diag < ProcessData::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = channel_amp2( numerators[i_page * page_size * ProcessData::ndiagrams + i_diag * page_size + i_vector], denominator );
          }
        }
        if( diagram_out != nullptr )
        {
          diagram_out[i_event + offset] = diagram_index[i_event] - 1;
        }
        if( color_out != nullptr )
        {
          color_out[i_event + offset] = color_index[i_event] - 1;
        }
        if( helicity_out != nullptr )
        {
          helicity_out[i_event + offset] = helicity_index[i_event] - 1;
        }
      }
    }
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
