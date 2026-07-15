// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: T. Heimel (Nov 2025) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro (2026).
// Integrated with the MadGraph7 project in Feb 2026.

#include "umami.h"

#include "CPPProcess.h"
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"

#include <cmath>
#include <vector>
#include <array>
#include <utility>

#ifdef MGONGPUCPP_GPUIMPL
using namespace mg5amcGpu;
#else
using namespace mg5amcCpu;
#endif

namespace
{

  void* initialize_impl(
    const fptype* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
#ifdef MGONGPUCPP_GPUIMPL
    fptype* color_jamps,
#endif
    fptype* numerators,
    fptype* denominators,
    std::size_t count )
  {
    bool is_good_hel[CPPProcess::ncomb];
    sigmaKin_getGoodHel(
      momenta, couplings, flavor_indices, matrix_elements, numerators, denominators,
#ifdef MGONGPUCPP_GPUIMPL
      color_jamps,
#endif
      is_good_hel,
      count );
    sigmaKin_setGoodHel( is_good_hel );
    return nullptr;
  }

  void initialize(
    const fptype* momenta,
    const fptype* couplings,
    const unsigned int* flavor_indices,
    fptype* matrix_elements,
#ifdef MGONGPUCPP_GPUIMPL
    fptype* color_jamps,
#endif
    fptype* numerators,
    fptype* denominators,
    std::size_t count )
  {
    // static local initialization is called exactly once in a thread-safe way
    static void* dummy = initialize_impl( momenta, couplings, flavor_indices, matrix_elements,
#ifdef MGONGPUCPP_GPUIMPL
                                          color_jamps,
#endif
                                          numerators,
                                          denominators,
                                          count );
  }

#ifdef MGONGPUCPP_GPUIMPL
  __device__
#endif
    void
    transpose_momenta( const double* momenta_in, fptype* momenta_out, std::size_t i_event_in, std::size_t i_event_out, std::size_t stride )
  {
    std::size_t page_size = MemoryAccessMomentaBase::neppM;
    std::size_t i_page = i_event_out / page_size;
    std::size_t i_vector = i_event_out % page_size;

    for( std::size_t i_part = 0; i_part < CPPProcess::npar; ++i_part )
    {
      for( std::size_t i_mom = 0; i_mom < 4; ++i_mom )
      {
        momenta_out[i_page * CPPProcess::npar * 4 * page_size +
                    i_part * 4 * page_size + i_mom * page_size + i_vector] = momenta_in[stride * ( CPPProcess::npar * i_mom + i_part ) + i_event_in];
      }
    }
  }

#ifdef MGONGPUCPP_GPUIMPL

  __global__ void copy_inputs(
    const double* momenta_in,
    const double* helicity_random_in,
    const double* color_random_in,
    const double* diagram_random_in,
    const double* alpha_s_in,
    const unsigned int* flavor_indices_in,
    fptype* momenta,
    fptype* helicity_random,
    fptype* color_random,
    fptype* diagram_random,
    fptype* g_s,
    unsigned int* flavor_indices,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_event >= count ) return;

    transpose_momenta( &momenta_in[offset], momenta, i_event, i_event, stride );
    diagram_random[i_event] = diagram_random_in ? diagram_random_in[i_event + offset] : 0.5;
    helicity_random[i_event] = helicity_random_in ? helicity_random_in[i_event + offset] : 0.5;
    color_random[i_event] = color_random_in ? color_random_in[i_event + offset] : 0.5;
    g_s[i_event] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_event + offset] ) : 1.2177157847767195;
    flavor_indices[i_event] = flavor_indices_in ? flavor_indices_in[i_event + offset] : 0;
  }

  __global__ void copy_outputs(
    fptype* denominators,
    fptype* numerators,
    fptype* matrix_elements,
    unsigned int* diagram_index,
    int* color_index,
    int* helicity_index,
    double* m2_out,
    double* amp2_out,
    int* diagram_out,
    int* color_out,
    int* helicity_out,
    std::size_t count,
    std::size_t stride,
    std::size_t offset )
  {
    std::size_t i_event = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_event >= count ) return;

    if( m2_out ) m2_out[i_event + offset] = matrix_elements[i_event];
    if( amp2_out )
    {
      double denominator = denominators[i_event];
      for( std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag )
      {
        amp2_out[stride * i_diag + i_event + offset] = numerators[i_event * CPPProcess::ndiagrams + i_diag] / denominator;
      }
    }
    if( diagram_out ) diagram_out[i_event + offset] = diagram_index[i_event] - 1;
    if( color_out ) color_out[i_event + offset] = color_index[i_event] - 1;
    if( helicity_out ) helicity_out[i_event + offset] = helicity_index[i_event] - 1;
  }

#endif // MGONGPUCPP_GPUIMPL

  struct InterfaceInstance
  {
    bool initialized = false;
  };

  std::vector<double> g_externalMasses;

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
#ifdef MGONGPUCPP_GPUIMPL
#ifdef __CUDACC__
        device = UMAMI_DEVICE_CUDA;
#elif defined( __HIPCC__ )
        device = UMAMI_DEVICE_HIP;
#endif
#else
        device = UMAMI_DEVICE_CPU;
#endif
        break;
      }
      case UMAMI_META_PARTICLE_COUNT:
        *static_cast<int*>( result ) = CPPProcess::npar;
        break;
      case UMAMI_META_DIAGRAM_COUNT:
        *static_cast<int*>( result ) = CPPProcess::ndiagrams;
        break;
      case UMAMI_META_HELICITY_COUNT:
        *static_cast<int*>( result ) = CPPProcess::ncomb;
        break;
      case UMAMI_META_COLOR_COUNT:
        return UMAMI_ERROR_UNSUPPORTED_META;
      case UMAMI_META_MASSES:
      {
        if( g_externalMasses.size() != (size_t)CPPProcess::npar ) return UMAMI_ERROR_UNINITIALIZED_META;

        for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
          static_cast<double*>( result )[ipar] = g_externalMasses[ipar];
        break;
      }
      default:
        return UMAMI_ERROR_UNSUPPORTED_META;
    }
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
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_set_parameter(
    [[maybe_unused]] UmamiHandle handle,
    [[maybe_unused]] char const* name,
    [[maybe_unused]] double parameter_real,
    [[maybe_unused]] double parameter_imag )
  {
    return UMAMI_ERROR_NOT_IMPLEMENTED;
  }

  UmamiStatus umami_get_parameter(
    [[maybe_unused]] UmamiHandle handle,
    [[maybe_unused]] char const* name,
    [[maybe_unused]] double* parameter_real,
    [[maybe_unused]] double* parameter_imag )
  {
    return UMAMI_ERROR_NOT_IMPLEMENTED;
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
          return UMAMI_ERROR_UNSUPPORTED_INPUT;
        case UMAMI_IN_DIAGRAM_INDEX:
          diagram_in = static_cast<const int*>( input );
          break;
        default:
          return UMAMI_ERROR_UNSUPPORTED_INPUT;
      }
    }
    if( !momenta_in ) return UMAMI_ERROR_MISSING_INPUT;

#ifdef MGONGPUCPP_GPUIMPL
    gpuStream_t gpu_stream = nullptr;
#endif
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
#ifdef MGONGPUCPP_GPUIMPL
        case UMAMI_OUT_GPU_STREAM:
          gpu_stream = static_cast<gpuStream_t>( output );
          break;
#endif
        default:
          return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
      }
    }

#ifdef MGONGPUCPP_GPUIMPL
    std::size_t n_threads = 256;
    std::size_t n_blocks = ( count + n_threads - 1 ) / n_threads;
    std::size_t rounded_count = n_blocks * n_threads;

    fptype *momenta, *couplings, *g_s, *helicity_random, *color_random, *diagram_random, *color_jamps;
    fptype *matrix_elements, *numerators, *denominators, *ghel_matrix_elements, *ghel_jamps;
    int *helicity_index, *color_index;
    unsigned int *flavor_indices, *diagram_index;

    std::size_t n_coup = mg5amcGpu::Parameters_dependentCouplings::ndcoup;
    std::array<std::pair<void**, std::size_t>, 16> ptrs_and_sizes = {{
        {reinterpret_cast<void**>(&momenta), rounded_count * CPPProcess::npar * 4 * sizeof( fptype )},
        {reinterpret_cast<void**>(&couplings), rounded_count * n_coup * 2 * sizeof( fptype )},
        {reinterpret_cast<void**>(&g_s), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&flavor_indices), rounded_count * sizeof( unsigned int )},
        {reinterpret_cast<void**>(&helicity_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&color_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_random), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&matrix_elements), rounded_count * sizeof( fptype )},
        {reinterpret_cast<void**>(&diagram_index), rounded_count * sizeof( unsigned int )},
        {reinterpret_cast<void**>(&color_jamps), rounded_count * CPPProcess::ncolor * mgOnGpu::nx2 * sizeof( fptype )},
        {reinterpret_cast<void**>(&numerators), rounded_count * CPPProcess::ndiagrams * CPPProcess::ncomb * sizeof( fptype )},
        {reinterpret_cast<void**>(&denominators), rounded_count * CPPProcess::ncomb * sizeof( fptype )},
        {reinterpret_cast<void**>(&helicity_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&color_index), rounded_count * sizeof( int )},
        {reinterpret_cast<void**>(&ghel_matrix_elements), rounded_count * CPPProcess::ncomb * sizeof( fptype )},
        {reinterpret_cast<void**>(&ghel_jamps), rounded_count * CPPProcess::ncomb * CPPProcess::ncolor * mgOnGpu::nx2 * sizeof( fptype )},
    }};
    std::size_t total_size = 0;
    constexpr std::size_t MAX_SIZE = std::max(sizeof(fptype), sizeof(int));
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
      momenta,
      helicity_random,
      color_random,
      diagram_random,
      g_s,
      flavor_indices,
      count,
      stride,
      offset );
    computeDependentCouplings<<<n_blocks, n_threads, 0, gpu_stream>>>( g_s, couplings );
    checkGpu( gpuPeekAtLastError() );

    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    if( !instance->initialized )
    {
      initialize(
        momenta, couplings, flavor_indices, matrix_elements, color_jamps, numerators, denominators, rounded_count );
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
      n_threads );

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
      count,
      stride,
      offset );
    checkGpu( gpuPeekAtLastError() );

    gpuFreeAsync( buffer, gpu_stream );
#else  // MGONGPUCPP_GPUIMPL
    constexpr std::size_t vector_size = MemoryAccessMomentaBase::neppM;
    // need to round to round to double page size for some reason
    constexpr std::size_t page_size2 = 2 * vector_size;
    std::vector<std::size_t> permutation;
    std::size_t rounded_count;

    constexpr std::size_t flavor_count = CPPProcess::nmaxflavor;
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
          for ( std::size_t i = 0; i < page_size2; ++i) {
            flavor_indices[voffset * page_size2 + i] = flav;
          }
          voffset += 1;
        }
        permutation[i_event] = vindex + vcount;
        vcount = (vcount + 1) % page_size2;
      }
      rounded_count = voffset * page_size2;
    } else {
      rounded_count = ( count + page_size2 - 1 ) / page_size2 * page_size2;
    }

    HostBufferBase<fptype, false> momenta( rounded_count * CPPProcess::npar * 4 );
    HostBufferBase<fptype, false> couplings( rounded_count * mg5amcCpu::Parameters_dependentCouplings::ndcoup * 2 );
    HostBufferBase<fptype, false> g_s( rounded_count );
    HostBufferBase<fptype, false> helicity_random( rounded_count );
    HostBufferBase<fptype, false> color_random( rounded_count );
    HostBufferBase<fptype, false> diagram_random( rounded_count );
    HostBufferBase<fptype, false> matrix_elements( rounded_count );
    HostBufferBase<unsigned int, false> diagram_index( rounded_count );
    HostBufferBase<fptype, false> numerators( rounded_count * CPPProcess::ndiagrams );
    HostBufferBase<fptype, false> denominators( rounded_count );
    HostBufferBase<int, false> helicity_index( rounded_count );
    HostBufferBase<int, false> color_index( rounded_count );
    if ( sort_flavors ) {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        std::size_t i_sorted = permutation[i_event];
        transpose_momenta( &momenta_in[offset], momenta.data(), i_event, i_sorted, stride );
        helicity_random[i_sorted] = random_helicity_in ? random_helicity_in[i_event + offset] : 0.5;
        color_random[i_sorted] = random_color_in ? random_color_in[i_event + offset] : 0.5;
        diagram_random[i_sorted] = random_diagram_in ? random_diagram_in[i_event + offset] : 0.5;
        g_s[i_sorted] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_event + offset] ) : 1.2177157847767195;
      }
    } else {
      for( std::size_t i_event = 0; i_event < count; ++i_event )
      {
        transpose_momenta( &momenta_in[offset], momenta.data(), i_event, i_event, stride );
        helicity_random[i_event] = random_helicity_in ? random_helicity_in[i_event + offset] : 0.5;
        color_random[i_event] = random_color_in ? random_color_in[i_event + offset] : 0.5;
        diagram_random[i_event] = random_diagram_in ? random_diagram_in[i_event + offset] : 0.5;
        g_s[i_event] = alpha_s_in ? sqrt( 4 * M_PI * alpha_s_in[i_event + offset] ) : 1.2177157847767195;
        flavor_indices[i_event] = flavor_indices_in ? flavor_indices_in[i_event + offset] : 0;
      }
      for ( std::size_t i_event = count; i_event < rounded_count; ++i_event ) {
        flavor_indices[i_event] = 0;
      }
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
      rounded_count );

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
          for( std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = numerators[i_page * page_size * CPPProcess::ndiagrams + i_diag * page_size + i_vector] / denominator;
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
          for( std::size_t i_diag = 0; i_diag < CPPProcess::ndiagrams; ++i_diag )
          {
            amp2_out[stride * i_diag + i_event + offset] = numerators[i_page * page_size * CPPProcess::ndiagrams + i_diag * page_size + i_vector] / denominator;
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
#endif // MGONGPUCPP_GPUIMPL
    return UMAMI_SUCCESS;
  }

  UmamiStatus umami_free( UmamiHandle handle )
  {
    InterfaceInstance* instance = static_cast<InterfaceInstance*>( handle );
    delete instance;
    return UMAMI_SUCCESS;
  }
}
