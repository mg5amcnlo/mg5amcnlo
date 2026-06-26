// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef RANDOMNUMBERKERNELS_H
#define RANDOMNUMBERKERNELS_H 1

#include "mgOnGpuConfig.h"

#include "MemoryBuffers.h"

// Forward definition from curand.h (the full header is only needed in CurandRandomKernel.cc)
struct curandGenerator_st;

// Forward definition from hiprand.h (the full header is only needed in HiprandRandomKernel.cc)
struct rocrand_generator_base_type;
typedef rocrand_generator_base_type hiprandGenerator_st;

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //--------------------------------------------------------------------------

  /*
  // An interface encapsulating random number generation on a CPU host or on a GPU device
  class IRandomNumberKernel
  {
  public:

    // Destructor
    virtual ~IRandomNumberKernel(){}

    // Seed the random number generator
    virtual void seedGenerator( const unsigned int seed ) = 0;

    // Generate the random number array
    virtual void generateRnarray() = 0;

    // Is this a host or device kernel?
    virtual bool isOnDevice() const = 0;

  };
  */

  //--------------------------------------------------------------------------

  // A base class encapsulating random number generation on a CPU host or on a GPU device
  class RandomNumberKernelBase //: virtual public IRandomNumberKernel
  {

  protected:

    // Constructor from an existing output buffer
    RandomNumberKernelBase( BufferRndNumMomenta& rnarray )
      : m_rnarray( rnarray ) {}

  public:

    // Destructor
    virtual ~RandomNumberKernelBase() {}

    // Seed the random number generator
    virtual void seedGenerator( const unsigned int seed ) = 0;

    // Generate the random number array
    virtual void generateRnarray() = 0;

    // Is this a host or device kernel?
    virtual bool isOnDevice() const = 0;

  protected:

    // The buffer for the output random numbers
    BufferRndNumMomenta& m_rnarray;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating common random number generation on a CPU host
  class CommonRandomNumberKernel final : public RandomNumberKernelBase
  {
  public:

    // Constructor from an existing output buffer
    CommonRandomNumberKernel( BufferRndNumMomenta& rnarray );

    // Destructor
    ~CommonRandomNumberKernel() {}

    // Seed the random number generator
    void seedGenerator( const unsigned int seed ) override final { m_seed = seed; };

    // Generate the random number array
    void generateRnarray() override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return false; }

  private:

    // The generator seed
    unsigned int m_seed;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating CURAND random number generation on a CPU host or on a GPU device
  class CurandRandomNumberKernel final : public RandomNumberKernelBase
  {
  public:

    // Constructor from an existing output buffer
    CurandRandomNumberKernel( BufferRndNumMomenta& rnarray, const bool onDevice );

    // Destructor
    ~CurandRandomNumberKernel();

    // Seed the random number generator
    void seedGenerator( const unsigned int seed ) override final;

    // Generate the random number array
    void generateRnarray() override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return m_isOnDevice; }

  private:

    // Create the generator (workaround for #429: do this in every seedGenerator call rather than only in the ctor)
    void createGenerator();

    // Destroy the generator (workaround for #429: do this in every seedGenerator call rather than only in the ctor)
    void destroyGenerator();

  private:

    // Is this a host or device kernel?
    const bool m_isOnDevice;

    // The curand generator
    // (NB: curand.h defines typedef curandGenerator_t as a pointer to forward-defined 'struct curandGenerator_st')
    curandGenerator_st* m_rnGen;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating HIPRAND random number generation on a CPU host or on a GPU device
  class HiprandRandomNumberKernel final : public RandomNumberKernelBase
  {
  public:

    // Constructor from an existing output buffer
    HiprandRandomNumberKernel( BufferRndNumMomenta& rnarray, const bool onDevice );

    // Destructor
    ~HiprandRandomNumberKernel();

    // Seed the random number generator
    void seedGenerator( const unsigned int seed ) override final;

    // Generate the random number array
    void generateRnarray() override final;

    // Is this a host or device kernel?
    bool isOnDevice() const override final { return m_isOnDevice; }

  private:

    // Create the generator (workaround for #429: do this in every seedGenerator call rather than only in the ctor)
    void createGenerator();

    // Destroy the generator (workaround for #429: do this in every seedGenerator call rather than only in the ctor)
    void destroyGenerator();

  private:

    // Is this a host or device kernel?
    const bool m_isOnDevice;

    // The hiprand generator
    // (NB: hiprand.h defines typedef hiprandGenerator_t as a pointer to forward-defined 'struct hiprandGenerator_st')
    hiprandGenerator_st* m_rnGen;
  };

  //--------------------------------------------------------------------------
}
#endif // RANDOMNUMBERKERNELS_H
