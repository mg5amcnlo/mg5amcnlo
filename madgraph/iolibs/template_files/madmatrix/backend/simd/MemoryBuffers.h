// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2021, based on earlier work by S. Hageboeck) for the MadGraph7 CUDACPP plugin.
// Further modified by: S. Roiser, J. Teig, A. Valassi (2021-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryBuffers_H
#define MemoryBuffers_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuCxtypes.h"

#include "ProcessData.h"
#include "Parameters.h"

#include <sstream>

namespace madmatrix
{
  //--------------------------------------------------------------------------

  namespace MemoryBuffers
  {
    // Process-independent compile-time constants
    static constexpr size_t np4 = ProcessData::np4;
    static constexpr size_t nw6 = ProcessData::nw6;
    static constexpr size_t nx2 = mgOnGpu::nx2;
    // Process-dependent compile-time constants
    static constexpr size_t nparf = ProcessData::nparf;
    static constexpr size_t npar = ProcessData::npar;
    static constexpr size_t ndcoup = Parameters_dependentCouplings::ndcoup;
    static constexpr size_t ncolor = ProcessData::ncolor;
  }

  //--------------------------------------------------------------------------

  // An abstract interface encapsulating a given number of events
  class INumberOfEvents
  {
  public:
    virtual ~INumberOfEvents() {}
    virtual size_t nevt() const = 0;
  };

  //--------------------------------------------------------------------------

  // A class encapsulating a given number of events
  class NumberOfEvents : virtual public INumberOfEvents
  {
  public:
    NumberOfEvents( const size_t nevt )
      : m_nevt( nevt ) {}
    virtual ~NumberOfEvents() {}
    virtual size_t nevt() const override { return m_nevt; }
  private:
    const size_t m_nevt;
  };

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer (not necessarily an event buffer)
  template<typename T>
  class BufferBase : virtual public INumberOfEvents
  {
  protected:
    BufferBase( const size_t size, const bool onDevice )
      : m_size( size ), m_data( nullptr ), m_isOnDevice( onDevice ) {}
  public:
    virtual ~BufferBase() {}
    T* data() { return m_data; }
    const T* data() const { return m_data; }
    T& operator[]( const size_t index ) { return m_data[index]; }
    const T& operator[]( const size_t index ) const { return m_data[index]; }
    size_t size() const { return m_size; }
    size_t bytes() const { return m_size * sizeof( T ); }
    bool isOnDevice() const { return m_isOnDevice; }
    virtual size_t nevt() const override { throw std::runtime_error( "This BufferBase is not an event buffer" ); }
  protected:
    const size_t m_size;
    T* m_data;
    const bool m_isOnDevice;
  };

  //--------------------------------------------------------------------------

  constexpr bool HostBufferALIGNED = false;   // ismisaligned=false
  constexpr bool HostBufferMISALIGNED = true; // ismisaligned=true

  // A class encapsulating a C++ host buffer
  template<typename T, bool ismisaligned>
  class HostBufferBase : public BufferBase<T>
  {
  public:
    HostBufferBase( const size_t size )
      : BufferBase<T>( size, false )
    {
      if constexpr( !ismisaligned )
        this->m_data = new( std::align_val_t( cppAlign ) ) T[size]();
      else
        this->m_data = new( std::align_val_t( cppAlign ) ) T[size + 1]() + 1; // TEST MISALIGNMENT!
    }
    virtual ~HostBufferBase()
    {
      if constexpr( !ismisaligned )
        ::operator delete[]( this->m_data, std::align_val_t( cppAlign ) );
      else
        ::operator delete[]( ( this->m_data ) - 1, std::align_val_t( cppAlign ) ); // TEST MISALIGNMENT!
    }
    static constexpr bool isaligned() { return !ismisaligned; }
  public:
    static constexpr size_t cppAlign = mgOnGpu::cppAlign;
  };

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------

  // A class encapsulating a C++ host buffer for a given number of events
  template<typename T, size_t sizePerEvent, bool ismisaligned>
  class HostBuffer : public HostBufferBase<T, ismisaligned>, virtual private NumberOfEvents
  {
  public:
    HostBuffer( const size_t nevt )
      : NumberOfEvents( nevt )
      , HostBufferBase<T, ismisaligned>( sizePerEvent * nevt )
    {
      //std::cout << "HostBuffer::ctor " << this << " " << nevt << std::endl;
    }
    virtual ~HostBuffer()
    {
      //std::cout << "HostBuffer::dtor " << this << std::endl;
    }
    virtual size_t nevt() const override final { return NumberOfEvents::nevt(); }
  };

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for momenta random numbers
  typedef BufferBase<fptype> BufferRndNumMomenta;

  // The size (number of elements) per event in a memory buffer for momenta random numbers
  constexpr size_t sizePerEventRndNumMomenta = MemoryBuffers::np4 * MemoryBuffers::nparf;

  // A class encapsulating a C++ host buffer for momenta random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumMomenta, HostBufferALIGNED> HostBufferRndNumMomenta;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for Gs (related to the event-by-event strength of running coupling constant alphas QCD)
  typedef BufferBase<fptype> BufferGs;

  // The size (number of elements) per event in a memory buffer for Gs
  constexpr size_t sizePerEventGs = 1;

  // A class encapsulating a C++ host buffer for gs
  typedef HostBuffer<fptype, sizePerEventGs, HostBufferALIGNED> HostBufferGs;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for numerators (of the multichannel single-diagram enhancement factors)
  typedef BufferBase<fptype_amp> BufferNumerators;

  // The size (number of elements) per event in a memory buffer for numerators
  // (should be equal to the number of diagrams in the process)
  constexpr size_t sizePerEventNumerators = ProcessData::ndiagrams;

  // A class encapsulating a C++ host buffer for numerators
  typedef HostBuffer<fptype_amp, sizePerEventNumerators, HostBufferALIGNED> HostBufferNumerators;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for denominators (of the multichannel single-diagram enhancement factors)
  typedef BufferBase<fptype_amp> BufferDenominators;

  // The size (number of elements) per event in a memory buffer for denominators
  constexpr size_t sizePerEventDenominators = 1;

  // A class encapsulating a C++ host buffer for denominators
  typedef HostBuffer<fptype_amp, sizePerEventDenominators, HostBufferALIGNED> HostBufferDenominators;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for couplings that depend on the event-by-event running coupling constant alphas QCD
  typedef BufferBase<fptype> BufferCouplings;

  // The size (number of elements) per event in a memory buffer for random numbers
  constexpr size_t sizePerEventCouplings = MemoryBuffers::ndcoup * MemoryBuffers::nx2;

  // A class encapsulating a C++ host buffer for couplings
  typedef HostBuffer<fptype, sizePerEventCouplings, HostBufferALIGNED> HostBufferCouplings;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for momenta
  typedef BufferBase<fptype_momenta> BufferMomenta;

  // The size (number of elements) per event in a memory buffer for momenta
  constexpr size_t sizePerEventMomenta = MemoryBuffers::np4 * MemoryBuffers::npar;

  // A class encapsulating a C++ host buffer for momenta
  typedef HostBuffer<fptype_momenta, sizePerEventMomenta, HostBufferALIGNED> HostBufferMomenta;
  //typedef HostBuffer<fptype_momenta, sizePerEventMomenta, HostBufferMISALIGNED> HostBufferMomenta; // TEST MISALIGNMENT!

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for sampling weights
  typedef BufferBase<fptype> BufferWeights;

  // The size (number of elements) per event in a memory buffer for sampling weights
  constexpr size_t sizePerEventWeights = 1;

  // A class encapsulating a C++ host buffer for sampling weights
  typedef HostBuffer<fptype, sizePerEventWeights, HostBufferALIGNED> HostBufferWeights;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for matrix elements
  typedef BufferBase<fptype> BufferMatrixElements;

  // The size (number of elements) per event in a memory buffer for matrix elements
  constexpr size_t sizePerEventMatrixElements = 1;

  // A class encapsulating a C++ host buffer for matrix elements
  typedef HostBuffer<fptype, sizePerEventMatrixElements, HostBufferALIGNED> HostBufferMatrixElements;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for the helicity mask
  typedef BufferBase<bool> BufferHelicityMask;

  // A class encapsulating a C++ host buffer for the helicity mask
  typedef HostBufferBase<bool, HostBufferALIGNED> HostBufferHelicityMask;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for wavefunctions
  typedef BufferBase<fptype_amp> BufferWavefunctions;

  // The size (number of elements) per event in a memory buffer for wavefunctions
  constexpr size_t sizePerEventWavefunctions = MemoryBuffers::nw6 * MemoryBuffers::nx2;

  // A class encapsulating a C++ host buffer for wavefunctions
  typedef HostBuffer<fptype_amp, sizePerEventWavefunctions, HostBufferALIGNED> HostBufferWavefunctions;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for helicity random numbers
  typedef BufferBase<fptype> BufferRndNumHelicity;

  // The size (number of elements) per event in a memory buffer for helicity random numbers
  constexpr size_t sizePerEventRndNumHelicity = 1;

  // A class encapsulating a C++ host buffer for helicity random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumHelicity, HostBufferALIGNED> HostBufferRndNumHelicity;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for color random numbers
  typedef BufferBase<fptype> BufferRndNumColor;

  // The size (number of elements) per event in a memory buffer for color random numbers
  constexpr size_t sizePerEventRndNumColor = 1;

  // A class encapsulating a C++ host buffer for color random numbers
  typedef HostBuffer<fptype, sizePerEventRndNumColor, HostBufferALIGNED> HostBufferRndNumColor;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for channel ids
  typedef BufferBase<unsigned int> BufferChannelIds;

  // The size (number of elements) per event in a memory buffer for channel ids
  constexpr size_t sizePerEventChannelId = 1;

  // A class encapsulating a C++ host buffer for channel ids
  typedef HostBuffer<unsigned int, sizePerEventChannelId, HostBufferALIGNED> HostBufferChannelIds;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for channel ids
  typedef BufferBase<unsigned int> BufferIflavorVec;

  // The size (number of elements) per event in a memory buffer for channel ids
  constexpr size_t sizePerEventIflavorVec = 1;

  // A class encapsulating a C++ host buffer for channel ids
  typedef HostBuffer<unsigned int, sizePerEventIflavorVec, HostBufferALIGNED> HostBufferIflavorVec;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for helicity selection
  typedef BufferBase<int> BufferSelectedHelicity;

  // The size (number of elements) per event in a memory buffer for helicity selection
  constexpr size_t sizePerEventSelectedHelicity = 1;

  // A class encapsulating a C++ host buffer for helicity selection
  typedef HostBuffer<int, sizePerEventSelectedHelicity, HostBufferALIGNED> HostBufferSelectedHelicity;

  //--------------------------------------------------------------------------

  // A base class encapsulating a memory buffer for color selection
  typedef BufferBase<int> BufferSelectedColor;

  // The size (number of elements) per event in a memory buffer for color selection
  constexpr size_t sizePerEventSelectedColor = 1;

  // A class encapsulating a C++ host buffer for color selection
  typedef HostBuffer<int, sizePerEventSelectedColor, HostBufferALIGNED> HostBufferSelectedColor;

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------
}

#endif // MemoryBuffers_H
