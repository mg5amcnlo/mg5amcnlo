#ifndef MGONGPUNVTX_H 
#define MGONGPUNVTX_H 1

// Provides macros for simply use of NVTX, if a compiler macro USE_NVTX is defined.
// Original author Peter Heywood <p.heywood@sheffield.ac.uk>
// With a few modifications by Andrea Valassi

//-------------------------------------------
// NVTX is enabled
//-------------------------------------------

#ifdef USE_NVTX

#include <stdio.h>

// This assumes CUDA 10.0+
#include "nvtx3/nvToolsExt.h"

// Scope some things into a namespace
namespace nvtx {

  // Colour palette (RGB): https://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
  const uint32_t palette[] = { 0xffa6cee3, 0xff1f78b4, 0xffb2df8a, 0xff33a02c, 0xfffb9a99, 0xffe31a1c,
                               0xfffdbf6f, 0xffff7f00, 0xffcab2d6, 0xff6a3d9a, 0xffffff99, 0xffb15928 };
  const uint32_t colourCount = sizeof( palette ) / sizeof( uint32_t );

  // Inline method to push an nvtx range
  inline void push( const char* str, const uint32_t nextColourIdx )
  {
    // Get the wrapped colour index
    uint32_t colourIdx = nextColourIdx % colourCount;
    // Build/populate the struct of nvtx event attributes
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.color = palette[colourIdx];
    eventAttrib.message.ascii = str;
    // Push the custom event.
    nvtxRangePushEx( &eventAttrib );
  }

  // Inline method to pop an nvtx range
  inline void pop()
  {
    nvtxRangePop();
  }

}

// Macro to push an arbitrary nvtx marker
#define NVTX_PUSH(str,idx) nvtx::push(str,idx)

// Macro to pop an arbitrary nvtx marker
#define NVTX_POP() nvtx::pop()

//-------------------------------------------
// NVTX is not enabled
//-------------------------------------------

#else

#define NVTX_PUSH(str,idx)
#define NVTX_POP()

#endif

#endif // MGONGPUNVTX_H 1
