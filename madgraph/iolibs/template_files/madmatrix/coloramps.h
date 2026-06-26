// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Dec 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: O. Mattelaer, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COLORAMPS_H
#define COLORAMPS_H 1

#include "CPPProcess.h"


namespace mgOnGpu
{
  // Summary of numbering and indexing conventions for the relevant concepts (see issue #826 and PR #852)
  // - Diagram number (no variable) in [0, N_diagrams-1]: all values are allowed (N_diagrams distinct values)
  //   It follows the same C-style indexing of MadSpace
  //   => this number is displayed for information before each block of code in CPPProcess.cc
  // - Channel number ("channelId" in C, CHANNEL_ID in F) in [1, N_channels]: not all values are allowed (N_config <= N_channels <= N_diagrams distinct values)
  //   *** NB channelId is a diagram number: but ALL diagrams > N_channels, and also some < N_channels, do not have an associated SDE config number (#919) ***
  //   => this number (with F indexing as in ps/pdf output) is passed around as an API argument between cudacpp functions
  //   Note: the old API passes around a single CHANNEL_ID (and uses CHANNEL_ID=0 to indicate no-multichannel mode, but this is not used in coloramps.h),
  //   while the new API passes around an array of CHANNEL_ID's (and uses a NULL array pointer to indicate no-multichannel mode)
  // - Channel number in C indexing: "channelID - 1"
  //   => this number (with C indexing) is used as the index of the channel2iconfig array below
  // - Config number ("iconfig" in C, ICONFIG in F) in [1, N_config]: all values are allowed (N_config <= N_channels <= N_diagrams distinct values)
  // - Config number in C indexing: "iconfig - 1"
  //   => this number (with C indexing) is used as the index of the icolamp array below

  // The number of channels in the channel2iconfig array below
  // *** NB this is not guaranteed to be equal to ndiagrams, it can be lower as the remaining diagrams all have no associated SDE iconfig (#919) ***
  constexpr unsigned int nchannels = %(nb_diag)i;
#ifdef MGONGPUCPP_GPUIMPL
  static_assert( nchannels <= mg5amcGpu::CPPProcess::ndiagrams, "nchannels should be <= ndiagrams" ); // sanity check #910 and #919
#else
  static_assert( nchannels <= mg5amcCpu::CPPProcess::ndiagrams, "nchannels should be <= ndiagrams" ); // sanity check #910 and #919
#endif
  
  // Map channel to iconfig (e.g. "iconfig = channel2iconfig[channelId - 1]": input index uses C indexing, output index uses F indexing)
  // Note: iconfig=-1 indicates channels/diagrams with no associated iconfig for single-diagram enhancement in the MadEvent sampling algorithm (presence of 4-point interaction?)
  // This array has N_diagrams elements, but only N_config <= N_diagrams valid values (iconfig>0)
  // (NB: this array is created on the host in C++ code and on the device in GPU code, but a host copy is also needed in runTest #917)
  __device__ constexpr int channel2iconfig[%(nb_diag)i] = { // note: a trailing comma in the initializer list is allowed
%(channelc2iconfig_lines)s
  };

  // Host copy of the channel2iconfig array (this is needed in runTest #917)
#ifndef MGONGPUCPP_GPUIMPL
  constexpr const int* hostChannel2iconfig = channel2iconfig;
#else
  constexpr int hostChannel2iconfig[%(nb_diag)i] = { // note: a trailing comma in the initializer list is allowed
%(channelc2iconfig_lines)s
  };
#endif

  // The number N_config of channels/diagrams with an associated iconfig for single-diagram enhancement in the MadEvent sampling algorithm (#917)
  constexpr unsigned int nconfigSDE = %(nb_channel)s;

  // Map iconfig to the mask of allowed colors (e.g. "colormask = icolamp[iconfig - 1]": input index uses C indexing)
  // This array has N_config <= N_diagrams elements
  // (NB: this array is created on the host in C++ code and on the device in GPU code)
  __device__ constexpr bool icolamp[%(nb_channel)s][%(nb_color)s] = { // note: a trailing comma in the initializer list is allowed
%(is_LC)s
  };

}

#endif // COLORAMPS_H
