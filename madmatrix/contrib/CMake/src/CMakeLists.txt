# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Roiser (Feb 2022) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Roiser (2022-2024).
# Integrated with the MadGraph7 project in Feb 2026.

file(GLOB_RECURSE HEADERS "*.h")
add_library(mg5amc_common Parameters.cc read_slha.cc ${HEADERS})

# some XCode specific stuff to make the executable run
set_property(TARGET mg5amc_common PROPERTY XCODE_GENERATE_SCHEME TRUE)
