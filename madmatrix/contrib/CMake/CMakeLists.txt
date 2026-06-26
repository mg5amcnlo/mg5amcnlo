# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Roiser (Feb 2022) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Roiser (2022-2024).
# Integrated with the MadGraph7 project in Feb 2026.
# Minimal CMake configuration to build a functional CPU version

cmake_minimum_required(VERSION 3.22)

project(Madgraph4GPU)

include(${PROJECT_SOURCE_DIR}/CMake/Platforms.txt)
include(${PROJECT_SOURCE_DIR}/CMake/Compilers.txt)
include(${PROJECT_SOURCE_DIR}/CMake/Macros.txt)

set(PROJECT_GITROOT_DIR ${PROJECT_SOURCE_DIR}/../../..)

add_subdirectory(src)
add_subdirectory(SubProcesses)
