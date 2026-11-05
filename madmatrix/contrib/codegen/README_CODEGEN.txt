# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Jun 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2023).
# Integrated with the MadGraph7 project in Feb 2026.

[Last updated Mon 19 June 2023]

This README contains **TEMPORARY** instructions for end users for generating CUDACPP code for MG5AMC.
It relies on using both the madgraph4gpu and mg5amcnlo github repositories.
Eventually, CUDACPP code generation will be fully integrated into the mg5amcnlo github repository.
NB: gridpack generation is not documented/supported yet.

0. Set up your favorite compilers

Example:
  echo $CXX
    /cvmfs/sft.cern.ch/lcg/releases/gcc/11.2.0-ad950/x86_64-centos8/bin/g++
  echo $FC
    /cvmfs/sft.cern.ch/lcg/releases/gcc/11.2.0-ad950/x86_64-centos8/bin/gfortran
  which nvcc
    /usr/local/cuda-12.0/bin/nvcc

1. Download mg5amcnlo

cd <userdir>
git clone -b gpucpp --single-branch git@github.com:mg5amcnlo/mg5amcnlo

export MG5AMC_HOME=$(pwd)/mg5amcnlo

2. Download madgraph4gpu

cd <userdir>
git clone -b master --single-branch git@github.com:madgraph5/madgraph4gpu.git

cd madgraph4gpu/epochX/cudacpp/

3. Generate your favorite process

./CODEGEN/generateAndCompare.sh --mad USER_gg_tt -c 'generate g g > t t~'

cd USER_gg_tt.mad

--------------------------------------------------------------------------------

EITHER: run the process directly

4a. Launch your process for FORTRAN

sed -i "s/.* = cudacpp_backend/FORTRAN = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
./bin/generate_events -f

4b. Launch your process for CPP (with AVX2)

sed -i "s/.* = cudacpp_backend/CPP = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
AVX=avx2 MG5AMC_CARD_PATH=$(pwd)/Cards ./bin/generate_events -f

4c. Launch your process for CUDA

sed -i "s/.* = cudacpp_backend/CUDA = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
MG5AMC_CARD_PATH=$(pwd)/Cards ./bin/generate_events -f

--------------------------------------------------------------------------------

OR: create gridpacks

5a. Create a gridpack using FORTRAN

sed -i "s/.* = gridpack/True = gridpack/" Cards/run_card.dat
sed -i "s/.* = cudacpp_backend/FORTRAN = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
./bin/generate_events -f

5b. Create a gridpack using CPP (with avx2)

sed -i "s/.* = gridpack/True = gridpack/" Cards/run_card.dat
sed -i "s/.* = cudacpp_backend/CPP = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
AVX=avx2 MG5AMC_CARD_PATH=$(pwd)/Cards ./bin/generate_events -f

5c. Create a gridpack using CUDA

sed -i "s/.* = gridpack/True = gridpack/" Cards/run_card.dat
sed -i "s/.* = cudacpp_backend/CUDA = cudacpp_backend/" Cards/run_card.dat
echo "r=21" > SubProcesses/randinit
MG5AMC_CARD_PATH=$(pwd)/Cards ./bin/generate_events -f

--------------------------------------------------------------------------------
