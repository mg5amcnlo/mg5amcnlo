#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Oct 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

set -e # fail on error

cd $(dirname $0)/..

bsm=
while [ "$1" != "" ]; do
  if [ "$1" == "-bsmonly" ] && [ "$bsm" != "-nobsm" ]; then
    bsm=$1
    shift
  elif [ "$1" == "-nobsm" ] && [ "$bsm" != "-bsmonly" ]; then
    bsm=$1
    shift
  else
    echo "Usage: $0 [-bsmonly|-nobsm]"
  fi
done

# SM processes (both mad and sa)

if [ "${bsm}" != "-bsmonly" ]; then

  ./CODEGEN/generateAndCompare.sh -q ee_mumu
  ./CODEGEN/generateAndCompare.sh -q ee_mumu --mad

  ./CODEGEN/generateAndCompare.sh -q gg_tt
  ./CODEGEN/generateAndCompare.sh -q gg_tt --mad

  ./CODEGEN/generateAndCompare.sh -q gg_ttg
  ./CODEGEN/generateAndCompare.sh -q gg_ttg --mad

  ./CODEGEN/generateAndCompare.sh -q gg_ttgg
  ./CODEGEN/generateAndCompare.sh -q gg_ttgg --mad

  ./CODEGEN/generateAndCompare.sh -q gg_ttggg
  ./CODEGEN/generateAndCompare.sh -q gg_ttggg --mad

  ./CODEGEN/generateAndCompare.sh -q gq_ttq
  ./CODEGEN/generateAndCompare.sh -q gq_ttq --mad

fi

# BSM processes

if [ "${bsm}" != "-nobsm" ]; then

  ./CODEGEN/generateAndCompare.sh -q heft_gg_bb
  ./CODEGEN/generateAndCompare.sh -q heft_gg_bb --mad

  ./CODEGEN/generateAndCompare.sh -q susy_gg_tt
  ./CODEGEN/generateAndCompare.sh -q susy_gg_tt --mad

  ./CODEGEN/generateAndCompare.sh -q susy_gg_t1t1
  ./CODEGEN/generateAndCompare.sh -q susy_gg_t1t1 --mad

  ./CODEGEN/generateAndCompare.sh -q smeft_gg_tttt
  ./CODEGEN/generateAndCompare.sh -q smeft_gg_tttt --mad

  ./CODEGEN/generateAndCompare.sh -q nobm_pp_ttW --mad

fi

# SM processes (mad only)

if [ "${bsm}" != "-bsmonly" ]; then

  ./CODEGEN/generateAndCompare.sh -q gg_tt01g --mad

  ./CODEGEN/generateAndCompare.sh -q pp_tt012j --mad

fi
