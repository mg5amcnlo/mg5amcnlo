#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Dec 2021) for the MG5aMC CUDACPP plugin.
# Integrated with the MadGraph7 project in Feb 2026.

direction=
eemumu=0
ggtt=0
ggttgg=0

function usage()
{
  echo "Usage: $0 <processes [-eemumu] [-ggtt] [-ggttgg]> <direction [-a2m|-m2a]>"
  exit 1
}

while [ "$1" != "" ]; do  
  if [ "$1" == "-a2m" ] || [ "$1" == "-m2a" ]; then
    if [ "$direction" != "" ] && [ "$direction" != "$1" ]; then echo "ERROR! Options $direction and $1 are incompatible"; usage; fi
    direction=$1
    shift
  elif [ "$1" == "-eemumu" ]; then
    eemumu=1
    shift
  elif [ "$1" == "-ggtt" ]; then
    ggtt=1
    shift
  elif [ "$1" == "-ggttgg" ]; then
    ggttgg=1
    shift
  else
    usage
  fi
done

# Check that a direction has been selected
if [ "${direction}" == "-a2m" ]; then
  src=auto; dst=manu
elif [ "${direction}" == "-m2a" ]; then
  src=manu; dst=auto
else
  usage
fi

# Check that at least one process has been selected
processes=
if [ "${ggttgg}" == "1" ]; then processes="ggttgg $processes"; fi
if [ "${ggtt}" == "1" ]; then processes="ggtt $processes"; fi
if [ "${eemumu}" == "1" ]; then processes="eemumu $processes"; fi
if [ "${processes}" == "" ]; then usage; fi

echo "direction: ${direction}"
echo "processes: ${processes}"

cd $(dirname $0)
for proc in ${processes}; do
  echo "------------------------------------------------------------------"
  for fs in logs_${proc}_${src}/log_${proc}_${src}_*; do
    fd=$(basename $fs)
    fd=logs_${proc}_${dst}/log_${proc}_${dst}_${fd#log_${proc}_${src}_}
    echo \cp -dpr ${fs} ${fd}
    \cp -dpr ${fs} ${fd}
  done
done
