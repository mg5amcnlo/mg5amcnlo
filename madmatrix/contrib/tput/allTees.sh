#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Apr 2022) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2022-2025).
# Integrated with the MadGraph7 project in Feb 2026.

scrdir=$(cd $(dirname $0); pwd)

# By default, use the madevent+cudacpp version of code and tee scripts (use -sa to use the standalone version instead)
# By default, build and run all tests (use -makeonly to only build all tests)
opts=
suff=".mad"
makeclean=-makeclean

# By default, build and run all backends
# (AV private config: on itgold91 build and run only the C++ backends)
bblds=
if [ "$(hostname)" == "itgold91.cern.ch" ]; then bblds=-cpponly; fi

# Usage
function usage()
{
  echo "Usage (1): $0 [-short] [-e] [-sa] [-makeonly] [-nomakeclean] [-hip|-nocuda|-cpponly] [-bsmonly|-nobsm|-scalingonly|-blasonly|-blasandscalingonly]"
  echo "Run tests and check all logs"
  echo ""
  echo "Usage (2): $0 -checkonly"
  echo "Check existing logs without running any tests"
  exit 1
}

# Parse command line arguments
checkonly=0
ggttggg=-ggttggg
rndhst=-curhst
sm=1
bsm=1
scaling=1
blas=1
if [ "$1" == "-checkonly" ]; then
  # Check existing logs without running any tests?
  checkonly=1
  shift
  if [ "$1" != "" ]; then usage; fi
fi
while [ "${checkonly}" == "0" ] && [ "$1" != "" ]; do
  if [ "$1" == "-short" ]; then
    # Short (no ggttggg) or long version?
    ggttggg=
    shift
  elif [ "$1" == "-e" ]; then
    # Fail on error?
    set -e
    shift
  elif [ "$1" == "-sa" ]; then
    # Use standalone_cudacpp builds instead of madevent+cudacpp?
    opts+=" -sa"
    suff=".sa"
    shift
  elif [ "$1" == "-makeonly" ]; then
    # Only build all tests instead of building and running them?
    opts+=" -makeonly"
    shift
  elif [ "$1" == "-nomakeclean" ]; then
    # Skip -makeclean (e.g. for brand new generated/downloaded code)
    makeclean=
    shift
  elif [ "$1" == "-hip" ]; then
    if [ "${bblds}" != "" ] && [ "${bblds}" != "$1" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="$1"
    shift
  elif [ "$1" == "-nocuda" ]; then
    if [ "${bblds}" != "" ] && [ "${bblds}" != "$1" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="$1"
    shift
  elif [ "$1" == "-cpponly" ]; then
    if [ "${bblds}" != "" ] && [ "${bblds}" != "$1" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="$1"
    shift
  elif [ "$1" == "-bsmonly" ] && [ "${sm}${scaling}${bsm}${blas}" == "1111" ]; then
    sm=0
    bsm=1
    scaling=0
    blas=0
    shift
  elif [ "$1" == "-nobsm" ] && [ "${sm}${scaling}${bsm}${blas}" == "1111" ]; then
    sm=1
    bsm=0
    scaling=1
    blas=1
    shift
  elif [ "$1" == "-scalingonly" ] && [ "${sm}${scaling}${bsm}${blas}" == "1111" ]; then
    sm=0
    bsm=0
    scaling=1
    blas=0
    shift
  elif [ "$1" == "-blasonly" ] && [ "${blas}${scaling}${bsm}${blas}" == "1111" ]; then
    sm=0
    bsm=0
    scaling=0
    blas=1
    shift
  elif [ "$1" == "-blasandscalingonly" ] && [ "${blas}${scaling}${bsm}${blas}" == "1111" ]; then
    sm=0
    bsm=0
    scaling=1
    blas=1
    shift
  else
    usage
  fi
done

# Check logs
function checklogs()
{
  cd $scrdir/..
  # Print out any errors in the logs (exclude scaling logs)
  if ! egrep -i '(error|fault|failed)' ./tput/logs_*/*.txt; then echo "No errors found in logs"; fi
  # Print out any FPEs or '{ }' in the logs
  echo
  if ! egrep '(^Floating Point Exception|{ })' tput/logs* -r; then echo "No FPEs or '{ }' found in logs"; fi
  # Print out any aborts in the logs (exclude scaling logs)
  echo
  txt=$(grep Abort ./tput/logs_*/*.txt | sed "s|\:.*SubProcesses/P|: P|")
  if [ "${txt}" == "" ]; then
    echo "No aborts found in logs"
  else
    echo "${txt}"
  fi
  # Print out any asserts/aborts in scaling logs
  echo
  txt=$(egrep -i '(abort|assert)' ./tput/logs_*/*.scaling | sed "s|\:.*SubProcesses/P|: P|" | sort -u)
  if [ "${txt}" == "" ]; then
    echo "No aborts or asserts found in scaling logs"
  else
    echo "${txt}"
  fi

  # Print out the MEK channelid debugging output (except for '{ }')
  echo
  \grep MEK ${scrdir}/logs_*/* | sed "s|${scrdir}/logs_||" | grep -v '{ }' | sed 's|_mad.*DEBUG:||' | sort -u
}
if [ "${checkonly}" != "0" ]; then checklogs; exit 0; fi

# Define builds
if [ "$bblds" == "-nocuda" ]; then
  # Random numbers use common instead of curand
  rndhst=-common
  opts+=" -nocuda"
elif [ "$bblds" == "-cpponly" ]; then
  # Random numbers use common instead of curand
  rndhst=-common
  opts+=" -cpponly"
elif [ "$bblds" == "-hip" ]; then # NB: currently (Sep 2024) this is identical to -nocuda
  #### Random numbers use hiprand instead of curand?
  #### This needs ROCm 6.2 (see https://github.com/ROCm/hipRAND/issues/76)
  ###rndhst=-hirhst
  # Random numbers use common (not hiprand) instead of curand
  rndhst=-common
  opts+=" -nocuda"
fi

# This is a script to launch in one go all tests for the main physics processes in this repository
# (Initially it reproduced the logs in tput at the time of commit c0c276840, just before the large alphas PR #434)

cd $scrdir/..
started="STARTED  AT $(date)"

# (+36: 36/144) Six logs (double/mixed/float x hrd0/hrd1 x inl0) in each of the six SM processes [sm==1]
\rm -rf gg_ttggg${suff}/lib/build.none_*
cmd="./tput/teeThroughputX.sh -dmf -hrd -makej -eemumu -ggtt -ggttg -ggttgg -gqttq $ggttggg ${makeclean} ${opts}"
tmp1=$(mktemp)
if [ "${sm}" == "1" ]; then
  $cmd; status=$?
  ls -ltr ee_mumu${suff}/lib/build.none_*_inl0_hrd* gg_tt${suff}/lib/build.none_*_inl0_hrd* gg_tt*g${suff}/lib/build.none_*_inl0_hrd* | egrep -v '(total|\./|\.build|_common|^$)' > $tmp1
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended1="$cmd\nENDED(1) AT $(date) [Status=$status]"

# (+18: 54/144) Three scaling logs (double/mixed/float x hrd0 x inl0) in each of the six SM processes [scaling==1]
if [ "${scaling}" == "1" ]; then
  if [ "${sm}" == "1" ]; then
    cmd="./tput/teeThroughputX.sh -dmf -makej -eemumu -ggtt -ggttg -ggttgg -gqttq $ggttggg -scaling ${opts}" # no rebuild needed
    $cmd; status=$?
  else
    cmd="./tput/teeThroughputX.sh -dmf -makej -eemumu -ggtt -ggttg -ggttgg -gqttq $ggttggg -scaling ${makeclean} ${opts}" # this is the first build
    $cmd; status=$?
  fi
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended1sc="$cmd\nENDED(1-scaling) AT $(date) [Status=$status]"

# (+6: 60/144) Three extra logs (double/mixed/float x hrd0 x inl0 + blasOn) only in two of the six SM processes (rebuild may be needed) [blas==1]
if [ "${blas}" == "1" ]; then
  if [ "${sm}" == "1" ] || [ "${scaling}" == "1" ]; then
    cmd="./tput/teeThroughputX.sh -ggtt -ggttgg -dmf -blasOn ${opts}" # no rebuild needed
    $cmd; status=$?
  else
    cmd="./tput/teeThroughputX.sh -ggtt -ggttgg -dmf -blasOn ${makeclean} ${opts}" # this is the first build
    $cmd; status=$?
  fi
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended2="$cmd\nENDED(2) AT $(date) [Status=$status]"

# (+12: 72/144) Three scaling logs (double/mixed/float x hrd0 x inl0 + blasOn) only in four of the six SM processes [blas==1 || scaling==1]
if [ "${blas}" == "1" ] || [ "${scaling}" == "1" ]; then
  cmd="./tput/teeThroughputX.sh -ggtt -ggttg -ggttgg -ggttggg -dmf -blasOn -scaling ${opts}" # no rebuild needed
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended2sc="$cmd\nENDED(2-scaling) AT $(date) [Status=$status]"

# (+12: 84/144) Four extra logs (double/float x hrd0/hrd1 x inl1) only in three of the six SM processes [sm==1]
\rm -rf gg_ttg${suff}/lib/build.none_*
\rm -rf gg_ttggg${suff}/lib/build.none_*
cmd="./tput/teeThroughputX.sh -d_f -hrd -makej -eemumu -ggtt -ggttgg -inlonly ${makeclean} ${opts}"
tmp3=$(mktemp)
if [ "${sm}" == "1" ]; then
  $cmd; status=$?
  ls -ltr ee_mumu${suff}/lib/build.none_*_inl1_hrd* gg_tt${suff}/lib/build.none_*_inl1_hrd* gg_tt*g${suff}/lib/build.none_*_inl1_hrd* | egrep -v '(total|\./|\.build|_common|^$)' > $tmp3
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended3="$cmd\nENDED(3) AT $(date) [Status=$status]"

# (+12: 96/144) Two extra logs (double/float x hrd0 x inl0 + bridge) in all six SM processes (rebuild from cache) [sm==1]
cmd="./tput/teeThroughputX.sh -makej -eemumu -ggtt -ggttg -gqttq -ggttgg $ggttggg -d_f -bridge ${makeclean} ${opts}"
if [ "${sm}" == "1" ]; then
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended4="$cmd\nENDED(4) AT $(date) [Status=$status]"

# (+6: 102/144) Two extra logs (double/float x hrd0 x inl0 + rmbhst) only in three of the six SM processes (no rebuild needed) [sm==1]
cmd="./tput/teeThroughputX.sh -eemumu -ggtt -ggttgg -d_f -rmbhst ${opts}"
if [ "${sm}" == "1" ]; then
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended5="$cmd\nENDED(5) AT $(date) [Status=$status]"

# (+6: 108/144) Two extra logs (double/float x hrd0 x inl0 + rndhst) only in three of the six SM processes (no rebuild needed) [sm==1]
cmd="./tput/teeThroughputX.sh -eemumu -ggtt -ggttgg -d_f ${rndhst} ${opts}"
if [ "${sm}" == "1" ] && [ "${rndhst}" != "-common" ]; then
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended6="$cmd\nENDED(6) AT $(date) [Status=$status]"

# (+6: 114/144) Two extra logs (double/float x hrd0 x inl0 + common) only in three of the six SM processes (no rebuild needed) [sm==1]
cmd="./tput/teeThroughputX.sh -eemumu -ggtt -ggttgg -d_f -common ${opts}"
if [ "${sm}" == "1" ]; then
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended7="$cmd\nENDED(7) AT $(date) [Status=$status]"

# (+6: 120/144) Three extra logs (double/float x hrd0 x inl0 + noBlas) only in two of the six SM processes (rebuild is needed) [blas==1]
cmd="./tput/teeThroughputX.sh -ggtt -ggttgg -dmf -noBlas ${makeclean} ${opts}"
if [ "${blas}" == "1" ]; then
  $cmd; status=$?
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended8="$cmd\nENDED(8) AT $(date) [Status=$status]"

# (+24: 144/144) Six extra logs (double/mixed/float x hrd0/hrd1 x inl0) only in the four BSM processes [bsm==1]
cmd="./tput/teeThroughputX.sh -dmf -hrd -makej -susyggtt -susyggt1t1 -smeftggtttt -heftggbb ${makeclean} ${opts}"
tmp9=$(mktemp)
if [ "${bsm}" == "1" ]; then
  $cmd; status=$?
  ls -ltr susy_gg_tt${suff}/lib/build.none_*_inl0_hrd* susy_gg_t1t1${suff}/lib/build.none_*_inl0_hrd* smeft_gg_tttt${suff}/lib/build.none_*_inl0_hrd* heft_gg_bb${suff}/lib/build.none_*_inl0_hrd* | egrep -v '(total|\./|\.build|_common|^$)' > $tmp9
else
  cmd="SKIP '$cmd'"; echo $cmd; status=$?
fi
ended9="$cmd\nENDED(9) AT $(date) [Status=$status]"

echo
echo "Build(1):"
cat $tmp1
echo
echo "Build(3):"
cat $tmp3
echo
echo "Build(9):"
cat $tmp9
echo
echo -e "$started"
echo -e "$ended1"
echo -e "$ended1sc"
echo -e "$ended2"
echo -e "$ended2sc"
echo -e "$ended3"
echo -e "$ended4"
echo -e "$ended5"
echo -e "$ended6"
echo -e "$ended7"
echo -e "$ended8"
echo -e "$ended9"

if [ "$ggttggg" == "" ]; then
  echo
  echo "To complete the test for ggttggg type:"
  echo "  ./tput/teeThroughputX.sh -dmf -hrd -makej -ggttggg ${makeclean} ${opts}"
  echo "  ./tput/teeThroughputX.sh -dmf -makej -ggttggg -scaling ${makeclean} ${opts}"
  echo "  ./tput/teeThroughputX.sh -makej -ggttggg -d_f -bridge ${makeclean} ${opts}"
fi

# Finally check all logs
echo
checklogs
