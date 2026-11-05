#!/bin/bash

if [ "$1" == "-h" ] || [ "$3" != "" ]; then
  echo "Usage:   $0 [<commit1>] [<commit2>]"
  echo "Example: $0 46356d6d 46356d6d~"
  exit 1
elif [ "$2" != "" ]; then
  commit1=" $1"
  commit2=" $2"
elif [ "$1" != "" ]; then
  commit1=" $1"
  commit2=""
else
  commit1=""
  commit2=""
fi 

cd $(dirname $0)
###diffpath=$(pwd)
diffpath="$(pwd)/logs*"
###diffpath=$(pwd)/logs_eemumu_mad/log_eemumu_mad_d_inl0_hrd0.txt 

# Performance lines which are expected to change
exclude1='(EvtsPerSec|COUNTERS|DATE|\-\-\-|\+\+\+|\-$|\+$)'
# Path-dependent lines which may change on different machines or directories
exclude2='(Entering|Leaving|BACKEND|USEBUILDDIR|Nothing to be done)'
# Lines (interesting) which may change on different software versions
exclude3='(XSECTION|UNWEIGHT|differ by less)'
# Lines (uninteresting) which change when missing some tests (unsupported platforms, early failures)
exclude4='(Executing|EXECUTE|Compare MAD|CUDACPP_RUNTIME|SIGMA|Workflow summary|OPENMPTH|NGOODHEL|are identical|INFO|WARNING|COMPLETED)'
# Lines (uninteresting) which change when missing some tests (unsupported platforms, early failures): input files
exclude5='(Number of events|Accuracy|Grid Adjustment|Suppress Amplitude|Helicity Sum|ICONFIG)'
# Lines (uninteresting) which appear during crashes from early failures
exclude6='(PDF set|alpha_s|Renormalization|Factorization|getting user|Enter number)'

echo "=== (*START*) git diff --no-ext-diff${commit1}${commit2} ${diffpath}"
git diff --no-ext-diff${commit1}${commit2} ${diffpath} |& egrep '^(\+|-|fatal)' \
  | egrep -v "${exclude1}" | egrep -v "${exclude2}" | egrep -v "${exclude3}" | egrep -v "${exclude4}" | egrep -v "${exclude5}" | egrep -v "${exclude6}"
echo "=== (**END**) git diff --no-ext-diff${commit1}${commit2} ${diffpath}"
