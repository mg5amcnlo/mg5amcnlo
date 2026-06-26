#!/bin/bash

if [ "$1" == "-h" ] || [ "$3" != "" ]; then
  echo "Usage:   $0 [<commit1>] [<commit2>]"
  echo "Example: $0 32a5b40a 32a5b40a~"
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
exclude1='(EvtsPerSec|elapsed|cycles|instructions|TOTAL|DATE|\-\-\-|\+\+\+)'
# Path-dependent lines which may change on different machines or directories
exclude2='(Entering|Leaving|Building|HASCURAND|BACKEND|USEBUILDDIR|Nothing to be done)'
# Lines (interesting) which may change on different software versions
exclude3='(Symbols|Avg|Relative|MeanMatrixElemValue)'
# Lines (uninteresting) which change when missing some tests (no avx512, cuda, hip...)
exclude4='(runTest|runExe|cmpExe|runNcu|SIGMA|Workflow|FP|Internal|OMP|Symbols|PASSED|INFO|WARNING|PROF|\+OK|\-OK|CPU:|===|\.\.\.|\-$|\+$)'
# Lines (interesting) which show that some tests are missing (no avx512, cuda, hip...)
exclude5='(Not found|no avx512vl)'

echo "=== (*START*) git diff --no-ext-diff${commit1}${commit2} ${diffpath}"
git diff --no-ext-diff${commit1}${commit2} ${diffpath} |& egrep '^(\+|-|fatal)' \
  | egrep -v "${exclude1}" | egrep -v "${exclude2}" | egrep -v "${exclude3}" | egrep -v "${exclude4}" | egrep -v "${exclude5}"
echo "=== (**END**) git diff --no-ext-diff${commit1}${commit2} ${diffpath}"
