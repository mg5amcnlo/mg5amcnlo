#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Apr 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2025).
# Integrated with the MadGraph7 project in Feb 2026.

set +x # not verbose
set -e # fail on error

scrdir=$(cd $(dirname $0); pwd)
bckend=$(basename $(cd $scrdir; cd ..; pwd)) # cudacpp or alpaka
topdir=$(cd $scrdir; cd ../../..; pwd)

# Enable OpenMP in tput tests? (#758)
###export USEOPENMP=1

# Debug channelid in MatrixElementKernelBase?
export MG5AMC_CHANNELID_DEBUG=1

function usage()
{
  echo "Usage: $0 <processes [-eemumu][-ggtt][-ggttg][-ggttgg][-ggttggg][-gqttq][-heftggbb][-susyggtt][-susyggt1t1][-smeftggtttt]> [-bldall|-nocuda|-cpponly|-cudaonly|-hiponly|-noneonly|-sse4only|-avx2only|-512yonly|-512zonly] [-sa] [-noalpaka] [-dblonly|-fltonly|-d_f|-dmf] [-inl|-inlonly] [-hrd|-hrdonly] [-common|-curhst] [-rmbhst|-bridge] [-noBlas|-blasOn] [-omp] [-makeonly|-makeclean|-makecleanonly|-dryrun] [-makej] [-3a3b] [-div] [-req] [-detailed] [-gtest(default)|-nogtest] [-scaling] [-v] [-dlp <dyld_library_path>]" # -nofpe is no longer supported
  exit 1
}

##########################################################################
# PART 0 - decode command line arguments
##########################################################################

procs=
eemumu=0
ggtt=0
ggttg=0
ggttgg=0
ggttggg=0
gqttq=0
heftggbb=0
susyggtt=0
susyggt1t1=0
smeftggtttt=0

suffs=".mad/"

omp=0
bblds=
alpaka=1

fptypes="m" # new default #995 (was "d")
helinls="0"
hrdcods="0"
rndgen=""
rmbsmp=""

blas="" # build with blas but disable it at runtime

maketype=
makej=

ab3=0
div=0
req=0
detailed=0
gtest=
scaling=0
###nofpe=0
verbose=0

dlp=

# Optional hack to build only the cudacpp plugin (without building the madevent code) in .mad directories
makef=
###makef="-f Makefile"

# (Was: workaround to allow 'make avxall' when '-avxall' is specified #536)
bbldsall="cuda hip cppnone cppsse4 cppavx2 cpp512y cpp512z"

if [ "$bckend" != "alpaka" ]; then alpaka=0; fi # alpaka mode is only available in the alpaka directory

while [ "$1" != "" ]; do
  if [ "$1" == "-eemumu" ]; then
    if [ "$eemumu" == "0" ]; then procs+=${procs:+ }$1; fi
    eemumu=1
    shift
  elif [ "$1" == "-ggtt" ]; then
    if [ "$ggtt" == "0" ]; then procs+=${procs:+ }$1; fi
    ggtt=1
    shift
  elif [ "$1" == "-ggttg" ]; then
    if [ "$ggttg" == "0" ]; then procs+=${procs:+ }$1; fi
    ggttg=1
    shift
  elif [ "$1" == "-ggttgg" ]; then
    if [ "$ggttgg" == "0" ]; then procs+=${procs:+ }$1; fi
    ggttgg=1
    shift
  elif [ "$1" == "-ggttggg" ]; then
    if [ "$ggttggg" == "0" ]; then procs+=${procs:+ }$1; fi
    ggttggg=1
    shift
  elif [ "$1" == "-gqttq" ]; then
    if [ "$gqttq" == "0" ]; then procs+=${procs:+ }$1; fi
    gqttq=1
    shift
  elif [ "$1" == "-heftggbb" ]; then
    if [ "$heftggbb" == "0" ]; then procs+=${procs:+ }$1; fi
    heftggbb=1
    shift
  elif [ "$1" == "-susyggtt" ]; then
    if [ "$susyggtt" == "0" ]; then procs+=${procs:+ }$1; fi
    susyggtt=1
    shift
  elif [ "$1" == "-susyggt1t1" ]; then
    if [ "$susyggt1t1" == "0" ]; then procs+=${procs:+ }$1; fi
    susyggt1t1=1
    shift
  elif [ "$1" == "-smeftggtttt" ]; then
    if [ "$smeftggtttt" == "0" ]; then procs+=${procs:+ }$1; fi
    smeftggtttt=1
    shift
  elif [ "$1" == "-sa" ]; then
    suffs=".sa/"
    shift
  elif [ "$1" == "-omp" ]; then
    omp=1
    shift
  elif [ "$1" == "-bldall" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="${bbldsall}"
    shift
  elif [ "$1" == "-nocuda" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="${bbldsall/cuda}"
    shift
  elif [ "$1" == "-cpponly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="${bbldsall/cuda}"
    bblds="${bblds/hip}"
    shift
  elif [ "$1" == "-cudaonly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="cuda"
    shift
  elif [ "$1" == "-hiponly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="hip"
    shift
  elif [ "$1" == "-noneonly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="none"
    shift
  elif [ "$1" == "-sse4only" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="sse4"
    shift
  elif [ "$1" == "-avx2only" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="avx2"
    shift
  elif [ "$1" == "-512yonly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="512y"
    shift
  elif [ "$1" == "-512zonly" ]; then
    if [ "${bblds}" != "" ]; then echo "ERROR! Incompatible option $1: backend builds are already defined as '$bblds'"; usage; fi
    bblds="512z"
    shift
  elif [ "$1" == "-noalpaka" ]; then
    alpaka=0
    shift
  elif [ "$1" == "-dblonly" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d"
    shift
  elif [ "$1" == "-fltonly" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="f"
    shift
  elif [ "$1" == "-d_f" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d f"
    shift
  elif [ "$1" == "-dmf" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d m f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d m f"
    shift
  elif [ "$1" == "-inl" ]; then
    if [ "${helinls}" == "1" ]; then echo "ERROR! Options -inl and -inlonly are incompatible"; usage; fi
    helinls="0 1"
    shift
  elif [ "$1" == "-inlonly" ]; then
    if [ "${helinls}" == "0 1" ]; then echo "ERROR! Options -inl and -inlonly are incompatible"; usage; fi
    helinls="1"
    shift
  elif [ "$1" == "-hrd" ]; then
    if [ "${hrdcods}" == "1" ]; then echo "ERROR! Options -hrd and -hrdonly are incompatible"; usage; fi
    hrdcods="0 1"
    shift
  elif [ "$1" == "-hrdonly" ]; then
    if [ "${hrdcods}" == "0 1" ]; then echo "ERROR! Options -hrd and -hrdonly are incompatible"; usage; fi
    hrdcods="1"
    shift
  elif [ "$1" == "-common" ]; then
    rndgen=" -${1}"
    shift
  elif [ "$1" == "-curhst" ]; then
    rndgen=" -${1}"
    shift
  ###elif [ "$1" == "-hirhst" ]; then
  ###  rndgen=" -${1}"
  ###  shift
  elif [ "$1" == "-rmbhst" ]; then
    rmbsmp=" -${1}"
    shift
  elif [ "$1" == "-bridge" ]; then
    rmbsmp=" -${1}"
    shift
  elif [ "$1" == "-noBlas" ]; then # build without blas
    if [ "${blas}" == "-blasOn" ]; then echo "ERROR! Options -noBlas and -blasOn are incompatible"; usage; fi
    blas=$1
    shift
  elif [ "$1" == "-blasOn" ]; then # build with blas and enable it at runtime
    if [ "${blas}" == "-noBlas" ]; then echo "ERROR! Options -noBlas and -blasOn are incompatible"; usage; fi
    blas=$1    
    shift
  elif [ "$1" == "-makeonly" ] || [ "$1" == "-makeclean" ] || [ "$1" == "-makecleanonly" ] || [ "$1" == "-dryrun" ]; then
    if [ "${maketype}" != "" ] && [ "${maketype}" != "$1" ]; then
      echo "ERROR! Options -makeonly, -makeclean, -makecleanonly and -dryrun are incompatible"; usage
    fi
    maketype="$1"
    shift
  elif [ "$1" == "-makej" ]; then
    ###makej=-j
    makej=-j5 # limit build parallelism to avoid "cudafe++ died due to signal 9" (#639)
    shift
  elif [ "$1" == "-3a3b" ]; then
    ab3=1
    shift
  elif [ "$1" == "-div" ]; then
    div=1
    shift
  elif [ "$1" == "-req" ]; then
    req=1
    shift
  elif [ "$1" == "-detailed" ]; then
    detailed=1
    shift
  elif [ "$1" == "-gtest" ]; then
    if [ "$gtest" == "0" ]; then
      echo "ERROR! Options -gtest and -nogtest are incompatible"; usage
    fi
    gtest=1
    shift
  elif [ "$1" == "-nogtest" ]; then
    if [ "$gtest" == "1" ]; then
      echo "ERROR! Options -gtest and -nogtest are incompatible"; usage
    fi
    gtest=0
    shift
  elif [ "$1" == "-scaling" ]; then
    scaling=1
    shift
  ###elif [ "$1" == "-nofpe" ]; then
  ###  nofpe=1
  ###  shift
  elif [ "$1" == "-v" ]; then
    verbose=1
    shift
  elif [ "$1" == "-dlp" ] && [ "$2" != "" ]; then
    dlp="$2"
    shift
    shift
  else
    echo "ERROR! Invalid option '$1'"
    usage
  fi
done
if [ "${gtest}" == "" ]; then gtest=1; fi
###echo procs=$procs
###exit 1

# Workaround for MacOS SIP (SystemIntegrity Protection): set DYLD_LIBRARY_PATH In subprocesses
if [ "${dlp}" != "" ]; then
  echo "export DYLD_LIBRARY_PATH=$dlp"
  export DYLD_LIBRARY_PATH=$dlp
fi

# FPEs are now enabled by default and cannot be disabled in the code (#831)
# The CUDACPP_RUNTIME_ENABLEFPE env variable (#733) is no longer used anywhere
###if [ "${nofpe}" == "0" ]; then
###  echo "export CUDACPP_RUNTIME_ENABLEFPE=on"
###  export CUDACPP_RUNTIME_ENABLEFPE=on
###else
###  echo "unset CUDACPP_RUNTIME_ENABLEFPE"
###  unset CUDACPP_RUNTIME_ENABLEFPE
###fi

# Check that at least one process has been selected
if [ "${eemumu}" == "0" ] && [ "${ggtt}" == "0" ] && [ "${ggttg}" == "0" ] && [ "${ggttgg}" == "0" ] && [ "${ggttggg}" == "0" ] && [ "${gqttq}" == "0" ] && [ "${heftggbb}" == "0" ] && [ "${susyggtt}" == "0" ] && [ "${susyggt1t1}" == "0" ] && [ "${smeftggtttt}" == "0" ]; then usage; fi

# Define the default bblds if none are defined (use ${bbldsall} which is translated back to 'make -bldall')
###if [ "${bblds}" == "" ]; then bblds="cuda avx2"; fi # this fails if cuda is not installed
if [ "${bblds}" == "" ]; then bblds="${bbldsall}"; fi # this succeeds if cuda is not installed because cudacpp.mk excludes it

# Use only the .auto process directories in the alpaka directory
if [ "$bckend" == "alpaka" ]; then
  echo "WARNING! alpaka directory: using .auto process directories only"
  suffs=".auto/"
fi

# Use only HRDCOD=0 in the alpaka directory (old epochX-golden2 code base)
if [ "$bckend" == "alpaka" ]; then
  echo "WARNING! alpaka directory: using HRDCOD=0 only"
  hrdcods="0"
fi

# Check whether Alpaka should and can be run
if [ "${alpaka}" == "1" ]; then
  if [ "${CUPLA_ROOT}" == "" ]; then echo "ERROR! CUPLA_ROOT is not set!"; exit 1; fi
  echo "CUPLA_ROOT=$CUPLA_ROOT"
  if [ ! -d "${CUPLA_ROOT}" ]; then echo "ERROR! $CUPLA_ROOT does not exist!"; exit 1; fi
  if [ "${ALPAKA_ROOT}" == "" ]; then echo "ERROR! ALPAKA_ROOT is not set!"; exit 1; fi
  echo "ALPAKA_ROOT=$ALPAKA_ROOT"
  if [ ! -d "${ALPAKA_ROOT}" ]; then echo "ERROR! $ALPAKA_ROOT does not exist!"; exit 1; fi
  if [ "${BOOSTINC}" == "" ]; then echo "ERROR! BOOSTINC is not set!"; exit 1; fi
  echo "BOOSTINC=$BOOSTINC"
  if [ ! -d "${BOOSTINC}" ]; then echo "ERROR! $BOOSTINC does not exist!"; exit 1; fi  
else
  export CUPLA_ROOT=none
fi

# Determine the O/S and the processor architecture for late decisions
unames=$(uname -s)
unamep=$(uname -p)

# Determine the working directory below topdir based on suff, bckend and proc
function showdir()
{
  if [ "${suff}" == ".mad/" ]; then
    if [ "${proc}" == "-eemumu" ]; then 
      dir=$topdir/epochX/${bckend}/ee_mumu${suff}SubProcesses/P1_epem_mupmum
    elif [ "${proc}" == "-ggtt" ]; then 
      dir=$topdir/epochX/${bckend}/gg_tt${suff}SubProcesses/P1_gg_ttx
    elif [ "${proc}" == "-ggttg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttg${suff}SubProcesses/P1_gg_ttxg
    elif [ "${proc}" == "-ggttgg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttgg${suff}SubProcesses/P1_gg_ttxgg
    elif [ "${proc}" == "-ggttggg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttggg${suff}SubProcesses/P1_gg_ttxggg
    elif [ "${proc}" == "-gqttq" ]; then 
      ###dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_gu_ttxu
      dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_gux_ttxux # only 1 out of 2 for now
    elif [ "${proc}" == "-heftggbb" ]; then 
      dir=$topdir/epochX/${bckend}/heft_gg_bb${suff}SubProcesses/P1_gg_bbx
    elif [ "${proc}" == "-susyggtt" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_tt${suff}SubProcesses/P1_gg_ttx
    elif [ "${proc}" == "-susyggt1t1" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_t1t1${suff}SubProcesses/P1_gg_t1t1x
    elif [ "${proc}" == "-smeftggtttt" ]; then 
      dir=$topdir/epochX/${bckend}/smeft_gg_tttt${suff}SubProcesses/P1_gg_ttxttx
    fi
  else
    if [ "${proc}" == "-eemumu" ]; then 
      dir=$topdir/epochX/${bckend}/ee_mumu${suff}SubProcesses/P1_Sigma_sm_epem_mupmum
    elif [ "${proc}" == "-ggtt" ]; then 
      dir=$topdir/epochX/${bckend}/gg_tt${suff}SubProcesses/P1_Sigma_sm_gg_ttx
    elif [ "${proc}" == "-ggttg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttg${suff}SubProcesses/P1_Sigma_sm_gg_ttxg
    elif [ "${proc}" == "-ggttgg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttgg${suff}SubProcesses/P1_Sigma_sm_gg_ttxgg
    elif [ "${proc}" == "-ggttggg" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttggg${suff}SubProcesses/P1_Sigma_sm_gg_ttxggg
    elif [ "${proc}" == "-gqttq" ]; then 
      ###dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_Sigma_sm_gu_ttxu
      dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_Sigma_sm_gux_ttxux # only 1 out of 2 for now
    elif [ "${proc}" == "-heftggbb" ]; then 
      dir=$topdir/epochX/${bckend}/heft_gg_bb${suff}SubProcesses/P1_Sigma_heft_gg_bbx
    elif [ "${proc}" == "-susyggtt" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_tt${suff}SubProcesses/P1_Sigma_MSSM_SLHA2_gg_ttx
    elif [ "${proc}" == "-susyggt1t1" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_t1t1${suff}SubProcesses/P1_Sigma_MSSM_SLHA2_gg_t1t1x
    elif [ "${proc}" == "-smeftggtttt" ]; then 
      dir=$topdir/epochX/${bckend}/smeft_gg_tttt${suff}SubProcesses/P1_Sigma_SMEFTsim_topU3l_MwScheme_UFO_gg_ttxttx
    fi
  fi
  echo $dir
}

echo MADGRAPH_CUDA_ARCHITECTURE=${MADGRAPH_CUDA_ARCHITECTURE}
echo MADGRAPH_HIP_ARCHITECTURE=${MADGRAPH_HIP_ARCHITECTURE}

###echo -e "\n********************************************************************************\n"
printf "\n"

##########################################################################
# PART 1 - compile the list of the executables which should be run
##########################################################################

dirs=
for proc in $procs; do
  for suff in $suffs; do
    dirs+=${dirs:+ }$(showdir)
  done
done
if [ "$dirs" == "" ]; then echo "ERROR! no valid directories found?"; exit 1; fi  
###echo dirs=$dirs

exes=
for dir in $dirs; do
  
  #=====================================
  # ALPAKA (epochX - manual/auto)
  #=====================================
  for hrdcod in $hrdcods; do
    hrdsuf=_hrd${hrdcod}
    if [ "$bckend" == "alpaka" ]; then hrdsuf=""; fi
    for helinl in $helinls; do
      for fptype in $fptypes; do
        if [ "${alpaka}" == "1" ]; then
          exes="$exes ${dir}/build.none_${fptype}_inl${helinl}${hrdsuf}/alpcheck.exe"
        fi
      done
    done
  done
  
  #=====================================
  # CUDA (epochX - manual/mad)
  # C++  (epochX - manual/mad)
  #=====================================
  for hrdcod in $hrdcods; do
    hrdsuf=_hrd${hrdcod}
    if [ "$bckend" == "alpaka" ]; then hrdsuf=""; fi
    for helinl in $helinls; do
      for fptype in $fptypes; do
        for bbld in cuda hip none sse4 avx2 512y 512z; do
          if [ "${bblds}" == "${bbldsall}" ] || [ "${bblds/${bbld}}" != "${bblds}" ]; then 
            if [ "${bbld}" == "cuda" ] || [ "${bbld}" == "hip" ]; then
              exes="$exes $dir/build.${bbld}_${fptype}_inl${helinl}${hrdsuf}/check_${bbld}.exe"
            else
              exes="$exes $dir/build.${bbld}_${fptype}_inl${helinl}${hrdsuf}/check_cpp.exe"
            fi
          fi
        done
      done
    done
  done

done
###echo "exes=$exes"; exit 1

##########################################################################
# PART 2 - build the executables which should be run
##########################################################################

if [ "${blas}" == "-noBlas" ]; then
  export HASBLAS=hasNoBlas
else
  export HASBLAS=hasBlas
fi
echo HASBLAS=${HASBLAS}

unset GTEST_ROOT
unset LOCALGTEST

if [ "${maketype}" == "-dryrun" ]; then

  printf "DRYRUN: SKIP MAKE\n\n"

else

  # Iterate over all directories (the first one will build googletest)
  gtestlibs=0
  for dir in $dirs; do
    bblds_dir=${bblds}
    if [ "${dir/\/gg_ttggg${suff}}" != ${dir} ]; then bblds_dir=${bblds/hip}; fi # skip ggttggg builds on HIP #933
    ###echo "Building in $dir" # FIXME: add a check that this $dir exists
    export USEBUILDDIR=1
    pushd $dir >& /dev/null
    echo "Building in $(pwd)"
    if [ "${maketype}" != "-makecleanonly" ] && [ "${gtestlibs}" == "0" ]; then
      # Build googletest once and for all to avoid issues in parallel builds
      # NB1: $topdir/test is NO LONGER RELEVANT and googletest must be built from one specific process
      # NB2: CXXNAMESUFFIX must be set by cudacpp.mk, so googletest must be built from one P1 directory
      gtestlibs=1
      make -f cudacpp.mk gtestlibs
    fi
    if [ "${maketype}" == "-makeclean" ]; then make cleanall; echo; fi
    if [ "${maketype}" == "-makecleanonly" ]; then make cleanall; echo; continue; fi
    for hrdcod in $hrdcods; do
      export HRDCOD=$hrdcod
      for helinl in $helinls; do
	export HELINL=$helinl
	for fptype in $fptypes; do
          export FPTYPE=$fptype
          if [ "${bblds_dir}" == "${bbldsall}" ]; then
            make ${makef} ${makej} bldall; echo # (was: allow 'make avxall' again #536)
          else
            for bbld in ${bblds_dir}; do
              make ${makef} ${makej} BACKEND=${bbld}; echo
            done
          fi
	done
      done
    done
    popd >& /dev/null
    export USEBUILDDIR=
    export HRDCOD=
    export HELINL=
    export FPTYPE=
  done

  if [ "${maketype}" == "-makecleanonly" ]; then printf "MAKE CLEANALL COMPLETED\n"; exit 0; fi
  if [ "${maketype}" == "-makeonly" ]; then printf "MAKE COMPLETED\n"; exit 0; fi

fi
  
##########################################################################
# PART 3 - run all the executables which should be run
##########################################################################

if [ "${maketype}" != "-dryrun" ]; then
  printf "DATE: $(date '+%Y-%m-%d_%H:%M:%S')\n\n"
fi

echo HASBLAS=${HASBLAS}

if [ "${blas}" == "-blasOn" ]; then
  export CUDACPP_RUNTIME_BLASCOLORSUM=1
else
  unset CUDACPP_RUNTIME_BLASCOLORSUM
fi
echo CUDACPP_RUNTIME_BLASCOLORSUM=${CUDACPP_RUNTIME_BLASCOLORSUM}

unset CUDACPP_RUNTIME_CUBLASTF32TENSOR
echo CUDACPP_RUNTIME_CUBLASTF32TENSOR=${CUDACPP_RUNTIME_CUBLASTF32TENSOR}

function runExe() {
  exe1=$1
  args="$2"
  args="$args$rndgen$rmbsmp"
  echo "runExe $exe1 $args OMP=$OMP_NUM_THREADS"
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  pattern="Process|fptype_sv|OMP threads|EvtsPerSec\[MECalc|MeanMatrix|FP precision|TOTAL       :"
  # Optionally add other patterns here for some specific configurations (e.g. clang)
  if [ "${exe1%%/check_cuda*}" != "${exe1}" ] || [ "${exe1%%/check_hip*}" != "${exe1}" ]; then pattern="${pattern}|EvtsPerSec\[Matrix"; fi
  pattern="${pattern}|Workflow"
  ###pattern="${pattern}|BLASCOLORSUM"
  ###pattern="${pattern}|CUCOMPLEX"
  ###pattern="${pattern}|COMMON RANDOM|CURAND HOST \(CUDA"
  pattern="${pattern}|ERROR"
  pattern="${pattern}|WARNING"
  pattern="${pattern}|Floating Point Exception"
  pattern="${pattern}|EvtsPerSec\[Rmb" # TEMPORARY! for rambo timing tests
  pattern="${pattern}|EvtsPerSec\[Matrix" # TEMPORARY! OLD C++/CUDA CODE
  if [ "${ab3}" == "1" ]; then pattern="${pattern}|3a|3b"; fi
  if [ "${req}" == "1" ]; then pattern="${pattern}|memory layout"; fi
  if perf stat -d date >& /dev/null; then # perf exists and CAP_SYS_ADMIN allows users to use it
    # -- Newer version using perf stat
    pattern="${pattern}|instructions|cycles"
    pattern="${pattern}|elapsed"
    if [ "${detailed}" == "1" ]; then pattern="${pattern}|#"; fi
    if [ "${verbose}" == "1" ]; then set -x; fi
    ###perf stat -d $exe1 $args 2>&1 | grep -v "Performance counter stats"
    perf stat -d $exe1 $args 2>&1 | egrep "(${pattern})" | grep -v "Performance counter stats" |& sed 's/.*rocdevice.cpp.*Aborting.*/rocdevice.cpp: Aborting/'
    set +x
  else
    # -- Older version using time
    # For TIMEFORMAT see https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html
    if [ "${verbose}" == "1" ]; then set -x; fi
    TIMEFORMAT=$'real\t%3lR' && time $exe1 $args 2>&1 | egrep "(${pattern})"
    set +x
  fi
}

function runTest() {
  exe1=$1
  echo "runTest $exe1"
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  pattern="PASS|FAIL"
  ###pattern="${pattern}|BLASCOLORSUM"
  pattern="${pattern}|ERROR"
  pattern="${pattern}|WARNING"
  pattern="${pattern}|Floating Point Exception"
  pattern="${pattern}|MEK"
  if [ "${verbose}" == "1" ]; then set -x; fi
  $exe1 2>&1 | egrep "(${pattern})" | sed "s/MEK 0x.* (/MEK (/"
  set +x
}

function cmpExe() {
  exe1=$1
  exef=${exe1/\/check//fcheck}
  argsf="2 64 2"
  args="--common -p ${argsf}"
  echo "cmpExe $exe1 $args"
  echo "cmpExe $exef $argsf"
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  if [ "${exe1%%/check_cuda*}" != "${exe1}" ] || [ "${exe1%%/check_hip*}" != "${exe1}" ]; then tag="/GPU)"; else tag="/C++) "; fi
  tmp1=$(mktemp)
  tmp2=$(mktemp)
  if ! ${exe1} ${args} 2>${tmp2} >${tmp1}; then
    echo "ERROR! C++ calculation (C++${tag} failed"; exit 1 # expose FPE crash #1003 on HIP
  fi
  me1=$(cat ${tmp1} | grep MeanMatrix | awk '{print $4}'); cat ${tmp2}
  ###cat ${tmp1} | grep BLASCOLORSUM
  if ! ${exef} ${argsf} 2>${tmp2} >${tmp1}; then
    echo "ERROR! Fortran calculation (F77${tag} failed"; exit 1
  fi
  me2=$(cat ${tmp1} | grep Average | awk '{print $4}'); cat ${tmp2}
  ###cat ${tmp1} | grep BLASCOLORSUM
  echo -e "Avg ME (C++${tag}   = ${me1}\nAvg ME (F77${tag}   = ${me2}"
  if [ "${me2}" == "NaN" ]; then
    echo "ERROR! Fortran calculation (F77${tag} returned NaN"; exit 1
  elif [ "${me2}" == "" ]; then
    echo "ERROR! Fortran calculation (F77${tag} crashed"; exit 1
  elif [ "${me1}" == "" ]; then
    echo "ERROR! C++ calculation (C++${tag} crashed"; exit 1
  else
    # NB skip python comparison if Fortran returned NaN or crashed, otherwise python returns an error status and the following tests are not executed
    python3 -c "me1=${me1}; me2=${me2}; reldif=abs((me2-me1)/me1); print('Relative difference =', reldif); ok = reldif <= 5E-3; print ( '%s (relative difference %s 5E-3)' % ( ('OK','<=') if ok else ('ERROR','>') ) )" 2>&1
  fi
}

# Profile #registers and %divergence only
function runNcu() {
  if ! ncu -v > /dev/null 2>&1; then return; fi
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  exe1=$1
  args="$2"
  args="$args$rndgen$rmbsmp"
  echo "runNcu $exe1 $args"
  ###echoblas=1
  kernels="calculate_jamps color_sum_kernel"
  ###if [ "${CUDACPP_RUNTIME_BLASCOLORSUM}" == "1" ]; then kernels="$kernels kernel"; fi # heavy to profile...
  ###if [ "${CUDACPP_RUNTIME_BLASCOLORSUM}" == "1" ]; then kernels="$kernels regex:gemm"; fi # output to be improved...
  for kernel in $kernels; do
    if [ "${verbose}" == "1" ]; then set -x; fi
    #$(which ncu) --metrics launch__registers_per_thread,sm__sass_average_branch_targets_threads_uniform.pct --target-processes all --kernel-id "::${kernel}:" --kernel-name-base function $exe1 $args | egrep '(calculate_jamps|registers| sm)' | tr "\n" " " | awk '{print $1, $2, $3, $15, $17; print $1, $2, $3, $18, $20$19}'
    set +e # do not fail on error
    out=$($(which ncu) --metrics launch__registers_per_thread,sm__sass_average_branch_targets_threads_uniform.pct --target-processes all --kernel-id "::${kernel}:" --kernel-name-base function $exe1 $args)
    echo "$out" | egrep '(ERROR|WARNING)' # NB must escape $out in between quotes
    ###if [ "${echoblas}" == "1" ]; then echo "$out" | egrep '(BLASCOLORSUM)'; echoblas=0; fi
    set -e # fail on error (after ncu and after egrep!)
    out=$(echo "${out}" | egrep "(${kernel}|registers| sm)" | tr "\n" " ") # NB must escape $out in between quotes
    echo $out | awk -v key1="launch__registers_per_thread" '{val1="N/A"; for (i=1; i<=NF; i++){if ($i==key1 && $(i+1)!="(!)") val1=$(i+2)}; print $1, $2, $3, key1, val1}'
    echo $out | awk -v key1="sm__sass_average_branch_targets_threads_uniform.pct" '{val1="N/A"; for (i=1; i<=NF; i++){if ($i==key1 && $(i+1)!="(!)") val1=$(i+2)$(i+1)}; print $1, $2, $3, key1, val1}'
    set +x
  done
}

# Profile divergence metrics more in detail
# See https://www.pgroup.com/resources/docs/18.10/pdf/pgi18profug.pdf
# See https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/branchstatistics.htm
# See https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/sourcelevel/divergentbranch.htm
function runNcuDiv() {
  if ! ncu -v > /dev/null 2>&1; then return; fi
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  exe1=$1
  args="-p 1 32 1"
  args="$args$rndgen$rmbsmp"
  ###echo "runNcuDiv $exe1 $args"
  if [ "${verbose}" == "1" ]; then set -x; fi
  ###$(which ncu) --query-metrics $exe1 $args
  ###$(which ncu) --metrics regex:.*branch_targets.* --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args
  ###$(which ncu) --metrics regex:.*stalled_barrier.* --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args
  ###$(which ncu) --metrics sm__sass_average_branch_targets_threads_uniform.pct,smsp__warps_launched.sum,smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_uniform.sum --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args | egrep '(calculate_jamps| sm)' | tr "\n" " " | awk '{printf "%29s: %-51s %s\n", "", $18, $19; printf "%29s: %-51s %s\n", "", $22, $23; printf "%29s: %-51s %s\n", "", $20, $21; printf "%29s: %-51s %s\n", "", $24, $26}'
  #$(which ncu) --metrics sm__sass_average_branch_targets_threads_uniform.pct,smsp__warps_launched.sum,smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_uniform.sum,smsp__sass_branch_targets.sum.per_second,smsp__sass_branch_targets_threads_divergent.sum.per_second,smsp__sass_branch_targets_threads_uniform.sum.per_second --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args | egrep '(calculate_jamps| sm)' | tr "\n" " " | awk '{printf "%29s: %-51s %-10s %s\n", "", $18, $19, $22$21; printf "%29s: %-51s %-10s %s\n", "", $28, $29, $32$31; printf "%29s: %-51s %-10s %s\n", "", $23, $24, $27$26; printf "%29s: %-51s %s\n", "", $33, $35}'
  out=$($(which ncu) --metrics sm__sass_average_branch_targets_threads_uniform.pct,smsp__warps_launched.sum,smsp__sass_branch_targets.sum,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_uniform.sum,smsp__sass_branch_targets.sum.per_second,smsp__sass_branch_targets_threads_divergent.sum.per_second,smsp__sass_branch_targets_threads_uniform.sum.per_second --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args | egrep '(calculate_jamps| sm)' | tr "\n" " ")
  ###echo $out
  echo $out | awk -v key1="smsp__sass_branch_targets.sum" '{key2=key1".per_second"; val1="N/A"; val2=""; for (i=1; i<=NF; i++){if ($i==key1 && $(i+1)!="(!)") val1=$(i+1); if ($i==key2 && $(i+1)!="(!)") val2=$(i+2)$(i+1)}; printf "%29s: %-51s %-10s %s\n", "", key1, val1, val2}'
  echo $out | awk -v key1="smsp__sass_branch_targets_threads_uniform.sum" '{key2=key1".per_second"; val1="N/A"; val2=""; for (i=1; i<=NF; i++){if ($i==key1 && $(i+1)!="(!)") val1=$(i+1); if ($i==key2 && $(i+1)!="(!)") val2=$(i+2)$(i+1)}; printf "%29s: %-51s %-10s %s\n", "", key1, val1, val2}'
  echo $out | awk -v key1="smsp__sass_branch_targets_threads_divergent.sum" '{key2=key1".per_second"; val1="N/A"; val2=""; for (i=1; i<=NF; i++){if ($i==key1 && $(i+1)!="(!)") val1=$(i+1); if ($i==key2 && $(i+1)!="(!)") val2=$(i+2)$(i+1)}; printf "%29s: %-51s %-10s %s\n", "", key1, val1, val2}'
  echo $out | awk -v key="smsp__warps_launched.sum" '{val1="N/A"; for (i=1; i<=NF; i++){if ($i==key && $(i+1)!="(!)") val1=$(i+2)}; printf "%29s: %-51s %s\n", "", key, val1}'
  set +x
}

# Profiles sectors and requests
function runNcuReq() {
  if ! ncu -v > /dev/null 2>&1; then return; fi
  if [ "${maketype}" == "-dryrun" ]; then return; fi
  exe1=$1
  ncuArgs="$2"
  ncuArgs="$ncuArgs$rndgen$rmbsmp"
  if [ "${verbose}" == "1" ]; then set -x; fi
  for args in "-p 1 1 1" "-p 1 4 1" "-p 1 8 1" "-p 1 32 1" "$ncuArgs"; do
    ###echo "runNcuReq $exe1 $args"
    # NB This will print nothing if $args are invalid (eg "-p 1 4 1" when neppR=8)
    $(which ncu) --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,launch__registers_per_thread,sm__sass_average_branch_targets_threads_uniform.pct --target-processes all --kernel-id "::calculate_jamps:" --kernel-name-base function $exe1 $args | egrep '(calculate_jamps|registers| sm|l1tex)' | tr "\n" " " | awk -vtag="[$args]" '{print $1, $2, $3, $16"s", $17";", $19"s", $20, tag}'
  done
  set +x
}

if nvidia-smi -L > /dev/null 2>&1; then
  gpuTxt="$(nvidia-smi -L | wc -l)x $(nvidia-smi -L | awk '{print $3,$4}' | sort -u)"
elif rocm-smi -i > /dev/null 2>&1; then
  gpuTxt="$(rocm-smi --showproductname | grep 'Card series' | awk '{print $5,$6,$7}')"
else
  gpuTxt=none
fi
if [ "${unames}" == "Darwin" ]; then 
  cpuTxt=$(sysctl -h machdep.cpu.brand_string)
  cpuTxt=${cpuTxt/machdep.cpu.brand_string: }
elif [ "${unamep}" == "ppc64le" ]; then 
  cpuTxt=$(cat /proc/cpuinfo | grep ^machine | awk '{print substr($0,index($0,"Power"))", "}')$(cat /proc/cpuinfo | grep ^cpu | head -1 | awk '{print substr($0,index($0,"POWER"))}')
else
  cpuTxt=$(cat /proc/cpuinfo | grep '^model name' |& head -1 | awk '{i0=index($0,"Intel"); if (i0==0) i0=index($0,"AMD"); i1=index($0," @"); if (i1>0) {print substr($0,i0,i1-i0)} else {print substr($0,i0)}}')
fi
echo -e "On $HOSTNAME [CPU: $cpuTxt] [GPU: $gpuTxt]:"

# Configure scaling tests
if [ "${scaling}" == "0" ]; then # no scaling tests (throughput tests only)
  exesSc=
elif [ "${scaling}" == "1" ]; then # scaling tests only (skip throughput tests)
  exesSc=$exes
  exes=
fi

# These two settings are needed by BMK containers: do not change them
BMKEXEARGS="" # if BMKEXEARGS is set, exeArgs is set equal to BMKEXEARGS, while exeArgs2 is set to ""
BMKMULTIPLIER=1 # the pre-defined numbers of iterations (including those in BMKEXEARGS) are multiplied by BMKMULTIPLIER

# (1) TRADITIONAL THROUGHPUT TESTS
###lastExe=
lastExeDir=
###echo "exes=$exes"
for exe in $exes; do
  ###echo EXE=$exe; continue
  exeArgs2=""
  ###if [ "$(basename $exe)" != "$lastExe" ]; then
  if [ "$(basename $(dirname $exe))" != "$lastExeDir" ]; then
    echo "========================================================================="
    ###lastExe=$(basename $exe)
    lastExeDir=$(basename $(dirname $exe))
  else
    echo "-------------------------------------------------------------------------"
  fi
  if [ ! -f $exe ]; then echo "Not found: $exe"; continue; fi
  if [ "${unamep}" != "x86_64" ]; then
    if [ "${exe/build.avx2}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
  elif [ "${unames}" == "Darwin" ]; then
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported on ${unames}"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported on ${unames}"; continue; fi
  elif [ "$(grep -m1 -c avx512vl /proc/cpuinfo)" != "1" ]; then
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported (no avx512vl in /proc/cpuinfo)"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported (no avx512vl in /proc/cpuinfo)"; continue; fi
  fi
  if [[ "${exe%%/check_cuda*}" != "${exe}" || "${exe%%/check_hip*}" != "${exe}" ]] && [ "$gpuTxt" == "none" ]; then pattern="${pattern}|EvtsPerSec\[Matrix"; fi
  if [ "${exe%%/heft_gg_bb*}" != "${exe}" ]; then 
    # For heftggbb, use the same settings as for ggtt
    exeArgs="-p 2048 256 2"
    ncuArgs="-p 2048 256 1"
  elif [ "${exe%%/smeft_gg_tttt*}" != "${exe}" ]; then 
    # For smeftggtttt, use the same settings as for ggttggg (may be far too short!)
    exeArgs="-p 1 256 2"
    ncuArgs="-p 1 256 1"
    # For smeftggtttt, use the same settings as for ggttggg (may be far too short!)
    exeArgs2="-p 64 256 1"
  elif [ "${exe%%/susy_gg_tt*}" != "${exe}" ]; then 
    # For susyggtt, use the same settings as for SM ggtt
    exeArgs="-p 2048 256 2"
    ncuArgs="-p 2048 256 1"
  elif [ "${exe%%/susy_gg_t1t1*}" != "${exe}" ]; then 
    # For susyggt1t1, use the same settings as for SM ggtt
    exeArgs="-p 2048 256 2"
    ncuArgs="-p 2048 256 1"
  elif [ "${exe%%/gq_ttq*}" != "${exe}" ]; then 
    # For gqttq, use the same settings as for ggttg
    exeArgs="-p 64 256 10"
    ncuArgs="-p 64 256 1"
    # For gqttq, use the same settings as for ggttg
    exeArgs2="-p 2048 256 1"
  elif [ "${exe%%/gg_ttggg*}" != "${exe}" ]; then 
    # For ggttggg: this is far too little for GPU (4.8E2), but it keeps the CPU to a manageble level (1sec with 512y)
    ###exeArgs="-p 1 256 1" # too short! see https://its.cern.ch/jira/browse/BMK-1056
    exeArgs="-p 1 256 2"
    ncuArgs="-p 1 256 1"
    # For ggttggg: on GPU test also "64 256" to reach the plateau (only ~5% lower than "2048 256": 1.18E4 vs 1.25E4 on cuda116/gcc102)
    exeArgs2="-p 64 256 1"
  elif [ "${exe%%/gg_ttgg*}" != "${exe}" ]; then 
    # For ggttgg (OLD): this is a good GPU middle point: tput is 1.5x lower with "32 256 1", only a few% higher with "128 256 1"
    exeArgs="-p 64 256 1"
    ncuArgs="-p 64 256 1"
    # For ggttgg (NEW): on GPU test both "64 256" and "2048 256" for ggttgg as the latter gives ~10% higher throughput on cuda110/gcc92
    ###exeArgs2="-p 2048 256 1" # Sep 2025: this aborts (and is not needed as the plateau is reached earlier) with helicity streams
  elif [ "${exe%%/gg_ttg*}" != "${exe}" ]; then 
    # For ggttg, as on ggttgg: this is a good GPU middle point: tput is 1.5x lower with "32 256 1", only a few% higher with "128 256 1"
    ###exeArgs="-p 64 256 1" # too short! see https://its.cern.ch/jira/browse/BMK-1056
    exeArgs="-p 64 256 10"
    ncuArgs="-p 64 256 1"
    # For ggttg, as on ggttgg: on GPU test both "64 256" and "2048 256" for ggttg as the latter gives ~10% higher throughput on cuda110/gcc92
    exeArgs2="-p 2048 256 1"
  elif [ "${exe%%/gg_tt*}" != "${exe}" ]; then 
    ###exeArgs="-p 2048 256 1" # too short! see https://its.cern.ch/jira/browse/BMK-1056
    exeArgs="-p 2048 256 2"
    ncuArgs="-p 2048 256 1"
  else # eemumu
    exeArgs="-p 2048 256 12"
    ncuArgs="-p 2048 256 1"
  fi
  if [ "$BMKEXEARGS" ]; then exeArgs="$BMKEXEARGS"; exeArgs2=""; fi
  exeArgs="${exeArgs% *} $((${exeArgs##* }*BMKMULTIPLIER))"
  if [ "${exeArgs2}" != "" ]; then exeArgs2="${exeArgs2% *} $((${exeArgs2##* }*BMKMULTIPLIER))"; fi  
  exeDir=$(dirname $exe)
  cd $exeDir/.. # workaround for reading '../../Cards/param_card.dat' without setting MG5AMC_CARD_PATH
  unset OMP_NUM_THREADS
  runExe $exe "$exeArgs"
  if [ "${exe%%/check_cpp*}" != "${exe}" ]; then 
    if [ "${maketype}" != "-dryrun" ]; then
      obj=${exe%%.exe}; obj=${obj/check/CPPProcess}.o; $scrdir/simdSymSummary.sh -stripdir ${obj} -dumptotmp # comment out -dumptotmp to keep full objdump
    fi
    if [ "${omp}" == "1" ]; then 
      echo "-------------------------------------------------------------------------"
      export OMP_NUM_THREADS=$(nproc --all)
      runExe $exe "$exeArgs"
      unset OMP_NUM_THREADS
    fi
  elif [[ "${exe%%/check_cuda*}" != "${exe}" || "${exe%%/check_hip*}" != "${exe}" ]] || [ "${exe%%/alpcheck*}" != "${exe}" ]; then
    echo "........................................................................."
    runNcu $exe "$ncuArgs"
    if [ "${div}" == "1" ]; then
      echo "........................................................................."
      runNcuDiv $exe
    fi
    if [ "${req}" == "1" ]; then
      echo "........................................................................."
      runNcuReq $exe "$ncuArgs"
    fi
    if [ "${exeArgs2}" != "" ]; then echo "........................................................................."; runExe $exe "$exeArgs2"; fi
  fi
  if [ "${gtest}" == "1" ]; then
    echo "-------------------------------------------------------------------------"
    # For simplicity a gtest runTest_xxx.exe is executed for each build where check_xxx.exe is executed
    exe2=${exe/check/runTest} # replace check_xxx.exe by runTest_xxx.exe
    runTest $exe2 2>&1
    if [ ${PIPESTATUS[0]} -ne "0" ]; then exit 1; fi 
  fi
  if [ "${bckend}" != "alpaka" ]; then
    echo "-------------------------------------------------------------------------"
    cmpExe $exe
  fi
done
###echo "========================================================================="

# (2) SCALING TESTS
lastExeDir=
for exe in $exesSc; do
  if [ "$(basename $(dirname $exe))" != "$lastExeDir" ]; then
    echo "========================================================================="
    lastExeDir=$(basename $(dirname $exe))
  else
    echo "-------------------------------------------------------------------------"
  fi
  echo "scalingTest $exe"
  if [ ! -f $exe ]; then echo "Not found: $exe"; continue; fi
  if [ "${unamep}" != "x86_64" ]; then
    if [ "${exe/build.avx2}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported on ${unamep}"; continue; fi
  elif [ "${unames}" == "Darwin" ]; then
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported on ${unames}"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported on ${unames}"; continue; fi
  elif [ "$(grep -m1 -c avx512vl /proc/cpuinfo)" != "1" ]; then
    if [ "${exe/build.512y}" != "${exe}" ]; then echo "$exe is not supported (no avx512vl in /proc/cpuinfo)"; continue; fi
    if [ "${exe/build.512z}" != "${exe}" ]; then echo "$exe is not supported (no avx512vl in /proc/cpuinfo)"; continue; fi
  fi
  exeDir=$(dirname $exe)
  cd $exeDir/.. # workaround for reading '../../Cards/param_card.dat' without setting MG5AMC_CARD_PATH
  unset OMP_NUM_THREADS
  # Scaling test with 256 threads per block
  if [[ "${exe%%/check_cuda*}" != "${exe}" || "${exe%%/check_hip*}" != "${exe}" ]]; then
    echo "### GPU: scaling test 256"
    for b in 1 2 4 8 16 32 64 128 256 512 1024; do ( $exe -p $b 256 1 | \grep "EvtsPerSec\[MECalcOnly\]" | awk -vb=$b "{printf \"%s %4d %3d\n\", \$5, b, 256}" ) |& sed "s/Gpu.*Assert/Assert/" |& sed 's/.*rocdevice.cpp.*Aborting.*/rocdevice.cpp: Aborting/'; done
    if [[ "${exe%%/check_hip*}" != "${exe}" ]]; then
      echo "### GPU: scaling test 64"
      for b in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do ( $exe -p $b 64 1 | \grep "EvtsPerSec\[MECalcOnly\]" | awk -vb=$b "{printf \"%s %4d %3d\n\", \$5, b, 64}" ) |& sed 's/.*rocdevice.cpp.*Aborting.*/rocdevice.cpp: Aborting/'; done # HIP (AMD GPU warp size is 32)
    else
      echo "### GPU: scaling test 32"
      for b in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192; do ( $exe -p $b 32 1 | \grep "EvtsPerSec\[MECalcOnly\]" | awk -vb=$b "{printf \"%s %4d %3d\n\", \$5, b, 32}" ) |& sed "s/Gpu.*Assert/Assert/"; done # CUDA (NVidia GPU warp size is 32)
    fi
  else
    echo "### CPU: scaling test 256"
    for b in 1 2 4; do ( $exe -p $b 256 1 | \grep "EvtsPerSec\[MECalcOnly\]" | awk -vb=$b "{printf \"%s %4d %3d\n\", \$5, b, 256}" ); done
    echo "### CPU: scaling test 32"
    for b in 1 2 4; do ( $exe -p $b 32 1 | \grep "EvtsPerSec\[MECalcOnly\]" | awk -vb=$b "{printf \"%s %4d %3d\n\", \$5, b, 32}" ); done
  fi
done
echo "========================================================================="

# Workaround for reading of data files
printf "\nTEST COMPLETED\n"
