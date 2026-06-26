#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Oct 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2023-2024).
# Integrated with the MadGraph7 project in Feb 2026.
# Verbose script
###set -x

# Automatic exit on error
###set -e

# Hack to force "echo" to leave a blank line
# (NB use a function because aliases are not inherited in subshells)
# [The issue is similar to https://gitlab.com/gitlab-org/gitlab/-/issues/217231]
function ECHO() { echo " "; }

# Path to the top directory of madgraphgpu
# In the CI this would be simply $(pwd), but allow the script to be run also outside the CI
echo "Executing $0 $*"; ECHO
topdir=$(cd $(dirname $0)/../..; pwd)

# Enable OpenMP in the CI tests? (#758)
###export USEOPENMP=1

# Debug channelid in MatrixElementKernelBase?
export MG5AMC_CHANNELID_DEBUG=1

# Bypass known issues?
BYPASS_KNOWN_ISSUES=0 # do not bypass known issues (fail)
###BYPASS_KNOWN_ISSUES=1 # bypass known issues (do not fail)

#----------------------------------------------------------------------------------------------------------------------------------

# Code generation stage
function codegen() {
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: $(basename $0) <process>"; exit 1; fi
  proc=$1
  if [ "${proc%.mad}" == "${proc}" ] && [ "${proc%.sa}" == "${proc}" ]; then echo "Usage: $(basename $0) <process.mad|process.sa>"; exit 1; fi
  # Generate code and check clang formatting
  cd ${topdir}/epochX/cudacpp
  echo "Current directory is $(pwd)"
  ECHO 
  echo "*******************************************************************************"
  echo "*** code generation for ${proc}"
  echo "*******************************************************************************"
  if [ "${proc%.mad}" != "${proc}" ]; then
    ./CODEGEN/generateAndCompare.sh -q ${proc%.mad} --mad
  else
    ./CODEGEN/generateAndCompare.sh -q ${proc%.sa}
  fi
  # Check if there are any differences to the current repo
  ###compare=1 # enable comparison to current git repo
  compare=0 # disable comparison to current git repo
  if [ "${compare}" != "0" ] && [ "$(git ls-tree --name-only HEAD ${proc})" != "" ]; then
    ECHO
    echo "Compare newly generated code for ${proc} to that in the madgraph4gpu github repository"
    git checkout HEAD ${proc}/CODEGEN*.txt
    if [ "${proc%.mad}" != "${proc}" ]; then
      git checkout HEAD ${proc}/Cards/me5_configuration.txt
      git checkout HEAD ${proc}/Source/make_opts
    fi
    echo "git diff (start)"
    git diff --exit-code
    echo "git diff (end)"
  else
    ECHO
    echo "(SKIP comparison of newly generated code for ${proc} to that in the madgraph4gpu github repository)"
  fi
}

#----------------------------------------------------------------------------------------------------------------------------------

function setup_ccache {
  # Set up ccache environment
  export PATH=${topdir}/BIN:$PATH
  export CCACHE_DIR=${topdir}/CCACHE_DIR
}

#----------------------------------------------------------------------------------------------------------------------------------

# Before-build stage (analyse data retrieved from cache, download ccache executable and googletest if not retrieved from cache)
function before_build() {
  # Install and configure ccache
  if [ -d ${topdir}/DOWNLOADS ]; then
    echo "Directory ${topdir}/DOWNLOADS already exists (retrieved from cache)"
  else
    echo "Directory ${topdir}/DOWNLOADS does not exist: create it"
    mkdir ${topdir}/DOWNLOADS
    cd ${topdir}/DOWNLOADS
    echo "Current directory is $(pwd)"
    ECHO
    echo "wget -q https://github.com/ccache/ccache/releases/download/v4.8.3/ccache-4.8.3-linux-x86_64.tar.xz"
    wget -q https://github.com/ccache/ccache/releases/download/v4.8.3/ccache-4.8.3-linux-x86_64.tar.xz
    ECHO
    echo "tar -xvf ccache-4.8.3-linux-x86_64.tar.xz"
    tar -xvf ccache-4.8.3-linux-x86_64.tar.xz
  fi
  mkdir ${topdir}/BIN
  cd ${topdir}/BIN
  ln -sf ${topdir}/DOWNLOADS/ccache-4.8.3-linux-x86_64/ccache .
  # Set up ccache environment
  setup_ccache
  # Create the CCACHE_DIR directory if it was not retrieved from the cache
  ECHO
  if [ -d ${CCACHE_DIR} ]; then
    echo "Directory CCACHE_DIR=${CCACHE_DIR} already exists (retrieved from cache)"
  else
    echo "Directory CCACHE_DIR=${CCACHE_DIR} does not exist: create it"
    mkdir ${CCACHE_DIR}
  fi
  # Dump ccache status before the builds
  ECHO
  echo "ccache --version | head -1"
  ccache --version | head -1
  ECHO
  echo "CCACHE_DIR=${CCACHE_DIR}"
  echo "du -sm ${CCACHE_DIR}"
  du -sm ${CCACHE_DIR}
  ECHO
  echo "ccache -s (before the builds)"
  ccache -s
  # Check if googletest has already been installed and configured
  ECHO
  if [ -d ${topdir}/test/googletest ]; then
    echo "Directory ${topdir}/test/googletest already exists (retrieved from cache)"
    echo "ls ${topdir}/test/googletest (start)"
    ls ${topdir}/test/googletest
    echo "ls ${topdir}/test/googletest (end)"
  else
    echo "Directory ${topdir}/test/googletest does not exist: it will be created during the build"
  fi
}

#----------------------------------------------------------------------------------------------------------------------------------

# Build stage
function build() {
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: $(basename $0) <process>"; exit 1; fi
  proc=$1
  if [ "${proc%.mad}" == "${proc}" ] && [ "${proc%.sa}" == "${proc}" ]; then echo "Usage: $(basename $0) <process.mad|process.sa>"; exit 1; fi
  # Set up build environment
  setup_ccache
  export USECCACHE=1 # enable ccache in madgraph4gpu builds
  export CXX=g++ # set up CXX that is needed by cudacpp.mk
  ###ECHO; echo "$CXX --version"; $CXX --version
  export USEBUILDDIR=1
  # Iterate over P* directories and build
  cd ${topdir}/epochX/cudacpp/${proc}
  echo "Current directory is $(pwd)"
  echo "FPTYPE=${FPTYPE}"
  gtestlibs=0
  pdirs="$(ls -d SubProcesses/P*_*)"
  for pdir in ${pdirs}; do
    pushd $pdir >& /dev/null
    ECHO
    echo "*******************************************************************************"
    echo "*** build ${proc} ($(basename $(pwd)))"
    echo "*******************************************************************************"
    ECHO
    echo "Building in $(pwd)"
    if [ "${gtestlibs}" == "0" ]; then
      # Build googletest once and for all to avoid issues in parallel builds
      gtestlibs=1
      make -f cudacpp.mk gtestlibs
    fi
    # NB: 'make bldall' internally checks if 'which nvcc' and 'which hipcc' succeed before attempting to build cuda and hip
    make -j bldall
    popd >& /dev/null
  done
}

#----------------------------------------------------------------------------------------------------------------------------------

# After-build stage (analyse data to be saved in updated cache)
function after_build() {
  # Set up ccache environment
  setup_ccache
  # Dump ccache status after the builds
  ECHO
  echo "CCACHE_DIR=${CCACHE_DIR}"
  echo "du -sm ${CCACHE_DIR}"
  du -sm ${CCACHE_DIR}
  ECHO
  echo "ccache -s (after the builds)"
  ccache -s
  # Check contents of googletest
  ECHO
  echo "ls ${topdir}/test/googletest (start)"
  ls ${topdir}/test/googletest
  echo "ls ${topdir}/test/googletest (end)"
  # Check contents of build directories
  ECHO
  echo "ls -d ${topdir}/epochX/cudacpp/*.*/SubProcesses/P*_*/build* (start)"
  ls -d ${topdir}/epochX/cudacpp/*.*/SubProcesses/P*_*/build*
  echo "ls -d ${topdir}/epochX/cudacpp/*.*/SubProcesses/P*_*/build* (end)"
}

#----------------------------------------------------------------------------------------------------------------------------------

function runExe() {
  ECHO
  echo "Execute $*"
  if [ -f $1 ]; then $*; else echo "(SKIP missing $1)"; fi
}

# Tput_test stage (runTest.exe, check.exe, gcheck.exe)
function tput_test() {
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: $(basename $0) <process>"; exit 1; fi
  proc=$1
  if [ "${proc%.mad}" == "${proc}" ] && [ "${proc%.sa}" == "${proc}" ]; then echo "Usage: $(basename $0) <process.mad|process.sa>"; exit 1; fi
  # Iterate over P* directories and run tests
  cd ${topdir}/epochX/cudacpp/${proc}
  echo "Current directory is $(pwd)"
  echo "FPTYPE=${FPTYPE}"
  pdirs="$(ls -d SubProcesses/P*_*)"
  for pdir in ${pdirs}; do
    pushd $pdir >& /dev/null
    ECHO
    echo "*******************************************************************************"
    echo "*** tput_test ${proc} ($(basename $(pwd)))"
    echo "*******************************************************************************"
    ECHO
    echo "Testing in $(pwd)"
    ECHO
    # FIXME1: this is just a quick test, eventually port here tput tests from throughputX.sh
    # (could move some throughputX.sh functions to a separate script included both here and there)
    # FIXME2: handle all d/f/m, inl0/1, hrd0/1 etc...
    # FIXME3: add fcheck.exe tests
    unamep=$(uname -p)
    unames=$(uname -s)
    for backend in cuda hip cppnone cppsse4 cppavx2 cpp512y cpp512z; do
      echo "-------------------------------------------------------------------------------"
      echo "--- tput_test ${proc} ($(basename $(pwd))) BACKEND=${backend}" # debug #937
      echo "-------------------------------------------------------------------------------"
      ECHO
      # Skip GPU tests for NVidia and AMD unless nvcc and hipcc, respectively, are in PATH
      if ! nvcc --version &> /dev/null; then
        if [ "${backend}" == "cuda" ]; then echo "(SKIP ${backend} because nvcc is missing on this node)"; ECHO; continue; fi
      elif ! hipcc --version &> /dev/null; then
        if [ "${backend}" == "hip" ]; then echo "(SKIP ${backend} because hipcc is missing on this node)"; ECHO; continue; fi
      fi
      # Skip C++ tests for unsupported simd modes as done in tput tests (prevent illegal instruction crashes #791)
      if [ "${unamep}" != "x86_64" ]; then
        if [ "${backend}" == "cppavx2" ]; then echo "(SKIP ${backend} which is not supported on ${unamep})"; ECHO; continue; fi
        if [ "${backend}" == "cpp512y" ]; then echo "(SKIP ${backend} which is not supported on ${unamep})"; ECHO; continue; fi
        if [ "${backend}" == "cpp512z" ]; then echo "(SKIP ${backend} which is not supported on ${unamep})"; ECHO; continue; fi
      elif [ "${unames}" == "Darwin" ]; then
        if [ "${backend}" == "cpp512y" ]; then echo "(SKIP ${backend} which is not supported on ${unames})"; ECHO; continue; fi
        if [ "${backend}" == "cpp512z" ]; then echo "(SKIP ${backend} which is not supported on ${unames})"; ECHO; continue; fi
      elif [ "$(grep -m1 -c avx512vl /proc/cpuinfo)" != "1" ]; then
        if [ "${backend}" == "cpp512y" ]; then echo "(SKIP ${backend} which is not supported - no avx512vl in /proc/cpuinfo)"; ECHO; continue; fi
        if [ "${backend}" == "cpp512z" ]; then echo "(SKIP ${backend} which is not supported - no avx512vl in /proc/cpuinfo)"; ECHO; continue; fi
      fi
      # Execute the tests
      if [ "${backend}" == "cuda" ]; then
        suffix=cuda
      elif [ "${backend}" == "hip" ]; then
        suffix=hip
      else
        suffix=cpp
      fi
      btag=${backend#cpp} # fix #937
      if ls -d build.${btag}* > /dev/null 2>&1; then
        echo "DEBUG: loop through directories: $(ls -d build.${btag}*)"
        bdirs="$(ls -d build.${btag}*)"
        for bdir in ${bdirs}; do
          ECHO
          echo "DEBUG: execute tests in directory ${bdir}"
          if [ ! -f ${bdir}/runTest_${suffix}.exe ]; then echo "ERROR! ${bdir}/runTest_${suffix}.exe not found?"; exit 1; fi
          runExe ${bdir}/runTest_${suffix}.exe
          if [ ! -f ${bdir}/check_${suffix}.exe ]; then echo "ERROR! ${bdir}/check_${suffix}.exe not found?"; exit 1; fi
          runExe ${bdir}/check_${suffix}.exe -p 1 32 1
        done
        ECHO
      else
        echo "(WARNING build.${btag}* directories not found)"
        ECHO
      fi
    done
    echo "-------------------------------------------------------------------------------"
    popd >& /dev/null
  done
}

#----------------------------------------------------------------------------------------------------------------------------------

# Determine the appropriate number of events for the specific process (fortran/cpp/cuda/hip)
function getnevt()
{
  nevt=32 # HARDCODED xfac=QUICK SETTING (tmad tests use 8192 for xfac=1 and 81920 for xfac=10)
  # FIXME? check that nevt is a multiple of NLOOP?
  echo $nevt
}

# Create an input file that is appropriate for the specific process
# (Simplified version: no argument $1 required, no need to multiply nevt by xfac)
function getinputfile()
{
  nevt=$(getnevt)
  tmp=$tmpdir/input_${proc%.mad}_${backend}
  iconfig=1
  if [ "${proc%.mad}" == "gg_ttgg" ]; then iconfig=104; fi # test iconfig=104 on gg_ttgg (LHE color mismatch #856?)
  cat << EOF >> ${tmp}
${nevt} 1 1 ! Number of events and max and min iterations
0.000001 ! Accuracy (ignored because max iterations = min iterations)
0 ! Grid Adjustment 0=none, 2=adjust (NB if = 0, ftn26 will still be used if present)
1 ! Suppress Amplitude 1=yes (i.e. use MadEvent single-diagram enhancement)
0 ! Helicity Sum/event 0=exact
${iconfig} ! ICONFIG number (1-N) for single-diagram enhancement multi-channel (NB used even if suppress amplitude is 0!)
EOF
  echo ${tmp}
}

# Run madevent_fortran (or madevent_cpp or madevent_cuda or madevent_hip, depending on $1) and parse its output
function runmadevent()
{
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: runmadevent <madevent executable>"; exit 1; fi
  cmd=$1
  NLOOP=32
  unset CUDACPP_RUNTIME_FBRIDGEMODE
  export CUDACPP_RUNTIME_VECSIZEUSED=${NLOOP}
  tmpin=$(getinputfile)
  if [ "${cmd/madevent_fortran}" == "$cmd" ]; then
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/} # skip this for fortran
  fi
  if [ ! -f $tmpin ]; then echo "ERROR! Missing input file $tmpin"; exit 1; fi
  tmp=${tmpin/input/output}
  \rm -f ${tmp}; touch ${tmp}  
  set +e # do not fail on error
  if [ "${debug}" == "1" ]; then
    echo "--------------------"
    echo CUDACPP_RUNTIME_FBRIDGEMODE = ${CUDACPP_RUNTIME_FBRIDGEMODE:-(not set)}
    echo CUDACPP_RUNTIME_VECSIZEUSED = ${CUDACPP_RUNTIME_VECSIZEUSED:-(not set)}
    echo "--------------------"; cat ${tmpin}; echo "--------------------"
    echo "Executing '$timecmd $cmd < ${tmpin} > ${tmp}'"
  fi
  $timecmd $cmd < ${tmpin} > ${tmp}
  if [ "$?" != "0" ]; then echo "ERROR! '$timecmd $cmd < ${tmpin} > ${tmp}' failed"; tail -10 $tmp; exit 1; fi
  cat ${tmp} | grep --binary-files=text '^DEBUG'
  omp=$(cat ${tmp} | grep --binary-files=text 'omp_get_max_threads() =' | awk '{print $NF}')
  if [ "${omp}" == "" ]; then omp=1; fi # _OPENMP not defined in the Fortran #579
  nghel=$(cat ${tmp} | grep --binary-files=text 'NGOODHEL =' | awk '{print $NF}')
  ncomb=$(cat ${tmp} | grep --binary-files=text 'NCOMB =' | awk '{print $NF}')
  fbm=$(cat ${tmp} | grep --binary-files=text 'FBRIDGE_MODE (.*) =' | awk '{print $NF}')
  nbp=$(cat ${tmp} | grep --binary-files=text 'VECSIZE_USED (.*) =' | awk '{print $NF}')
  mch=$(cat ${tmp} | grep --binary-files=text 'MULTI_CHANNEL =' | awk '{print $NF}')
  conf=$(cat ${tmp} | grep --binary-files=text 'Running Configuration Number:' | awk '{print $NF}')
  chid=$(cat ${tmp} | grep --binary-files=text 'CHANNEL_ID =' | awk '{print $NF}')
  echo " [OPENMPTH] omp_get_max_threads/nproc = ${omp}/$(nproc --all)"
  echo " [NGOODHEL] ngoodhel/ncomb = ${nghel}/${ncomb}"
  echo " [XSECTION] VECSIZE_USED = ${nbp}"
  echo " [XSECTION] MultiChannel = ${mch}"
  echo " [XSECTION] Configuration = ${conf}"
  echo " [XSECTION] ChannelId = ${chid}"
  xsec=$(cat ${tmp} | grep --binary-files=text 'Cross sec =' | awk '{print 0+$NF}')
  xsec2=$(cat ${tmp} | grep --binary-files=text 'Actual xsec' | awk '{print $NF}')
  if [ "${xsec}" == "" ]; then
    echo -e " [XSECTION] ERROR! No cross section in log file:\n   $tmp\n   ..."
    tail -10 $tmp
    exit 1
  elif [ "${fbm}" != "" ]; then
    echo " [XSECTION] Cross section = ${xsec} [${xsec2}] fbridge_mode=${fbm}"
  elif [ "${xsec2}" != "" ]; then
    echo " [XSECTION] Cross section = ${xsec} [${xsec2}]"
  elif [ "${xsec}" != "" ]; then
    echo " [XSECTION] Cross section = ${xsec}"
  fi
  evtf=$(cat ${tmp} | grep --binary-files=text 'events.' | grep 'Found' | awk '{print $2}')
  evtw=$(cat ${tmp} | grep --binary-files=text 'events.' | grep 'Wrote' | awk '{print $2}')
  if [ "${evtf}" != "" ] && [ "${evtw}" != "" ]; then
    echo " [UNWEIGHT] Wrote ${evtw} events (found ${evtf} events)"  
  fi
  if [ "${cmd/madevent_cpp}" != "$cmd" ] || [ "${cmd/madevent_cuda}" != "$cmd" ]; then # this is madevent_fortran
    # Hack: use awk to convert Fortran's 0.42E-01 into 4.20e-02
    cat ${tmp} | grep --binary-files=text MERATIOS \
      | awk -v sep=" 1 - " '{i=index($0,sep); if(i>0){print substr($0,0,i-1) sep 0+substr($0,i+length(sep))} else print $0}' \
      | awk -v sep=" 1 + " '{i=index($0,sep); if(i>0){print substr($0,0,i-1) sep 0+substr($0,i+length(sep))} else print $0}' \
      | awk -v sep1=" AVG = " -v sep2=" +- " '{i1=index($0,sep1); i2=index($0,sep2); if(i1>0 && i2>0){print substr($0,0,i1-1) sep1 0+substr($0,i1+length(sep1),i2-i1) sep2 0+substr($0,i2+length(sep2))} else print $0}'
  fi
  cat ${tmp} | grep --binary-files=text COUNTERS
  set -e # fail on error
  xsecnew=${xsec2}
  ###echo "Extracted results from $tmp"
}

# Tmad_test stage
function tmad_test() {
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: $(basename $0) <process>"; exit 1; fi
  proc=$1
  if [ "${proc%.sa}" != "${proc}" ]; then echo "(SKIP tmad_test for SA process ${proc})"; exit 0; fi
  if [ "${proc%.mad}" == "${proc}" ]; then echo "Usage: $(basename $0) <process.mad>"; exit 1; fi
  # Iterate over P* directories and run tests
  cd ${topdir}/epochX/cudacpp/${proc}
  echo "Current directory is $(pwd)"
  echo "FPTYPE=${FPTYPE}"
  fptype=${FPTYPE}
  if [ "${fptype}" == "f" ]; then
    ###xsecthr="2E-4" # fails for ggttggg with clang14 (2.8E-4)
    ###xsecthr="4E-4" # xsec differs by 5.2E-4  ggttggg sse4 in master_june24 #889
    ###xsecthr="6E-4" # xsec differs by 1.1E-3  ggttggg avx2 in master_june24 #889
    xsecthr="2E-3"
  elif [ "${fptype}" == "m" ]; then
    xsecthr="2E-4" # FIXME #537 (AV: by "fixme" I probably meant a stricter tolerance could be used, maybe E-5?)
  else
    ###xsecthr="2E-14" # fails when updating gpucpp in PR #811
    ###xsecthr="3E-14" # xsec differs by 2E-13 for heft_gg_bb in the CI #871
    xsecthr="3E-13"
  fi
  scrdir=$(cd $(pwd)/../tmad; pwd) # tmad script dir
  pdirs="$(ls -d SubProcesses/P*_*)"
  for pdir in ${pdirs}; do
    pushd $pdir >& /dev/null
    tmpdir=$(pwd) # use a local directory instead of /tmp as in the original tmad scripts
    ECHO
    echo "*******************************************************************************"
    echo "*** tmad_test ${proc} ($(basename $(pwd)))"
    echo "*******************************************************************************"
    ECHO
    echo "Testing in $(pwd)"
    echo "r=21" > ../randinit # hardcode randinit (just in case it is disturbed by the test in the previous pdir)
    # -----------------------------------
    # (1) MADEVENT_FORTRAN
    # -----------------------------------
    \rm -f results.dat # ALWAYS remove results.dat before the first madevent execution
    echo -e "\n*** (1) EXECUTE MADEVENT_FORTRAN (create results.dat) ***"
    \rm -f ftn26
    runmadevent ./madevent_fortran
    \cp -p results.dat results.dat.ref
    xfac=QUICK
    echo -e "\n*** (1) EXECUTE MADEVENT_FORTRAN x$xfac (create events.lhe) ***"
    \rm -f ftn26
    runmadevent ./madevent_fortran
    xsecrefQUICK=$xsecnew
    \cp events.lhe events.lhe0
    if [ "${fptype}" == "f" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi
    if [ "${fptype}" == "m" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi # FIXME #537
    \mv events.lhe events.lhe.ref.$xfac
    # -----------------------------------
    # (2) MADEVENT_CPP
    # ----------------------------------- 
    unamep=$(uname -p)
    unames=$(uname -s)
    for backend in none sse4 avx2 512y 512z; do # NB other parts of this script use backend name with stripped off leading 'cpp' 
      # Skip C++ tests for unsupported simd modes as done in tput tests (prevent illegal instruction crashes #791)
      if [ "$backend" == "512y" ] || [ "$backend" == "512z" ]; then 
        if [ "${unames}" == "Darwin" ]; then echo -e "\n*** (2-$backend) WARNING! SKIP MADEVENT_CPP (${backend} is not supported on ${unames}) ***"; continue; fi
        if ! grep avx512vl /proc/cpuinfo >& /dev/null; then echo -e "\n*** (2-$backend) WARNING! SKIP MADEVENT_CPP (no avx512vl in /proc/cpuinfo) ***"; continue; fi
      fi
      # Execute the tests
      \cp -p results.dat.ref results.dat # default for rmrdat=0
      echo -e "\n*** (2-$backend) EXECUTE MADEVENT_CPP x$xfac (create events.lhe) ***"
      \rm -f ftn26
      runmadevent ./madevent_cpp
      echo -e "\n*** (2-$backend) Compare MADEVENT_CPP x$xfac xsec to MADEVENT_FORTRAN xsec ***"
      xsecref=$xsecrefQUICK
      delta=$(python3 -c "print(abs(1-$xsecnew/$xsecref))")
      if python3 -c "assert(${delta}<${xsecthr})" 2>/dev/null; then
        echo -e "\nOK! xsec from fortran ($xsecref) and cpp ($xsecnew) differ by less than ${xsecthr} ($delta)"
      else
        echo -e "\nERROR! xsec from fortran ($xsecref) and cpp ($xsecnew) differ by more than ${xsecthr} ($delta)"
        exit 1
      fi
      echo -e "\n*** (2-$backend) Compare MADEVENT_CPP x$xfac events.lhe to MADEVENT_FORTRAN events.lhe reference (including colors and helicities) ***"
      \cp events.lhe events.lhe0
      if [ "${fptype}" == "f" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi
      if [ "${fptype}" == "m" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi # FIXME #537
      \mv events.lhe events.lhe.${backend}.$xfac
      if ! diff events.lhe.${backend}.$xfac events.lhe.ref.$xfac &> /dev/null; then echo "ERROR! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac differ!"; echo "diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20"; diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20; exit 1; else echo -e "\nOK! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac are identical"; fi
    done
    # -----------------------------------
    # (3) MADEVENT_CUDA and MADEVENT_HIP
    # -----------------------------------
    if nvidia-smi -L > /dev/null 2>&1; then
      gpuTxt="$(nvidia-smi -L | wc -l)x $(nvidia-smi -L | awk '{print $3,$4}' | sort -u)"
    elif rocm-smi -i > /dev/null 2>&1; then
      gpuTxt="$(rocm-smi --showproductname | grep 'Card series' | awk '{print $5,$6,$7}')"
    else
      gpuTxt=none
    fi
    for backend in cuda hip; do
      # Skip GPU tests for NVidia and AMD unless nvcc and hipcc, respectively, are in PATH
      if [ "$backend" == "cuda" ]; then
        MADEVENT_GPU=MADEVENT_CUDA
        if [ "$gpuTxt" == "none" ]; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (there is no GPU on this node) ***"; continue; fi
        if ! nvcc --version >& /dev/null; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (${backend} is not supported on this node) ***"; continue; fi
      elif [ "$backend" == "hip" ]; then
        MADEVENT_GPU=MADEVENT_HIP
        if [ "$gpuTxt" == "none" ]; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (there is no GPU on this node) ***"; continue; fi
        if ! hipcc --version >& /dev/null; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (${backend} is not supported on this node) ***"; continue; fi
      else
        echo "INTERNAL ERROR! Unknown backend ${backend}"; exit 1
      fi
      # Execute the tests
      \cp -p results.dat.ref results.dat # default for rmrdat=0
      echo -e "\n*** (3-${backend}) EXECUTE ${MADEVENT_GPU} x$xfac (create events.lhe) ***"
      \rm -f ftn26
      runmadevent ./madevent_${backend}
      echo -e "\n*** (3-${backend}) Compare ${MADEVENT_GPU} x$xfac xsec to MADEVENT_FORTRAN xsec ***"
      xsecref=$xsecrefQUICK
      delta=$(python3 -c "print(abs(1-$xsecnew/$xsecref))")
      if python3 -c "assert(${delta}<${xsecthr})" 2>/dev/null; then
        echo -e "\nOK! xsec from fortran ($xsecref) and $backend ($xsecnew) differ by less than ${xsecthr} ($delta)"
      else
        echo -e "\nERROR! xsec from fortran ($xsecref) and $backend ($xsecnew) differ by more than ${xsecthr} ($delta)"
        exit 1
      fi
      echo -e "\n*** (3-${backend}) Compare ${MADEVENT_GPU} x$xfac events.lhe to MADEVENT_FORTRAN events.lhe reference (including colors and helicities) ***"
      \cp events.lhe events.lhe0
      if [ "${fptype}" == "f" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi
      if [ "${fptype}" == "m" ]; then ${scrdir}/lheFloat.sh events.lhe0 events.lhe; fi # FIXME #537
      \mv events.lhe events.lhe.${backend}.$xfac
      if ! diff events.lhe.${backend}.$xfac events.lhe.ref.$xfac &> /dev/null; then echo "ERROR! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac differ!"; echo "diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20"; diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20; exit 1; else echo -e "\nOK! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac are identical"; fi
    done
    popd >& /dev/null
  done
}

#----------------------------------------------------------------------------------------------------------------------------------

# Usage
function usage() {
  echo "Usage: $(basename $0) <${stages// /|}> <proc.sa|proc.mad>"
  exit 1
}

#----------------------------------------------------------------------------------------------------------------------------------

# Valid stages
stages="codegen before_build build after_build tput_test tmad_test"

# Check input arguments
for astage in $stages; do
  if [ "$1" == "$astage" ]; then
    stage=$1; proc=$2; shift; shift; break
  fi
done
if [ "$stage" == "" ] || [ "$proc" == "" ] || [ "$1" != "" ]; then usage; fi

# Start
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "[testsuite_oneprocess.sh] $stage ($proc) starting at $(date)"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

# Execute stage
( set -e; $stage $proc ) # execute this within a subprocess and fail immediately on error
status=$?

# Finish
ECHO
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
if [ $status -eq 0 ]; then
  echo "[testsuite_oneprocess.sh] $stage ($proc) finished with status=$status (OK) at $(date)"
else
  echo "[testsuite_oneprocess.sh] $stage ($proc) finished with status=$status (NOT OK) at $(date)"
fi
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

# Optionally bypass known issues
function bypassIssue(){
  echo "[testsuite_oneprocess.sh] $stage ($proc) FPTYPE=${FPTYPE}: bypass known issue '$1'"
  status=0
}
if [ $BYPASS_KNOWN_ISSUES -eq 1 ] && [ $status -ne 0 ]; then
  # Known issues in tmad_test
  if [ "$stage" == "tmad_test" ]; then
    # No cross section in susy_gg_t1t1 (#826)
    if [ "${proc%.mad}" == "susy_gg_t1t1" ]; then bypassIssue "No cross section in ${proc%.mad} for FPTYPE=d,f,m (#826)"; fi
    # Cross section mismatch in pp_tt012j for P2_gu_ttxgu (#872)
    if [ "${proc%.mad}" == "pp_tt012j" ]; then bypassIssue "Cross section mismatch for P2_gu_ttxgu in ${proc%.mad} for FPTYPE=d,f,m (#856)"; fi
    # Final printout
    if [ $status -ne 0 ]; then echo "[testsuite_oneprocess.sh] $stage ($proc) FPTYPE=${FPTYPE}: issue will not be bypassed, test has FAILED"; fi
  fi
fi
exit $status
