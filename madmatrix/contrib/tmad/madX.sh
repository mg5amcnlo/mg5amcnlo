#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Mar 2022) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

set +x # not verbose

# [NB: set -e may lead to unexpected silent failures if ((..)) arithmetic expressions have a result equal to 0]
# [See https://www.gnu.org/software/bash/manual/html_node/Conditional-Constructs.html and https://stackoverflow.com/a/66824545]
set -e # fail on error

scrdir=$(cd $(dirname $0); pwd)
bckend=$(basename $(cd $scrdir; cd ..; pwd)) # cudacpp or alpaka
topdir=$(cd $scrdir; cd ../../..; pwd)

# Enable OpenMP in tmad tests? (#758)
###export USEOPENMP=1

# Debug channelid in MatrixElementKernelBase?
export MG5AMC_CHANNELID_DEBUG=1

# HARDCODE NLOOP HERE (may improve this eventually...)
NLOOP=8192

# Workaround for #498 on juwels
host=$(hostname)
if [ "${host/juwels}" != "${host}" ]; then NLOOP=32; fi # workaround for #498

# These two environment variables used to be input parameters to madevent (#658)
# (Possible values for FBRIDGEMODE: CppOnly=1, FortranOnly=0, BothQuiet=-1, BothDebug=-2)
unset CUDACPP_RUNTIME_FBRIDGEMODE
export CUDACPP_RUNTIME_VECSIZEUSED=${NLOOP}

function usage()
{
  echo "Usage: $0 <processes [-eemumu][-ggtt][-ggttg][-ggttgg][-ggttggg][-gguu][-gqttq][-heftggbb][-susyggtt][-susyggt1t1][-smeftggtttt]> [-d] [-hip] [-dblonly|-fltonly] [-makeonly|-makeclean|-makecleanonly] [-rmrdat] [+10x] [-checkonly] [-nocleanup][-iconfig <iconfig>]" > /dev/stderr
  echo "(NB: OMP_NUM_THREADS is taken as-is from the caller's environment)"
  exit 1
}

##########################################################################
# PART 0 - decode command line arguments
##########################################################################

debug=0

eemumu=0
ggtt=0
ggttg=0
ggttgg=0
ggttggg=0
gguu=0
gqttq=0
heftggbb=0
susyggtt=0
susyggt1t1=0
smeftggtttt=0

hip=0

fptype="m" # new default #995 (was "d")

maketype=
###makej=

rmrdat=0

xfacs="1"

checkonly=0

nocleanup=0

iconfig=

while [ "$1" != "" ]; do
  if [ "$1" == "-d" ]; then
    debug=1
    shift
  elif [ "$1" == "-eemumu" ]; then
    eemumu=1
    shift
  elif [ "$1" == "-ggtt" ]; then
    ggtt=1
    shift
  elif [ "$1" == "-ggttg" ]; then
    ggttg=1
    shift
  elif [ "$1" == "-ggttgg" ]; then
    ggttgg=1
    shift
  elif [ "$1" == "-ggttggg" ]; then
    ggttggg=1
    shift
  elif [ "$1" == "-gguu" ]; then
    gguu=1
    shift
  elif [ "$1" == "-gqttq" ]; then
    gqttq=1
    shift
  elif [ "$1" == "-heftggbb" ]; then
    heftggbb=1
    shift
  elif [ "$1" == "-susyggtt" ]; then
    susyggtt=1
    shift
  elif [ "$1" == "-susyggt1t1" ]; then
    susyggt1t1=1
    shift
  elif [ "$1" == "-smeftggtttt" ]; then
    smeftggtttt=1
    shift
  elif [ "$1" == "-hip" ]; then
    hip=1
    shift
  elif [ "$1" == "-dblonly" ]; then
    if [ "${fptype}" != "m" ] && [ "${fptype}" != "$1" ]; then
      echo "ERROR! Options -fltonly and -dblonly are incompatible"; usage
    fi
    fptype="d"
    shift
  elif [ "$1" == "-fltonly" ]; then
    if [ "${fptype}" != "m" ] && [ "${fptype}" != "$1" ]; then
      echo "ERROR! Options -fltonly and -dblonly are incompatible"; usage
    fi
    fptype="f"
    shift
  elif [ "$1" == "-makeonly" ] || [ "$1" == "-makeclean" ] || [ "$1" == "-makecleanonly" ]; then
    if [ "${maketype}" != "" ] && [ "${maketype}" != "$1" ]; then
      echo "ERROR! Options -makeonly, -makeclean and -makecleanonly are incompatible"; usage
    fi
    maketype="$1"
    shift
  elif [ "$1" == "-rmrdat" ]; then
    rmrdat=1
    shift
  elif [ "$1" == "+10x" ]; then
    xfacs="$xfacs 10"
    shift
  elif [ "$1" == "-checkonly" ]; then
    checkonly=1
    shift
  elif [ "$1" == "-nocleanup" ]; then
    nocleanup=1
    shift
  elif [ "$1" == "-iconfig" ] && [ "$2" != "" ]; then
    iconfig=$2
    shift; shift
  else
    usage
  fi
done
###exit 1

# Check that at least one process has been selected
if [ "${eemumu}" == "0" ] && [ "${ggtt}" == "0" ] && [ "${ggttg}" == "0" ] && [ "${ggttgg}" == "0" ] && [ "${ggttggg}" == "0" ] && [ "${gguu}" == "0" ] && [ "${gqttq}" == "0" ] && [ "${heftggbb}" == "0" ] && [ "${susyggtt}" == "0" ] && [ "${susyggt1t1}" == "0" ] && [ "${smeftggtttt}" == "0" ]; then usage; fi

# Always test only the .mad/ directories (hardcoded)
suffs=".mad/"

# Switch between double and float builds
export FPTYPE=$fptype
if [ "${fptype}" == "f" ]; then
  ###xsecthr="2E-4" # fails for ggttggg with clang14 (2.8E-4)
  xsecthr="4E-4"
elif [ "${fptype}" == "m" ]; then
  xsecthr="2E-4" # FIXME #537 (AV: by "fixme" I probably meant a stricter tolerance could be used, maybe E-5?)
elif [ "${fptype}" == "d" ]; then
  ###xsecthr="2E-14" # fails when updating gpucpp in PR #811
  xsecthr="3E-14"
else
  echo "INTERNAL ERROR! Unknown FPTYPE='${fptype}'" > /dev/stderr; exit 1
fi

# Determine the working directory below topdir based on suff, bckend and <process>
function showdir()
{
  if [ "${suff}" == ".mad/" ]; then
    # FIXME? These checks should rather be 'if [ "${proc}" == "-ggtt" ]; then'?
    # FIXME? More generally, this script now accepts several processes but is only able to handle one?
    if [ "${eemumu}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/ee_mumu${suff}SubProcesses/P1_epem_mupmum
    elif [ "${ggtt}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gg_tt${suff}SubProcesses/P1_gg_ttx
    elif [ "${ggttg}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttg${suff}SubProcesses/P1_gg_ttxg
    elif [ "${ggttgg}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttgg${suff}SubProcesses/P1_gg_ttxgg
    elif [ "${ggttggg}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gg_ttggg${suff}SubProcesses/P1_gg_ttxggg
    elif [ "${gqttq}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_gu_ttxu # 1st of two (test only one for now)
      ###dir=$topdir/epochX/${bckend}/gq_ttq${suff}SubProcesses/P1_gux_ttxux # 2nd of two (test only one for now)
    elif [ "${gguu}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/gg_uu${suff}SubProcesses/P1_gg_uux
    elif [ "${heftggbb}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/heft_gg_bb${suff}SubProcesses/P1_gg_bbx
    elif [ "${susyggtt}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_tt${suff}SubProcesses/P1_gg_ttx
    elif [ "${susyggt1t1}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/susy_gg_t1t1${suff}SubProcesses/P1_gg_t1t1x
    elif [ "${smeftggtttt}" == "1" ]; then 
      dir=$topdir/epochX/${bckend}/smeft_gg_tttt${suff}SubProcesses/P1_gg_ttxttx
    ###else
    ###  echo "INTERNAL ERROR! Unknown process '${proc}'" > /dev/stderr; exit 1 # this should never happen
    fi
  else
    echo "INTERNAL ERROR! tmad tests only make sense in .mad directories" > /dev/stderr; exit 1 # this should never happen (suff=.mad/ is hardcoded)
  fi
  echo $dir
}

# Determine the appropriate number of events for the specific process (fortran/cpp/cuda/hip)
function getnevt()
{
  if [ "${eemumu}" == "1" ]; then
    nevt=8192 # Fortran (x1, x10) computes (8192, 90112) MEs and writes to file (1611, 1827) events in (0.3s, 0.8s)
  elif [ "${ggtt}" == "1" ]; then 
    nevt=8192 # Fortran (x1, x10) computes (8192, 90112) MEs and writes to file (434, 1690) events in (0.4s, 2.5s)
  elif [ "${ggttg}" == "1" ]; then
    nevt=8192 # Fortran (x1, x10) computes (8192, 90112) MEs and writes to file (40, 633) events in (0.8s, 6.8s)
  elif [ "${ggttgg}" == "1" ]; then
    nevt=8192 # Fortran (x1, x10) computes (8192, 90112) MEs and writes to file (49, 217) events in (5.8s, 58s)
  elif [ "${ggttggg}" == "1" ]; then
    nevt=8192 # Fortran (x1, x10) computes (8192, 90112) MEs and writes to file (14, 97) events in (121s, 1222s)
  elif [ "${gguu}" == "1" ]; then
    nevt=8192 # use the same settings as for ggttg
  elif [ "${gqttq}" == "1" ]; then
    nevt=8192 # use the same settings as for ggttg
  elif [ "${heftggbb}" == "1" ]; then
    nevt=8192 # use the same settings as for SM ggtt
  elif [ "${susyggtt}" == "1" ]; then
    nevt=8192 # use the same settings as for SM ggtt
  elif [ "${susyggt1t1}" == "1" ]; then
    nevt=8192 # use the same settings as for SM ggtt
  elif [ "${smeftggtttt}" == "1" ]; then
    nevt=8192 # use the same settings as for SM ggttg
  else
    echo "ERROR! Unknown process" > /dev/stderr; usage
  fi
  # FIXME? check that nevt is a multiple of NLOOP?
  echo $nevt
}

# Determine the appropriate CUDA/HIP grid dimension for the specific process (to run the fastest check_cuda or check_hip)
function getgridmax()
{
  if [ "${eemumu}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256, but seems faster
  elif [ "${ggtt}" == "1" ]; then 
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${ggttg}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${ggttgg}" == "1" ]; then
    echo 512 32 # same total grid dimension as 64 256 (new sep2025: even 1024/32 aborts in max8thr mode)
  elif [ "${ggttggg}" == "1" ]; then
    echo 512 32 # same total grid dimension as 64 256
  elif [ "${gguu}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${gqttq}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${heftggbb}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${susyggtt}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${susyggt1t1}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  elif [ "${smeftggtttt}" == "1" ]; then
    echo 16384 32 # same total grid dimension as 2048 256
  else
    echo "ERROR! Unknown process" > /dev/stderr; usage
  fi
}

# Create an input file that is appropriate for the specific process
function getinputfile()
{
  iconfig_proc=1 # use iconfig=1 by default (NB: this does not mean channel_id=1 i.e. the first diagram, see #826)
  nevt=$(getnevt)
  tmpdir=/tmp/$USER
  mkdir -p $tmpdir
  if [ "${eemumu}" == "1" ]; then 
    tmp=$tmpdir/input_eemumu
  elif [ "${ggtt}" == "1" ]; then 
    tmp=$tmpdir/input_ggtt
  elif [ "${ggttg}" == "1" ]; then 
    tmp=$tmpdir/input_ggttg
  elif [ "${ggttgg}" == "1" ]; then 
    tmp=$tmpdir/input_ggttgg
    iconfig_proc=104 # use iconfig=104 in ggttgg to check #855 SIGFPE fix (but issue #856 is pending: LHE color mismatch!)
  elif [ "${ggttggg}" == "1" ]; then 
    tmp=$tmpdir/input_ggttggg
  elif [ "${gguu}" == "1" ]; then 
    tmp=$tmpdir/input_gguu
  elif [ "${gqttq}" == "1" ]; then 
    tmp=$tmpdir/input_gqttq
  elif [ "${heftggbb}" == "1" ]; then 
    tmp=$tmpdir/input_heftggbb
  elif [ "${susyggtt}" == "1" ]; then 
    tmp=$tmpdir/input_susyggtt
  elif [ "${susyggt1t1}" == "1" ]; then 
    tmp=$tmpdir/input_susyggt1t1
    iconfig_proc=2 # use iconfig=2 in susyggt1t1 to check #855 SIGFPE fix (but issue #826 is pending: no cross section!)
  elif [ "${smeftggtttt}" == "1" ]; then 
    tmp=$tmpdir/input_smeftggtttt
  else
    echo "ERROR! cannot determine input file name"; exit 1
  fi
  tmp=${tmp}_x${xfac}
  \rm -f ${tmp}; touch ${tmp}
  if [ "$1" == "-fortran" ]; then
    # Keep the argument but there is nothing to do specific to fortran
    # Previously fbridge_mode=0 was set here (#658)
    mv ${tmp} ${tmp}_fortran
    tmp=${tmp}_fortran
  elif [ "$1" == "-cuda" ] || [ "$1" == "-hip" ] || [ "$1" == "-cpp" ]; then # NB: new script, use the same input for cuda/hip/cpp
    # Keep the argument but there is nothing to do specific to cuda/hip/cpp
    # Previously fbridge_mode=1 was set here (#658)
    mv ${tmp} ${tmp}_cudacpp
    tmp=${tmp}_cudacpp
  else
    echo "Usage: getinputfile <backend [-fortran][-cuda][-hip][-cpp]>"
    exit 1
  fi
  if [ "${iconfig}" == "" ]; then iconfig=${iconfig_proc}; fi
  (( nevt = nevt*$xfac ))
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

# Run check_(cpp|cuda|hip).exe (depending on $1) and parse its output
function runcheck()
{
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: runcheck <check_(cpp|cuda|hip) executable>"; exit 1; fi
  cmd=$1
  if [ "${cmd/gcheckmax128thr}" != "$cmd" ]; then
    txt="GCHECK(MAX128THR)"
    cmd=${cmd/gcheckmax128thr/check_${backend}} # hack: run cuda/hip check with tput fastest settings
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
    nblk=$(getgridmax | cut -d ' ' -f1)
    nthr=$(getgridmax | cut -d ' ' -f2)
    while [ $nthr -lt 128 ]; do (( nthr = nthr * 2 )); (( nblk = nblk / 2 )); done
    (( nevt = nblk*nthr ))
  elif [ "${cmd/gcheckmax8thr}" != "$cmd" ]; then
    txt="GCHECK(MAX8THR)"
    cmd=${cmd/gcheckmax8thr/check_${backend}} # hack: run cuda/hip check with tput fastest settings
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
    nblk=$(getgridmax | cut -d ' ' -f1)
    nthr=$(getgridmax | cut -d ' ' -f2)
    while [ $nthr -gt 8 ]; do (( nthr = nthr / 2 )); (( nblk = nblk * 2 )); done
    (( nevt = nblk*nthr ))
  elif [ "${cmd/gcheckmax}" != "$cmd" ]; then
    txt="GCHECK(MAX)"
    cmd=${cmd/gcheckmax/check_${backend}} # hack: run cuda/hip check with tput fastest settings
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
    nblk=$(getgridmax | cut -d ' ' -f1)
    nthr=$(getgridmax | cut -d ' ' -f2)
    (( nevt = nblk*nthr ))
  elif [ "${cmd/gcheck}" != "$cmd" ]; then
    txt="GCHECK($NLOOP)"
    cmd=${cmd/gcheck/check_${backend}}
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
    nthr=32
    (( nblk = NLOOP/nthr )) || true # integer division (NB: bash double parenthesis fails if the result is 0)
    (( nloop2 = nblk*nthr )) || true
    if [ "$NLOOP" != "$nloop2" ]; then echo "ERROR! NLOOP($nloop) != nloop2($nloop2)"; exit 1; fi
    nevt=$(getnevt)
  elif [ "${cmd/check}" != "$cmd" ]; then
    txt="CHECK($NLOOP)"
    cmd=${cmd/check/check_cpp}
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
    nthr=32
    (( nblk = NLOOP/nthr )) || true # integer division (NB: bash double parenthesis fails if the result is 0)
    (( nloop2 = nblk*nthr )) || true
    if [ "$NLOOP" != "$nloop2" ]; then echo "ERROR! NLOOP($NLOOP) != nloop2($nloop2)"; exit 1; fi
    nevt=$(getnevt)
  else
    echo "ERROR! Unknown check executable '$cmd'"; exit 1
  fi
  (( ngrid = nthr*nblk ))
  if [ $ngrid -gt $nevt ]; then nevt=$ngrid; fi # do run at least 8192 events in gcheck8192
  (( nite = nevt/ngrid )) || true # integer division (NB: bash double parenthesis fails if the result is 0)
  (( nevt2 = ngrid*nite )) || true
  if [ "$nevt" != "$nevt2" ]; then echo "ERROR! nevt($nevt) != nevt2($nevt2)=ngrid($ngrid)*nite($nite)"; exit 1; fi
  pattern="Process|Workflow|EvtsPerSec\[MECalc"
  echo -e "\n*** EXECUTE $txt -p $nblk $nthr $nite --bridge ***"
  $cmd -p $nblk $nthr $nite --bridge | egrep "(${pattern})"
  echo -e "\n*** EXECUTE $txt -p $nblk $nthr $nite ***"
  $cmd -p $nblk $nthr $nite | egrep "(${pattern})"
}

# Run madevent_fortran (or madevent_cpp or madevent_cuda or madevent_hip, depending on $1) and parse its output
function runmadevent()
{
  if [ "$1" == "" ] || [ "$2" != "" ]; then echo "Usage: runmadevent <madevent executable>"; exit 1; fi
  cmd=$1
  if [ "${cmd/madevent_cpp}" != "$cmd" ]; then
    tmpin=$(getinputfile -cpp)
    cmd=${cmd/.\//.\/build.${backend}_${fptype}_inl0_hrd0\/}
  elif [ "${cmd/madevent_cuda}" != "$cmd" ]; then
    cmd=${cmd/.\//.\/build.cuda_${fptype}_inl0_hrd0\/}
    tmpin=$(getinputfile -cuda)
  elif [ "${cmd/madevent_hip}" != "$cmd" ]; then
    cmd=${cmd/.\//.\/build.hip_${fptype}_inl0_hrd0\/}
    tmpin=$(getinputfile -hip)
  else # assume this is madevent_fortran (do not check)
    tmpin=$(getinputfile -fortran)
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
  cat ${tmp} | grep --binary-files=text '^DEBUG' | sed "s/MEK 0x.* processed/MEK processed/"
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

##########################################################################
# PART 1 - build madevent
##########################################################################

echo MADGRAPH_CUDA_ARCHITECTURE=${MADGRAPH_CUDA_ARCHITECTURE}
echo MADGRAPH_HIP_ARCHITECTURE=${MADGRAPH_HIP_ARCHITECTURE}

unset GTEST_ROOT
unset LOCALGTEST

export HASBLAS=hasBlas
echo HASBLAS=${HASBLAS}

for suff in $suffs; do

  dir=$(showdir)
  if [ ! -d $dir ]; then echo "WARNING! Skip missing directory $dir"; continue; fi
  echo "Working directory (build): $dir"
  cd $dir

  if [ "${maketype}" == "-makeclean" ]; then make cleanall; echo; fi
  if [ "${maketype}" == "-makecleanonly" ]; then make cleanall; echo; continue; fi
  if [ "${hip}" == "0" ] || [ "${dir/\/gg_ttggg${suff}}" == ${dir} ]; then # skip parallel bldall builds only for ggttggg on HIP #933
    ###make -j5 avxall # limit build parallelism of the old 'make avxall' to avoid "cudafe++ died due to signal 9" (#639)
    make -j6 bldall # limit build parallelism also with the new 'make bldall'
    ###make -j bldall
  else
    export USEBUILDDIR=1
    bblds="cppnone cppsse4 cppavx2 cpp512y cpp512z" # skip cuda on LUMI always; skip HIP for ggttggg builds #933
    for bbld in ${bblds}; do
      make ${makef} -j BACKEND=${bbld}; echo
    done
    unset USEBUILDDIR
  fi
done

if [ "${maketype}" == "-makecleanonly" ]; then printf "\nMAKE CLEANALL COMPLETED\n"; exit 0; fi
if [ "${maketype}" == "-makeonly" ]; then printf "\nMAKE COMPLETED\n"; exit 0; fi

##########################################################################
# PART 2 - run madevent
##########################################################################

unset CUDACPP_RUNTIME_BLASCOLORSUM
printf "\nCUDACPP_RUNTIME_BLASCOLORSUM=$CUDACPP_RUNTIME_BLASCOLORSUM\n"

unset CUDACPP_RUNTIME_CUBLASTF32TENSOR
printf "\nCUDACPP_RUNTIME_CUBLASTF32TENSOR=$CUDACPP_RUNTIME_CUBLASTF32TENSOR\n"

printf "\nOMP_NUM_THREADS=$OMP_NUM_THREADS\n"

printf "\nDATE: $(date '+%Y-%m-%d_%H:%M:%S')\n\n"

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

for suff in $suffs; do

  # DEFAULT IMPLEMENTATION : compute cross section and then generate events
  dir=$(showdir)
  if [ ! -d $dir ]; then echo "WARNING! Skip missing directory $dir"; continue; fi
  echo "Working directory (run): $dir"
  cd $dir

  # Hardcode randinit (just in case it is disturbed by tlau/lauX.sh tests)
  echo "r=21" > ../randinit

  # Use the time command?
  ###timecmd=time
  timecmd=

  # Show results.dat?
  ###rdatcmd="stat results.dat"
  rdatcmd="echo"

  # (1) MADEVENT_FORTRAN
  if [ "${checkonly}" == "0" ]; then
    xfac=1
    \rm -f results.dat # ALWAYS remove results.dat before the first madevent execution
    if [ ! -f results.dat ]; then
      echo -e "\n*** (1) EXECUTE MADEVENT_FORTRAN (create results.dat) ***"
      \rm -f ftn26
      runmadevent ./madevent_fortran
      \cp -p results.dat results.dat.ref
    fi
    for xfac in $xfacs; do
      echo -e "\n*** (1) EXECUTE MADEVENT_FORTRAN x$xfac (create events.lhe) ***"
      ${rdatcmd} | grep Modify | sed 's/Modify/results.dat /'
      \rm -f ftn26
      runmadevent ./madevent_fortran
      if [ "${xfac}" == "1" ]; then
        xsecref1=$xsecnew
      elif [ "${xfac}" == "10" ]; then
        xsecref10=$xsecnew
      else
        echo "ERROR! Unknown xfac=$xfac"; exit 1
      fi
      \cp events.lhe events.lhe0
      if [ "${fptype}" == "f" ]; then
	${scrdir}/lheFloat.sh events.lhe0 events.lhe
      fi
      if [ "${fptype}" == "m" ]; then
	${scrdir}/lheFloat.sh events.lhe0 events.lhe # FIXME #537
      fi
      \mv events.lhe events.lhe.ref.$xfac
    done
  fi

  # (2) MADEVENT_CPP
  for backend in none sse4 avx2 512y 512z; do
    if [ "$backend" == "512y" ] || [ "$backend" == "512z" ]; then 
      if ! grep avx512vl /proc/cpuinfo >& /dev/null; then echo -e "\n*** (2-$backend) WARNING! SKIP MADEVENT_CPP (${backend} is not supported on this node) ***"; continue; fi
    fi
    if [ "${checkonly}" == "0" ]; then      
      xfac=1
      if [ "${rmrdat}" == "0" ]; then \cp -p results.dat.ref results.dat; else \rm -f results.dat; fi  
      if [ ! -f results.dat ]; then
        echo -e "\n*** (2-$backend) EXECUTE MADEVENT_CPP (create results.dat) ***"
        \rm -f ftn26
        runmadevent ./madevent_cpp
      fi
      for xfac in $xfacs; do
        echo -e "\n*** (2-$backend) EXECUTE MADEVENT_CPP x$xfac (create events.lhe) ***"
        ${rdatcmd} | grep Modify | sed 's/Modify/results.dat /'
        \rm -f ftn26
        runmadevent ./madevent_cpp
        echo -e "\n*** (2-$backend) Compare MADEVENT_CPP x$xfac xsec to MADEVENT_FORTRAN xsec ***"
        if [ "${xfac}" == "1" ]; then
          xsecref=$xsecref1
        elif [ "${xfac}" == "10" ]; then
          xsecref=$xsecref10
        else
          echo "ERROR! Unknown xfac=$xfac"; exit 1
        fi
        delta=$(python3 -c "print(abs(1-$xsecnew/$xsecref))")
        if python3 -c "assert(${delta}<${xsecthr})" 2>/dev/null; then
          echo -e "\nOK! xsec from fortran ($xsecref) and cpp ($xsecnew) differ by less than ${xsecthr} ($delta)"
        else
          echo -e "\nERROR! xsec from fortran ($xsecref) and cpp ($xsecnew) differ by more than ${xsecthr} ($delta)"
          exit 1
        fi
        echo -e "\n*** (2-$backend) Compare MADEVENT_CPP x$xfac events.lhe to MADEVENT_FORTRAN events.lhe reference (including colors and helicities) ***"
	\cp events.lhe events.lhe0
	if [ "${fptype}" == "f" ]; then
	  ${scrdir}/lheFloat.sh events.lhe0 events.lhe
	fi
	if [ "${fptype}" == "m" ]; then
	  ${scrdir}/lheFloat.sh events.lhe0 events.lhe # FIXME #537
	fi
        \mv events.lhe events.lhe.cpp.$xfac
        if ! diff events.lhe.cpp.$xfac events.lhe.ref.$xfac &> /dev/null; then echo "ERROR! events.lhe.cpp.$xfac and events.lhe.ref.$xfac differ!"; echo "diff $(pwd)/events.lhe.cpp.$xfac $(pwd)/events.lhe.ref.$xfac | head -20"; diff $(pwd)/events.lhe.cpp.$xfac $(pwd)/events.lhe.ref.$xfac | head -20; exit 1; else echo -e "\nOK! events.lhe.cpp.$xfac and events.lhe.ref.$xfac are identical"; fi
      done
    fi
    runcheck ./check.exe
  done

  # (3) MADEVENT_CUDA and MADEVENT_HIP
  for backend in cuda hip; do
    if [ "$backend" == "cuda" ]; then
      MADEVENT_GPU=MADEVENT_CUDA
      if [ "$gpuTxt" == "none" ]; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (there is no GPU on this node) ***"; continue; fi
      if ! nvcc --version >& /dev/null; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (${backend} is not supported on this node) ***"; continue; fi
    elif [ "$backend" == "hip" ]; then
      MADEVENT_GPU=MADEVENT_HIP
      if [ "$gpuTxt" == "none" ]; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (there is no GPU on this node) ***"; continue; fi
      if ! hipcc --version >& /dev/null; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (${backend} is not supported on this node) ***"; continue; fi
      if [ "${dir/\/gg_ttggg${suff}}" != ${dir} ]; then echo -e "\n*** (3-$backend) WARNING! SKIP ${MADEVENT_GPU} (gg_ttggg is not supported on hip #933) ***"; continue; fi
    else
      echo "INTERNAL ERROR! Unknown backend ${backend}"; exit 1
    fi
    if [ "${checkonly}" == "0" ]; then      
      xfac=1
      if [ "${rmrdat}" == "0" ]; then \cp -p results.dat.ref results.dat; else \rm -f results.dat; fi  
      if [ ! -f results.dat ]; then
        echo -e "\n*** (3-${backend}) EXECUTE ${MADEVENT_GPU} (create results.dat) ***"
        \rm -f ftn26
        runmadevent ./madevent_${backend}
      fi
      for xfac in $xfacs; do
        echo -e "\n*** (3-${backend}) EXECUTE ${MADEVENT_GPU} x$xfac (create events.lhe) ***"
        ${rdatcmd} | grep Modify | sed 's/Modify/results.dat /'
        \rm -f ftn26
        runmadevent ./madevent_${backend}
        echo -e "\n*** (3-${backend}) Compare ${MADEVENT_GPU} x$xfac xsec to MADEVENT_FORTRAN xsec ***"
        if [ "${xfac}" == "1" ]; then
          xsecref=$xsecref1
        elif [ "${xfac}" == "10" ]; then
          xsecref=$xsecref10
        else
          echo "ERROR! Unknown xfac=$xfac"; exit 1
        fi
        delta=$(python3 -c "print(abs(1-$xsecnew/$xsecref))")
        if python3 -c "assert(${delta}<${xsecthr})" 2>/dev/null; then
          echo -e "\nOK! xsec from fortran ($xsecref) and $backend ($xsecnew) differ by less than ${xsecthr} ($delta)"
        else
          echo -e "\nERROR! xsec from fortran ($xsecref) and $backend ($xsecnew) differ by more than ${xsecthr} ($delta)"
          exit 1
        fi
        echo -e "\n*** (3-${backend}) Compare ${MADEVENT_GPU} x$xfac events.lhe to MADEVENT_FORTRAN events.lhe reference (including colors and helicities) ***"
        \cp events.lhe events.lhe0
        if [ "${fptype}" == "f" ]; then
          ${scrdir}/lheFloat.sh events.lhe0 events.lhe
        fi
        if [ "${fptype}" == "m" ]; then
          ${scrdir}/lheFloat.sh events.lhe0 events.lhe # FIXME #537
        fi
        \mv events.lhe events.lhe.${backend}.$xfac
        if ! diff events.lhe.${backend}.$xfac events.lhe.ref.$xfac &> /dev/null; then echo "ERROR! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac differ!"; echo "diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20"; diff $(pwd)/events.lhe.${backend}.$xfac $(pwd)/events.lhe.ref.$xfac | head -20; exit 1; else echo -e "\nOK! events.lhe.${backend}.$xfac and events.lhe.ref.$xfac are identical"; fi
      done
    fi
    runcheck ./gcheck.exe
    runcheck ./gcheckmax.exe
    runcheck ./gcheckmax128thr.exe
    runcheck ./gcheckmax8thr.exe
  done
done

# Cleanup
if [ "$nocleanup" == "0" ]; then
  \rm -f results.dat
  \rm -f results.dat.ref
  \rm -f events.lhe*
else
  echo
  ls -l $(pwd)/results.dat*
  ls -l $(pwd)/events.lhe*
fi

printf "\nTEST COMPLETED\n"
