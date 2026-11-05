#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2025).
# Integrated with the MadGraph7 project in Feb 2026.

scrdir=$(cd $(dirname $0); pwd)
bckend=$(basename $(cd $scrdir; cd ..; pwd)) # cudacpp or alpaka
cd $scrdir

function usage()
{
  echo "Usage: $0 <processes [-eemumu][-ggtt][-ggttg][-ggttgg][-ggttggg][-gqttq][-heftggbb][-susyggtt][-susyggt1t1][-smeftggtttt]> [-nocuda] [-sa] [-noalpaka] [-dblonly|-fltonly|-d_f|-dmf] [-inl|-inlonly] [-hrd|-hrdonly] [-common|-curhst] [-rmbhst|-bridge] [-noBlas|-blasOn] [-makeonly] [-makeclean] [-makej] [-scaling] [-dlp <dyld_library_path>]" # -nofpe is no longer supported
  exit 1
}

procs=
eemumu=
ggtt=
ggttg=
ggttgg=
ggttggg=
gqttq=
heftggbb=
susyggtt=
susyggt1t1=
smeftggtttt=
bldall=-bldall
suffs="mad" # DEFAULT code base: madevent + cudacpp as 2nd exporter (logs_*_mad)
alpaka=
fptypes="m" # new default #995 (was "d")
helinls="0"
hrdcods="0"
rndgen=
rmbsmp=
blas="" # build with blas but disable it at runtime
steps="make test"
makej=
scaling=
###nofpe=
dlp=
dlpset=0

for arg in $*; do
  if [ "${dlpset}" == "1" ]; then
    dlpset=2
    dlp="-dlp $arg"
  elif [ "$arg" == "-dlp" ] && [ "${dlpset}" == "0" ]; then
    dlpset=1
  elif [ "$arg" == "-eemumu" ]; then
    if [ "$eemumu" == "" ]; then procs+=${procs:+ }${arg}; fi
    eemumu=$arg
  elif [ "$arg" == "-ggtt" ]; then
    if [ "$ggtt" == "" ]; then procs+=${procs:+ }${arg}; fi
    ggtt=$arg
  elif [ "$arg" == "-ggttg" ]; then
    if [ "$ggttg" == "" ]; then procs+=${procs:+ }${arg}; fi
    ggttg=$arg
  elif [ "$arg" == "-ggttgg" ]; then
    if [ "$ggttgg" == "" ]; then procs+=${procs:+ }${arg}; fi
    ggttgg=$arg
  elif [ "$arg" == "-ggttggg" ]; then
    if [ "$ggttggg" == "" ]; then procs+=${procs:+ }${arg}; fi
    ggttggg=$arg
  elif [ "$arg" == "-gqttq" ]; then
    if [ "$gqttq" == "" ]; then procs+=${procs:+ }${arg}; fi
    gqttq=$arg
  elif [ "$arg" == "-heftggbb" ]; then
    if [ "$heftggbb" == "" ]; then procs+=${procs:+ }${arg}; fi
    heftggbb=$arg
  elif [ "$arg" == "-susyggtt" ]; then
    if [ "$susyggtt" == "" ]; then procs+=${procs:+ }${arg}; fi
    susyggtt=$arg
  elif [ "$arg" == "-susyggt1t1" ]; then
    if [ "$susyggt1t1" == "" ]; then procs+=${procs:+ }${arg}; fi
    susyggt1t1=$arg
  elif [ "$arg" == "-smeftggtttt" ]; then
    if [ "$smeftggtttt" == "" ]; then procs+=${procs:+ }${arg}; fi
    smeftggtttt=$arg
  elif [ "$arg" == "-cpponly" ]; then
    bldall=-cpponly
  elif [ "$arg" == "-nocuda" ]; then
    bldall=-nocuda
  elif [ "$arg" == "-sa" ]; then
    suffs="sa" # standalone_cudacpp code base (logs_*_sa)
  elif [ "$arg" == "-noalpaka" ]; then
    alpaka=$arg
  elif [ "$arg" == "-dblonly" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d"
  elif [ "$arg" == "-fltonly" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="f"
  elif [ "$arg" == "-d_f" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d f"
  elif [ "$arg" == "-dmf" ]; then
    if [ "${fptypes}" != "m" ] && [ "${fptypes}" != "d m f" ]; then echo "ERROR! Options -dblonly, -fltonly, -d_f and -dmf are incompatible"; usage; fi
    fptypes="d m f"
  elif [ "$arg" == "-inl" ]; then
    if [ "${helinls}" == "1" ]; then echo "ERROR! Options -inl and -inlonly are incompatible"; usage; fi
    helinls="0 1"
  elif [ "$arg" == "-inlonly" ]; then
    if [ "${helinls}" == "0 1" ]; then echo "ERROR! Options -inl and -inlonly are incompatible"; usage; fi
    helinls="1"
  elif [ "$arg" == "-hrd" ]; then
    if [ "${hrdcods}" == "1" ]; then echo "ERROR! Options -hrd and -hrdonly are incompatible"; usage; fi
    hrdcods="0 1"
  elif [ "$arg" == "-hrdonly" ]; then
    if [ "${hrdcods}" == "0 1" ]; then echo "ERROR! Options -hrd and -hrdonly are incompatible"; usage; fi
    hrdcods="1"
  elif [ "$arg" == "-common" ]; then
    rndgen=$arg
  elif [ "$arg" == "-curhst" ]; then
    rndgen=$arg
  ###elif [ "$arg" == "-hirhst" ]; then
  ###  rndgen=$arg
  elif [ "$arg" == "-rmbhst" ]; then
    rmbsmp=$arg
  elif [ "$arg" == "-bridge" ]; then
    rmbsmp=$arg
  elif [ "$arg" == "-noBlas" ]; then # build with blas but disable it at runtime
    if [ "${blas}" == "-blasOn" ]; then echo "ERROR! Options -noBlas and -blasOn are incompatible"; usage; fi
    blas=$arg
  elif [ "$arg" == "-blasOn" ]; then # build with blas and enable it at runtime
    if [ "${blas}" == "-noBlas" ]; then echo "ERROR! Options -noBlas and -blasOn are incompatible"; usage; fi
    blas=$arg
  elif [ "$arg" == "-makeonly" ]; then
    if [ "${steps}" == "make test" ]; then
      steps="make"
    elif [ "${steps}" == "makeclean make test" ]; then
      steps="makeclean make"
    fi
  elif [ "$arg" == "-makeclean" ]; then
    if [ "${steps}" == "make test" ]; then
      steps="makeclean make test"
    elif [ "${steps}" == "make" ]; then
      steps="makeclean make"
    fi
  elif [ "$arg" == "-makej" ]; then
    makej=-makej
  elif [ "$arg" == "-scaling" ]; then
    scaling=$arg
  ###elif [ "$arg" == "-nofpe" ]; then
  ###  nofpe=-nofpe
  else
    echo "ERROR! Invalid option '$arg'"; usage
  fi  
done

# Workaround for MacOS SIP (SystemIntegrity Protection): set DYLD_LIBRARY_PATH In subprocesses
if [ "${dlpset}" == "1" ]; then usage; fi

# Use only the .auto process directories in the alpaka directory
if [ "$bckend" == "alpaka" ]; then
  echo "WARNING! alpaka directory: using .auto process directories only"
  suffs="auto"
fi

#echo "procs=$procs"
#echo "suffs=$suffs"
#echo "df=$df"
#echo "inl=$inl"
#echo "hrd=$hrd"
#echo "steps=$steps"
###exit 0

# Check that at least one process has been selected
if [ "${procs}" == "" ]; then usage; fi

status=0
started="STARTED AT $(date)"

for step in $steps; do
  for proc in $procs; do
    for suff in $suffs; do
      sa=; if [ "${suff}" == "sa" ]; then sa=" -sa"; sufflog=sa; else sufflog=${suff}; fi
      for fptype in $fptypes; do
        flt=; if [ "${fptype}" == "f" ]; then flt=" -fltonly"; elif [ "${fptype}" == "d" ]; then flt=" -dblonly"; fi
        for helinl in $helinls; do
          inl=; if [ "${helinl}" == "1" ]; then inl=" -inlonly"; fi
          for hrdcod in $hrdcods; do
            hrd=; if [ "${hrdcod}" == "1" ]; then hrd=" -hrdonly"; fi
            args="${proc}${sa}${flt}${inl}${hrd} ${dlp}"
            args="${args} ${alpaka}" # optionally disable alpaka tests
            args="${args} ${rndgen}" # optionally use common random numbers or curand on host
            args="${args} ${rmbsmp}" # optionally use rambo or bridge on host
            args="${args} ${scaling}" # optionally run scaling tests
            args="${args} ${blas}" # optionally build with no blas or instead enable it at runtime
            ###args="${args} ${nofpe}" # optionally disable FPEs
            args="${args} ${bldall}" # avx, fptype, helinl and hrdcod are now supported for all processes
            if [ "${step}" == "makeclean" ]; then
              printf "\n%80s\n" |tr " " "*"
              printf "*** ./throughputX.sh -makecleanonly $args"
              printf "\n%80s\n" |tr " " "*"
              if ! ./throughputX.sh -makecleanonly $args; then exit 1; fi
            elif [ "${step}" == "make" ]; then
              printf "\n%80s\n" |tr " " "*"
              printf "*** ./throughputX.sh -makeonly ${makej} $args"
              printf "\n%80s\n" |tr " " "*"
              if ! ./throughputX.sh -makeonly ${makej} $args; then exit 1; fi
            else
              logfile=logs_${proc#-}_${sufflog}/log_${proc#-}_${sufflog}_${fptype}_inl${helinl}_hrd${hrdcod}.txt
              if [ "${rndgen}" != "" ]; then logfile=${logfile%.txt}_${rndgen#-}.txt; fi
              if [ "${rmbsmp}" != "" ]; then logfile=${logfile%.txt}_${rmbsmp#-}.txt; fi
              if [ "${blas}" != "" ]; then logfile=${logfile%.txt}_${blas#-}.txt; fi
              if [ "${scaling}" != "" ]; then logfile=${logfile%.txt}.scaling; fi
              printf "\n%80s\n" |tr " " "*"
              printf "*** ./throughputX.sh $args | tee $logfile"
              printf "\n%80s\n" |tr " " "*"
              mkdir -p $(dirname $logfile)
              ./throughputX.sh $args -gtest | tee $logfile 
              if [ ${PIPESTATUS[0]} -ne "0" ]; then status=2; fi
            fi
          done
        done
      done
    done
  done
  printf "\n%80s\n" |tr " " "#"
done

ended="ENDED   AT $(date)"
echo
echo "$started"
echo "$ended"
exit $status
