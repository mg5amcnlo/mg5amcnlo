#!/bin/bash

#############################################################################
#                                                                          ##
#                    MadGraph/MadEvent                                     ##
#                                                                          ##
# FILE : run.sh                                                            ##
# VERSION : 1.0                                                            ##
# DATE : 29 January 2008                                                   ##
# AUTHOR : Michel Herquet (UCL-CP3)                                        ##
#                                                                          ##
# DESCRIPTION : script to save command line param in a grid card and       ##
#   call gridrun                                                           ##
# USAGE : run [num_events] [iseed]                                         ##
#############################################################################

function usage() {
    local retcode="${1:-1}"  # default return code is 1
    echo "Usage:"
    echo "  run.sh [options] [num events] [seed]"
    echo "  run.sh [options] [num events] [seed] [granularity]"
    echo "Options:"
    echo "  -h, --help                  print this message and exit"
    echo "  -p, --parallel [num procs]  number of processes to run in parallel"
    echo "  -m, --maxevts [num events]  maximum number of unweighted events per job"
    exit $retcode
}

if [[ -d ./madevent ]]; then
    DIR='./madevent'
else
    # find the path to the gridpack (https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within)
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
	DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
	SOURCE="$(readlink "$SOURCE")"
	[[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
    done
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )/madevent"
fi

# For Linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/madevent/lib:${PWD}/HELAS/lib
# For Mac OS X
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${PWD}/madevent/lib:${PWD}/HELAS/lib

pos_args=()
nprocs=1
maxevts=2500 

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage 0 ;;
    -p|--parallel)
      nprocs="$2" && shift && shift ;;
    -m|--maxevts)
      maxevts="$2" && shift && shift ;;
    -*)
      echo "Error: Unknown option $1" && usage ;;
    *)
      pos_args+=("$1") && shift ;;
  esac
done

case `echo "${pos_args[@]}" | wc -w | tr -d " "`  in
    "2")
      num_events=${pos_args[0]}
      seed=${pos_args[1]}
      gran=1
      ;;
    "3")
      num_events=${pos_args[0]}
      seed=${pos_args[1]}
      gran=${pos_args[2]}
      ;;
    *)
      echo "Error: number of arguments is not correct"
      usage
      ;;
esac

echo "Now generating $num_events events with random seed $seed and granularity $gran using $nprocs processes"

############    RUN THE PYTHON CODE #####################
${DIR}/bin/gridrun $num_events $seed $gran $nprocs $maxevts
########################################################

###########    POSTPROCESSING      #####################

if [[ -e ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe.gz ]]; then
    mv ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe.gz events.lhe.gz
else
    mv ./Events/GridRun_${seed}/unweighted_events.lhe.gz events.lhe.gz
    rm -rf Events Cards P* *.dat randinit &> /dev/null
fi
echo "write ./events.lhe.gz"
exit
