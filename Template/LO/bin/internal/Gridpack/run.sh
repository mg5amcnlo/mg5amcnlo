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


if [[  ($1 != "") && ("$2" != "") && ("$3" == "") ]]; then
   num_events=$1
   seed=$2
   gran=1
elif [[  ($1 != "") && ("$2" != "") && ("$3" != "") ]]; then
   num_events=$1
   seed=$2
   gran=$3
else
   echo "Warning: input is not correct. script requires two arguments: NB_EVENT SEED"
fi

echo "Now generating $num_events events with random seed $seed and granularity $gran"

############    RUN THE PYTHON CODE #####################
${DIR}/bin/gridrun $num_events $seed $gran
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
