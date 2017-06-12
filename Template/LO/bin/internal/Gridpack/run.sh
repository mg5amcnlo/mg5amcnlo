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
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
fi

# For Linux
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/madevent/lib:${PWD}/HELAS/lib
# For Mac OS X
export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${PWD}/madevent/lib:${PWD}/HELAS/lib

echo "Now generating $num_events events with random seed $seed and granularity $gran"

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



############    RUN THE PYTHON CODE #####################
${DIR}/bin/gridrun $num_events $seed $gran
########################################################

###########    POSTPROCESSING      #####################

echo "search for ./Events/GridRun_${seed}/unweighted_events.lhe.gz"
if [[ -e ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe.gz ]]; then
	gunzip ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe.gz
fi

if [[ ! -e  ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe ]]; then
    echo "Error: event file not found !"
    exit
else
    echo "Moving events from  events.lhe"
    mv ${DIR}/Events/GridRun_${seed}/unweighted_events.lhe ./events.lhe
    cd ..
fi

if [[ -e ${DIR}/DECAY/decay ]]; then
    cd DECAY
    echo -$seed > iseed.dat
    for ((i = 1 ;  i <= 20;  i++)) ; do
	if [[ -e decay_$i\.in ]]; then
	    echo "Decaying events..."
	    mv ../events.lhe ../events_in.lhe
	    ./decay < decay_$i\.in
	fi
    done
    cd ..
fi

if [[ -e ./REPLACE/replace.pl ]]; then
    for ((i = 1 ;  i <= 20;  i++)) ; do
	if [[ -e ./REPLACE/replace_card$i\.dat ]];then
	    echo "Adding flavors..."
	    mv ./events.lhe ./events_in.lhe
	    cd ./REPLACE
	    ./replace.pl ../events_in.lhe ../events.lhe < replace_card$i\.dat
	    cd ..
	fi
    done
fi

# part added by Stephen Mrenna to correct the kinematics of the replaced
#  particles
if [[ -e ./madevent/bin/internal/addmasses.py ]]; then
  mv ./events.lhe ./events.lhe.0
  python ./madevent/bin/internal/addmasses.py ./events.lhe.0 ./events.lhe
  if [[ $? -eq 0 ]]; then
     echo "Mass added"
     rm -rf ./events.lhe.0 &> /dev/null
  else
     mv ./events.lhe.0 ./events.lhe
  fi
fi  

gzip -f events.lhe
exit
