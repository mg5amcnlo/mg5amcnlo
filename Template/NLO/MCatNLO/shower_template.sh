#!/bin/bash

SHOWER=$1
#HEP-> hepmc/stdhep
#TOP-> top
OUTPUT=$2
RUN_NAME=$3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:%(extralibs)s

# this is for a cluster run
if [ -e events.lhe.gz ] ; then
    gunzip $RUN_NAME/events.lhe.gz
fi

if [ "$SHOWER" == "HERWIG6" ] || [ "$SHOWER" == "PYTHIA6Q" ] || [ "$SHOWER" == "PYTHIA6PT" ] || [ "$SHOWER" == "HERWIGPP" ] ; then
    ./MCATNLO_$SHOWER\_EXE < MCATNLO_$SHOWER\_input > mcatnlo_run.log 2>&1

elif [ $SHOWER == "PYTHIA8" ] ; then
    source config.sh
    ./Pythia8.exe Pythia8.cmd > mcatnlo_run.log 2>&1
fi

if [ "$OUTPUT" == "HEP" ] ; then
    # hep or hepmc output
    # at the end a file called events.hep.gz or events.hepmc.gz will be delivered
    if [ "$SHOWER" == "HERWIG6" ] || [ "$SHOWER" == "PYTHIA6Q" ] || [ "$SHOWER" == "PYTHIA6PT" ] ; then
        mv events.lhe.hep events.hep
        gzip events.hep
    elif [ "$SHOWER" == "HERWIGPP" ] ; then
        mv MCATNLO_HERWIGPP.hepmc events.hepmc
        gzip events.hepmc
    elif [ "$SHOWER" == "PYTHIA8" ] ; then
        mv Pythia8.hep events.hepmc
        gzip events.hepmc
    fi

elif [ "$OUTPUT" == "TOP" ] ; then
    #top, just tar all the topfiles which are found
    tar -cf topfiles.tar *.top
fi

