#!/bin/bash

# loop over itype
for itype in $(seq 1 1 4)
do
    # loop over ipart
    for ipart in $(seq 1 1 7)
    do
        ./csub-local.sh nextweek 1 ./run_gridsudgen.sh $itype $ipart
        let jcount=jcount+1
        mv submit-condor.sh submit-condor.sh-$jcount
        ./submit-condor.sh-$jcount
    done
done
