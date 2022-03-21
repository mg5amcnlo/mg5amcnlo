#!/bin/bash

# loop over itype
for itype in $(seq 1 1 4)
do
    # loop over ipart
    for ipart in $(seq 1 1 7)
    do
	# loop over ipart
	for imass in $(seq 1 1 50)
	do
            ./csub-local.sh espresso 1 ./run_gridsudgen.sh $itype $ipart $imass
            let jcount=jcount+1
            mv submit-condor.sh submit-condor.sh-$jcount
            ./submit-condor.sh-$jcount
	done
    done
done
