#!/bin/bash

# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: combine_jobs.sh must be executed from the main, or bin directory"
	exit
    fi
fi

cd SubProcesses
for pdir in P* ; do
    echo $pdir
    cd $pdir
    make combine_jobs
    for gdir in G* ; do
	if [[ -e $gdir/events_1.lhe ]]; then
	    cd $gdir
	    ../combine_jobs
	    cd ..
	fi
    done
    cd ..
done
