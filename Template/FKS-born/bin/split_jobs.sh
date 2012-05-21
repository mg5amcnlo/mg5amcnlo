#!/bin/bash

# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: split_jobs.sh must be executed from the main, or bin directory"
	exit
    fi
fi

cd SubProcesses

# Check if nevents_unweighted exists
if [[ ! -e nevents_unweighted ]] ; then
    echo 'ERROR, nevents_unweighted not found'
    exit
fi

# clean the old nevts and event files
rm -f P*/G*/nevts__*
rm -f P*/G*/events_*.lhe

# Compile and run the fortran code that writes the updated version of nevents_unweighted
# and also writes the nevts_?? files in the P*/G* directories
gfortran -o split_jobs split_jobs.f
./split_jobs

# Loop over the directories
for pdir in P*; do
    echo $pdir
    cd $pdir
# remove old genE's and cmd files
    rm -f genE*
    rm -f me*.cmd
# compile and run the code that writes the new genE and cmd files
    gfortran -o write_ajob_basic ../write_ajob_basic.f ../write_ajob.f
    ./write_ajob_basic
# put them far away
    mv -f genE1 genE9999
    mv -f me1.cmd me9999.cmd
# Loop over the G* directories
    j=0
    for gdir in G*; do
	j=`expr $j + 1`
# Look in the G* directories and see which have a nevts_?? files
	for i in `seq 999` ; do
	    if [[ -e $gdir/nevts__$i ]]; then
# For those files, copy the genE and cmd files with the updated diagram information
# Using 1 genE per G* directory.
		if [[ ! -e genE$j ]]; then
		    sed "s/ <<TAG>> / ${gdir:2} /g" < genE9999  > genE$j
		    sed "s/executable = genE1/executable = genE$j/g" < me9999.cmd  > me$j.cmd
                    sed -i.bak "7s/.*/Arguments = 2 ${gdir:1:1} $i/" me$j.cmd
# if there are more than 1 nevts_?? file in a given G* directory, just append it to the cmd file
		else 
		    echo "Arguments = 2 ${gdir:1:1} $i" >> me$j.cmd
		    echo "queue" >> me$j.cmd
		fi
	    fi
	done
    done
# Delete the intermediate files
    rm -f genE9999
    rm -f me9999.cmd
    rm -f me*.bak
# Submit the jobs...
    chmod +x genE*
    for job in me*.cmd ; do
	condor_submit $job
    done
    cd ..
done
