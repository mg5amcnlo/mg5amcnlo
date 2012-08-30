#!/bin/bash


#
# If runmode is 1: run n-body only and 1 fks directory per subprocess
#                  S-functions are set to one
# If runmode is 2: run all fks dir's for subprocesses with soft singularity
#                  S-functions are not set to one
# If runmode is 3: run all fks dir's for all subprocesses
#                  This is needed to get NLO
#

#
# Two possible arguments:
# 'mode' can be 'cluster' to run on cluster or empty to run locally
# 'mode2' can be used to use presetted grids from a previous run.
# Make sure to also give the correct inputs in madin.2.
#


#---------------------------
# Update random number seed
r=0
if [[ -e randinit ]]; then
    source ./randinit
fi
r=`expr $r + 1`
echo "r=$r" >& randinit
#--------------------------


run_mode=$1
use_preset=$2
run_cluster=$3

if [[ $run_mode == "" ]] ; then
    echo 'Enter run mode (born, grid, virt, novi, all, etc.)'
    read run_mode
fi
if [[ -e madin.$run_mode ]] ; then
    string=`tail -n 1 madin.$run_mode`
    len=${#run_mode}
    strings=`echo ${string:0:len+1}`
    if [[ $run_mode != $strings ]] ; then
	echo 'ERROR: run mode in file is not consistent' $run_mode $strings
    else
	echo 'run mode is "'$run_mode'", inputs read from madin.'$run_mode
    fi
else
    echo 'Cannot read the inputs. File not found: madin.'$run_mode
    exit
fi

if [[ $use_preset == "" ]] ; then
    echo "Enter presets used for integration grids (none, born0, virt0, novi, all, etc.)"
    echo "   [Default is 'none']"
    read use_preset
fi
if [[ $use_preset == "none" ]] ; then
    echo "No preset used"
    use_preset=""
else
    echo "Using preset:" $use_preset
fi

if [[ $run_cluster == "" ]] ; then
    echo "Local run (0), cluster running (1) or ganga (2)?"
#    echo "Cluster running needs a configured condor batching system"
    read run_cluster
fi
if [[ $run_cluster == 0 ]] ; then
    echo "Running locally"
elif [[ $run_cluster == 1 ]] ; then
    echo "submitting jobs to cluster"
elif [[ $run_cluster == 2 ]] ; then
    echo "using ganga to submit jobs"
else
    echo "ERROR" $run_cluster
    exit
fi

# Always run madevent_vegas: set vegas_mint to '0'.
vegas_mint='0'

if [[ $run_mode == "virt" || $run_mode == "viSC" || $run_mode == "viSA" || $run_mode == "viSB" || $run_mode == "viLC" || $run_mode == "born" || $run_mode == "novi" || $run_mode == "novA" || $run_mode == "novB" || $run_mode == "grid" || $run_mode == "real" || $run_mode == "all" ]] ; then
    runmode=3
else
    echo "don't know what to do:   " $run_mode
    exit
fi

string=`head -n 4 madin.$run_mode | tail -n1`
suppress=`echo ${string:0:1}`
echo 'suppressing amplitude:'  $suppress
if [[ $run_cluster == 1 ]] ; then
    if [[ $suppress == '1' ]] ; then
	jobs=mg[1-9]*.cmd
    else
	jobs=mg1.cmd
    fi
else
    if [[ $suppress == '1' ]] ; then
	jobs=ajob[1-9]*
    else
	jobs=ajob1
    fi
fi

if [[ $run_cluster == 2 ]] ; then
    args='[\n'
    for dir in P*_* ; do
	cd $dir 
	for job in ajob* ; do
	    graph=`fgrep "for i in" $job | cut -d" " -f4`
	    args="$args""['madevent_vegas','"$dir"','"$job"','"$vegas_mint"','"$run_mode"','"$use_preset"','"$run_mode"_G"$graph"'],\n"
	done
	cd ..
    done
    args="$args"]
    echo -e $args > ganga_arguments.txt
    echo "Put the contents of ganga_arguments.txt into Massimo's driver.py and execute it"
    exit
fi

for p in P*_* ; do
    cd $p
    echo "Running in" $p
    chmod +x ajob*
    for job in $jobs ; do
	if [[ $run_cluster == 1 ]] ; then
            sed -i.bak "7s/.*/Arguments = $vegas_mint $run_mode 0 $use_preset/" $job
	    if [[ -e done.$job.$vegas_mint.$run_mode.$use_preset ]] ; then
		rm -f done.$job.$vegas_mint.$run_mode.$use_preset
	    fi
	    touch wait.$job.$vegas_mint.$run_mode.$use_preset
            condor_submit $job
            rm $job.bak
	elif [[ $run_cluster == 0 ]] ; then
   	    ./$job $vegas_mint $run_mode 0 $use_preset
	fi
    done
    cd ..
done
