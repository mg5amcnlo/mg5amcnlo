#! /bin/bash

echo '****************************************************'
echo 'This script runs a madfks process'
echo '****************************************************'

# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: run_madfks.sh must be executed from the main, or bin directory"
	exit
    fi
fi

Maindir=`pwd`

run_mode=$1
use_preset=$2
run_cluster=$3

if [[ $run_mode == "" ]] ; then
    echo 'Enter run mode (H, S, V or B)'
    read run_mode
fi
if [[ -e! madinMMC_$run_mode.2 ]] ; then
    echo 'Cannot read the inputs. File not found: madinMMC_'$run_mode'.2'
    exit
fi

if [[ $use_preset == "" ]] ; then
    echo "Enter presets used for integration grids (none, H, S, B, V)"
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


#---------------------------
# Update random number seed
cd $Maindir/SubProcesses/
r=0
if [[ -e randinit ]]; then
    source ./randinit
fi
for i in P*_* ; do
    r=`expr $r + 1`
done
echo "r=$r" >& randinit
cd $Maindir


vegas_mint="2"

cd SubProcesses

for dir in P*_* ; do
    cd $dir
    echo $dir
    if [[ -e madevent_mintMC ]]; then
	chmod +x ajob*
	if [[ $run_cluster == 1 ]] ; then
	    for job in mg*.cmd ; do
		sed -i "7s/.*/Arguments = $vegas_mint $run_mode $use_preset/" $job
		condor_submit $job
	    done
	elif [[ $run_cluster == 0 ]] ; then
	    echo "Doing "$run_mode"-events in this dir"
	    for job in ajob* ; do
		./$job $vegas_mint $run_mode $use_preset
	    done
	fi
    else
	echo 'madevent_mintMC does not exist. Skipping directory'
    fi
    echo ''
    cd ..
done

cd ..



echo ""
echo "Execute ./SubProcesses/combine_results.sh to collect results and"
echo "compute how many events are needed per channel."
echo ""
echo "./SubProcesses/combine_results.sh i n"
echo ""
echo "where 'i' is 0 after grid setting and '1' after integration and"
echo "'n' is the total number of unweigted events you want."
echo "Update madinMMC_H.2, madinMMC_S.2 and madinMMC_V.2 before before"
echo "executing the next integration or event generation step."
echo ""
