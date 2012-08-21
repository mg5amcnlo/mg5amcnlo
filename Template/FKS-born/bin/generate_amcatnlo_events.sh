#! /bin/bash

echo '****************************************************'
echo 'This script generates events for an amcatnlo process'
echo '****************************************************'

# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: run_amcatnlo.sh must be executed from the main, or bin directory"
	exit
    fi
fi

Maindir=`pwd`

if [[ "$@" == "" ]]; then
    echo "Please give event types that should be calculated"
    echo "e.g. 'F V', as arguments of this script" 
    exit
fi

nevents=`awk '/^[^#].*=.*nevents/{print $1}' $Maindir/Cards/run_card.dat`

for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 0
done
$Maindir/SubProcesses/combine_results.sh 0 $nevents G$@\*

for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 1
done
$Maindir/SubProcesses/combine_results.sh 1 $nevents G$@\*

for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 2
done

cd $Maindir/SubProcesses
make collect_events
echo "1" | ./collect_events

