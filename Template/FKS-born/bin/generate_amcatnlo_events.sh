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

for run_mode in "$@" ; do
    echo 'Cleaning previous results in the SubProcesses/P*/G'$run_mode'* directories'
    rm -rf SubProcesses/P*/G$run_mode*/
done

nevents=`awk '/^[^#].*=.*nevents/{print $1}' $Maindir/Cards/run_card.dat`
parton_shower=`awk '/^[^#].*=.*parton_shower/{print $1}' $Maindir/Cards/run_card.dat`
echo "Generating" $nevents "events for" $parton_shower



for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 0 $parton_shower
done
$Maindir/SubProcesses/combine_results.sh 0 $nevents G$@\*

for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 1 $parton_shower
done
$Maindir/SubProcesses/combine_results.sh 1 $nevents G$@\*

for run_mode in "$@" ; do
    $Maindir/bin/run_amcatnlo.sh $run_mode none 0 2 $parton_shower
done

reweight_scale=`awk '/^[^#].*=.*reweight_scale/{print $1}' $Maindir/Cards/run_card.dat`
reweight_PDF=`awk '/^[^#].*=.*reweight_PDF/{print $1}' $Maindir/Cards/run_card.dat`
PDF_set_min=`awk '/^[^#].*=.*PDF_set_min/{print $1}' $Maindir/Cards/run_card.dat`
PDF_set_max=`awk '/^[^#].*=.*PDF_set_max/{print $1}' $Maindir/Cards/run_card.dat`
muR_over_ref=`awk '/^[^#].*=.*muR_over_ref/{print $1}' $Maindir/Cards/run_card.dat`
muF1_over_ref=`awk '/^[^#].*=.*muF1_over_ref/{print $1}' $Maindir/Cards/run_card.dat`
muF2_over_ref=`awk '/^[^#].*=.*muF2_over_ref/{print $1}' $Maindir/Cards/run_card.dat`
QES_over_ref=`awk '/^[^#].*=.*QES_over_ref/{print $1}' $Maindir/Cards/run_card.dat`
lhaid=`awk '/^[^#].*=.*lhaid/{print $1}' $Maindir/Cards/run_card.dat`

if [[ $muF1_over_ref == $muF2_over_ref ]] ; then
    echo "0" > rwgt.temp
    if [[ $reweight_scale == ".true." ]] ; then
	echo "1" >> rwgt.temp
    else
	echo "0" >> rwgt.temp
    fi
    if [[ $reweight_PDF == ".true." ]] ; then
	echo "1" >> rwgt.temp
    else
	echo "0" >> rwgt.temp
    fi
    echo $QES_over_ref >> rwgt.temp
    echo $muR_over_ref $muF1_over_ref >> rwgt.temp
    if [[ $reweight_scale == ".true." ]] ; then
	echo "0.5 2.0" >> rwgt.temp
	echo "0.5 2.0" >> rwgt.temp
    fi
    if [[ $reweight_PDF == ".true." ]] ; then
	echo "lhaid" >> rwgt.temp
	echo $PDF_set_min $PDF_set_max >> rwgt.temp
    fi
else
    echo "Cannot include reweight information: muF1_over_ref != muF2_over_ref"
fi


./SubProcesses/reweight_xsec_events.sh < rwgt.temp

rm -f rwgt.temp

cd $Maindir/SubProcesses
make collect_events
echo "1" | ./collect_events

