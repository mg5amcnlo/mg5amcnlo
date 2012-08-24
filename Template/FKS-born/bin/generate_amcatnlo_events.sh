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

if [[ "$1" == "" ]]; then
    echo "Please, enter the run mode"
    echo " 1 : Born only"
    echo " 2 : NLO (combined tree-level and virtual contributions)"
    echo " 3 : NLO (separate channels for tree-level and virtual contributions)"
    read run_mode_num
else
    run_mode_num=$1
fi

if [[ $run_mode_num == 1 ]]; then
    echo 'Cleaning previous results in the SubProcesses/P*/GB* directories'
    rm -rf SubProcesses/P*/GB*/
    echo "Setting inputs to run for 'born' in madinMMC_B.2"
    sed -i.bak "11s/.*/born /" $Maindir/SubProcesses/madinMMC_B.2
elif [[ $run_mode_num == 2 ]]; then
    echo 'Cleaning previous results in the SubProcesses/P*/GF* directories'
    rm -rf SubProcesses/P*/GF*/
    echo "Setting inputs to run 'all' in madinMMC_F.2"
    sed -i.bak "11s/.*/all  /" $Maindir/SubProcesses/madinMMC_F.2
elif [[ $run_mode_num == 3 ]]; then
    echo 'Cleaning previous results in the SubProcesses/P*/GF* directories'
    rm -rf SubProcesses/P*/GF*/
    echo "Setting inputs to run 'novB' in madinMMC_F.2"
    sed -i.bak "11s/.*/novB /" $Maindir/SubProcesses/madinMMC_F.2
    echo 'Cleaning previous results in the SubProcesses/P*/GV* directories'
    rm -rf SubProcesses/P*/GV*/
    echo "Setting inputs to run 'viSB' in madinMMC_V.2"
    sed -i.bak "11s/.*/viSB /" $Maindir/SubProcesses/madinMMC_V.2
else
    echo "Input not understood:" $run_mode_num
    exit
fi
    
nevents=`awk '/^[^#].*=.*nevents/{print $1}' $Maindir/Cards/run_card.dat`
parton_shower=`awk '/^[^#].*=.*parton_shower/{print $1}' $Maindir/Cards/run_card.dat`
echo "Generating" $nevents "events for" $parton_shower

if [[ $run_mode_num == 1 ]]; then
    $Maindir/bin/run_amcatnlo.sh B none 0 0 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 0 $nevents GB*
    $Maindir/bin/run_amcatnlo.sh B none 0 1 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 1 $nevents GB*
    $Maindir/bin/run_amcatnlo.sh B none 0 2 $parton_shower
elif [[ $run_mode_num == 2 ]]; then
    $Maindir/bin/run_amcatnlo.sh F none 0 0 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 0 $nevents GF*
    $Maindir/bin/run_amcatnlo.sh F none 0 1 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 1 $nevents GF*
    $Maindir/bin/run_amcatnlo.sh F none 0 2 $parton_shower
elif [[ $run_mode_num == 3 ]]; then
    $Maindir/bin/run_amcatnlo.sh F none 0 0 $parton_shower
    $Maindir/bin/run_amcatnlo.sh V none 0 0 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 0 $nevents GF* GV*
    $Maindir/bin/run_amcatnlo.sh F none 0 1 $parton_shower
    $Maindir/bin/run_amcatnlo.sh V none 0 1 $parton_shower
    $Maindir/SubProcesses/combine_results.sh 1 $nevents GF* GV*
    $Maindir/bin/run_amcatnlo.sh F none 0 2 $parton_shower
    $Maindir/bin/run_amcatnlo.sh V none 0 2 $parton_shower
else
    echo "Input not understood" $run_mode_num
    exit
fi

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

