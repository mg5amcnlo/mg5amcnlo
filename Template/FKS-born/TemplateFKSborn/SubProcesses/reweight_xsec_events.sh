#!/bin/bash

# Script to compile and execute reweight_xsec_events in all the
# SubProcesses/P*/G* directories. It asks for inputs (see the
# reweight_doc.txt for more details). This program takes a standard
# event file (in the SubProcesses/P*/G* directories) and can compute
# the scale and PDF uncertainties by reweighting.
# Note that when not using LHAPDF, it's not possible to do PDF error
# estimations.

# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: reweight_xsec_events.sh must be executed from the SubProcesses directory"
	exit
    fi
fi
cd SubProcesses



if [[ $run_cluster == "" ]] ; then
    echo "Local run (0) or cluster running (1)?"
    read run_cluster
fi
if [[ $run_cluster == 0 ]] ; then
    echo "Running locally"
    echo 'Enter event file name (default is "events.lhe")'
    read event_file
    echo $event_file > reweight_xsec_events.input
elif [[ $run_cluster == 1 ]] ; then
    echo "submitting jobs to cluster"
    echo 'Can only reweight files called "events.lhe"'
    event_file=events.lhe
    echo $event_file > reweight_xsec_events.input
else
    echo "ERROR" $run_cluster
    exit
fi

#echo 'Enter 1 to save all cross sections on tape'
#echo '      0 otherwise'
#read isave
echo 1 >> reweight_xsec_events.input

echo 'Enter 1 to compute scale uncertainty'
echo '      0 otherwise'
read imu
echo $imu >> reweight_xsec_events.input

echo 'Enter 1 to compute PDF uncertainty'
echo '      0 otherwise'
read ipdf
echo $ipdf >> reweight_xsec_events.input

echo 'Enter QES_over_ref used in the reference computation'
read xQES_over_ref
echo $xQES_over_ref >> reweight_xsec_events.input

echo 'Enter muR_over_ref, muF1_over_ref(=muF2_over_ref)'
echo '  used in the reference computation'
read -e xmuR_over_ref
#echo $xmuR_over_ref $xmuF1_over_ref >> reweight_xsec_events.input
echo $xmuR_over_ref >> reweight_xsec_events.input

if [[ ! $imu == 0 ]]; then
    echo 'Enter renormalization scale variation range'
    echo '  (e.g., 0.5 2.0)'
    read -e yfactR2
    echo $yfactR2 >> reweight_xsec_events.input
    echo 'Enter factorization scale variation range'
    echo '  (e.g., 0.5 2.0)'
    read -e yfactF2
    echo $yfactF2 >> reweight_xsec_events.input
fi

if [[ ! $ipdf == 0 ]]; then
    echo 'Enter id number of central set'
    read idpdf0
    echo $idpdf0 >> reweight_xsec_events.input
    echo 'Enter id numbers of first and last error set'
    read -e idpdf1
    echo $idpdf1 >> reweight_xsec_events.input
fi

if [[ $run_cluster == 1 ]] ; then
    chmod +x reweight_xsec_events.cluster
fi

for p in P*_[1-9]* ; do
    cd $p
    echo "Running in" $p
    make reweight_xsec_events >/dev/null
    for G in G* ; do
	if [[ -e $G ]] ; then
	    cd $G
	    if [[ -e $event_file ]] ; then
		if [[ $run_cluster == 1 ]] ; then
		    ln -sf ../../reweight_xsec_events.cmd .
		    ln -sf ../../reweight_xsec_events.cluster .
		    condor_submit reweight_xsec_events.cmd
		elif [[ $run_cluster == 0 ]] ; then
   		    ../reweight_xsec_events < ../../reweight_xsec_events.input > reweight_xsec_event.output
		fi
	    fi
	    cd ..
	fi
    done
    cd ..
done
