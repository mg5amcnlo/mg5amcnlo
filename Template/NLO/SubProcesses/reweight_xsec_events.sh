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

source Source/fj_lhapdf_opts
Maindir=`pwd`
libdir=$Maindir/lib
c=`awk '/^[^#].*=.*pdlabel/{print $1}' Cards/run_card.dat`
if [[ $c == "'lhapdf'" ]]; then
    echo Using LHAPDF interface!
    LHAPDF=`$lhapdf_config --libdir`
    LHAPDFSETS=`$lhapdf_config --pdfsets-path`
    export lhapdf=true
    if [ ! -f $libdir/libLHAPDF.a ]; then 
      ln -s $LHAPDF/libLHAPDF.a $libdir/libLHAPDF.a 
    fi
    if [ ! -d $libdir/PDFsets ]; then 
      ln -s $LHAPDFSETS $libdir/. 
    fi
else
    unset lhapdf
fi

cd SubProcesses

echo "This script compiles the reweight_xsec_code and runs it on the event files found in the nevents_unweighted file"
rm -f reweight_xsec_events.input
touch reweight_xsec_events.input

if [[ $run_cluster == "" ]] ; then
    echo "Local run (0) or cluster running (1)?"
    read run_cluster
fi
#if [[ $run_cluster == 0 ]] ; then
#    echo "Running locally"
#    echo 'Enter event file name (default is "events.lhe")'
#    read event_file
#    echo $event_file >> reweight_xsec_events.input
#elif [[ $run_cluster == 1 ]] ; then
#    echo "submitting jobs to cluster"
#    echo 'Can only reweight files called "events.lhe"'
#    event_file=events.lhe
#    echo $event_file >> reweight_xsec_events.input
#else
#    echo "ERROR" $run_cluster
#    exit
#fi

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

echo 'Compiling reweight code in P* directories'
for p in P* ; do
    cd $p
    make reweight_xsec_events >/dev/null
    cd ..
done


# Check the nevents_unweighted file to check which event files need reweighting
echo 'Looking in the nevents_unweighted file to see which event files are needed for reweighting'

i=0
for line in $(cat nevents_unweighted) ; do
    i=`expr $i + 1`
    if [[ $i == 1 ]] ; then
	pdir=$(echo $line | cut -d '/' -f1)
	gdir=$(echo $line | cut -d '/' -f2)
	event_file=$(echo $line | cut -d '/' -f3)
	echo 'found in nevents_unweighted:' $pdir'/'$gdir'/'$event_file
	cd $pdir
	make reweight_xsec_events >/dev/null
	cd $gdir
	if [[ -e $event_file ]] ; then
	    echo $event_file > reweight_xsec_events.input	
	    cat ../../reweight_xsec_events.input >> reweight_xsec_events.input
	    if [[ $run_cluster == 1 ]] ; then
		ln -sf ../../reweight_xsec_events.cmd .
		ln -sf ../../reweight_xsec_events.cluster .
		condor_submit reweight_xsec_events.cmd
	    elif [[ $run_cluster == 0 ]] ; then
   		../reweight_xsec_events < reweight_xsec_events.input > reweight_xsec_events.output
		if [[ `tail -n1 $event_file.rwgt` != '</LesHouchesEvents>' ]] ; then
		    echo 'ERROR in reweighting for file' $pdir'/'$gdir'/'$event_file
		    echo '      check the' $pdir'/'$gdir'/reweight_xsec_events.output file for more details'
		    echo 'quitting'
		    exit
		fi
	    fi
    	else
	    echo 'ERROR event file not found:' $pdir'/'$gdir'/'$event_file
	    echo 'quitting...'
	    exit
	fi
	cd ../..
    fi
    if [[ $i == 3 ]] ; then
	i=0
    fi
done

echo 'Updating the nevents_unweighted file with the new event file names'
i=0
rm -f nevents_unweighted_reweight
touch nevents_unweighted_reweight
for line in $(cat nevents_unweighted) ; do
    i=`expr $i + 1`
    if [[ $i == 1 ]] ; then
	pdir=$(echo $line | cut -d '/' -f1)
	gdir=$(echo $line | cut -d '/' -f2)
	event_file=$(echo $line | cut -d '/' -f3)
    fi
    if [[ $i == 2 ]] ; then
	nevents=$line
    fi
    if [[ $i == 3 ]] ; then
	cross_section=$line
	echo ' '$pdir'/'$gdir'/'$event_file'.rwgt' '      ' $nevents '      ' $cross_section >> nevents_unweighted_reweight
	i=0
    fi
done

mv nevents_unweighted_reweight nevents_unweighted
