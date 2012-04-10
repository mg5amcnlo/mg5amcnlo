#! /bin/bash

echo '****************************************************'
echo 'This script compiles, tests and runs a madfks process'
echo '****************************************************'


# find the correct directory
if [[  ! -d ./SubProcesses  ]]; then
    cd ../
    if [[ ! -d ./SubProcesses ]]; then
	echo "Error: compile_madfks.sh must be executed from the main, or bin directory"
	exit
    fi
fi


source Source/fj_lhapdf_opts
Maindir=`pwd`

libdir=$Maindir/lib
CutToolsdir=$Maindir/../CutTools
carddir=$Maindir/Cards


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


echo ''
echo 'Press "0" for "NO" and "1" for "YES"'

echo 'Compile and run tests?'
read test
if [[ $test != "1" ]] ; then
    test="0"
fi

if [[ $test == "1" ]] ; then
    echo 'Enter number of points for tests'
    read points
fi

echo 'Compile and run gensym?'
read gensym
if [[ $gensym != "1" ]] ; then
    gensym="0"
fi

echo 'Compile madevent?'
read madevent_compile
if [[ $madevent_compile != "1" ]] ; then
    madevent_compile="0"
fi

#echo 'Run madevent?'
#read madevent_run
madevent_run="0"
if [[ $madevent_run != "1" ]] ; then
    madevent_run="0"
fi


echo 'FKS directories to do n-body only (with S-functions = 1)?'
echo '  0 for all dirs, 1 for n-body only, 2 for all but n-body only'
read nbodyonly
if [[ $nbodyonly != "1" && $nbodyonly != "2" ]] ; then
    nbodyonly="0"
fi

if [[ $gensym == "1" || $madevent_run == "1" ]] ; then
    echo 'press: "0" for local run, "1" for Condor cluster'
    read run_cluster
    if [[ $run_cluster != "1" ]] ; then
	run_cluster="0"
    fi
fi

if [[ $gensym == "1" || $madevent_compile == "1" ]] ; then
    echo 'press: "0" for Vegas, "1" for Mint, "2" for MintMC"'
    read vegas_mint
    if [[ $vegas_mint == "0" ]] ; then
	executable='madevent_vegas'
    elif [[ $vegas_mint == "1" ]] ; then
	executable='madevent_mint'
    elif [[ $vegas_mint == "2" ]] ; then
	executable='madevent_mintMC'
    fi
    if [[ $vegas_mint != "1" && $vegas_mint != "2" ]] ; then
	vegas_mint="0"
    fi
fi

echo 'all FKS directories/SubProcesses?'
read all_dirs

if [[ $all_dirs != '1' ]] ; then
    echo 'Give P* directory(ies)'
    read dirs
else
    cd SubProcesses
    dirs=`ls -d P*/R*`
    born_dirs=`ls -d P0_*`
    cd ..
fi

echo 'Enter number of cores to use for compilation'
read j

if [[ $test == "1" ]] ; then
    echo 'Running the tests for' $points 'points'
# input for the tests (test_ME, test_Sij & testMC)
    echo '1' > $Maindir/input_MC
    echo '1 -0.1' >> $Maindir/input_MC
    echo '-1 -0.1' >> $Maindir/input_MC
    echo '-2 -2' > $Maindir/input_Sij
    echo '-2 -2' > $Maindir/input_ME
    echo '-2 -2' >> $Maindir/input_MC
    echo $points $points >> $Maindir/input_Sij
    echo $points $points >> $Maindir/input_ME
    echo $points $points >> $Maindir/input_MC
    echo '-1' >> $Maindir/input_Sij
    dir=`ls -d SubProcesses/P*/R* | tail -n1`
    if [[ -e $dir"/helicities.inc" ]]; then
	echo '1' >> $Maindir/input_ME
	echo '1' >> $Maindir/input_MC
    else
	echo '0' >> $Maindir/input_ME
	echo '0' >> $Maindir/input_MC
    fi
    echo '-1' >> $Maindir/input_ME
    echo '-1' >> $Maindir/input_MC
    echo 'results from test_Sij' > $Maindir/test_Sij.log
    echo 'results from test_ME' > $Maindir/test_ME.log
    echo 'results from test_MC' > $Maindir/test_MC.log
fi

if [[ $gensym == '1' ]] ; then
    echo 'results from gensym' > $Maindir/gensym.log
fi
echo 'compilation results' > $Maindir/compile_madfks.log
echo 'compilation results for madloop' > $Maindir/compile_madloop.log

# Source directory
if [[  $gensym == '1' || $madevent_compile == '1' ]]; then
    # Source
    cd Source
    echo 'compiling Source...'
    echo 'compiling Source' >>  $Maindir/compile_madfks.log
    make -j$j >>  $Maindir/compile_madfks.log 2>&1
    echo '...done'
    cd ..

fi

cd $Maindir/SubProcesses

echo 'continuing with the P* directories...'
echo ''

if [[ $gensym == '1' || $madevent_compile == '1' ]]; then
    dir=`ls -d P*/R* | tail -n1`
    if [[ -e $dir"/helicities.inc" ]]; then
	echo 'helicities.inc found: it is recommended to MC over helicities'
	echo 'converting fks_singular.f and fks_singularMC.f'
	sed -i.hel 's/HelSum=.true./HelSum=.false./g;s/chel  include "helicities.inc"/      include "helicities.inc"/g' fks_singular.f
	sed -i.hel 's/HelSum=.true./HelSum=.false./g;s/chel  include "helicities.inc"/      include "helicities.inc"/g' fks_singularMC.f
	echo "originals are backup'ed as fks_singular.f.hel and fks_singularMC.f.hel"
    else
	echo 'helicities.inc NOT found, can only do explicit helicity sum'
        echo 'converting fks_singular.f and fks_singularMC.f'
	sed -i.hel 's/HelSum=.false./HelSum=.true./g;s/      include "helicities.inc"/chel  include "helicities.inc"/g' fks_singular.f
	sed -i.hel 's/HelSum=.false./HelSum=.true./g;s/      include "helicities.inc"/chel  include "helicities.inc"/g' fks_singularMC.f
	echo "originals are backup'ed as fks_singular.f.hel and fks_singularMC.f.hel"
    fi
fi

#
# TODO: NEED TO CONVERT MAKEFILES HERE
#
# First compile the virtuals which are in the P0*/V0* directory and linke the
# libMadLoop.a to the R* folders whi need it
if [[ $madevent_compile == '1' ]]; then
echo "Compiling virtual maxtrix elements"
    for dir in $born_dirs ; do
        cd $dir
        v_dirs=`ls -d V0*`
        for vdir in $v_dirs ; do
            cd $vdir
            echo "Compiling MadLoop MatrixElement in " $vdir
            make >> $Maindir/compile_madloop.log 2>&1
            cd ..
        done
        r_dirs=`ls -d R_*`
        for rdir in $r_dirs ; do
            if [[ ("$(tail -n 1 $rdir/integrate.fks)" == "I") && $(ls -d V0_* | wc -l) != 0  ]] ; then
                if [ -L $rdir/libMadLoop.a ] ; then 
                  rm -f  $rdir/libMadLoop.a
                fi
                ln -s `pwd`/libMadLoop.a `pwd`/$rdir/
            fi
        done

        cd ..
    done
fi



for dir in $dirs ; do
    cd $dir
    echo $dir

    if [[ ($nbodyonly == "1" && "$(head -n 1 nbodyonly.fks)" == "Y") || $nbodyonly == "0" || ($nbodyonly == "2" && "$(head -n 1 nbodyonly.fks)" == "N") ]] ; then
	echo $dir >> $Maindir/compile_madfks.log


#
# COMPILE AND RUN TESTS
#
    if [[ $test == '1' ]]; then
	echo $dir >> $Maindir/test_Sij.log
	echo $dir >> $Maindir/test_ME.log
	echo $dir >> $Maindir/test_MC.log
	echo '     make test_Sij...'
	make -j$j test_Sij >> $Maindir/compile_madfks.log 2>&1
	if [[ -e "test_Sij" ]]; then
	    echo '     ...running test_Sij'
	    ./test_Sij < $Maindir/input_Sij | tee -a $Maindir/test_Sij.log | grep 'Failures (fraction)'
	else
	    echo 'ERROR in compilation, see compile_madfks.log for details'
	fi
	echo '     make test_ME...'
	make -j$j test_ME >> $Maindir/compile_madfks.log 2>&1
	if [[ -e "test_ME" ]]; then
	    echo '     ...running test_ME'
	    ./test_ME < $Maindir/input_ME | tee -a $Maindir/test_ME.log | grep 'Failures (fraction)'
	else
	    echo 'ERROR in compilation, see compile_madfks.log for details'
	fi
	echo '     make test_MC...'
	make -j$j test_MC >> $Maindir/compile_madfks.log 2>&1
	if [[ -e "test_MC" ]]; then
	    echo '     ...running test_MC'
	    ./test_MC < $Maindir/input_MC | tee -a $Maindir/test_MC.log | grep 'Failures (fraction)'
	else
	    echo 'ERROR in compilation, see compile_madfks.log for details'
	fi
    fi

#
# COMPILE AND RUN GENSYM
#
    if [[ $gensym == '1' ]] ; then
	echo $dir >> $Maindir/gensym.log
	echo '     make gensym...'
	if [[ -e "gensym" ]]; then
	    rm -f gensym
	fi
	make -j$j gensym >> $Maindir/compile_madfks.log 2>&1
	if [[ -e "gensym" ]]; then
	    echo '     ...running gensym for' $executable
	    echo $run_cluster $vegas_mint | ./gensym >> $Maindir/gensym.log
	else
	    echo 'ERROR in compilation, see compile_madfks.log for details'
	fi
    fi

#
# COMPILE MADEVENT
#
    if [[ $madevent_compile == "1" ]] ; then
        if [[ ("$(tail -n 1 integrate.fks)" == "I") && $(ls -d ../V0_* | wc -l) != 0  ]] ; then
            export madloop=true
        else
            unset madloop
        fi
	echo '     make' $executable...
	if [[ -e $executable ]]; then
	    rm -f $executable
	fi
	make -j$j $executable >> $Maindir/compile_madfks.log 2>&1
	if [[ -e "madevent_vegas" ]]; then
	    echo "     madevent_vegas compiled"
	else
	    echo 'ERROR in compilation, see compile_madfks.log for details'
	fi
    fi

#
# TODO: RUN MADEVENT
#
    if [[ $madevent_run == "1" ]] ; then
	echo 'automatic running not yet implemented.'
	echo 'To run, goto "./SubProcesses/" direcotry and execute "./run.sh"'
    fi


    else
	echo '     No need for this dir: doing n-body only'
    fi
    cd ../..
done

if [[ $test == "1" ]]; then
    rm  $Maindir/input_Sij
    rm  $Maindir/input_ME
    rm  $Maindir/input_MC
fi

echo '*******************************************************************'
echo 'Please, check the log files in the main directory for details:'
echo 'link_fks.log, test_*.log, NLOComp.log and gensym.log'
echo 'Compilation warnings/errors can be found in compile_madfks.log'
echo '*******************************************************************'
