#!/bin/bash
TD=td_mac_intel
PSOPEN=open

# $? is the value of last executed command. A call to this function
# after a failure will cause the program to quit the script
function teststatus {
rc=$?
if [ 0 = $rc ]
then
:
else
echo $*, exit status=$rc 
exit $rc
fi
}

thisdir=`pwd`
if [ -f $thisdir/MADatNLO.top ]
then
rm -f $thisdir/MADatNLO*.top
fi
make read40
EXENAME=$thisdir/read40
echo -n "" > dir
counterp=0
for p in P* ; do
    cd $thisdir
    cd $p
    rm -f MADatNLO*
    rm -f *read40*.out
    echo -n "" > dir
    counter=0
    i=1
    for g in $* ; do
        if [ ! -f $g/MADatNLO*.top ]
        then
          echo "No topdrawer file in $p/$g"
	else
	counter=$(($counter + 1))
	echo $g"/MADatNLO.top" >> dir
	if [[ $(($counter % 40)) == 0 ]]; then
            echo -n "" > dir2
	    echo "40" > dir2
	    cat dir >> dir2
	    $EXENAME < dir2
            teststatus Failure in step 1
	    rm -f dir
	    rm -f dir2
	    mv fort.88 MADatNLO$i\.top 
	    mv read40.out "S1read40_"$i\.out
	    i=$(($i + 1))
	    echo -n "" > dir
        fi
	fi
    done
    if [[ $(($counter % 40)) != 0 ]] ; then
	echo $(($counter % 40)) > dir2
	cat dir >> dir2
	while [[ $(($counter % 40)) -ne 0 ]]; do
	    counter=$[$counter-1]
	done
	$EXENAME < dir2
	teststatus Failure in step 2
	rm -f dir
	rm -f dir2
	mv fort.88 MADatNLO$i\.top 
	mv read40.out "S2read40_"$i\.out
    fi
	
    counter=0
    for m in MADatNLO*.top ; do
	counter=$[$counter+1]
	echo $m >> dir
    done
    if [[ $counter -gt 1 ]] ; then
	echo $counter > dir2
	cat dir >> dir2
	$EXENAME < dir2
	teststatus Failure in step 3
	rm -f dir2
	mv fort.88 MADatNLO.top 
	mv read40.out S3read40.out
    else
	mv MADatNLO1.top MADatNLO.top
    fi
    rm -f dir
    cd $thisdir
    counterp=$(($counterp + 1))
    echo $p"/MADatNLO.top" >> dir
    echo $counterp $p "done"
done

cd $thisdir
make split40
./split40

i=0
#pdir* are created by split
for p in pdir* ; do
    i=$(($i+1))
    $EXENAME < $p    
    teststatus Failure in step 4
    mv fort.88 MADatNLO$i\.top
    mv read40.out "S4read40_"$i\.out
    rm $p
    echo $i $p "done"
done

if [[ $i -gt 1 ]] ; then
    echo $i > dir2
    for p in MADatNLO*.top ; do
	echo $p >>dir2
    done
    $EXENAME < dir2
    teststatus Failure in step 5
    rm -f dir
    rm -f dir2
    mv fort.88 MADatNLO.top 
    mv read40.out S5read40.out
else
    mv MADatNLO1.top MADatNLO.top
fi

if [ -f $thisdir/read40.errors ]
then
rm -f $thisdir/read40.errors
fi
find . -name S"*"read40"*".out -exec fgrep -il "READ40 ERROR READ40 ERROR" {} \; > $thisdir/read40.errors

if [ ! -s $thisdir/read40.errors ] ; then
    echo 'no errors found'
else
    echo 'ERRORS! see read40.errors for details'
fi

#./NLO_Born3.py
#$TD MADatNLO_combined.top "POSTSCR,ORIENT=3"
#$PSOPEN MADatNLO_combined.ps
##td2pdf MADatNLO_con
