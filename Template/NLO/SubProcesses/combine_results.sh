#!/bin/bash

# find the correct directory
if [[ ! -d ./SubProcesses ]]; then
    cd ../
fi
if [[ -d ./SubProcesses ]]; then
    cd SubProcesses
fi

if [[ -e res.txt ]]; then
    rm -f res.txt
fi
if [[ -e dirs.txt ]]; then
    rm -f dirs.txt
fi
if [[ -e nevents_unweighted ]]; then
    rm -f nevents_unweighted
fi

arg1=$1
arg2=$2
arg3=$3
# shift the list of arguments by 3
shift
shift
shift
if [[ "$@" == "" ]]; then
    echo "Please give the G directories that should be combined,"
    echo "e.g. 'GF* GV*', as final arguments of this script" 
    exit
fi

touch res.txt
touch dirs.txt
NTOT=0
for dir in "$@" ; do
    N=`ls -d P*/$dir | wc -l`
    NTOT=`expr $NTOT + $N`
    ls -d P*/$dir >> dirs.txt
    grep -H 'Final result' P*/$dir/res_$arg1 >> res.txt
done
echo N of directories: $NTOT
if [[ $arg1 == '0' ]] ; then
    echo 'Determining the number of unweighted events per channel'
elif [[ $arg1 == '1' ]] ; then
    echo 'Updating the number of unweighted events per channel'
fi
./sumres.py $NTOT $arg2 $arg3

echo 'Integrated abs(cross-section)' 
tail -n2 res.txt | head -n1
echo 'Integrated cross-section' 
tail -n1 res.txt
mv res.txt res_$arg1.txt
