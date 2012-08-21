#!/bin/bash

# find the correct directory
if [[ ! -d ./SubProcesses ]]; then
    cd ../
fi
if [[ -d ./SubProcesses ]]; then
    cd SubProcesses
fi

if [[ $1 == "0" ]] ; then
    mint_mode=_MINT0
    shift
elif [[ $1 == "1" ]] ; then
    mint_mode=_MINT1
    shift
elif [[ $1 == "2" ]] ; then
    echo "Cannot combine results for mint_mode 2"
    exit
else
    mint_mode=
fi

if [[ -e res.txt ]]; then
    rm -f res.txt
fi
if [[ -e dirs.txt ]]; then
    rm -f dirs.txt
fi

touch res.txt
touch dirs.txt
NTOT=0
for dir in "$@" ; do
    N=`ls -d P*/$dir | wc -l`
    NTOT=`expr $NTOT + $N`
    ls -d P0*/$dir >> dirs.txt
    grep -H 'Final result:' P0*/$dir/log$mint_mode.txt >> res.txt
done

sed -i.bak s/"\+\/\-"/" \+\/\-"/ res.txt

echo N of directories: $NTOT

./sumres.py $NTOT -1

rm -r res.txt.bak

tail -n1 res.txt
