#!/bin/bash


if [[ -e res.txt ]]; then
    rm -r res.txt
fi

if [[ -e dirs.txt ]]; then
    rm -r dirs.txt
fi

touch res.txt
touch dirs.txt
NTOT=0
for dir in "$@" ; do
    let NTOT=$NTOT+`ls -d P*/$dir | wc -l`
    ls -d P*/$dir >> dirs.txt
    grep -H 'Final result' P*_[1-9]*/$dir/log.txt >> res.txt
done


sed -i.bak s/"\+\/\-"/" \+\/\-"/ res.txt

echo N of directories: $NTOT

./sumres.py $NTOT -1

rm -r res.txt.bak

tail -n1 res.txt
