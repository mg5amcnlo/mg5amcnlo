#!/bin/bash

find . -name *_G* | wc -l

if [[ -e res.txt ]]; then
    rm -r res.txt
fi
touch res.txt

for dir in $* ; do
    grep 'Final result' P*/$dir/log.txt >> res.txt
done


sed -i.bak s/"\+\/\-"/" \+\/\-"/ res.txt


./sumres.py

rm -r res.txt.bak

tail -n1 res.txt
