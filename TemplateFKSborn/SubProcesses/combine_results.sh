#!/bin/bash

# find the correct directory
if [[ ! -d ./SubProcesses ]]; then
    cd ../
fi
if [[ -d ./SubProcesses ]]; then
    cd SubProcesses
fi

grep 'Final result \[ABS\]' P*/G*/res_$1 > res.txt
if [[ $1 == '0' ]] ; then
    echo 'Determining the number of unweighted events per channel'
elif [[ $1 == '1' ]] ; then
    echo 'Updating the number of unweighted events per channel'
fi
echo $2 > nunwgt
f77 -o sumres2 sumres2.f
./sumres2 < nunwgt
rm -f nunwgt
tail -n1 res.txt
mv res.txt res_$1_abs.txt

grep 'Final result:' P*/G*/res_$1 > res.txt
f77 -o sumres sumres.f
./sumres
tail -n1 res.txt
mv res.txt res_$1_tot.txt
