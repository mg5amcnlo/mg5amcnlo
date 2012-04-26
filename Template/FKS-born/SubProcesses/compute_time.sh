#!/bin/bash

time=0
njobs=0
for pdir in P*_* ; do
    for dir in $pdir/$@ ; do
	if [[ -e $dir/log.txt ]] ; then
	    x1=`grep 'Time in seconds' $dir/log.txt`
	    time=`expr $time + ${x1##* }`
	    njobs=`expr $njobs + 1`
	fi
    done
done

echo 'number of jobs found is' $njobs
printf "total time: %02d days %02d hours %02d minutes %02d seconds\n" "$((time/86400))" "$((time/3600%24))" "$((time/60%60))" "$((time%60))"
time=`expr $time/$njobs`
printf "average time per integration channel: %02d days %02d hours %02d minutes %02d seconds\n" "$((time/86400))" "$((time/3600%24))" "$((time/60%60))" "$((time%60))"
