#!/bin/bash


time=0
njobs=0
for pdir in P0*_* ; do
    for dir in $pdir/$@ ; do
	x1=`grep 'Time in seconds' $dir/log.txt`
	time=`expr $time + ${x1##* }`
	njobs=`expr $njobs + 1`
    done
done

printf "total time: %02d days %02d hours %02d minutes %02d seconds\n" "$((time/86400))" "$((time/3600%24))" "$((time/60%60))" "$((time%60))"

time=`expr $time/$njobs`
printf "average time per integration channel: %02d days %02d hours %02d minutes %02d seconds\n" "$((time/86400))" "$((time/3600%24))" "$((time/60%60))" "$((time%60))"
