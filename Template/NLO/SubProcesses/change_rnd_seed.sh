#!/bin/bash

max=0
min=9999999
for p in P*_[1-9]*/vegas_rnd_seed ; do
   new=( $(head -n1 $p) )
   if [ $new -gt $max ] ; then
      max=$new
   fi
  if [ $new -lt $min ] ; then
      min=$new
   fi
done
echo 'largest and smallest seeds found:' $max $min

for p in P*_[1-9]*/vegas_rnd_seed ; do
   old=( $(head -n1 $p) )
   let new=old+max-min
   echo $new > $p
done

