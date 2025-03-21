#!/bin/bash

# loop over itype
for (( itype=1; itype<=4; itype++ ))
do
  # loop over ipart
  for (( ipart=1; ipart<=7; ipart++ ))
  do
    rm -rf grid_$itype\_$ipart.txt
    touch grid_$itype\_$ipart.txt
    # loop over dipole mass
    for f in grid_$itype\_$ipart\_???.txt
    do
      cat $f >> grid_$itype\_$ipart.txt
    done
  done
done
