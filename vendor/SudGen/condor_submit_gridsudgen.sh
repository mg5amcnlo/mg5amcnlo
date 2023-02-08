#!/bin/bash
nmass_dn=1
nmass_up=50
nmasses=$((nmass_up - nmass_dn + 1))

# generate file with dipole-mass nodal points
rm -f masses-condor.txt
touch masses-condor.txt
for (( imass=$nmass_dn; imass<=$nmass_up; imass++ ))
do
  echo $imass >> masses-condor.txt
done

# loop over itype
for (( itype=1; itype<=4; itype++ )) #itype in $(seq 1 1 4)
do
  # loop over ipart
  for (( ipart=1; ipart<=7; ipart++ )) #ipart in $(seq 1 1 7)
  do
    ./csub-local.sh microcentury $nmasses ./run_gridsudgen.sh $itype $ipart MASS
    mv submit-condor.sh submit-condor.sh-$itype\_$ipart
    ./submit-condor.sh-$itype\_$ipart
  done
done
