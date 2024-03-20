#!/bin/bash
nmass_dn=1
nmass_up=50
nmasses=$((nmass_up - nmass_dn + 1))

# input parameters
stlow=1.0
stupp=7000.0
xmlow=1.0
xmupp=7000.0
xlowthrs=0.001
ifk88seed=-1
exe=$(pwd)/gridsudgen_clust

# write script to that will be executed on cluster nodes
rm -rf run_gridsudgen.sh
touch run_gridsudgen.sh
chmod +x run_gridsudgen.sh

echo "#!/bin/bash" > run_gridsudgen.sh
echo " " >> run_gridsudgen.sh
echo $exe "<<EOF" >> run_gridsudgen.sh
cat <<EOF >> run_gridsudgen.sh
$stlow $stupp
$xmlow $xmupp
$xlowthrs
$ifk88seed
EOF
echo "$"1 "$"2 "$"3 >> run_gridsudgen.sh
echo "EOF" >> run_gridsudgen.sh

# write input file for grid combination
rm -rf gridsudcomb_input
touch gridsudcomb_input
cat <<EOF >> gridsudcomb_input
$stlow $stupp
$xmlow $xmupp
$xlowthrs
$ifk88seed
1 1
EOF

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
