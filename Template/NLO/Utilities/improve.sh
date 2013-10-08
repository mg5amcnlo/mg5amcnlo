#! /bin/bash


#---------------------------
# Update random number seed
r=0
if [[ -e randinit ]]; then
    source ./randinit
fi
for i in P*_* ; do
    r=`expr $r + 1`
done
echo "r=$r" >& randinit
#--------------------------

run_cluster=1

for p in P*_[1-9]* ; do
    echo "Processing $p"
    cd $p
    rm -f ajob[!0-9]*
    rm -f mg[!0-9]*.cmd
    if [[ ! -e "improve.f" ]] ; then
	ln -s  ../improve.f .
    fi
    gfortran -ffixed-line-length-132 -o improve improve.f
    echo $run_cluster | ./improve
    if [[ $run_cluster == 1 ]] ; then
	chmod +x ajob*
	for job in mg[!0-9]*.cmd ; do
	    if [ -e $job ] ; then
                condor_submit $job
	    fi
	done
    else
	echo 'can only run on cluster' $run_cluster
	exit
    fi
    cd ..
done