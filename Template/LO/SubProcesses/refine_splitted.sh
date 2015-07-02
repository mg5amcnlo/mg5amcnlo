#!/bin/bash

# For support of LHAPATH in cluster mode
if [ $CLUSTER_LHAPATH ]; then
  export LHAPATH=$CLUSTER_LHAPATH;
fi
if [[ -e MadLoop5_resources.tar.gz && ! -e MadLoop5_resources ]]; then
tar -xzf MadLoop5_resources.tar.gz
fi
k=run1_app.log
script=refine_splitted.sh
# Argument
# 1st argument the Directory name
grid_directory=$1;
# 2st argument Directory where to find the grid input
base_directory=$2; 
# 3st argument the offset
offset=$3;


# prepare the directory where to run
if [[ ! -e $grid_directory ]]; then
    # Do not exists     
    mkdir $grid_directory;
else
   rm -rf $grid_directory/$k;
   rm -rf $grid_directory/input_app.txt;
   rm -rf $grid_directory/ftn25;
   rm -rf $grid_directory/ftn26;
fi
# handle input file
if [[ -e $base_directory ]]; then
    cp $base_directory/ftn26 $grid_directory/ftn25;
    cp $base_directory/input_app.txt $grid_directory/input_app.txt;
elif [[ -e ./ftn26 ]]; then
    cp ./ftn26 $grid_directory/ftn25;
    cp ./input_app.txt $grid_directory/input_app.txt;
else
    exit 1;
fi

# Move to the running directory
cd $grid_directory;

# Put the correct offset
rm -f moffset.dat >& /dev/null;
echo   $offset > moffset.dat;

# run the executable. The loop is design to avoid
# filesystem problem (executable not found)
for((try=1;try<=16;try+=1)); 
do
    ../madevent 2>&1 >> $k <input_app.txt | tee -a $k;
    status_code=${PIPESTATUS[0]};
    if [ -s $k ]
    then
        break
    else
        echo $try > fail.log 
    fi
done
echo "" >> $k; echo "ls status:" >> $k; ls >> $k
# Perform some cleaning to keep less file on disk/transfer less file.
subdir=${grid_directory##*_}
if [[ $subdir -ne 1 &&  -s results.dat && $MG5DEBUG != true ]]; then
	 rm -f ftn25 &> /dev/null
         rm -f ftn26 &> /dev/null
	 rm -f log.txt &> /dev/null
         rm -f *.log &> /dev/null
	 rm -f moffset.dat &> /dev/null
	 rm -f fail.log &> /dev/null
fi
if [[ $status_code -ne 0 ]]; then 
	 rm results.dat
	 echo "ERROR DETECTED"
	 echo "end code not correct $status_code" > results.dat
fi

cd ../