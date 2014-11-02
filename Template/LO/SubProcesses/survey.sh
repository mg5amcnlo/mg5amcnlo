#!/bin/bash
if [[ -e MadLoop5_resources.tar && ! -e MadLoop5_resources ]]; then
tar -xf MadLoop5_resources.tar
fi
k=run1_app.log
script=ajob1                         
offset=$1
shift
for i in $@ ; do
     j=G$i
     echo $offset  > moffset.dat
     if [[ $offset -gt 0 ]]; then
	 j=G${i}_${offset}
     fi
     if [[ ! -e $j ]]; then
         mkdir $j
     fi
     cd $j
     rm -f ftn25 ftn26 ftn99
     rm -f $k
     cat ../input_app.txt >& input_app.txt
     echo $i >> input_app.txt
     for((try=1;try<=10;try+=1)); 
     do
     ../madevent > $k <input_app.txt
     if [ -s $k ]
     then
         break
     else
     sleep 1
     fi
     done
     rm -f ftn25 ftn99
     rm -f ftn26
     echo "" >> $k; echo "ls status:" >> $k; ls >> $k
     cp $k log.txt
     cd ../
done
