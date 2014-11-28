#!/bin/bash
if [[ -e MadLoop5_resources.tar && ! -e MadLoop5_resources ]]; then
tar -xf MadLoop5_resources.tar
fi
k=run1_app.log
script=ajob1                         
offset=$1
shift
subdir=$offset
for i in $@ ; do
     j=G$i
     if [[ $offset == *.* ]];then
	 subdir=${offset%.*}
	 offset=${offset##*.}
	 j=G${i}_${subdir}	     
     elif [[ $offset -gt 0 ]]; then
	 j=G${i}_${subdir}
     fi
     if [[ ! -e $j ]]; then
         mkdir $j
     fi
     cd $j
     if [[ $offset -eq 0 ]]; then
	 rm -f ftn25 ftn26 ftn99
	 rm -f $k
     else
	 echo   "$offset"  > moffset.dat
     fi
     if [[ $offset -eq $subdir ]]; then
	 rm -f ftn25 ftn26 ftn99
	 rm -f $k
     else
        if [[ -e ../ftn25 ]]; then
	    cp ../ftn25 .
	fi
     fi
     if [[ ! -e input_app.txt  ]]; then
	 cat ../input_app.txt >& input_app.txt
     fi
     echo $i >> input_app.txt

     for((try=1;try<=10;try+=1)); 
     do
     ../madevent >> $k <input_app.txt
     if [ -s $k ]
     then
         break
     else
     sleep 1
     fi
     done
#     rm -f ftn25 ftn99
#     rm -f ftn26
     echo "" >> $k; echo "ls status:" >> $k; ls >> $k
     cp $k log.txt
     cd ../
done
