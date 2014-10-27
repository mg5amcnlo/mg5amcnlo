#!/bin/bash

if [[ -e MadLoop5_resources.tar && ! -e MadLoop5_resources ]]; then
tar -xf MadLoop5_resources.tar
fi
k=%(name)s_app.log
script=%(script_name)s                         

j=%(directory)s
     if [[ ! -e $j ]]; then
          mkdir $j
     fi
     cd $j
     rm -f $k
     rm -f moffset.dat >& /dev/null
      echo   %(offset)s  > moffset.dat
     if  [[ -e ftn26 ]]; then
          cp ftn26 ftn25
     fi
     # create the input file
         echo "    %(nevents)s       %(maxiter)s       %(miniter)s" >& input_sg.txt
         echo "    %(precision)s" >> input_sg.txt
     if [[ ! -e ftn25 ]]; then
         echo "2" >> input_sg.txt   # grid refinement
         echo "1" >> input_sg.txt   # suppress amplitude

     else
         echo "%(grid_refinment)s" >> input_sg.txt
         echo "1" >> input_sg.txt
     fi
     echo "%(nhel)s" >> input_sg.txt
     echo "%(channel)s" >> input_sg.txt

     # run the executable. The loop is design to avoid
     # filesystem problem (executable not found)
     for((try=1;try<=16;try+=1)); 
     do
         ../madevent >> $k <input_sg.txt
         if [ -s $k ]
         then
             break
         else
             echo $try > fail.log 
         fi
     done
     echo "" >> $k; echo "ls status:" >> $k; ls >> $k
     cat $k >> log.txt
     
     if [[ -e ftn26 ]]; then
         cp ftn26 ftn25
     fi
     cd ../

