#!/bin/bash
# Main driver for the combination. 
# To be executed in ./SubProcesses; the upper-level command
# (the only one to be executed) is:
#
#    steerall born_G*
#    steerall all_G*
#    steerall $1
#

#It is recommended to use a very recent version of root.
#On SLC6 maching, from a tcsh shell execute e.g.
# setenv ROOTSYS /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.11/x86_64-slc6-gcc46-dbg/root/
# set path= ( $ROOTSYS/bin $path )
# setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${ROOTSYS}/lib
#

function combine_root_files {
#Removes target file MADatNLO.root if present,
#and writes a file (temp_root_files.txt) with the list
#of input files. These are searched in the directories
# ./P*/$1

if [ -f temp_root_files.txt ]
then
  rm -f temp_root_files.txt
fi

# Remove target file if already present: no warning is issued
if [ -f MADatNLO.root ]
then
  rm -f MADatNLO.root
fi

thisdir=`pwd`
i=0
for p in P* ; do
  cd $p
  for el in $* ; do
       i=`expr $i + 1`
       echo $thisdir/$p/$el"/MADatNLO.root" >> ../temp_root_files.txt
  done
  cd ../
done

if [ -f definitely_temporary.txt ]
then
  \rm definitely_temporary.txt
fi

echo $i >> definitely_temporary.txt
echo $thisdir >> definitely_temporary.txt
cat temp_root_files.txt >> definitely_temporary.txt
mv -f definitely_temporary.txt temp_root_files.txt
}


function steerall {
combine_root_files $1

if [ -f rootinput.txt ]
then
  rm -f rootinput.txt
fi
echo ".x combine_root.C" >> rootinput.txt
echo ".q" >> rootinput.txt

root -b < rootinput.txt

rm -f rootinput.txt
}

steerall $1

rm -f AutoDict_vector_TH1D*

