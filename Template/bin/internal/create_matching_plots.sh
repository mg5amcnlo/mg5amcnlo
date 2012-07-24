#!/bin/bash

if [[ "$2" == "" ]];then
    echo "Error: Need run prefix and path to MadAnalysis"
    exit
fi

if [[ "$3" != "" ]];then
    MAdir=$3
else
    MAdir=../../../../MadAnalysis
fi

if [[ ! -e `which root` ]];then
    if [[ ! -e "$ROOTSYS/bin/root" ]];then
        echo "Error: root executable not found"
        exit
    fi
    export PATH=$ROOTSYS/bin:$PATH
fi

if [[ ! -e events.tree || ! -e xsecs.tree ]];then
    echo "No events.tree or xsecs.tree files found"
    exit
fi

echo Running root
root -q -b -l ../bin/internal/read_tree_files.C &> /dev/null
echo Creating plots
root -q -b -l ../bin/internal/create_matching_plots.C &> /dev/null
mv pythia.root $1/$2_pythia.root

dir=../HTML/$1/plots_pythia_$2

if [[ ! -d  $dir ]];then
  mkdir $dir
fi
for i in DJR*.eps; do mv $i $dir/${i%.*}.ps;done

cd $dir

if [[ ! -d $MAdir ]];then exit; fi

for file in DJR?.ps ; do
  echo ">> Converting file $file" >> log.convert
  $MAdir/epstosmth --gsopt='-r60x60 -dGraphicsAlphaBits=4' --gsdev=jpeg $file
done

