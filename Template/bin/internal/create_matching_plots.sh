#!/bin/bash

if [[ "$1" == "" ]];then
    echo "Error: Need run prefix"
    exit
fi
if [[ ! -e `which root` ]];then
    if [[ "$ROOTSYS" == "" ]];then
        echo "Error: root executable not found"
        exit
    fi
    export PATH=$ROOTSYS/bin:$PATH
fi
donerun=0
if [[ -e $1/$2_pythia_events.tree.gz ]];then
    donerun=1
    echo gunzip $1/$2_pythia_events.tree.gz
    gunzip -c $1/$2_pythia_events.tree.gz > events.tree
    cp $1/$2_pythia_xsecs.tree xsecs.tree
fi

if [[ ! -e events.tree || ! -e xsecs.tree ]];then
    echo "No events.tree or xsecs.tree files found"
    exit
fi
echo Running root
root -q -b ../bin/internal/read_tree_files.C &> /dev/null
echo Creating plots
root -q -b ../bin/internal/create_matching_plots.C &> /dev/null
mv pythia.root $1/$2_pythia.root

if [[ ! -d ../HTML/$1/plots_pythia_$2 ]];then
  mkdir ../HTML/$1/plots_pythia_$2
fi
for i in DJR*.eps; do mv $i ../HTML/$1/plots_pythia_$2/${i%.*}.ps;done
if [[ donerun -eq 1 ]];then
    rm events.tree xsecs.tree
fi


