#!/bin/bash

if [[ "$1" == "" ]];then
    echo "Error: Need run prefix"
    exit
fi
if [[ ! -e `which root` ]];then
    echo "Error: root executable not found"
    exit
fi
donerun=0
if [[ -e $1_events.tree.gz ]];then
    donerun=1
    echo gunzip $1_events.tree.gz
    gunzip -c $1_events.tree.gz > events.tree
    cp $1_xsecs.tree xsecs.tree
fi
if [[ ! -e events.tree || ! -e xsecs.tree ]];then
    echo "No events.tree or xsecs.tree files found"
    exit
fi
echo Running root
root -q -b ../bin/read_tree_files.C
echo Creating plots
root -q -b ../bin/create_matching_plots.C
mv pythia.root $1_pythia.root
if [[ ! -d $1_pythia ]];then
  mkdir $1_pythia
fi
for i in DJR*.eps; do mv $i $1_pythia/${i%.*}.ps;done
if [[ donerun -eq 1 ]];then
    rm events.tree xsecs.tree
fi


