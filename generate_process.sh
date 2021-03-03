#!/bin/bash

PROCESS=$1
OUTNAME=$2

rm -rf $OUTNAME-def
rm -rf $OUTNAME-FA4

MODEL1=loop_qcd_qed_sm-no_widths
MODEL2=loop_qcd_qed_sm_FAconv4-no_widths

MG=./bin/mg5_aMC


LAUNCH="launch -i\ncompile FO\n0\nset iseed 10\nset mt 174.3\nset ymt 174.3"

printf "import model $MODEL1\n define heavy = w+ w- z h t t~\n define peasy = u u~ d\n generate $PROCESS\n output $OUTNAME-def \n$LAUNCH" |$MG > log_$OUTNAME-def.txt &
printf "import model $MODEL2\n define heavy = w+ w- z h t t~\n define peasy = u u~ d\n generate $PROCESS\n output $OUTNAME-FA4 \n$LAUNCH" |$MG > log_$OUTNAME-fa4.txt 

sleep 10

python ./merge_sud_approx.py $OUTNAME-def $OUTNAME-FA4
