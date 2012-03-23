#!/bin/bash

listoffile="MGMEVersion.txt Cards lib/Pdfdata"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/madevent_vegas"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/*_G*/"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/config.fks"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/iproc.dat"
listoffile="$listoffile"" SubProcesses/randinit"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/integrate.fks"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/nbodyonly.fks"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/symfact.dat"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/HelasNLO.input"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/order.file"
listoffile="$listoffile"" SubProcesses/P*_[1-9]*/ajob*"
listoffile="$listoffile"" SubProcesses/madin.*"

echo "make tar of:" $listoffile

tar -cvzf ganga_madfks.tar.gz $listoffile
