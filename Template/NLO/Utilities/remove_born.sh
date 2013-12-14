#!/bin/bash

for  p in P1* ; do
    cd $p
    cp born/G*/integrate.fks .
    if [[ "$(tail -n 1 integrate.fks)" == "E" ]] ; then
	mv born XXX_born
    fi
    cd ..
done
