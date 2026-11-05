#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Oct 2022) for the MG5aMC CUDACPP plugin.
# Integrated with the MadGraph7 project in Feb 2026.

# Temporary workaround for #403 (NB precision must be increased however see #542)
# Replace event-by-event helicity information in <events.lhe.in> by a dummy helicity "0." (floating point) in <events.lhe.out>
if [ "$1" == "$2" ] || [ "$2" == "" ] || [ "$3" != "" ]; then
  echo "Usage: $0 <events.lhe.in> <events.lhe.out>"
  exit 1
fi
infile=$1
outfile=$2

if [ ! -f ${infile} ]; then
  echo "ERROR! File not found: ${infile}"
  exit 1
fi

cat ${infile} | awk \
  'BEGIN{line=1000; npar=0}
   {if ($1=="<event>"){line=0; print $0}
    else {line=line+1;
          if (line==1){$3=sprintf("%.0E",$3); print $0} # keep only 1 digit "1." in XWGTUP (first one from sprintf is non-0) #542
          else print $0}}' > ${outfile}
