#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Apr 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2024).
SCRDIR=$(cd $(dirname $0); pwd)
TOPDIR=$(dirname $SCRDIR) # e.g. epochX/cudacpp if $SCRDIR=epochX/cudacpp/CODEGEN

if [ "$1" == "" ]; then
  echo "Usage: $0 <procdir1> [<procdir2>...]"
  exit 1
fi
procdirs=$@

for procdir in $procdirs; do
  if [ ! -d $TOPDIR/$procdir ]; then echo "ERROR! Directory not found $TOPDIR/$procdir"; exit 1; fi
done

# Add to the git repo COPYRIGHT, COPYING and COPYING.LESSER (but not AUTHORS) in <procdir> if they exist
for procdir in $procdirs; do
  echo "=== $TOPDIR/$procdir"
  cd $TOPDIR/$procdir
  for fcopy in COPYRIGHT COPYING COPYING.LESSER; do
    if [ -f ${fcopy} ]; then git add ${fcopy}; fi
  done
done

