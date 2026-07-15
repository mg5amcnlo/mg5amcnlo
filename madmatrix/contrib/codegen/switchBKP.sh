#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Apr 2022) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2022-2024).
# Integrated with the MadGraph7 project in Feb 2026.

status=0

scrdir=$(cd $(dirname $0); pwd)

if [ "$1" == "" ] || [ "$2" != "" ]; then
  echo "Usage: $0 <dir>"
  exit 1 
fi
dir=$1

if [ ! -d ${dir} ]; then echo "ERROR! Directory ${dir} does not exist"; exit 1; fi
if [ ! -d ${dir}.BKP ]; then echo "ERROR! Directory ${dir}.BKP does not exist"; exit 1; fi

set -x

mv ${dir}.BKP ${dir}.BKP.tmp
mv ${dir} ${dir}.BKP
mv ${dir}.BKP.tmp ${dir}
