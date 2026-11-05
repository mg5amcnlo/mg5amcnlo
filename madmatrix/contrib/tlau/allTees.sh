#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (June 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2023).
# Integrated with the MadGraph7 project in Feb 2026.

for proc in gg_tt gg_ttg gg_ttgg gg_ttggg; do
  ./tlau/lauX.sh -cuda ${proc}.mad
  ./tlau/lauX.sh -fortran ${proc}.mad
  ./tlau/lauX.sh -cpp512y ${proc}.mad
done

grep START $(ls -1tr tlau/logs_g*/output.txt | head -1)
grep END $(ls -1t tlau/logs_g*/output.txt | head -1)
