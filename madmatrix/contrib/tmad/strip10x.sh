#!/bin/sh
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2025) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2025).
# Integrated with the MadGraph7 project in Feb 2026.

cd $(dirname $0)
for log in logs_*/log*.txt ; do
  cat $log | awk 'BEGIN{ok=1}; /^\*\*\*/{if ($5=="x10") ok=0; else ok=1}; {if (ok==1) print $0}' > ${log}.new
  mv ${log}.new ${log}
done
