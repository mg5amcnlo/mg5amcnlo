#!/bin/bash

/afs/cern.ch/work/t/torriell/Work/MG5_aMC/PY8meetsMG5aMC_release/vendor/SudGen/gridsudgen_clust <<EOF
1.0 7000.0
1.0 7000.0
0.001
-1
$1 $2 $3
EOF
