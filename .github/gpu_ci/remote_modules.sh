#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate) for
# gpu_cluster_modules.yml:
#     ssh <cluster> bash -l -s -- PATTERN [PARTITION [GRES]] < remote_modules.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate).
# Lists the modules matching PATTERN (case-insensitive extended regexp), on the
# login node or, if PARTITION is given, on a compute node of that partition
# (GPU nodes can have a different software stack than the login node).
export PATTERN=${1:-.}
PARTITION=$2
GRES=$3
LIST='echo "== modules matching \"$PATTERN\" on $(hostname)"; module -t avail 2>&1 | grep -i -E "$PATTERN" | sort -u'
if [ -z "$PARTITION" ]; then
    eval "$LIST"
else
    srun --partition="$PARTITION" ${GRES:+--gres="$GRES"} --ntasks=1 --time=00:05:00 bash -l -c "$LIST"
fi
