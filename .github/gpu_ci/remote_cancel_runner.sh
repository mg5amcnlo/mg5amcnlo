#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate):
#     ssh <cluster> bash -l -s -- LABEL [RUN_TAG...] < remote_cancel_runner.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate):
# install it there again after a change (README.md, "One-time setup").
# Lists the runner jobs of the cluster, i.e. the Slurm jobs called mg7-gh-runner-LABEL, of any
# run, one line each:
#     RUNNER_JOB JOBID STATE RUN_TAG
# and scancels those whose comment is one of the RUN_TAG arguments (the comment is the RUN_TAG
# of the run attempt that submitted the job, see remote_start_runner.sh):
#     CANCELLED JOBID STATE RUN_TAG
# Without RUN_TAG, it only lists. No other job is ever touched: scancel gets the name too.
LABEL=${1:?missing LABEL}
shift
NAME=mg7-gh-runner-$LABEL

JOBS=$(squeue -h -u "$USER" -n "$NAME" -o '%i %T %k') || exit 1
STATUS=0
while read -r ID STATE TAG _; do
    [ -n "$ID" ] || continue
    echo "RUNNER_JOB $ID $STATE $TAG"
    for WANTED in "$@"; do
        [ "$TAG" = "$WANTED" ] || continue
        if scancel --name="$NAME" "$ID"; then
            echo "CANCELLED $ID $STATE $TAG"
        else
            STATUS=1
        fi
    done
done <<< "$JOBS"
exit $STATUS
