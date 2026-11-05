#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate) for the
# start_runner job of gpu_runner_ci.yml:
#     ssh <cluster> bash -l -s -- LABEL RUN_ID RUN_TAG IDLE_MINUTES [sbatch options...] < remote_start_runner.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate):
# install it there again after a change (README.md, "One-time setup").
# Submits runner_batch.sh (copied beforehand into ~/mg7-gpu-ci/runner-LABEL, next to
# job_started_hook.sh) and waits until the runner is connected to GitHub.
#  - every runner job of a cluster is called mg7-gh-runner-LABEL and uses
#    --dependency=singleton: a runner only starts once the previous one has stopped;
#  - RUN_TAG is stored as the Slurm comment, to recognise the job in squeue.
# Exits non-zero, with the runner log, if the runner job ends before being ready
# (e.g. registration deleted by GitHub: run install_runner.sh again).
LABEL=${1:?missing LABEL}
RUN_ID=${2:?missing RUN_ID}
RUN_TAG=${3:?missing RUN_TAG}
IDLE_MINUTES=${4:?missing IDLE_MINUTES}
shift 4
RUNNER_DIR=$HOME/mg7-gpu-ci/runner-$LABEL
if [ ! -f "$RUNNER_DIR/.runner" ]; then
    echo "ERROR: no runner configured in $RUNNER_DIR, run .github/gpu_ci/install_runner.sh first"
    exit 1
fi

JOBID=$(sbatch --parsable --job-name="mg7-gh-runner-$LABEL" --comment="$RUN_TAG" \
               --dependency=singleton --chdir="$RUNNER_DIR" --output="$RUNNER_DIR/slurm-%j.out" \
               "$@" "$RUNNER_DIR/runner_batch.sh" "$RUN_ID" "$RUNNER_DIR/stop-$RUN_TAG" "$IDLE_MINUTES") || exit 1
JOBID=${JOBID%%;*}
LOG=$RUNNER_DIR/slurm-$JOBID.out
echo "Submitted runner job $JOBID ($*)"

LAST=
while true; do
    STATE=$(squeue -h -j "$JOBID" -o %T 2> /dev/null)
    case $STATE in
        PENDING|CONFIGURING|RUNNING) ;;
        *) echo "ERROR: runner job $JOBID ended (${STATE:-no longer in squeue}) before the runner was ready:"
           cat "$LOG" 2> /dev/null
           exit 1 ;;
    esac
    if grep -q 'Listening for Jobs' "$LOG" 2> /dev/null; then
        cat "$LOG"
        echo "Runner $LABEL online in Slurm job $JOBID on $(squeue -h -j "$JOBID" -o %N)"
        exit 0
    fi
    if [ "$STATE" != "$LAST" ]; then
        echo "$(date +%T) job $JOBID: $STATE $(squeue -h -j "$JOBID" -o %r)"
        LAST=$STATE
    fi
    sleep 20
done
