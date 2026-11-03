#!/bin/bash
# Slurm batch script, submitted by remote_start_runner.sh from the runner directory, where
# ceci-gate copies it at every start from the vmadgraph VM (/usr/local/share/ceci-gate):
# runs the self-hosted GitHub runner for ONE workflow run, then ends the allocation,
# so that all the CI jobs of that run share this single allocation.
#     sbatch [options] runner_batch.sh RUN_ID STOP_FILE IDLE_MINUTES
#   RUN_ID        the only GitHub workflow run whose jobs are accepted (job_started_hook.sh)
#   STOP_FILE     created by the last CI job (stop_runner): the runner is then stopped
#   IDLE_MINUTES  stop anyway when no job ran for that long (e.g. the run was cancelled)
# The job output (this file's and run.sh's) goes to slurm-<jobid>.out in this directory.
export CI_RUN_ID=${1:?missing RUN_ID}
export CI_STOP_FILE=${2:?missing STOP_FILE}
IDLE_MINUTES=${3:-10}
export ACTIONS_RUNNER_HOOK_JOB_STARTED=$PWD/job_started_hook.sh
LOG=$PWD/slurm-$SLURM_JOB_ID.out
rm -f "$CI_STOP_FILE"
echo "Runner for workflow run $CI_RUN_ID, Slurm job $SLURM_JOB_ID on $(hostname)"

# job control: run.sh gets its own process group (PGID = its PID), so that the runner
# and all its children can be stopped with one signal below
set -m
./run.sh &
RUNNER_PID=$!
while kill -0 "$RUNNER_PID" 2> /dev/null; do
    sleep 15
    if [ -e "$CI_STOP_FILE" ]; then
        # give the runner the time to report the end of the stop_runner job: it uploads the
        # logs first, which can take a minute. Stopped before, the job stays in progress on
        # GitHub, and the watchdog takes the end of the allocation for an early one
        for _ in $(seq 300); do
            grep -q -F ' stop_runner completed with result' "$LOG" && break
            sleep 1
        done
        echo "Stop requested by the workflow"
        break
    fi
    IDLE=$(( ($(date +%s) - $(stat -c %Y "$LOG")) / 60 ))
    if [ "$IDLE" -ge "$IDLE_MINUTES" ] && tail -n 1 "$LOG" | grep -q -E 'Listening for Jobs|completed with result'; then
        echo "No job for $IDLE minutes, stopping"
        break
    fi
done
kill -TERM -- "-$RUNNER_PID" 2> /dev/null || kill -TERM "$RUNNER_PID" 2> /dev/null
wait "$RUNNER_PID"
rm -f "$CI_STOP_FILE"
