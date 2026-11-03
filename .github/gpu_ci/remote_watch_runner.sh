#!/bin/bash
# Runs on the cluster LOGIN node, piped through ssh by ceci-gate (vm_gate/ceci-gate) for the
# watchdog job of gpu_runner_ci.yml:
#     ssh <cluster> bash -l -s -- LABEL RUN_TAG < remote_watch_runner.sh
# The copy that runs is the one installed on the vmadgraph VM (/usr/local/share/ceci-gate):
# install it there again after a change (README.md, "One-time setup").
# Waits until the runner job of the workflow run has ended: the Slurm job called
# mg7-gh-runner-LABEL whose comment is RUN_TAG (set by remote_start_runner.sh). Then prints
#     ALLOCATION_ENDED JOBID STATE ELAPSED TIMELIMIT
# with a final state from sacct (COMPLETED, TIMEOUT, NODE_FAIL, PREEMPTED, CANCELLED...), and
# exits 0. Whether that end was the normal one (after stop_runner) is decided by the watchdog,
# from GitHub. Other endings:
#     ALLOCATION_ENDED ? ...                no job with comment RUN_TAG in squeue for 2 minutes
#     ALLOCATION_ENDED JOBID PAST_TIME_LIMIT ...   gone from squeue for longer than its time
#                                           limit allowed, with no final state in sacct
# Exits non-zero, without that line, if squeue keeps failing: the watchdog calls again.
# squeue can leave out a running job now and then (seen on manneback, with exit status 0): a
# job missing from squeue has only ended once sacct says so.
LABEL=${1:?missing LABEL}
RUN_TAG=${2:?missing RUN_TAG}
NAME=mg7-gh-runner-$LABEL
POLL=20

# a Slurm duration ([D-][HH:]MM:SS) in seconds; nothing for UNLIMITED, INVALID...
seconds() {
    local days=0 t=$1 h=0 m s
    case $t in *-*) days=${t%%-*}; t=${t#*-} ;; esac
    case $t in
        *:*:*) IFS=: read -r h m s <<< "$t" ;;
        *:*) IFS=: read -r m s <<< "$t" ;;
        *) return ;;
    esac
    case $days$h$m$s in *[!0-9]*) return ;; esac
    echo $(( ((10#$days * 24 + 10#$h) * 60 + 10#$m) * 60 + 10#$s ))
}

JOBID=
LAST=
SHOWN=0
FAILS=0
MISSES=0
DEADLINE=
COMMENT=
while true; do
    # by name, not with -j: squeue -j fails once the job has left the controller
    if ! JOBS=$(squeue -h -u "$USER" -n "$NAME" -o '%i %T %k %M %l' 2>&1); then
        FAILS=$((FAILS + 1))
        echo "$(date +%T) squeue failed ($FAILS): $JOBS"
        [ "$FAILS" -lt 30 ] || exit 1
        sleep "$POLL"
        continue
    fi
    FAILS=0
    if [ -z "$JOBID" ]; then
        # the runner job of this run attempt, from its comment; from then on, by its id
        read -r ID STATE TAG USED LIMIT <<< "$(awk -v tag="$RUN_TAG" '$3 == tag' <<< "$JOBS")"
        if [ -z "$ID" ]; then
            MISSES=$((MISSES + 1))
            echo "$(date +%T) no job $NAME with comment $RUN_TAG in squeue ($MISSES): ${JOBS:-empty list}"
            if [ "$MISSES" -ge 6 ]; then
                echo "ALLOCATION_ENDED ? no job $NAME with comment $RUN_TAG in squeue"
                exit 0
            fi
            sleep "$POLL"
            continue
        fi
        JOBID=$ID
    else
        read -r ID STATE TAG USED LIMIT <<< "$(awk -v id="$JOBID" '$1 == id' <<< "$JOBS")"
    fi
    case $STATE in
        PENDING|CONFIGURING|RUNNING|SUSPENDED)
            if [ "$TAG" != "${COMMENT:-$RUN_TAG}" ]; then
                echo "$(date +%T) runner job $JOBID: its comment is now '$TAG', still watching it by its id"
                COMMENT=$TAG
            fi
            # it cannot run past its time limit (plus a margin for the end of the job)
            L=$(seconds "$LIMIT")
            U=$(seconds "$USED")
            DEADLINE=
            [ -z "$L" ] || [ -z "$U" ] || DEADLINE=$(( $(date +%s) + L - U + 600 ))
            # on every change of state, and every 10 minutes: how much of the time limit is used
            if [ "$STATE" != "$LAST" ] || [ $(($(date +%s) - SHOWN)) -ge 600 ]; then
                echo "$(date +%T) runner job $JOBID: $STATE, $USED of $LIMIT"
                LAST=$STATE
                SHOWN=$(date +%s)
            fi
            sleep "$POLL"
            continue ;;
    esac

    # gone from squeue, or ending (COMPLETING...): sacct has the last word
    FINAL=$(sacct -n -X -P -j "$JOBID" -o State,Elapsed,Timelimit 2> /dev/null | head -n 1 | tr '|' ' ')
    case ${FINAL%% *} in
        PENDING|CONFIGURING|RUNNING|SUSPENDED|COMPLETING|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|SIGNALING|STAGE_OUT|'')
            ;;
        *)  echo "$(date +%T) runner job $JOBID: ${STATE:-no longer in squeue}"
            echo "ALLOCATION_ENDED $JOBID $FINAL"
            exit 0 ;;
    esac
    # not over for sacct (or the accounting database lags behind, or sacct failed): watch on
    WHAT="squeue: ${STATE:-not listed}, sacct: ${FINAL:-no state}"
    if [ "$WHAT" != "$LAST" ] || [ $(($(date +%s) - SHOWN)) -ge 600 ]; then
        echo "$(date +%T) runner job $JOBID: $WHAT; still watching (squeue listed: ${JOBS:-nothing})"
        LAST=$WHAT
        SHOWN=$(date +%s)
    fi
    if [ -n "$DEADLINE" ] && [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "ALLOCATION_ENDED $JOBID PAST_TIME_LIMIT ($WHAT)"
        exit 0
    fi
    sleep "$POLL"
done
