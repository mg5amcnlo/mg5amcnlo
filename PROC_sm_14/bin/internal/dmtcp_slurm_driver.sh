#!/bin/bash

export PATH="$DMTCP_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$DMTCP_PATH/lib:$LD_LIBRARY_PATH"

count=0
while [ -d "$RUN_DIR/dmtcp_fail" ] && [ $count -lt 10 ]; do
    echo "$(date) - Waiting for $RUN_DIR/dmtcp_fail to disappear..."
    sleep 20
    ((count++))
done

export DMTCP_CHECKPOINT_DIR="$RUN_DIR/dmtcp_$SLURM_JOB_ID"
mkdir -p "$DMTCP_CHECKPOINT_DIR"

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "$(date) - Approaching walltime. Creating checkpoint..."
    if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
    fi
    script -qfc "dmtcp_command -bcheckpoint"
    count=0
    while [ ! -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ] && [ $count -lt 10 ]; do
        echo "$(date) - Waiting for checkpoint..."
        sleep 20
        ((count++))
    done
    if [ $count -eq 10 ]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        echo "$(date) - Checkpoint creation failed. Requeuing..."
    else
        rm $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
        echo "$(date) - Checkpoint created. Requeuing..."
    fi
    script -qfc "dmtcp_command --quit"
    sleep 2
    scontrol requeue $SLURM_JOB_ID
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" USR1

if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${SLURM_RESTART_COUNT}"
    script -qfc "srun /bin/bash $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT" &
else
    export EXECUTE="srun dmtcp_launch --allow-file-overwrite $@"
    script -qfc "$EXECUTE" &
fi

wait

echo "$(date) - Calculation finished. Cleanup..."

link="$DMTCP_CHECKPOINT_DIR"

while [ -L "$link" ]; do
    next=$(readlink "$link")
    rm "$link"
    link="$next"
done

rm -r "$link"

echo "$(date) - Exit"
exit 0
