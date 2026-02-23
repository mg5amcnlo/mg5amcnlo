#!/bin/bash

export PATH="$DMTCP_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$DMTCP_PATH/lib:$LD_LIBRARY_PATH"

# Check if shared directory is set
if [ -z "${SHARED_DIR+x}" ]; then
    out="/dev/null"
    echo "$(date) - Shared disk disabled" | tee -a $out
    export DMTCP_CHECKPOINT_DIR="$PWD/dmtcp_$CONDOR_ID"
else
    if [ -d "$SHARED_DIR" ]; then
        out="$SHARED_DIR/condor_$CONDOR_ID.out"
        echo "$(date) - Shared disk enabled" | tee -a $out
        export DMTCP_CHECKPOINT_DIR="$SHARED_DIR/dmtcp_$CONDOR_ID"
        cd "$SHARED_DIR"
    else
        exit 0
    fi
fi

tarCounter=0
while [[ (-f MadLoop5_resources.tar.gz) && (! -f MadLoop5_resources/HelConfigs.dat) && ($tarCounter < 10) ]]; do
    if [[ $tarCounter > 0 ]]; then
	sleep 2s
    fi
    tar -xzf MadLoop5_resources.tar.gz >/dev/null 2>&1
    tarCounter=$[$tarCounter+1]
done

if [[ (-e MadLoop5_resources.tar.gz) && (! -e MadLoop5_resources/HelConfigs.dat) ]]; then
    echo "Cannot untar and unzip file `pwd`/MadLoop5_resources.tar.gz." > log.txt
    exit
fi

mkdir -p "$DMTCP_CHECKPOINT_DIR"
if compgen -G "G*" > /dev/null; then
    cd G* || exit 1
fi

dmtcp_coordinator -i 86400 --daemon --exit-on-last -p 0 --port-file "$DMTCP_CHECKPOINT_DIR/dmtcp.port" 1>/dev/null 2>&1
export DMTCP_COORD_HOST=$(hostname)
export DMTCP_COORD_PORT=$(cat "$DMTCP_CHECKPOINT_DIR/dmtcp.port")

timeout() {
    echo "$(date) - Approaching walltime. Creating checkpoint..." | tee -a $out
    if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
    fi
    script -qfc "dmtcp_command -bcheckpoint" | tee -a $out 2>&1
    count=0
    while [ ! -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ] && [ $count -lt 10 ]; do
        echo "$(date) - Waiting for checkpoint..." | tee -a $out
        sleep 20
        ((count++))
    done
    if [ $count -eq 10 ]; then
        mv $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh
        echo "$(date) - Checkpoint creation failed. Requeuing..." | tee -a $out
    else
        rm $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script_prev.sh
        echo "$(date) - Checkpoint created. Requeuing..." | tee -a $out
    fi
    script -qfc "dmtcp_command --quit" | tee -a $out 2>&1
    sleep 10
    exit 85
}

# Trap signals
trap "timeout" SIGTERM

if [[ -e "$DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh" ]]; then
    echo "$(date) - Resuming from checkpoint. Restart: ${CONDOR_RESTART_COUNT}" | tee -a $out
    script -qfc "/bin/bash $DMTCP_CHECKPOINT_DIR/dmtcp_restart_script.sh \
        -d $DMTCP_CHECKPOINT_DIR -h $DMTCP_COORD_HOST -p $DMTCP_COORD_PORT" | tee -a $out 2>&1 &
else
    export EXECUTE="dmtcp_launch --allow-file-overwrite $@"
    script -qfc "$EXECUTE" | tee -a $out 2>&1 &
fi

wait

echo "$(date) - Exit" | tee -a $out
exit 0
