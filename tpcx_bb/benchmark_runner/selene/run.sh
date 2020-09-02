#!/bin/bash

ACCOUNT=rapids
PARTITION=luna
WORKERS=0
IMAGE=randerzander/selene:8-27_0

DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/tpcx-bb"
TPCX_BB_HOME="$HOME/shared/tpcx-bb"

rm *.out
rm -rf $HOME/dask-local-directory/*

sbatch \
    --account $ACCOUNT \
    --partition $PARTITION \
    --nodes 1 \
    --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:/home/root/ \
    --container-image $IMAGE \
    bash -c "$TPCX_BB_HOME/tpcx_bb/benchmark_runner/selene/scheduler-client.sh"
    #bash -c "ls $HOME" -- as expected
    #bash -c "whoami" -- rgelhausen

if [ "$WORKERS" -gt "0" ]; then
	sbatch \
	    --account $ACCOUNT \
	    --partition $PARTITION \
	    --nodes $WORKERS
	    --container-mounts=$DATA_PATH:$MOUNT_PATH \
	    --container-image $IMAGE \
	    bash -c "$TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup-selene.sh"
fi
