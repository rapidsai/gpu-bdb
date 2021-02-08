#!/bin/bash

ACCOUNT=rapids
PARTITION=partition
WORKERS=1

IMAGE="/path/to/container/"
DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/gpu-bdb-data/"
GPU_BDB_HOME="$HOME/gpu-bdb"

rm *.out
rm -rf $HOME/dask-local-directory/*

srun \
    --account $ACCOUNT \
    --partition $PARTITION \
    --nodes 1 \
    --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
    --container-image=$IMAGE \
    bash -c "$GPU_BDB_HOME/gpu_bdb/benchmark_runner/slurm/scheduler-client.sh"


if [ "$WORKERS" -gt "0" ]; then
    srun \
        --account $ACCOUNT \
        --partition $PARTITION \
        --nodes $WORKERS \
        --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
        --container-image $IMAGE \
        bash -c "$GPU_BDB_HOME/gpu_bdb/cluster_configuration/cluster-startup-slurm.sh"
fi
