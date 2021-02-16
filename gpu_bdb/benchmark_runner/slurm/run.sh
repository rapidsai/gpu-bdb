#!/bin/bash

ACCOUNT=rapids
PARTITION=partition
NODES=1
let "WORKER_NODES = $NODES - 1"

IMAGE=/lustre/fsw/rapids/gpu-bdb/containers/gpu-bdb-20210211.sqsh
DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/gpu-bdb-data/"
GPU_BDB_HOME="$HOME/gpu-bdb"

rm *.out
rm -rf $HOME/dask-local-directory/*

srun \
    --account $ACCOUNT \
    --partition $PARTITION \
    --nodes 1 \
    --job-name gpubdb-sched \
    --time 120 \
    --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
    --container-image=$IMAGE \
    bash -c "$GPU_BDB_HOME/gpu_bdb/benchmark_runner/slurm/scheduler-client.sh" &

sleep 15

SCHEDULER_JOBID=$(sacct --starttime $(date -d "-30 seconds" +%FT%H:%M:%S) --endtime $(date -d "+1 days" +%F) -n -X --format jobid --name gpubdb-sched)

if [ "$WORKER_NODES" -gt "0" ]
then
    srun \
        --account $ACCOUNT \
        --partition $PARTITION \
        --nodes $WORKER_NODES \
        --time 120 \
        --dependency after:$SCHEDULER_JOBID \
        --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
        --container-image $IMAGE \
        bash -c "$GPU_BDB_HOME/gpu_bdb/benchmark_runner/slurm/spawn-workers.sh"
fi
