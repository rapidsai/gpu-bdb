#!/bin/bash

ACCOUNT=rapids
PARTITION=backfill
NODES=1
GPUS_PER_NODE=16
export NUM_WORKERS=$((NODES*GPUS_PER_NODE))

DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/gpu-bdb-data/"
IMAGE=${IMAGE:-/lustre/fsw/rapids/gpu-bdb/containers/gpu-bdb-20210421.sqsh}
RUN_BENCH_PATH=${RUN_BENCH_PATH:-$HOME/gpu-bdb/gpu_bdb/benchmark_runner/slurm/run_bench.sh}

rm *.out
rm -rf $HOME/dask-local-directory/*

srun \
    --account $ACCOUNT \
    --partition $PARTITION \
    --nodes $NODES \
    --exclusive \
    --job-name ${ACCOUNT}-gpubdb:run_bench \
    --gpus-per-node $GPUS_PER_NODE \
    --time 120 \
    --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
    --container-image=$IMAGE \
    bash -c "$RUN_BENCH_PATH"
