#!/bin/bash

ACCOUNT=rapids
PARTITION=luna
WORKERS=15

IMAGE="/path/to/container/"
DATA_PATH="/lustre/fsw/rapids"
MOUNT_PATH="/tpcx-bb-data/"
TPCX_BB_HOME="$HOME/tpcx-bb"

srun \
    --account $ACCOUNT \
    --partition $PARTITION \
    --nodes $WORKERS \
    --container-mounts $DATA_PATH:$MOUNT_PATH,$HOME:$HOME \
    --container-image $IMAGE \
    bash -c "$TPCX_BB_HOME/tpcx_bb/benchmark_runner/selene/spawn-workers.sh"
