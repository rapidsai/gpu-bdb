#!/bin/bash

PROJECT=sw_rapids_testing
PARTITION=batch_32GB_short
# PARTITION=batch_16GB_short
# PARTITION=batch_dgx2_singlenode
WORKERS=0
IMAGE=nickb500/draco-tpcxbb:2020_09_10

DATA_PATH="/fs/sjc1-gcl01/datasets/tpcx-bb"
TPCX_BB_HOME="/root/tpcx-bb"


rm *.out
rm -rf $HOME/dask-local-directory/*

nvs run \
    --project $PROJECT \
    --partition $PARTITION \
    --gpus 8 \
    --cores_per_node 40 \
    --nodes 1 \
    --container-mounts=$DATA_PATH:/tpcx-bb-data \
    --container-image $IMAGE \
    "bash $TPCX_BB_HOME/tpcx_bb/benchmark_runner/draco/scheduler-client.sh"

if [ "$WORKERS" -gt "0" ]; then
	nvs batch \
	    --project $PROJECT \
	    --partition $PARTITION \
	    --gpus 8 \
	    --cores_per_node 40 \
	    --nodes $WORKERS \
	    --container-mounts=$DATA_PATH:/tpcx-bb-data \
	    --container-image $IMAGE \
	    "bash $TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup.sh"
fi
