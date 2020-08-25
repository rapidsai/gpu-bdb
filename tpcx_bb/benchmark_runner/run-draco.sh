#!/bin/bash

PROJECT=sw_rapids_testing
PARTITION=batch_32GB_short
IMAGE=randerzander/draco:8-24_0

DATA_PATH="/fs/sjc1-gcl01/datasets/tpcx-bb"
STARTUP_SCRIPT="/root/tpcx-bb/tpcx_bb/cluster_configuration/cluster-startup.sh"

WORKERS=1

rm *.out
rm -rf $HOME/dask-local-directory/*
rm -rf $HOME/sched-out.txt

nvs batch \
    --project $PROJECT \
    --partition $PARTITION \
    --gpus 8 \
    --cores_per_node 40 \
    --nodes 1 \
    --container-mounts=$DATA_PATH:/tpcx-bb \
    --container-image $IMAGE \
    "echo STARTING SCHED && bash $STARTUP_SCRIPT SCHEDULER && echo DONE"

nvs batch \
    --project $PROJECT \
    --partition $PARTITION \
    --gpus 8 \
    --cores_per_node 40 \
    --nodes $WORKERS \
    --container-mounts=$DATA_PATH:/tpcx-bb \
    --container-image $IMAGE \
    "echo STARTING WORKER && bash $STARTUP_SCRIPT && echo FINISHED"
