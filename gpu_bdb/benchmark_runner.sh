#!/bin/bash

USERNAME=$(whoami)
GPU_BDB_HOME=/raid/$USERNAME/prod/gpu-bdb

INCLUDE_DASK=True
INCLUDE_BLAZING=True
N_REPEATS=1

# Dask queries
if [ $INCLUDE_DASK = "True" ]; then
    for qnum in {01..30}
    do
        cd $GPU_BDB_HOME/gpu_bdb/queries/q$qnum/
        for j in $(seq 1 $N_REPEATS)
        do
            python gpu_bdb_query_$qnum.py --config_file ../../benchmark_runner/benchmark_config.yaml
            sleep 3
        done
        sleep 3
    done
fi

# BlazingSQL Queries
if [ $INCLUDE_BLAZING = "True" ]; then
    for qnum in {01..30}
    do
        cd $GPU_BDB_HOME/gpu_bdb/queries/q$qnum/
        for j in $(seq 1 $N_REPEATS)
        do
            python gpu_bdb_query_$qnum\_sql.py --config_file ../../benchmark_runner/benchmark_config.yaml
            sleep 3
        done
        sleep 3
    done
fi
