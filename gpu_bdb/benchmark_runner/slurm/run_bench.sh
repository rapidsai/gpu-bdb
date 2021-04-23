#!/bin/bash
set -e pipefail

#########################################################
### Run Configuration ###
#########################################################
export USERNAME=$(whoami)

# Repository home
export GPU_BDB_HOME=$HOME/gpu-bdb

# Conda environment information
export CONDA_ENV_NAME="rapids-gpu-bdb"
export CONDA_ENV_PATH="/opt/conda/etc/profile.d/conda.sh"

# Logging and scratch space for your machine or cluster
export LOGDIR=$HOME/dask-local-directory/logs
export STATUS_FILE=${LOGDIR}/status.txt

# Communication protocol
export CLUSTER_MODE=TCP

# Scheduler configuration for your machine or cluster
export INTERFACE="enp97s0f1"

# Cluster memory configuration
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT=70GB
POOL_SIZE=78GB

# Dask-cuda optional configuration
JIT_SPILLING=False
EXPLICIT_COMMS=False

# BSQL
export BLAZING_ALLOCATOR_MODE="existing"
export BLAZING_LOGGING_DIRECTORY=/gpu-bdb-data/gpu-bdb/blazing_log
rm -rf $BLAZING_LOGGING_DIRECTORY/*

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"


#########################################################
### Launch the Run ###
#########################################################

source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME

if [[ "$SLURM_NODEID" -eq 0 ]]; then
    bash $GPU_BDB_HOME/gpu_bdb/cluster_configuration/cluster-startup-slurm.sh SCHEDULER
    echo "STARTED SCHEDULER"
    sleep 10

    echo "STARTED" > ${STATUS_FILE}

    cd $GPU_BDB_HOME/gpu_bdb
    echo "Starting waiter.."
    python benchmark_runner/wait.py benchmark_runner/benchmark_config.yaml > $LOGDIR/wait.log
    # echo "Starting load test.."
    # python queries/load_test/gpu_bdb_load_test.py --config_file benchmark_runner/benchmark_config.yaml > $LOGDIR/load_test.log
    echo "Starting E2E run.."
    python benchmark_runner.py --config_file benchmark_runner/benchmark_config.yaml > $LOGDIR/benchmark_runner.log

    echo "FINISHED" > ${STATUS_FILE}
else
    sleep 15 # Sleep and wait for the scheduler to spin up
    echo $LOGDIR
    echo ls -l $LOGDIR
    bash $GPU_BDB_HOME/gpu_bdb/cluster_configuration/cluster-startup-slurm.sh &
    echo "STARTING WORKERS"
    sleep 10
fi

# Keep polling status_file until job is done

until grep -q "FINISHED" "${STATUS_FILE}"
do
    sleep 30
done

pkill dask
