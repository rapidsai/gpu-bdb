#!/bin/bash

#########################################################
### Configuration to (possibly) tweak ###
#########################################################
USERNAME=$(whoami)

# Scheduler configuration for your machine or cluster
SCHEDULER_PORT=${SCHEDULER_PORT:-8786}
DASHBOARD_ADDRESS=${DASHBOARD_ADDRESS:-8787}
INTERFACE=${INTERFACE:-ib0}

# Logging and scratch space for your machine or cluster
LOCAL_DIRECTORY=${LOCAL_DIRECTORY:-/raid/$USERNAME/dask-local-directory}
SCHEDULER_FILE=${SCHEDULER_FILE:-$LOCAL_DIRECTORY/scheduler.json}
LOGDIR=${LOGDIR:-$LOCAL_DIRECTORY/logs}
WORKER_DIR=${WORKER_DIR:-/raid/$USERNAME/gpu-bdb-dask-workers/}

# Communication protocol
CLUSTER_MODE=${CLUSTER_MODE:-TCP}

# Cluster memory configuration
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT=${DEVICE_MEMORY_LIMIT:-18GB}
POOL_SIZE=${POOL_SIZE:-29GB}

# Conda environment information
CONDA_ENV_NAME=${CONDA_ENV_NAME:-rapids-gpu-bdb}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/raid/$USERNAME/miniconda3/etc/profile.d/conda.sh}

# Repository home
GPU_BDB_HOME=${GPU_BDB_HOME:-/raid/$USERNAME/prod/gpu-bdb}

# Dask-cuda optional configuration
JIT_SPILLING=${DASK_JIT_UNSPILL:-False}
EXPLICIT_COMMS=${DASK_EXPLICIT_COMMS:-False}


#########################################################
### Configuration to (generally) leave as default ### #########################################################
ROLE=$1

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

# Purge Dask config directories
rm -rf ~/.config/dask


#########################################################
### Launch the cluster ### #########################################################

# Activate conda environment and install the local library
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME

cd $GPU_BDB_HOME/gpu_bdb
python -m pip install .


if [ "$ROLE" = "SCHEDULER" ]; then
  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     echo "Starting UCX scheduler.."
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False DASK_JIT_UNSPILL=$JIT_SPILLING DASK_EXPLICIT_COMMS=$EXPLICIT_COMMS nohup dask-scheduler --dashboard-address $DASHBOARD_ADDRESS --port $SCHEDULER_PORT --interface $INTERFACE --protocol ucx --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
  
  if [ "$CLUSTER_MODE" = "TCP" ]; then
     echo "Starting TCP scheduler.."
     CUDA_VISIBLE_DEVICES='0' DASK_JIT_UNSPILL=$JIT_SPILLING DASK_EXPLICIT_COMMS=$EXPLICIT_COMMS nohup dask-scheduler --dashboard-address $DASHBOARD_ADDRESS --port $SCHEDULER_PORT --interface $INTERFACE --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
fi

# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    echo "Starting workers.."
    DASK_JIT_UNSPILL=$JIT_SPILLING DASK_EXPLICIT_COMMS=$EXPLICIT_COMMS dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size $POOL_SIZE --memory-limit $MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    echo "Starting workers.."
    DASK_JIT_UNSPILL=$JIT_SPILLING DASK_EXPLICIT_COMMS=$EXPLICIT_COMMS dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size $POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi
