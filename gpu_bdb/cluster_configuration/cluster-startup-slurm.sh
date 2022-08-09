#!/bin/bash

#########################################################
### Configuration to (possibly) tweak ###
#########################################################
USERNAME=$(whoami)

# Logging and scratch space for your machine or cluster
LOCAL_DIRECTORY=${LOCAL_DIRECTORY:-$HOME/dask-local-directory}
SCHEDULER_FILE=${SCHEDULER_FILE:-$LOCAL_DIRECTORY/scheduler.json}
LOGDIR=${LOGDIR:-$LOCAL_DIRECTORY/logs}
WORKER_DIR=${WORKER_DIR:-/tmp/gpu-bdb-dask-workers/}

# Communication protocol
CLUSTER_MODE=${CLUSTER_MODE:-TCP}

# Cluster memory configuration
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT=${DEVICE_MEMORY_LIMIT:-70GB}
POOL_SIZE=${POOL_SIZE:-78GB}

# Conda environment information
CONDA_ENV_NAME=${CONDA_ENV_NAME:-rapids-gpu-bdb}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/opt/conda/etc/profile.d/conda.sh}

# Repository home
GPU_BDB_HOME=${GPU_BDB_HOME:-$HOME/gpu-bdb}

# Dask-cuda optional configuration
export DASK_JIT_UNSPILL=${DASK_JIT_UNSPILL:-True}
export DASK_EXPLICIT_COMMS=${DASK_EXPLICIT_COMMS:-False}


#########################################################
### Configuration to (generally) leave as default ### #########################################################
ROLE=$1
HOSTNAME=$HOSTNAME

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=${DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT:-100s}
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=${DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP:-600s}
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN=${DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN:-1s}
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX=${DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX:-60s}

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOCAL_DIRECTORY/*
    mkdir -p $LOCAL_DIRECTORY
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

# Purge Dask config directories
rm -rf ~/.config/dask


#########################################################
### Launch the cluster ###
#########################################################

# Activate conda environment and install the local library
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME

cd $GPU_BDB_HOME/gpu_bdb
python -m pip install .

# Setup scheduler
if [ "$ROLE" = "SCHEDULER" ]; then

  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False nohup dask-scheduler --dashboard-address 8787 --protocol ucx --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi
  
  if [ "$CLUSTER_MODE" = "IB" ]; then
     DASK_RMM__POOL_SIZE=1GB CUDA_VISIBLE_DEVICES='0' DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True nohup dask-scheduler --dashboard-address 8787 --protocol ucx --interface ibp18s0 --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi

  if [ "$CLUSTER_MODE" = "TCP" ]; then
     CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address 8787 --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi
fi


# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY  --rmm-pool-size $POOL_SIZE --memory-limit $MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1 &
fi

if [ "$CLUSTER_MODE" = "IB" ]; then
    python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size $POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp18s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT 2>&1 | tee $LOGDIR/$HOSTNAME-worker.log &
fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY  --rmm-pool-size $POOL_SIZE --memory-limit $MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1 &
fi

