ROLE=$1
USERNAME=$(whoami)

# NVLINK or TCP, based on environment variable (or manual setting)
if [ -z "$CLUSTER_MODE" ]
then
    CLUSTER_MODE="TCP"
else
    CLUSTER_MODE=$CLUSTER_MODE
fi

MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT="15GB"
POOL_SIZE="30GB"

GPU_BDB_HOME=$HOME/gpu-bdb
CONDA_ENV_NAME="rapids-gpu-bdb"
CONDA_ENV_PATH="/home/$USERNAME/conda/etc/profile.d/conda.sh"

# Used for writing scheduler file and logs to shared storage
LOCAL_DIRECTORY=$HOME/dask-local-directory
SCHEDULER_FILE=$LOCAL_DIRECTORY/scheduler.json
LOGDIR="$LOCAL_DIRECTORY/logs"
WORKER_DIR=/tmp/$USERNAME/gpu-bdb-dask-workers/

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

# Purge Dask config directories
rm -rf ~/.config/dask

# Activate conda environment and install the local library
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME
 
cd $GPU_BDB_HOME/gpu_bdb
python -m pip install .

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"

# Select an interface appropriate for your cluster or machine.
INTERFACE="ib0"

# Setup scheduler
SCHEDULER_PORT=8786
DASHBOARD_ADDRESS=8787

if [ "$ROLE" = "SCHEDULER" ]; then
  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     echo "Starting UCX scheduler.."
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False nohup dask-scheduler --dashboard-address $DASHBOARD_ADDRESS --port $SCHEDULER_PORT --interface $INTERFACE --protocol ucx --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
  
  if [ "$CLUSTER_MODE" = "TCP" ]; then
     echo "Starting TCP scheduler.."
     CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address $DASHBOARD_ADDRESS --port $SCHEDULER_PORT --interface $INTERFACE --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
fi

# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    echo "Starting workers.."
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    echo "Starting workers.."
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi
