#IB, NVLINK, or TCP
ROLE=$1
CLUSTER_MODE="NVLINK"
USERNAME=$(whoami)

MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT="35GB"
POOL_SIZE="38GB"

TPCX_BB_HOME=$HOME/shared/tpcx-bb
CONDA_ENV_NAME="rapids-tpcx-bb"
CONDA_ENV_PATH="/conda/etc/profile.d/conda.sh"

# Used for writing scheduler file to shared storage
LOCAL_DIRECTORY=$HOME/dask-local-directory
SCHEDULER_FILE=$LOCAL_DIRECTORY/scheduler.json

# change to $LOCAL_DIRECTORY/logs for visibility into scheduler & worker logs
#LOGDIR="/tmp/tpcx-bb-dask-logs/"
LOGDIR="$LOCAL_DIRECTORY/logs"
WORKER_DIR="/tmp/tpcx-bb-dask-workers/"

# Purge Dask worker and log directories
rm -rf $LOGDIR/*
mkdir -p $LOGDIR
rm -rf $WORKER_DIR/*
mkdir -p $WORKER_DIR

# Purge Dask config directories
rm -rf ~/.config/dask

# Activate conda environment

source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME
 
cd $TPCX_BB_HOME/tpcx_bb
python -m pip install .

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"


# Setup scheduler
if [ "$ROLE" = "SCHEDULER" ]; then
  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False nohup dask-scheduler --dashboard-address 8787 --protocol ucx --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
  
  if [ "$CLUSTER_MODE" = "TCP" ]; then
     CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address 8787 --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
  fi
fi

# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1

    for WORKER in 0 1 2 3
    do
      CUDA_VISIBLE_DEVICES=$WORKER UCX_NET_DEVICES=mlx5_$WORKER:1 \
      dask_cuda_worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY \
      --enable-tcp-over-ucx --enable-nvlink --enable-infiniband --enable-rdmacm --scheduler-file $SCHEDULER_FILE --interface ib$WORKER \
      --enable-rdmacm 2>&1 | tee $LOGDIR/worker-$WORKER.log &
    done

    for WORKER in 4 5 6 7
    do
      CUDA_VISIBLE_DEVICES=$WORKER UCX_NET_DEVICES=mlx5_$(WORKER+2):1 \
      dask_cuda_worker \
      --scheduler-file $LOCAL_DIRECTORY/scheduler.json --local-directory $LOCAL_DIRECTORY --interface ib$(WORKER+2) \
      --enable-tcp-over-ucx --enable-nvlink --enable-infiniband --enable-rdmacm $RMM_POOL 2>&1 | tee $LOGDIR/worker-$WORKER.log &
    done

fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1
fi
