#IB, NVLINK, or TCP
ROLE=$1
CLUSTER_MODE="NVLINK"
USERNAME=$(whoami)
# HOSTNAME=$(hostname -i)
HOSTNAME=$HOSTNAME

MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT="32GB"
POOL_SIZE="36GB"

TPCX_BB_HOME=$HOME/tpcx-bb
CONDA_ENV_NAME="rapids"
#CONDA_ENV_PATH="/conda/etc/profile.d/conda.sh"
CONDA_ENV_PATH="/opt/conda/etc/profile.d/conda.sh"

# Used for writing scheduler file to shared storage
LOCAL_DIRECTORY=$HOME/dask-local-directory
SCHEDULER_FILE=$LOCAL_DIRECTORY/scheduler.json

# change to $LOCAL_DIRECTORY/logs for visibility into scheduler & worker logs
#LOGDIR="/tmp/tpcx-bb-dask-logs/"
LOGDIR="$LOCAL_DIRECTORY/logs"
WORKER_DIR="/tmp/tpcx-bb-dask-workers/"

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

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
export DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False"


# Setup scheduler
if [ "$ROLE" = "SCHEDULER" ]; then
  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=True DASK_UCX__RDMACM=False UCX_NET_DEVICES=mlx5_0:1 nohup dask-scheduler --dashboard-address 8787 --protocol ucx --interface ibp12s0 --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi

  if [ "$CLUSTER_MODE" = "TCP" ]; then
     CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address 8787 --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi
fi

# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    # GPU 0
    CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=mlx5_0:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp12s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-0.log &

    # GPU 1
    CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=mlx5_1:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp18s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-1.log &

    # GPU 2
    CUDA_VISIBLE_DEVICES=2 UCX_NET_DEVICES=mlx5_2:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp75s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-2.log &

    # GPU 3
    CUDA_VISIBLE_DEVICES=3 UCX_NET_DEVICES=mlx5_3:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp84s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-3.log &

    # GPU 4
    CUDA_VISIBLE_DEVICES=4 UCX_NET_DEVICES=mlx5_6:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp141s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-4.log &

    # GPU 5
    CUDA_VISIBLE_DEVICES=5 UCX_NET_DEVICES=mlx5_7:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp148s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-5.log &

    # GPU 6
    CUDA_VISIBLE_DEVICES=6 UCX_NET_DEVICES=mlx5_8:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp186s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-6.log &

    # GPU 7
    CUDA_VISIBLE_DEVICES=7 UCX_NET_DEVICES=mlx5_9:1 python -m dask_cuda.cli.dask_cuda_worker --rmm-pool-size=$POOL_SIZE --scheduler-file $SCHEDULER_FILE --local-directory $LOCAL_DIRECTORY --interface ibp204s0 --enable-tcp-over-ucx --device-memory-limit $DEVICE_MEMORY_LIMIT --enable-nvlink --enable-infiniband --disable-rdmacm 2>&1 | tee $LOGDIR/$HOSTNAME-worker-7.log &

fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1
fi

