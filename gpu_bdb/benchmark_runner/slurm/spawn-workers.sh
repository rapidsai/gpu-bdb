set -e pipefail

USERNAME=$(whoami)
GPU_BDB_HOME=$HOME/gpu-bdb
LOGDIR=$HOME/dask-local-directory/logs

# BSQL setup
export INTERFACE="enp97s0f1"
export BLAZING_ALLOCATOR_MODE="existing"
export BLAZING_LOGGING_DIRECTORY=/gpu-bdb-data/gpu-bdb/blazing_log

# sleep for 15 seconds to handle cases in which this job spins up workers
# before the scheduler is ready
sleep 15

echo $LOGDIR
echo ls -l $LOGDIR
bash $GPU_BDB_HOME/gpu_bdb/cluster_configuration/cluster-startup-slurm.sh &
echo "STARTING WORKERS"
sleep 10000

