set -e pipefail

USERNAME=$(whoami)
TPCX_BB_HOME=$HOME/tpcx-bb
LOGDIR=$HOME/dask-local-directory/logs

# BSQL setup
export INTERFACE="enp97s0f1"
export BLAZING_ALLOCATOR_MODE="existing"
export BLAZING_LOGGING_DIRECTORY=/tpcx-bb-data/tpcx-bb/blazing_log

echo $LOGDIR
echo ls -l $LOGDIR
bash $TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup-slurm.sh &
echo "STARTING WORKERS"
sleep 10000

