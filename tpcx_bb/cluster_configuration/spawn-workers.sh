set -e pipefail

USERNAME=$(whoami)
TPCX_BB_HOME=$HOME/tpcx-bb
LOGDIR=$HOME/dask-local-directory/logs

echo $LOGDIR
echo ls -l $LOGDIR
bash $TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup-selene.sh &
echo "STARTING WORKERS"
sleep 10000

