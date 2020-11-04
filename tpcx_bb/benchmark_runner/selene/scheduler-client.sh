set -e pipefail

USERNAME=$(whoami)
TPCX_BB_HOME=$HOME/tpcx-bb
LOGDIR=$HOME/dask-local-directory/logs


# BSQL setup
export INTERFACE="enp97s0f1"
export BLAZING_ALLOCATOR_MODE="existing"

bash $TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup-selene.sh SCHEDULER &
echo "STARTED SCHEDULER"
sleep 10

CONDA_ENV_NAME="rapids"
CONDA_ENV_PATH="/opt/conda/etc/profile.d/conda.sh"
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME
 
cd $TPCX_BB_HOME/tpcx_bb
echo "Starting waiter.."
python benchmark_runner/wait.py benchmark_runner/config.yaml > $LOGDIR/wait.log
# echo "Starting load test.."
# python queries/load_test/tpcx_bb_load_test.py --config_file benchmark_runner/draco/config.yaml > $LOGDIR/load_test.log
echo "Starting E2E run.."
python benchmark_runner.py --config_file benchmark_runner/config.yaml > $LOGDIR/benchmark_runner.log
