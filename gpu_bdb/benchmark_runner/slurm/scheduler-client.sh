set -e pipefail

USERNAME=$(whoami)
GPU_BDB_HOME=$HOME/gpu-bdb
LOGDIR=$HOME/dask-local-directory/logs


# BSQL setup
export INTERFACE="enp97s0f1"
export BLAZING_ALLOCATOR_MODE="existing"
export BLAZING_LOGGING_DIRECTORY=/gpu-bdb-data/gpu-bdb/blazing_log
rm -rf $BLAZING_LOGGING_DIRECTORY/*

bash $GPU_BDB_HOME/gpu_bdb/cluster_configuration/cluster-startup-slurm.sh SCHEDULER &
echo "STARTED SCHEDULER"
sleep 10

CONDA_ENV_NAME="rapids-gpu-bdb"
CONDA_ENV_PATH="/opt/conda/etc/profile.d/conda.sh"
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME

cd $GPU_BDB_HOME/gpu_bdb
echo "Starting waiter.."
python benchmark_runner/wait.py benchmark_runner/benchmark_config.yaml > $LOGDIR/wait.log
# echo "Starting load test.."
# python queries/load_test/gpu_bdb_load_test.py --config_file benchmark_runner/benchmark_config.yaml > $LOGDIR/load_test.log
echo "Starting E2E run.."
python benchmark_runner.py --config_file benchmark_runner/benchmark_config.yaml > $LOGDIR/benchmark_runner.log
