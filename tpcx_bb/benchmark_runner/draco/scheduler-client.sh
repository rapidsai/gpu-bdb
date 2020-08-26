set -e pipefail

TPCX_BB_HOME=/root/shared/tpcx-bb

bash $TPCX_BB_HOME/tpcx_bb/cluster_configuration/cluster-startup.sh SCHEDULER &
echo "STARTED SCHEDULER"
sleep 10

CONDA_ENV_NAME="rapids-tpcx-bb"
CONDA_ENV_PATH="/home/root/conda/etc/profile.d/conda.sh"
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME
 
cd $TPCX_BB_HOME/tpcx_bb
echo "Starting waiter.."
python benchmark_runner/wait.py benchmark_runner/draco/config.yaml
echo "Starting load test.."
python queries/load_test/tpcx_bb_load_test.py --config_file benchmark_runner/draco/config.yaml > $HOME/load_test.log
echo "Starting E2E run.."
python benchmark_runner.py --config_file benchmark_runner/draco/config.yaml > $HOME/benchmark_runner.log
