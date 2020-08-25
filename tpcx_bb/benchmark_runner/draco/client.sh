bash /root/tpcx-bb/tpcx_bb/cluster_configuration/cluster-startup.sh SCHEDULER &

CONDA_ENV_NAME="rapids-tpcx-bb"
CONDA_ENV_PATH="/conda/etc/profile.d/conda.sh"
source $CONDA_ENV_PATH
conda activate $CONDA_ENV_NAME
 
echo "STARTED SCHEDULER"
sleep 10
echo "Starting load test.."
cd /root/tpcx-bb/tpcx_bb
python queries/load_test/tpcx_bb_load_test.py --config_file benchmark_runner/draco_config.yaml > $HOME/row-counts.txt
echo "Starting E2E run.."
python benchmark_runner.py --config_file benchmark_runner/draco_config.yaml
