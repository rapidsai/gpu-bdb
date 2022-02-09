# RAPIDS GPU-BDB

## Disclaimer

gpu-bdb is derived from [TPCx-BB](http://www.tpc.org/tpcx-bb/). Any results based on gpu-bdb are considered unofficial results, and per [TPC](http://www.tpc.org/) policy, cannot be compared against official TPCx-BB results.

## Overview

The GPU Big Data Benchmark (gpu-bdb) is a [RAPIDS](https://rapids.ai) library based benchmark for enterprises that includes 30 queries representing real-world ETL & ML workflows at various "scale factors": SF1000 is 1 TB of data, SF10000 is 10TB. Each “query” is in fact a model workflow that can include SQL, user-defined functions, careful sub-setting and aggregation, and machine learning.

## Conda Environment Setup

We provide a conda environment definition specifying all RAPIDS dependencies needed to run our query implementations. To install and activate it:

```bash
CONDA_ENV="rapids-gpu-bdb"
conda env create --name $CONDA_ENV -f gpu-bdb/conda/rapids-gpu-bdb.yml
conda activate rapids-gpu-bdb
```

### Installing RAPIDS bdb Tools
This repository includes a small local module containing utility functions for running the queries. You can install it with the following:

```bash
cd gpu-bdb/gpu_bdb
python -m pip install .

```

This will install a package named `bdb-tools` into your Conda environment. It should look like this:

```bash
conda list | grep bdb
bdb-tools                 0.2                      pypi_0    pypi
```

Note that this Conda environment needs to be replicated or installed manually on all nodes, which will allow starting one dask-cuda-worker per node.

## NLP Query Setup

Queries 10, 18, and 19 depend on two static (negativeSentiment.txt, positiveSentiment.txt) files. As we cannot redistribute those files, you should [download the tpcx-bb toolkit](http://www.tpc.org/tpc_documents_current_versions/download_programs/tools-download-request5.asp?bm_type=TPCX-BB&bm_vers=1.3.1&mode=CURRENT-ONLY) and extract them to your data directory on your shared filesystem:
```
jar xf bigbenchqueriesmr.jar
cp tpcx-bb1.3.1/distributions/Resources/io/bigdatabenchmark/v1/queries/q10/*.txt ${DATA_DIR}/sentiment_files/
```

For Query 27, we rely on [spacy](https://spacy.io/). To download the necessary language model after activating the Conda environment:

```bash
python -m spacy download en_core_web_sm
````

## Starting Your Cluster

We use the `dask-scheduler` and `dask-cuda-worker` command line interfaces to start a Dask cluster. We provide a `cluster_configuration` directory with a bash script to help you set up an NVLink-enabled cluster using UCX.

Before running the script, you'll make changes specific to your environment.

In `cluster_configuration/cluster-startup.sh`:

    - Update `GPU_BDB_HOME=...` to location on disk of this repo
    - Update `CONDA_ENV_PATH=...` to refer to your conda environment path.
    - Update `CONDA_ENV_NAME=...` to refer to the name of the conda environment you created, perhaps using the `yml` files provided in this repository.
    - Update `INTERFACE=...` to refer to the relevant network interface present on your cluster.
    - Update `CLUSTER_MODE="TCP"` to refer to your communication method, either "TCP" or "NVLINK". You can also configure this as an environment variable.
    - You may also need to change the `LOCAL_DIRECTORY` and `WORKER_DIR` depending on your filesystem. Make sure that these point to a location to which you have write access and that `LOCAL_DIRECTORY` is accessible from all nodes.


To start up the cluster on your scheduler node, please run the following from `gpu_bdb/cluster_configuration/`. This will spin up a scheduler and one Dask worker per GPU.

```bash
DASK_JIT_UNSPILL=True CLUSTER_MODE=NVLINK bash cluster-startup.sh SCHEDULER
```

Note: Don't use DASK_JIT_UNSPILL when running BlazingSQL queries.

Then run the following on every other node from `gpu_bdb/cluster_configuration/`.

```bash
bash cluster-startup.sh
```

This will spin up one Dask worker per GPU. If you are running on a single node, you will only need to run `bash cluster-startup.sh SCHEDULER`.

If you are using a Slurm cluster, please adapt the example Slurm setup in `gpu_bdb/benchmark_runner/slurm/` which uses `gpu_bdb/cluster_configuration/cluster-startup-slurm.sh`.


## Running the Queries

To run a query, starting from the repository root, go to the query specific subdirectory. For example, to run q07:

```bash
cd gpu_bdb/queries/q07/
```

The queries assume that they can attach to a running Dask cluster. Cluster address and other benchmark configuration lives in a yaml file (`gpu_bdb/benchmark_runner/becnhmark_config.yaml`). You will need to fill this out as appropriate if you are not using the Slurm cluster configuration.

```bash
conda activate rapids-gpu-bdb
python gpu_bdb_query_07.py --config_file=../../benchmark_runner/benchmark_config.yaml
```

To NSYS profile a gpu-bdb query, change `start_local_cluster` in benchmark_config.yaml to `True` and run:

```bash
nsys profile -t cuda,nvtx python gpu_bdb_query_07_dask_sql.py --config_file=../../benchmark_runner/benchmark_config.yaml
```

Note: There is no need to start workers with `cluster-startup.sh` as
there is a `LocalCUDACluster` being started in `attach_to_cluster` API.

## Performance Tracking

This repository includes optional performance-tracking automation using Google Sheets. To enable logging query runtimes, on the client node:
```
export GOOGLE_SHEETS_CREDENTIALS_PATH=<path to creds.json>
```
Then configure the `--sheet` and `--tab` arguments in `benchmark_config.yaml`.

### Running all of the Queries

The included `benchmark_runner.py` script will run all queries sequentially. Configuration for this type of end-to-end run is specified in `benchmark_runner/benchmark_config.yaml`.

To run all queries, cd to `gpu_bdb/` and:

```python
python benchmark_runner.py --config_file benchmark_runner/benchmark_config.yaml
```

By default, this will run each Dask query five times, and, if BlazingSQL queries are enabled in `benchmark_config.yaml`, each BlazingSQL query five times. You can control the number of repeats by changing the `N_REPEATS` variable in the script.


## BlazingSQL

BlazingSQL implementations of all queries are included. BlazingSQL currently supports communication via TCP. To run BlazingSQL queries, please follow the instructions above to create a cluster using `CLUSTER_MODE=TCP`.


## Data Generation

The RAPIDS queries expect [Apache Parquet](http://parquet.apache.org/) formatted data. We provide a [script](gpu_bdb/queries/load_test/gpu_bdb_load_test.py) which can be used to convert bigBench dataGen's raw CSV files to optimally sized Parquet partitions.
