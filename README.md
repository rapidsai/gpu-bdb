# RAPIDS TPCx-BB

## Overview

TPCx-BB is a Big Data benchmark for enterprises that includes 30 queries representing real-world ETL & ML workflows at various "scale factors": SF1000 is 1 TB of data, SF10000 is 10TB. Each “query” is in fact a model workflow that can include SQL, user-defined functions, careful sub-setting and aggregation, and machine learning. To date, these queries have been run with [Apache Hive](http://hive.apache.org/) and [Apache Spark](http://spark.apache.org/).

This repository provides implementations of the TPCx-BB queries using [RAPIDS](https://rapids.ai/) libraries. For more information about the TPCx-BB Benchmark, please visit the [TPCx-BB homepage](http://www.tpc.org/tpcx-bb/).


## Conda Environment Setup

We provide a conda environment definition specifying all RAPIDS dependencies needed to run our query implementations. To install and activate it:

```bash
CONDA_ENV="rapids-tpcx-bb"
conda env create --name $CONDA_ENV -f tpcx-bb/conda/rapids-tpcx-bb.yml
conda activate rapids-tpcx-bb
```

### Installing RAPIDS TPCxBB Tools
This repository includes a small local module containing utility functions for running the queries. You can install it with the following:

```bash
cd tpcx-bb/tpcx_bb
python -m pip install .

```

This will install a package named `xbb-tools` into your Conda environment. It should look like this:

```bash
conda list | grep xbb
xbb-tools                 0.2                      pypi_0    pypi
```

Note that this Conda environment needs to be replicated or installed manually on all nodes, which will allow starting one dask-cuda-worker per node.

## NLP Query Setup

Queries 10, 18, and 19 depend on two static (negativeSentiment.txt, positiveSentiment.txt) files. As we cannot redistribute those files, you should [download the tpcx-bb toolkit](http://www.tpc.org/tpc_documents_current_versions/download_programs/tools-download-request5.asp?bm_type=TPCX-BB&bm_vers=1.3.1&mode=CURRENT-ONLY) and extract them to your shared filesystem:
```
jar xf bigbenchqueriesmr.jar
cp tpcx-bb1.3.1/distributions/Resources/io/bigdatabenchmark/v1/queries/q10/*.txt ${DATA_DIR}/tpcx-bb/sentiment_files/
```

For Query 27, we rely on [spacy](https://spacy.io/). To download the necessary language model after activating the Conda environment:

```bash
python -m spacy download en_core_web_sm
````

## Starting Your Cluster

We use the `dask-scheduler` and `dask-cuda-worker` command line interfaces to start a Dask cluster. We provide a `cluster_configuration` directory with a bash script to help you set up an NVLink-enabled cluster using UCX.

Before running the script, you'll make changes specific to your environment.

In `cluster_configuration/cluster-startup.sh`:

    - Update `TPCX_BB_HOME=...` to location on disk of this repo
    - Update `INTERFACE=...` to refer to the relevant network interface present on your cluster.
    - Update `CONDA_ENV_PATH=...` to refer to your conda environment path.
    - Update `CONDA_ENV_NAME=...` to refer to the name of the conda environment you created, perhaps using the `yml` files provided in this repository.
    - Update `SCHEDULER=...` to refer to the host name of the node you intend to use as the scheduler.
    - Update `SCHEDULER_FILE=...` to refer to the location of your scheduler file

In `cluster_configuration/example-cluster-scheduler.json`:
Update the scheduler address to be the address for the network interface you chose for `INTERFACE=...` above. If you are not using UCX, you'll need to adjust the address to be `tcp://...` rather than `ucx://...`. Note that `example-cluster-scheduler.json` is just an example scheduler configuration. See [the Dask docs](https://docs.dask.org/en/latest/setup/hpc.html#using-a-shared-network-file-system-and-a-job-scheduler) for more details on how you can generate your own and make it available to all cluster nodes.

To start up the cluster, please run the following on every node from `tpcx_bb/cluster_configuration/`.

```bash
bash cluster-startup.sh NVLINK
```


## Running the Queries

To run a query, starting from the repository root, go to the query specific subdirectory. For example, to run q07:

```bash
cd tpcx_bb/queries/q07/
```

The queries assume that they can attach to a running Dask cluster. Cluster address and other benchmark configuration lives in a yaml file.

```bash
conda activate rapids-tpcx-bb
python tpcx_bb_query_07.py --config_file=../../benchmark_runner/benchmark_config.yaml
```

## Performance Tracking

This repository includes optional performance-tracking automation using Google Sheets. To enable logging query runtimes, on the client node:
```
export GOOGLE_SHEETS_CREDENTIALS_PATH=<path to creds.json>
```
Then configure the `--sheet` and `--tab` arguments in benchmark_config.yaml.

### Running all of the Queries

The included `benchmark_runner.py` script will run all queries sequentially. Configuration for this type of end-to-end run is specified in `benchmark_runner/benchmark_config.yaml`.

To run all queries, cd to `tpcx_bb/` and:

```python
python benchmark_runner.py --config_file benchmark_runner/benchmark_config.yaml
```

By default, this will run each query once. You can control the number of repeats by changing the `N_REPEATS` variable in the script.


## BlazingSQL

We include BlazingSQL implementations of several queries. As we continue scale testing BlazingSQL implementations, we'll add them to the `queries` folder in the appropriate locations.


### Cluster Configuration for TCP

BlazingSQL currently supports clusters using TCP. Please follow the instructions above, making sure to use the InfiniBand interface as the `INTERFACE` variable. Then, start the cluster with:

```bash
bash cluster_configuration/bsql-cluster-startup.sh TCP
```

### Additional useful parameters

BlazingSQL supports some useful parameters which you can set it up manually, it could achieve better performance in some cases. These parameters are by default defined in the `tpcx_bb/xbb_tools/cluster_startup.py` file.

For more context about this check it out [config options](https://docs.blazingdb.com/docs/config_options).

## Data Generation

The RAPIDS queries expect [Apache Parquet](http://parquet.apache.org/) formatted data. We provide a [script](tpcx_bb/queries/load_test/tpcx_bb_load_test.py) which can be used to convert bigBench dataGen's raw CSV files to optimally sized Parquet partitions.
