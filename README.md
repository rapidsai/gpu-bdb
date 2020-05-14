# RAPIDS TPCx-BB

## Overview

TPCx-BB is a Big Data benchmark for enterprises that includes 30 queries representing real-world ETL & ML workflows at various "scale factors": SF1000 is 1 TB of data, SF10000 is 10TB. Each “query” is in fact a model workflow that can include SQL, user-defined functions, careful sub-setting and aggregation, and machine learning. To date, these queries have been run with [Apache Hive](http://hive.apache.org/) and [Apache Spark](http://spark.apache.org/).

This repository provides implementations of the TPCx-BB queries using [RAPIDS](https://rapids.ai/) libraries. For more information about the TPCx-BB Benchmark, please visit the [TPCx-BB homepage](http://www.tpc.org/tpcx-bb/default.asp).


## Conda Environment Setup

We provide a conda environment definition specifying all RAPIDS dependencies needed to run our query implementations. To install and activate it:

```bash
CONDA_ENV="rapids-tpcx-bb"
conda env create --name $CONDA_ENV -f tpcx-bb/conda/rapids-tpcx-bb.yml
conda activate rapids-tpcx-bb
```

For Query 27, we rely on [spacy](https://spacy.io/). To download the necessary language model after activating the Conda environment:

```bash
python -m spacy download en_core_web_sm
````


### Installing RAPIDS TPCxBB Tools
This repository includes a small local module containing utility functions for running the queries. You can install it with the following:

```bash
cd tpcx-bb/tpcx_bb
python setup.py install --force
```

This will install a package named `xbb-tools` into your Conda environment. It should look like this:

```bash
conda list | grep xbb
xbb-tools                 0.1                      pypi_0    pypi
```

Note that this Conda environment needs to be replicated or installed manually on all nodes, which will allow starting one dask-cuda-worker per node.



## Cluster Configuration
We use the `dask-scheduler` and `dask-cuda-worker` command line interfaces to start a Dask cluster. Before spinning up the scheduler, set the following environment variables:

```bash
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"
export DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING=True
```

Next, run the `dask-scheduler` and `dask-cuda-worker` commands with several additional environment variables, depending on your desired networking and communication pattern.


### Configuration for UCX + NVLink

For the `dask-scheduler`, use the following:

```bash
LOGDIR="/raid/logs"

DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False nohup dask-scheduler --dashboard-address 8787 --interface ib0 --protocol ucx > $LOGDIR/scheduler.log 2>&1 &
```

We use `--interface ib0`. You'll need to change this to the name of a network interface present on your cluster. 

For the `dask-cuda-workers`, use the following:

```bash
DEVICE_MEMORY_LIMIT="25GB"
POOL_SIZE="30GB"
LOGDIR="/raid/logs"
WORKER_DIR="/raid/dask"
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M


dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file lab-sched.json >> $LOGDIR/worker.log 2>&1 &
```

Note that we also pass a scheduler file to `--scheduler-file`, indicating where the scheduler is running. You can read more about all of these configuration variables in the [Dask documentation](https://docs.dask.org/en/latest/setup/cli.html).


### Configuration for TCP over UCX

To use UCX without NVLink, start the scheduler with the following:

```bash
LOGDIR="/raid/logs"

DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True nohup dask-scheduler --interface ib0 --protocol ucx  > $LOGDIR/scheduler.log 2>&1 &
```

Start the workers with the following:

```bash
DEVICE_MEMORY_LIMIT="25GB"
POOL_SIZE="30GB"
LOGDIR="/raid/logs"
WORKER_DIR="/raid/dask"
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M


dask-cuda-worker --rmm-pool-size=$POOL_SIZE --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --scheduler-file lab-sched.json  >> $LOGDIR/worker.log 2>&1 &
```

## Running the Queries

To run the query, starting from the repository root, go to the `queries` directory:

```bash
cd tpcx_bb/rapids-queries/
```

Choose a query to run, and `cd` to that directory. We'll pick query 07.

```bash
cd q07
```

Activate the conda environment with `conda activate rapids-tpcx-bb`.

The queries assume that they can attach to a running Dask cluster. Command line arguments are used to determine the cluster and dataset on which to run the queries. The following is an example of running query 07.

```bash
SCHEDULER_IP=$YOUR_SCHEDULER_NODE_IP

python tpcx_bb_query_07.py --data_dir=$DATA_DIR --cluster_host=$SCHEDULER_IP --output_dir=$OUTPUT_DIR
```

- `data_dir` points to the location of the data
- `cluster_host` corresponds to the address of the running Dask cluster
    - In this case, this query would attempt to connect to a cluster running at `$SCHEDULER_IP`, which would have been configured beforehand
- `output_dir` points to where the queries should write output files


### Running all of the Queries

You can run all the queries with the provided `benchmark_runner.sh` bash script. It is parameterized, and expects the first argument to be either `dask` or `blazing`. The following arguments correspond to the same ones listed above. 


## BlazingSQL

We include BlazingSQL implementations of several queries. As we continue scale testing BlazingSQL implementations, we'll add them to the `queries` folder in the appropriate locations.

We provide a conda environment definition specifying all RAPIDS dependencies needed to run the BSQL query implementations. To install and activate it:

```bash
CONDA_ENV="rapids-bsql-tpcx-bb"
conda env create --name $CONDA_ENV -f tpcx-bb/conda/rapids-bsql-tpcx-bb.yml
conda activate rapids-bsql-tpcx-bb
```
The environment will also require installation of the `xbb_tools` module. More steps on installing this [here](#installing-rapids-tpcxbb-tools).


### Cluster Configuration for TCP

Before spinning up the scheduler setup the following environment variables on all nodes
```bash
export INTERFACE="ib0"
```

**Note**: `ib0` must be replaced by a network interface available on your cluster

Start the `dask-scheduler`:
```bash
nohup dask-scheduler --interface ib0 > $LOGDIR/dask-scheduler.log 2>&1 &
```

Start the `dask-cuda-workers`:
```bash
nohup dask-cuda-worker --local-directory $WORKER_DIR  --interface ib0 --scheduler-file lab-sched.json >> $LOGDIR/dask-worker.log 2>&1 &
```
More information on cluster setup and configurations [here](#cluster-configuration).

### Running Queries

The BlazingSQL + Dask query implementations can be run the same way as described for Dask only implementations. However, you must set the `INTERFACE` environment variable on the client node:
```bash
$ export INTERFACE="ib0"
$ export SCHEDULER_IP=$YOUR_SCHEDULER_NODE_IP

$ python tpcx_bb_query_07_sql.py --data_dir=$DATA_DIR --cluster_host=$SCHEDULER_IP --output_dir=$OUTPUT_DIR
```


## Data Generation

The RAPIDS queries expect [Apache Parquet](http://parquet.apache.org/) formatted data. We provide a Jupyter notebook which can be used to convert bigBench dataGen's raw CSV files to optimally sized Parquet partitions.
