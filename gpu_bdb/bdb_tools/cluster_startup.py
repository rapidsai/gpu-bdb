#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import requests
import sys
import importlib

import dask
from dask.distributed import Client
from dask.utils import parse_bytes


def attach_to_cluster(config, create_sql_context=False):
    """Attaches to an existing cluster if available.
    By default, tries to attach to a cluster running on localhost:8786 (dask's default).

    This is currently hardcoded to assume the dashboard is running on port 8787.

    Optionally, this will also create a Dask-SQL Context.
    """
    scheduler_file = config.get("scheduler_file_path")
    host = config.get("cluster_host")
    port = config.get("cluster_port", "8786")
    start_local_cluster = config.get("start_local_cluster", False)

    if start_local_cluster:
        from dask_cuda import LocalCUDACluster
        cluster = LocalCUDACluster(
            n_workers=int(os.environ.get("NUM_WORKERS", 16)),
            device_memory_limit=os.environ.get("DEVICE_MEMORY_LIMIT", "20GB"),
            local_directory=os.environ.get("LOCAL_DIRECTORY"),
            rmm_pool_size=os.environ.get("POOL_SIZE", "29GB"),
            memory_limit=os.environ.get("DEVICE_MEMORY_LIMIT", "1546828M"),
            enable_tcp_over_ucx=os.environ.get("CLUSTER_MODE", "TCP")=="NVLINK",
            enable_nvlink=os.environ.get("CLUSTER_MODE", "TCP")=="NVLINK",
            protocol="ucx" if os.environ.get("CLUSTER_MODE", "TCP")=="NVLINK" else "tcp",
            enable_infiniband=False,
           enable_rdmacm=False,
           jit_unspill=True
        )
        client = Client(cluster)

    elif scheduler_file is not None:
        try:
            with open(scheduler_file) as fp:
                print(fp.read())
            client = Client(scheduler_file=scheduler_file)
            print('Connected!')
        except OSError as e:
            sys.exit(f"Unable to create a Dask Client connection: {e}")

    elif host is not None:
        try:
            content = requests.get(
                "http://" + host + ":8787/info/main/workers.html"
            ).content.decode("utf-8")
            url = content.split("Scheduler ")[1].split(":" + str(port))[0]
            client = Client(address=f"{url}:{port}")
            print(f"Connected to {url}:{port}")
            config["protocol"] = str(url)[0:3]
        except requests.exceptions.ConnectionError as e:
            sys.exit(
                f"Unable to connect to existing dask scheduler dashboard to determine cluster type: {e}"
            )
        except OSError as e:
            sys.exit(f"Unable to create a Dask Client connection: {e}")

    else:
        raise ValueError("Must pass a scheduler file or cluster address to the host argument.")

    def maybe_create_worker_directories(dask_worker):
        worker_dir = dask_worker.local_directory
        if not os.path.exists(worker_dir):
            os.mkdir(worker_dir)

    client.run(maybe_create_worker_directories)

    # Get ucx config variables
    ucx_config = client.submit(_get_ucx_config).result()
    config.update(ucx_config)
    
    # CuPy should use RMM on all worker and client processes
    import cupy as cp
    import rmm
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
    client.run(cp.cuda.set_allocator, rmm.rmm_cupy_allocator)

    # Save worker information
    # Assumes all GPUs are the same size
    expected_workers = int(os.environ.get("NUM_WORKERS", 16))
    worker_counts = worker_count_info(client)
    for gpu_size, count in worker_counts.items():
        if count != 0:
            current_workers = worker_counts.pop(gpu_size)
            break

    if expected_workers is not None and expected_workers != current_workers:
        print(
            f"Expected {expected_workers} {gpu_size} workers in your cluster, but got {current_workers}. It can take a moment for all workers to join the cluster. You may also have misconfigred hosts."
        )
        sys.exit(-1)

    config["16GB_workers"] = worker_counts.get("16GB", 0)
    config["32GB_workers"] = worker_counts.get("32GB", 0)
    config["40GB_workers"] = worker_counts.get("40GB", 0)
    config["80GB_workers"] = worker_counts.get("80GB", 0)

    c = None
    if create_sql_context:
        from dask_sql import Context
        c = Context()

    return client, c


def worker_count_info(client):
    """
    Method accepts the Client object and returns a dictionary
    containing number of workers per GPU size specified

    Assumes all GPUs are of the same type.
    """
    gpu_sizes = ["16GB", "32GB", "40GB", "80GB"]
    counts_by_gpu_size = dict.fromkeys(gpu_sizes, 0)
    tolerance = "6.3GB"

    worker_info = client.scheduler_info()["workers"]
    for worker, info in worker_info.items():
        worker_device_memory = info["gpu"]["memory-total"]
        for gpu_size in gpu_sizes:
            if abs(parse_bytes(gpu_size) - worker_device_memory) < parse_bytes(tolerance):
                counts_by_gpu_size[gpu_size] += 1
                break

    return counts_by_gpu_size


def _get_ucx_config():
    """
    Get a subset of ucx config variables relevant for benchmarking
    """
    relevant_configs = ["infiniband", "nvlink"]
    ucx_config = dask.config.get("distributed.comm.ucx")
    # Doing this since when relevant configs are not enabled the value is `None` instead of `False`
    filtered_ucx_config = {
        config: ucx_config.get(config) if ucx_config.get(config) else False
        for config in relevant_configs
    }

    return filtered_ucx_config


def import_query_libs():
    library_list = [
        "rmm",
        "cudf",
        "cuml",
        "cupy",
        "sklearn",
        "dask_cudf",
        "pandas",
        "numpy",
        "spacy",
    ]

    for lib in library_list:
        importlib.import_module(lib)
