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


def attach_to_cluster(cli_args):
    """Attaches to an existing cluster if available.
    By default, tries to attach to a cluster running on localhost:8786 (dask's default).

    This is currently hardcoded to assume the dashboard is running on port 8787.
    """
    host = cli_args.get("cluster_host")
    port = cli_args.get("cluster_port", "8786")

    if host is not None:
        try:
            content = requests.get(
                "http://" + host + ":8787/info/main/workers.html"
            ).content.decode("utf-8")
            url = content.split("Scheduler ")[1].split(":" + str(port))[0]
            client = Client(address=f"{url}:{port}")
            print(f"Connected to {url}:{port}")
        except requests.exceptions.ConnectionError as e:
            sys.exit(
                f"Unable to connect to existing dask scheduler dashboard to determine cluster type: {e}"
            )
        except OSError as e:
            sys.exit(f"Unable to create a Dask Client connection: {e}")

    else:
        raise ValueError("Must pass a cluster address to the host argument.")

    def maybe_create_worker_directories(dask_worker):
        worker_dir = dask_worker.local_directory
        if not os.path.exists(worker_dir):
            os.mkdir(worker_dir)

    client.run(maybe_create_worker_directories)

    # Get ucx config variables
    ucx_config = client.submit(_get_ucx_config).result()
    cli_args.update(ucx_config)

    # Save worker information
    gpu_sizes = ["16GB", "32GB", "40GB"]
    worker_counts = worker_count_info(client, gpu_sizes=gpu_sizes)
    for size in gpu_sizes:
        key = size + "_workers"
        if cli_args.get(key) is not None and cli_args.get(key) != worker_counts[size]:
            print(
                f"Expected {cli_args.get(key)} {size} workers in your cluster, but got {worker_counts[size]}. It can take a moment for all workers to join the cluster. You may also have misconfigred hosts."
            )
            sys.exit(-1)

    cli_args["16GB_workers"] = worker_counts["16GB"]
    cli_args["32GB_workers"] = worker_counts["32GB"]
    cli_args["40GB_workers"] = worker_counts["40GB"]

    return client


def worker_count_info(client, gpu_sizes=["16GB", "32GB", "40GB"], tol="2.1GB"):
    """
    Method accepts the Client object, GPU sizes and tolerance limit and returns
    a dictionary containing number of workers per GPU size specified
    """
    counts_by_gpu_size = dict.fromkeys(gpu_sizes, 0)
    worker_info = client.scheduler_info()["workers"]
    for worker, info in worker_info.items():
        # Assumption is that a node is homogeneous (on a specific node all gpus have the same size)
        worker_device_memory = info["gpu"]["memory-total"][0]
        for gpu_size in gpu_sizes:
            if abs(parse_bytes(gpu_size) - worker_device_memory) < parse_bytes(tol):
                counts_by_gpu_size[gpu_size] += 1
                break

    return counts_by_gpu_size


def _get_ucx_config():
    """
    Get a subset of ucx config variables relevant for benchmarking
    """
    relevant_configs = ["infiniband", "nvlink"]
    ucx_config = dask.config.get("ucx")
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
