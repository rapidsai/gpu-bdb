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


def get_bsql_config_options():
    """Loads configuration environment variables.
    In case it is not previously set, returns a default value for each one.

    Returns a dictionary object.
    For more info: https://docs.blazingdb.com/docs/config_options
    """
    config_options = {}
    config_options['JOIN_PARTITION_SIZE_THRESHOLD'] = os.environ.get("JOIN_PARTITION_SIZE_THRESHOLD", 300000000)
    config_options['MAX_DATA_LOAD_CONCAT_CACHE_BYTE_SIZE'] =  os.environ.get("MAX_DATA_LOAD_CONCAT_CACHE_BYTE_SIZE", 400000000)
    config_options['BLAZING_DEVICE_MEM_CONSUMPTION_THRESHOLD'] = os.environ.get("BLAZING_DEVICE_MEM_CONSUMPTION_THRESHOLD", 0.6)
    config_options['BLAZ_HOST_MEM_CONSUMPTION_THRESHOLD'] = os.environ.get("BLAZ_HOST_MEM_CONSUMPTION_THRESHOLD", 0.6)
    config_options['MAX_KERNEL_RUN_THREADS'] = os.environ.get("MAX_KERNEL_RUN_THREADS", 3)
    config_options['TABLE_SCAN_KERNEL_NUM_THREADS'] = os.environ.get("TABLE_SCAN_KERNEL_NUM_THREADS", 1)
    config_options['MAX_NUM_ORDER_BY_PARTITIONS_PER_NODE'] = os.environ.get("MAX_NUM_ORDER_BY_PARTITIONS_PER_NODE", 20)
    config_options['NUM_BYTES_PER_ORDER_BY_PARTITION'] = os.environ.get("NUM_BYTES_PER_ORDER_BY_PARTITION", 400000000)
    config_options['MAX_ORDER_BY_SAMPLES_PER_NODE'] = os.environ.get("MAX_ORDER_BY_SAMPLES_PER_NODE", 10000)
    config_options['MAX_SEND_MESSAGE_THREADS'] = os.environ.get("MAX_SEND_MESSAGE_THREADS", 20)
    config_options['MEMORY_MONITOR_PERIOD'] = os.environ.get("MEMORY_MONITOR_PERIOD", 50)
    config_options['TRANSPORT_BUFFER_BYTE_SIZE'] = os.environ.get("TRANSPORT_BUFFER_BYTE_SIZE", 1048576) # 1 MBs
    config_options['TRANSPORT_POOL_NUM_BUFFERS'] = os.environ.get("TRANSPORT_POOL_NUM_BUFFERS", 1000)
    config_options['BLAZING_LOGGING_DIRECTORY'] = os.environ.get("BLAZING_LOGGING_DIRECTORY", 'blazing_log')
    config_options['BLAZING_CACHE_DIRECTORY'] = os.environ.get("BLAZING_CACHE_DIRECTORY", '/tmp/')
    config_options['LOGGING_LEVEL'] = os.environ.get("LOGGING_LEVEL", "trace")
    config_options['MAX_JOIN_SCATTER_MEM_OVERHEAD'] = os.environ.get("MAX_JOIN_SCATTER_MEM_OVERHEAD", 500000000)
    config_options['PROTOCOL'] = os.environ.get("PROTOCOL", "AUTO")

    return config_options


def disable_gc():
    import gc

    gc.disable()
    gc.set_threshold(0)


def attach_to_cluster(config, create_blazing_context=False):
    """Attaches to an existing cluster if available.
    By default, tries to attach to a cluster running on localhost:8786 (dask's default).

    This is currently hardcoded to assume the dashboard is running on port 8787.

    Optionally, this will also create a BlazingContext.
    """
    scheduler_file = config.get("scheduler_file_path")
    host = config.get("cluster_host")
    port = config.get("cluster_port", "8786")

    if scheduler_file is not None:
        try:
            with open(scheduler_file) as fp:
                print(fp.read())
            client = Client(scheduler_file=scheduler_file)
            print('Connected!')
            
            print("Disabling GC on scheduler")
            client.run_on_scheduler(disable_gc)
            print("Disabled!")
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

    bc = None
    if create_blazing_context:
        from blazingsql import BlazingContext
        bc = BlazingContext(
            dask_client=client,
            pool=os.environ.get("BLAZING_POOL", False),
            network_interface=os.environ.get("INTERFACE", "ib0"),
            config_options=get_bsql_config_options(),
            allocator=os.environ.get("BLAZING_ALLOCATOR_MODE", "existing"),
            initial_pool_size=os.environ.get("BLAZING_INITIAL_POOL_SIZE", None)
        )

    return client, bc


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

    # optionally include blazingsql
    # this is brittle, but it resolves breaking change
    # issues as we can control the environment
    if os.environ.get("RUNNER_INCLUDE_BSQL"):
        library_list.append("blazingsql")

    for lib in library_list:
        importlib.import_module(lib)
