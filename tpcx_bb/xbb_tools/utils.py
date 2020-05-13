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
import shutil
import argparse
import time
from collections.abc import Iterable, MutableMapping
import traceback

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, wait, performance_report, SSHCluster


def tpcxbb_argparser():
    parser = argparse.ArgumentParser(description="Run TPCx-BB query")
    parser.add_argument(
        "--data_dir",
        default="/datasets/tpcx-bb/sf1000/new_parquet/",
        type=str,
        help="Data Dir",
    )
    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="Query Output Directry. Defaults to the directory of the query script.",
    )
    parser.add_argument("--dask_dir", default="./", type=str, help="Dask Dir")
    parser.add_argument(
        "--repartition_small_table",
        action="store_true",
        help="Repartition small tables",
    )
    parser.add_argument(
        "--dask_profile", action="store_true", help="Include Dask Performance Report"
    )
    parser.add_argument(
        "--cluster_host",
        default=None,
        type=str,
        help="Hostname to use for the cluster scheduler. If you are trying to spin up a fresh SSHCluster, please ignore this and use --hosts instead.",
    )
    parser.add_argument(
        "--cluster_port",
        default=8786,
        type=int,
        help="Which port to use for the cluster scheduler. If you are trying to spin up a fresh SSHCluster, please ignore this and use --hosts instead.",
    )

    args = parser.parse_args()
    args = vars(args)
    return args


def benchmark(csv=True, dask_profile=False):
    def decorate(func):
        def profiled(*args, **kwargs):
            name = func.__name__
            t0 = time.time()
            if dask_profile:
                with performance_report(filename=f"profiled-{name}.html"):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            elapsed_time = time.time() - t0

            logging_info = {}
            logging_info["elapsed_time_seconds"] = elapsed_time
            logging_info["function_name"] = name

            logdf = pd.DataFrame.from_dict(logging_info, orient="index").T

            if csv:
                logdf.to_csv(f"benchmarked_{name}.csv", index=False)
            else:
                print(logdf)
            return result

        return profiled

    return decorate


@benchmark()
def write_result(payload, output_directory="./"):
    """
    """
    import cudf

    if isinstance(payload, MutableMapping):
        write_clustering_result(result_dict=payload, output_directory=output_directory)
    elif isinstance(payload, cudf.DataFrame) or isinstance(payload, dd.DataFrame):
        write_etl_result(df=payload, output_directory=output_directory)
    else:
        raise ValueError("payload must be a dict or a dataframe.")


def write_etl_result(df, output_directory="./"):
    QUERY_NUM = get_query_number()

    output_path = f"{output_directory}q{QUERY_NUM}-results.parquet"
    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            ## to remove existing  directory
            shutil.rmtree(output_path)
        else:
            ## to remove existing single parquet file
            os.remove(output_path)

    if isinstance(df, dd.DataFrame):
        df.to_parquet(output_path, write_index=False)

    else:
        df.to_parquet(f"{output_directory}q{QUERY_NUM}-results.parquet", index=False)


def write_clustering_result(result_dict, output_directory="./"):
    """Results are a text file AND a csv or parquet file.
    This works because we are down to a single partition dataframe.
    """
    QUERY_NUM = get_query_number()
    clustering_info_name = f"{QUERY_NUM}-results-cluster-info.txt"

    with open(f"{output_directory}q{clustering_info_name}", "w") as fh:
        fh.write("Clusters:\n\n")
        fh.write(f"Number of Clusters: {result_dict.get('nclusters')}\n")
        fh.write(f"WSSSE: {result_dict.get('wssse')}\n")

        centers = result_dict.get("cluster_centers")
        for center in centers.values.tolist():
            fh.write(f"{center}\n")

    # this is a single partition dataframe, with cid_labels hard coded
    # as the label column
    data = result_dict.get("cid_labels")

    clustering_result_name = f"q{QUERY_NUM}-results.parquet"
    data.to_parquet(f"{output_directory}{clustering_result_name}", index=False)

    return 0


def train_clustering_model(training_df, n_clusters, max_iter, n_init):
    """Trains a KMeans clustering model on the 
    given dataframe and returns the resulting
    labels and WSSSE"""

    from cuml.cluster.kmeans import KMeans

    best_sse = 0
    best_model = None

    # Optimizing by doing multiple seeding iterations.
    for i in range(n_init):
        model = KMeans(
            oversampling_factor=0,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=np.random.randint(0, 500),
            init="k-means++",
        )
        model.fit(training_df)

        score = model.inertia_

        if best_model is None:
            best_sse = score
            best_model = model

        elif abs(score) < abs(best_sse):
            best_sse = score
            best_model = model

    return {
        "cid_labels": best_model.labels_,
        "wssse": best_model.inertia_,
        "cluster_centers": best_model.cluster_centers_,
        "nclusters": n_clusters,
    }


def get_query_number():
    """This assumes a directory structure like:
    - rapids-queries
        - q01
        - q02
        ...
    and that it is being executed in one of the sub-directories.
    """
    QUERY_NUM = os.getcwd().split("/")[-1][1:]
    return QUERY_NUM


def left_semi_join(df_1, df_2, left_on, right_on):
    """
        Pefrorm left semi join b/w tables
    """
    left_merge = lambda df_1, df_2: df_1.merge(
        df_2, left_on=left_on, right_on=right_on, how="leftsemi"
    )

    ## asserting that number of partitions of the right frame is always 1
    assert df_2.npartitions == 1

    return df_1.map_partitions(left_merge, df_2.to_delayed()[0], meta=df_1._meta)


def run_dask_cudf_query(cli_args, client, query_func, write_func=write_result):
    """
    Common utility to perform all steps needed to execute a dask-cudf version
    of the query. Includes attaching to cluster, running the query and writing results
    """
    try:
        results = query_func(client=client)

        write_func(
            results, output_directory=cli_args["output_dir"],
        )

        client.close()

    except:
        print("Encountered Exception while running query")
        print(traceback.format_exc())
