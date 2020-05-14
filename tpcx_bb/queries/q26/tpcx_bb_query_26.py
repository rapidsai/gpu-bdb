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

import sys


import numpy as np
import rmm


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    train_clustering_model,
    run_dask_cudf_query,
)
from xbb_tools.readers import build_reader
from dask import delayed

try:
    cli_args = tpcxbb_argparser()
except:
    cli_args = {}

# q26 parameters
Q26_CATEGORY = "Books"
Q26_ITEM_COUNT = 5
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


@benchmark(dask_profile=cli_args.get("dask_profile"),)
def read_tables():
    table_reader = build_reader(basepath=cli_args["data_dir"])

    ss_cols = ["ss_customer_sk", "ss_item_sk"]
    items_cols = ["i_item_sk", "i_category", "i_class_id"]

    ss_ddf = table_reader.read("store_sales", relevant_cols=ss_cols, index=False)
    items_ddf = table_reader.read("item", relevant_cols=items_cols, index=False)

    return (ss_ddf, items_ddf)


def agg_count_distinct(df, group_key, counted_key):
    """Returns a Series that is the result of counting distinct instances of 'counted_key' within each 'group_key'.
    The series' index will have one entry per unique 'group_key' value.
    Workaround for lack of nunique aggregate function on Dask df.
    """
    return (
        df.drop_duplicates([group_key, counted_key])
        .groupby(group_key)[counted_key]
        .count()
    )


def convert_datestring_to_days(df, date_col="d_date", date_format="%Y-%m-%d"):
    """
    Utility to convert datestring to int representing days
    """
    datetime_array = rmm.device_array(len(df), dtype=np.int64)
    df[date_col].str.timestamp2int(
        format=date_format, units="D", devptr=datetime_array.device_ctypes_pointer.value
    )
    df[date_col] = datetime_array
    return df


@benchmark(dask_profile=cli_args.get("dask_profile"))
def get_clusters(client, kmeans_input_df):
    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in kmeans_input_df.to_delayed()
    ]
    results_dict = client.compute(*ml_tasks, sync=True)

    output = kmeans_input_df.index.to_frame().reset_index(drop=True)

    labels_final = dask_cudf.from_cudf(
        results_dict["cid_labels"], npartitions=output.npartitions
    )
    output["label"] = labels_final.reset_index()[0]

    # Based on CDH6.1 q26-result formatting
    results_dict["cid_labels"] = output
    return results_dict


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(client):
    ss_ddf, items_ddf = read_tables()

    items_filtered = items_ddf[items_ddf.i_category == Q26_CATEGORY].reset_index(
        drop=True
    )
    items_filtered = items_filtered[["i_item_sk", "i_class_id"]]

    f_ss_ddf = ss_ddf[ss_ddf["ss_customer_sk"].notnull()].reset_index(drop=True)
    merged_ddf = f_ss_ddf.merge(
        items_filtered, left_on="ss_item_sk", right_on="i_item_sk", how="inner"
    )
    keep_cols = ["ss_customer_sk", "i_class_id"]
    merged_ddf = merged_ddf[keep_cols]

    # One-Hot-Encode i_class_id
    merged_ddf = merged_ddf.map_partitions(
        cudf.DataFrame.one_hot_encoding,
        column="i_class_id",
        prefix="id",
        cats=[i for i in range(1, 16)],
        prefix_sep="",
        dtype="float32",
    )
    merged_ddf["total"] = 1.0  # Will keep track of total count
    all_categories = ["total"] + ["id%d" % i for i in range(1, 16)]

    # Aggregate using agg to get sorted ss_customer_sk
    agg_dict = dict.fromkeys(all_categories, "sum")
    rollup_ddf = merged_ddf.groupby("ss_customer_sk").agg(agg_dict)
    rollup_ddf = rollup_ddf[rollup_ddf.total > Q26_ITEM_COUNT][all_categories[1:]]

    # Prepare data for KMeans clustering
    rollup_ddf = rollup_ddf.astype("float64")

    kmeans_input_df = rollup_ddf.persist()

    results_dict = get_clusters(client=client, kmeans_input_df=kmeans_input_df)
    return results_dict


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
