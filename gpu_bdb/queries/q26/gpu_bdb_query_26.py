#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    train_clustering_model,
    run_query,
)
from bdb_tools.q26_utils import (
    Q26_CATEGORY,
    Q26_ITEM_COUNT,
    N_CLUSTERS,
    CLUSTER_ITERATIONS,
    N_ITER,
    read_tables
)
import numpy as np
from dask import delayed

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


def get_clusters(client, kmeans_input_df):
    import dask_cudf

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

    # Sort based on CDH6.1 q26-result formatting
    output = output.sort_values(["ss_customer_sk"])

    # Based on CDH6.1 q26-result formatting
    results_dict["cid_labels"] = output
    return results_dict


def main(client, config):
    import cudf

    ss_ddf, items_ddf = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )

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
        cudf.get_dummies,
        columns=["i_class_id"],
        prefix="id",
        cats={"i_class_id": np.arange(1, 16, dtype="int32")},
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
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
