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
import cupy as cp
import rmm
import numpy as np


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    train_clustering_model,
    run_query,
)
from xbb_tools.readers import build_reader
from dask import delayed
from dask.distributed import wait


# q20 parameters
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_cols = [
        "ss_customer_sk",
        "ss_ticket_number",
        "ss_item_sk",
        "ss_net_paid",
    ]
    store_returns_cols = [
        "sr_item_sk",
        "sr_customer_sk",
        "sr_ticket_number",
        "sr_return_amt",
    ]

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    store_returns_df = table_reader.read(
        "store_returns", relevant_cols=store_returns_cols
    )
    return store_sales_df, store_returns_df


def get_clusters(client, ml_input_df, feature_cols):
    """
    Takes the dask client, kmeans_input_df and feature columns.
    Returns a dictionary matching the output required for q20
    """
    import dask_cudf

    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in ml_input_df[feature_cols].to_delayed()
    ]

    results_dict = client.compute(*ml_tasks, sync=True)

    labels = results_dict["cid_labels"]

    labels_final = dask_cudf.from_cudf(labels, npartitions=ml_input_df.npartitions)
    ml_input_df["label"] = labels_final.reset_index()[0]

    output = ml_input_df[["user_sk", "label"]]

    results_dict["cid_labels"] = output
    return results_dict


def remove_inf_and_nulls(df, column_names, value=0.0):
    """
    Replace all nulls, inf, -inf with value column_name from df 
    """

    to_replace_dict = dict.fromkeys(column_names, [np.inf, -np.inf])
    value_dict = dict.fromkeys(column_names, 0.0)

    # Fill nulls for ratio columns with 0.0
    df.fillna(value_dict, inplace=True)
    # Replace inf and -inf with 0.0
    df.replace(to_replace_dict, value_dict, inplace=True)

    return df


def main(client, config):
    store_sales_df, store_returns_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    n_workers = len(client.scheduler_info()["workers"])

    ### going via repartition for split_out drop duplicates
    ### see issue: https://github.com/rapidsai/tpcx-bb-internal/issues/492

    unique_sales = store_sales_df[
        ["ss_ticket_number", "ss_customer_sk"]
    ].map_partitions(lambda df: df.drop_duplicates())
    unique_sales = unique_sales.shuffle(on=["ss_customer_sk"])
    unique_sales = unique_sales.map_partitions(lambda df: df.drop_duplicates())

    unique_sales = unique_sales.persist()
    wait(unique_sales)

    orders_count = (
        unique_sales.groupby(by="ss_customer_sk")
        .agg({"ss_ticket_number": "count"})
        .reset_index()
    )

    orders_df = (
        store_sales_df.groupby(by="ss_customer_sk")
        .agg({"ss_item_sk": "count", "ss_net_paid": "sum"})
        .reset_index()
    )

    ### free up memory no longer needed
    del store_sales_df

    orders_df = orders_df.merge(orders_count, how="inner", on="ss_customer_sk")
    orders_df = orders_df.rename(
        columns={
            "ss_customer_sk": "user_sk",
            "ss_ticket_number": "orders_count",
            "ss_item_sk": "orders_items",
            "ss_net_paid": "orders_money",
        }
    )

    orders_df = orders_df.persist()
    wait(orders_df)
    del unique_sales

    returns_count = (
        store_returns_df[["sr_ticket_number", "sr_customer_sk"]]
        .drop_duplicates(split_out=n_workers)
        .groupby(by="sr_customer_sk")
        .agg({"sr_ticket_number": "count"})
        .reset_index()
    )
    returns_df = (
        store_returns_df.groupby(by="sr_customer_sk")
        .agg({"sr_item_sk": "count", "sr_return_amt": "sum"})
        .reset_index()
    )
    ### free up memory no longer needed
    del store_returns_df

    returns_df = returns_df.merge(returns_count, how="inner", on="sr_customer_sk")

    returns_df = returns_df.rename(
        columns={
            "sr_customer_sk": "user_sk",
            "sr_ticket_number": "returns_count",
            "sr_item_sk": "returns_items",
            "sr_return_amt": "returns_money",
        }
    )

    returns_df = returns_df.persist()
    wait(returns_df)

    final_df = orders_df.merge(returns_df, how="left", on="user_sk")

    final_df["orderRatio"] = (
        final_df["returns_count"] / final_df["orders_count"]
    ).round(7)
    final_df["itemsRatio"] = (
        final_df["returns_items"] / final_df["orders_items"]
    ).round(7)
    final_df["monetaryRatio"] = (
        final_df["returns_money"] / final_df["orders_money"]
    ).round(7)

    ratio_columns = ["orderRatio", "itemsRatio", "monetaryRatio"]
    final_df = final_df.map_partitions(
        remove_inf_and_nulls, column_names=ratio_columns, value=0.0
    )

    final_df = final_df.rename(columns={"returns_count": "frequency"})

    keep_cols = ["user_sk", "orderRatio", "itemsRatio", "monetaryRatio", "frequency"]
    final_df = final_df[keep_cols]

    final_df = final_df.fillna(0)
    final_df = final_df.repartition(npartitions=1).persist()
    wait(final_df)

    final_df = final_df.sort_values(["user_sk"]).reset_index(drop=True)
    final_df = final_df.persist()
    wait(final_df)

    feature_cols = ["orderRatio", "itemsRatio", "monetaryRatio", "frequency"]

    results_dict = get_clusters(
        client=client, ml_input_df=final_df, feature_cols=feature_cols
    )
    return results_dict


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
