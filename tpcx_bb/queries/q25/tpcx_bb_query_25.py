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

# q25 parameters
Q25_DATE = "2002-01-02"
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


@benchmark(dask_profile=cli_args.get("dask_profile"),)
def read_tables():
    table_reader = build_reader(
        basepath=cli_args["data_dir"],
        repartition_small_table=cli_args["repartition_small_table"],
    )

    ss_cols = ["ss_customer_sk", "ss_sold_date_sk", "ss_ticket_number", "ss_net_paid"]
    ws_cols = [
        "ws_bill_customer_sk",
        "ws_sold_date_sk",
        "ws_order_number",
        "ws_net_paid",
    ]
    datedim_cols = ["d_date_sk", "d_date"]

    ss_ddf = table_reader.read("store_sales", relevant_cols=ss_cols, index=False)
    ws_ddf = table_reader.read("web_sales", relevant_cols=ws_cols, index=False)
    datedim_ddf = table_reader.read("date_dim", relevant_cols=datedim_cols, index=False)

    return (ss_ddf, ws_ddf, datedim_ddf)


def agg_count_distinct(df, group_key, counted_key, client):
    """Returns a Series that is the result of counting distinct instances of 'counted_key' within each 'group_key'.
    The series' index will have one entry per unique 'group_key' value.
    Workaround for lack of nunique aggregate function on Dask df.
    """

    ### going via repartition for split_out drop duplicates
    ### see issue: https://github.com/rapidsai/tpcx-bb-internal/issues/492
    unique_df = df[[group_key, counted_key]].map_partitions(
        lambda df: df.drop_duplicates()
    )
    unique_df = unique_df.repartition(columns=[group_key])
    unique_df = unique_df.map_partitions(lambda df: df.drop_duplicates())

    return unique_df.groupby(group_key)[counted_key].count(split_every=2)


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
def get_clusters(client, ml_input_df):
    ml_tasks = [
        delayed(train_clustering_model)(df, N_CLUSTERS, CLUSTER_ITERATIONS, N_ITER)
        for df in ml_input_df.to_delayed()
    ]
    results_dict = client.compute(*ml_tasks, sync=True)

    output = ml_input_df.index.to_frame().reset_index(drop=True)

    labels_final = dask_cudf.from_cudf(
        results_dict["cid_labels"], npartitions=output.npartitions
    )
    output["label"] = labels_final.reset_index()[0]

    # Based on CDH6.1 q25-result formatting
    results_dict["cid_labels"] = output
    return results_dict


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(client):
    ss_ddf, ws_ddf, datedim_ddf = read_tables()
    datedim_ddf = datedim_ddf.map_partitions(convert_datestring_to_days)
    min_date = np.datetime64(Q25_DATE, "D").astype(int)
    # Filter by date
    valid_dates_ddf = datedim_ddf[datedim_ddf["d_date"] > min_date].reset_index(
        drop=True
    )

    f_ss_ddf = ss_ddf[ss_ddf["ss_customer_sk"].notnull()].reset_index(drop=True)
    f_ws_ddf = ws_ddf[ws_ddf["ws_bill_customer_sk"].notnull()].reset_index(drop=True)

    # Merge
    ss_merged_df = f_ss_ddf.merge(
        valid_dates_ddf, left_on="ss_sold_date_sk", right_on="d_date_sk", how="inner"
    )
    ws_merged_df = f_ws_ddf.merge(
        valid_dates_ddf, left_on="ws_sold_date_sk", right_on="d_date_sk", how="inner"
    )

    # Roll up store sales
    agg_store_sales_ddf = ss_merged_df.groupby("ss_customer_sk").agg(
        {"ss_sold_date_sk": "max", "ss_net_paid": "sum"}
    )

    agg_store_sales_ddf["frequency"] = agg_count_distinct(
        ss_merged_df, "ss_customer_sk", "ss_ticket_number", client=client
    )  # Simulate count distinct

    # Same rollup, just different columns for web sales
    agg_web_sales_ddf = ws_merged_df.groupby("ws_bill_customer_sk").agg(
        {"ws_sold_date_sk": "max", "ws_net_paid": "sum"}
    )

    agg_web_sales_ddf["frequency"] = agg_count_distinct(
        ws_merged_df, "ws_bill_customer_sk", "ws_order_number", client=client
    )  # Simulate count distinct

    agg_store_sales_ddf = agg_store_sales_ddf.reset_index()
    agg_web_sales_ddf = agg_web_sales_ddf.reset_index()

    shared_columns = ["cid", "most_recent_date", "amount", "frequency"]
    agg_store_sales_ddf.columns = shared_columns
    agg_web_sales_ddf.columns = shared_columns
    agg_sales_ddf = dask_cudf.concat([agg_store_sales_ddf, agg_web_sales_ddf])

    cluster_input_ddf = agg_sales_ddf.groupby("cid").agg(
        {"most_recent_date": "max", "frequency": "sum", "amount": "sum"}
    )

    cluster_input_ddf["recency"] = (37621 - cluster_input_ddf["most_recent_date"]) < 60

    # Reorder to match refererence examples
    cluster_input_ddf = cluster_input_ddf[["recency", "frequency", "amount"]]

    # Prepare df for KMeans clustering
    cluster_input_ddf["recency"] = cluster_input_ddf["recency"].astype("int64")
    cluster_input_ddf["amount"] = cluster_input_ddf["amount"].astype("float64")

    cluster_input_ddf = cluster_input_ddf.persist()

    results_dict = get_clusters(client=client, ml_input_df=cluster_input_ddf)
    return results_dict


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    client = attach_to_cluster(cli_args)

    run_dask_cudf_query(cli_args=cli_args, client=client, query_func=main)
