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
import numpy as np

import dask_cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
    convert_datestring_to_days,
    get_clusters
)
from bdb_tools.q25_utils import (
    q25_date,
    read_tables
)

def agg_count_distinct(df, group_key, counted_key, client):
    """Returns a Series that is the result of counting distinct instances of 'counted_key' within each 'group_key'.
    The series' index will have one entry per unique 'group_key' value.
    Workaround for lack of nunique aggregate function on Dask df.
    """

    ### going via repartition for split_out drop duplicates
    unique_df = df[[group_key, counted_key]].map_partitions(
        lambda df: df.drop_duplicates()
    )
    unique_df = unique_df.shuffle(on=[group_key])
    unique_df = unique_df.map_partitions(lambda df: df.drop_duplicates())

    return unique_df.groupby(group_key)[counted_key].count(split_every=2)


def main(client, config):

    ss_ddf, ws_ddf, datedim_ddf = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
    )
    date_meta_df = datedim_ddf._meta
    date_meta_df["d_date"] = date_meta_df["d_date"].astype("int64")
    datedim_ddf = datedim_ddf.map_partitions(convert_datestring_to_days, meta=date_meta_df)
    min_date = np.datetime64(q25_date, "D").astype(int)
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

    cluster_input_ddf = agg_sales_ddf.groupby("cid", as_index=False).agg(
        {"most_recent_date": "max", "frequency": "sum", "amount": "sum"}
    )

    cluster_input_ddf["recency"] = (37621 - cluster_input_ddf["most_recent_date"]) < 60

    cluster_input_ddf = cluster_input_ddf.sort_values(["cid"])
    cluster_input_ddf = cluster_input_ddf.set_index("cid")

    # Reorder to match refererence examples
    cluster_input_ddf = cluster_input_ddf[["recency", "frequency", "amount"]]

    # Prepare df for KMeans clustering
    cluster_input_ddf["recency"] = cluster_input_ddf["recency"].astype("int64")
    cluster_input_ddf["amount"] = cluster_input_ddf["amount"].astype("float64")

    cluster_input_ddf = cluster_input_ddf.persist()

    results_dict = get_clusters(client=client, kmeans_input_df=cluster_input_ddf)
    return results_dict


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
