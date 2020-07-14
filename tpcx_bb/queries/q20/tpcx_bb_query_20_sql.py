#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
# Copyright (c) 2019-2020, BlazingSQL, Inc.
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

from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask import delayed
from dask.distributed import wait
import os
import numpy as np

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
    train_clustering_model
)

cli_args = tpcxbb_argparser()

# q20 parameters
N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


@benchmark(dask_profile=cli_args.get("dask_profile"))
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


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("store_returns", data_dir + "store_returns/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = """
        SELECT
            ss_customer_sk AS user_sk,
            round(CASE WHEN ((returns_count IS NULL) OR (orders_count IS NULL)
                OR ((returns_count / orders_count) IS NULL) ) THEN 0.0
                ELSE (returns_count / orders_count) END, 7) AS orderRatio,
            round(CASE WHEN ((returns_items IS NULL) OR (orders_items IS NULL)
                OR ((returns_items / orders_items) IS NULL) ) THEN 0.0
                ELSE (returns_items / orders_items) END, 7) AS itemsRatio,
            round(CASE WHEN ((returns_money IS NULL) OR (orders_money IS NULL)
                OR ((returns_money / orders_money) IS NULL) ) THEN 0.0
                ELSE (returns_money / orders_money) END, 7) AS monetaryRatio,
            round(CASE WHEN ( returns_count IS NULL) THEN 0.0
                ELSE returns_count END, 0) AS frequency
        FROM
        (
            SELECT
                ss_customer_sk,
                -- return order ratio
                CAST (COUNT(distinct(ss_ticket_number)) AS DOUBLE)
                    AS orders_count,
                -- return ss_item_sk ratio
                CAST (COUNT(ss_item_sk) AS DOUBLE) AS orders_items,
                -- return monetary amount ratio
                CAST(SUM( ss_net_paid ) AS DOUBLE) AS orders_money
            FROM store_sales s
            GROUP BY ss_customer_sk
        ) orders
        LEFT OUTER JOIN
        (
            SELECT
                sr_customer_sk,
                -- return order ratio
                CAST(count(distinct(sr_ticket_number)) AS DOUBLE)
                    AS returns_count,
                -- return ss_item_sk ratio
                CAST (COUNT(sr_item_sk) AS DOUBLE) AS returns_items,
                -- return monetary amount ratio
                CAST( SUM( sr_return_amt ) AS DOUBLE) AS returns_money
            FROM store_returns
            GROUP BY sr_customer_sk
        ) returned ON ss_customer_sk=sr_customer_sk
    """
    final_df = bc.sql(query)

    ratio_columns = ["orderRatio", "itemsRatio", "monetaryRatio"]
    final_df = final_df.map_partitions(
        remove_inf_and_nulls, column_names=ratio_columns, value=0.0
    )

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
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        dask_client=client,
        pool=True,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    run_bsql_query(
        cli_args=cli_args, client=client, query_func=main
    )
