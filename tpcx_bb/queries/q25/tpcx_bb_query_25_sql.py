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
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
    train_clustering_model
)
from dask import delayed

cli_args = tpcxbb_argparser()


# -------- Q25 -----------
# -- store_sales and web_sales date
q25_date = "2002-01-02"

N_CLUSTERS = 8
CLUSTER_ITERATIONS = 20
N_ITER = 5


def get_clusters(client, ml_input_df):
    import dask_cudf

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


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("web_sales", data_dir + "web_sales/*.parquet")
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "date_dim/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = f"""
        WITH concat_table AS
        (
            (
                SELECT
                    ss_customer_sk AS cid,
                    count(distinct ss_ticket_number) AS frequency,
                    max(ss_sold_date_sk) AS most_recent_date,
                    CAST( SUM(ss_net_paid) AS DOUBLE) AS amount
                FROM store_sales ss
                JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
                WHERE CAST(d.d_date AS DATE) > DATE '{q25_date}'
                AND ss_customer_sk IS NOT NULL
                GROUP BY ss_customer_sk
            ) union all
            (
                SELECT
                    ws_bill_customer_sk AS cid,
                    count(distinct ws_order_number) AS frequency,
                    max(ws_sold_date_sk)   AS most_recent_date,
                    CAST( SUM(ws_net_paid) AS DOUBLE) AS amount
                FROM web_sales ws
                JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
                WHERE CAST(d.d_date AS DATE) > DATE '{q25_date}'
                AND ws_bill_customer_sk IS NOT NULL
                GROUP BY ws_bill_customer_sk
            )
        )
        SELECT
            cid AS cid,
            CASE WHEN 37621 - max(most_recent_date) < 60 THEN 1.0
                ELSE 0.0 END AS recency, -- 37621 == 2003-01-02
            CAST( SUM(frequency) AS BIGINT) AS frequency, --total frequency
            CAST( SUM(amount) AS DOUBLE)    AS amount --total amount
        FROM concat_table
        GROUP BY cid
        ORDER BY cid
    """
    result = bc.sql(query)

    # Prepare df for KMeans clustering
    cluster_input_ddf = result.set_index('cid')
    cluster_input_ddf["recency"] = cluster_input_ddf["recency"].astype("int64")

    cluster_input_ddf = cluster_input_ddf.repartition(npartitions=1)
    cluster_input_ddf = cluster_input_ddf.persist()
    results_dict = get_clusters(client=client, ml_input_df=cluster_input_ddf)

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
