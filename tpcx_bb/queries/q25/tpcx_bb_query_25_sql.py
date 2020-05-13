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


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)
from tpcx_bb_query_25 import get_clusters

cli_args = tpcxbb_argparser()


@benchmark()
def read_tables(data_dir):
    bc.create_table("web_sales", data_dir + "web_sales/*.parquet")
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "date_dim/*.parquet")


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(client, data_dir):
    read_tables(data_dir)

    query = """
        WITH concat_table AS 
        (
            (
                SELECT
                    ss_customer_sk           AS cid,
                    count(distinct ss_ticket_number)  AS frequency,
                    max(ss_sold_date_sk)     AS most_recent_date,
                    CAST( SUM(ss_net_paid) AS DOUBLE) AS amount
                FROM store_sales ss
                JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
                WHERE CAST(d.d_date AS DATE) > DATE '2002-01-02'
                AND ss_customer_sk IS NOT NULL
                GROUP BY ss_customer_sk
            ) union all 
            (
                SELECT
                ws_bill_customer_sk    AS cid,
                count(distinct ws_order_number) AS frequency,
                max(ws_sold_date_sk)   AS most_recent_date,
                CAST( SUM(ws_net_paid) AS DOUBLE) AS amount 
            FROM web_sales ws
            JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
            WHERE CAST(d.d_date AS DATE) > DATE '2002-01-02'
            AND ws_bill_customer_sk IS NOT NULL
            GROUP BY ws_bill_customer_sk
            )
        )
        SELECT
            cid            AS cid,
            CASE WHEN 37621 - max(most_recent_date) < 60 THEN 1.0 ELSE 0.0 END AS recency, -- 37621 == 2003-01-02
            CAST( SUM(frequency) AS DOUBLE) AS frequency, --total frequency
            CAST( SUM(amount) AS DOUBLE)    AS totalspend --total amount
        FROM concat_table
        GROUP BY cid 
        ORDER BY cid
    """

    result = bc.sql(query)
    result = result.repartition(npartitions=1)
    ml_result_dict = get_clusters(client=client, ml_input_df=result)
    return ml_result_dict


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        allocator="existing",
        dask_client=client,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    ml_result_dict = main(client=client, data_dir=cli_args["data_dir"])
    write_result(
        ml_result_dict, output_directory=cli_args["output_dir"],
    )

    if cli_args["verify_results"]:
        result_verified = verify_results(cli_args["verify_dir"])
    cli_args["result_verified"] = result_verified
