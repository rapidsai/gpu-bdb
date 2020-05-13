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
from dask.distributed import Client, wait
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)
from tpcx_bb_query_20 import get_clusters

cli_args = tpcxbb_argparser()


@benchmark()
def read_tables(data_dir):
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("store_returns", data_dir + "store_returns/*.parquet")


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(client, data_dir):
    read_tables(data_dir)

    query = """
        SELECT
            ss_customer_sk AS user_sk,
            round(CASE WHEN ((returns_count IS NULL) OR (orders_count IS NULL) OR ((returns_count / orders_count) IS NULL) ) THEN 0.0 ELSE (returns_count / orders_count) END, 7) AS orderRatio,
            round(CASE WHEN ((returns_items IS NULL) OR (orders_items IS NULL) OR ((returns_items / orders_items) IS NULL) ) THEN 0.0 ELSE (returns_items / orders_items) END, 7) AS itemsRatio,
            round(CASE WHEN ((returns_money IS NULL) OR (orders_money IS NULL) OR ((returns_money / orders_money) IS NULL) ) THEN 0.0 ELSE (returns_money / orders_money) END, 7) AS monetaryRatio,
            round(CASE WHEN ( returns_count IS NULL                                                                        ) THEN 0.0 ELSE  returns_count                 END, 0) AS frequency
        FROM
        (
            SELECT
                ss_customer_sk,
                -- return order ratio
                COUNT(distinct(ss_ticket_number)) AS orders_count,
                -- return ss_item_sk ratio
                COUNT(ss_item_sk) AS orders_items,
                -- return monetary amount ratio
                SUM( ss_net_paid ) AS orders_money
            FROM store_sales s
            GROUP BY ss_customer_sk
        ) orders
        LEFT OUTER JOIN
        (
            SELECT
                sr_customer_sk,
                -- return order ratio
                count(distinct(sr_ticket_number)) as returns_count,
                -- return ss_item_sk ratio
                COUNT(sr_item_sk) as returns_items,
                -- return monetary amount ratio
                SUM( sr_return_amt ) AS returns_money
            FROM store_returns
            GROUP BY sr_customer_sk
        ) returned ON ss_customer_sk=sr_customer_sk
        ORDER BY user_sk
    """

    result = bc.sql(query)
    result = result.repartition(npartitions=1)
    result = result.persist()
    wait(result)
    feature_cols = ["orderRatio", "itemsRatio", "monetaryRatio", "frequency"]
    ml_result_dict = get_clusters(
        client=client, ml_input_df=result, feature_cols=feature_cols
    )
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
