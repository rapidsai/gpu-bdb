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
)

cli_args = tpcxbb_argparser()


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("item", data_dir + "/item/*.parquet")
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query_distinct = """
        SELECT DISTINCT ss_item_sk, ss_ticket_number
        FROM store_sales s, item i
        WHERE s.ss_item_sk = i.i_item_sk
        AND i.i_category_id IN (1, 2 ,3)
        AND s.ss_store_sk IN (10, 20, 33, 40, 50)
    """
    result_distinct = bc.sql(query_distinct)

    bc.create_table("distinct_table", result_distinct)

    query = """
        SELECT item_sk_1, item_sk_2, COUNT(*) AS cnt
        FROM
        (
            SELECT CAST(t1.ss_item_sk as BIGINT) AS item_sk_1,
                CAST(t2.ss_item_sk AS BIGINT) AS item_sk_2
            FROM distinct_table t1
            INNER JOIN distinct_table t2
            ON t1.ss_ticket_number = t2.ss_ticket_number
            WHERE t1.ss_item_sk < t2.ss_item_sk
        )
        GROUP BY item_sk_1, item_sk_2
        HAVING  COUNT(*) > 50
        ORDER BY cnt DESC, CAST(item_sk_1 AS VARCHAR),
                 CAST(item_sk_2 AS VARCHAR)
        LIMIT 100
    """
    result = bc.sql(query)
    return result


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
