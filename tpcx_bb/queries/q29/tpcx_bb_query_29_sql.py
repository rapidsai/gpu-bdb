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


# -------- Q29 -----------
q29_limit = 100


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table('item', data_dir + "item/*.parquet")
    bc.create_table('web_sales', data_dir + "web_sales/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query_distinct = """
        SELECT DISTINCT i_category_id, ws_order_number
        FROM web_sales ws, item i
        WHERE ws.ws_item_sk = i.i_item_sk
        AND i.i_category_id IS NOT NULL
    """

    result_distinct = bc.sql(query_distinct)

    bc.create_table('distinct_table', result_distinct)

    query = f"""
        SELECT category_id_1, category_id_2, COUNT (*) AS cnt
        FROM
        (
            SELECT CAST(t1.i_category_id as BIGINT) AS category_id_1,
                CAST(t2.i_category_id as BIGINT) AS category_id_2
            FROM distinct_table t1
            INNER JOIN distinct_table t2
            ON t1.ws_order_number = t2.ws_order_number
            WHERE t1.i_category_id < t2.i_category_id
        )
        GROUP BY category_id_1, category_id_2
        ORDER BY cnt DESC, category_id_1, category_id_2
        LIMIT {q29_limit}
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
