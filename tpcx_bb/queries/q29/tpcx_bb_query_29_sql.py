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

from xbb_tools.cluster_startup import attach_to_cluster

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

# -------- Q29 -----------
q29_limit = 100


def read_tables(data_dir, bc):
    bc.create_table('item', data_dir + "item/*.parquet")
    bc.create_table('web_sales', data_dir + "web_sales/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query_distinct = """
        SELECT DISTINCT i_category_id, ws_order_number
        FROM web_sales ws, item i
        WHERE ws.ws_item_sk = i.i_item_sk
        AND i.i_category_id IS NOT NULL
    """
    result_distinct = bc.sql(query_distinct)

    result_distinct = result_distinct.persist()
    wait(result_distinct)
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

    bc.drop_table("distinct_table")
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
