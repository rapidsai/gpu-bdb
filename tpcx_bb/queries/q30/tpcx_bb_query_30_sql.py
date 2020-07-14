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

from xbb_tools.sessionization import (
    get_distinct_sessions,
    get_pairs
)

cli_args = tpcxbb_argparser()

# -------- Q30 -----------
# session timeout in secs
q30_session_timeout_inSec = 3600
# query output limit
q30_limit = 40


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table('web_clickstreams', data_dir + "web_clickstreams/*.parquet")
    bc.create_table('item', data_dir + "item/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query_1 = """
        SELECT i_item_sk,
            CAST(i_category_id AS TINYINT) AS i_category_id
        FROM item
    """
    item_df = bc.sql(query_1)

    bc.create_table("item_df", item_df)

    query_2 = """
        SELECT wcs_user_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec,
            i_category_id
        FROM web_clickstreams wcs, item_df i
        WHERE wcs.wcs_item_sk = i.i_item_sk
        AND i.i_category_id IS NOT NULL
        AND wcs.wcs_user_sk IS NOT NULL
        ORDER BY wcs.wcs_user_sk, tstamp_inSec, i_category_id
    """
    merged_df = bc.sql(query_2)

    distinct_session_df = merged_df.map_partitions(get_distinct_sessions,
            keep_cols=["wcs_user_sk", "i_category_id"],
            time_out=q30_session_timeout_inSec)

    del merged_df
    pair_df = distinct_session_df.map_partitions(
        get_pairs,
        pair_col="i_category_id",
        output_col_1="category_id_1",
        output_col_2="category_id_2")
    del distinct_session_df

    bc.create_table('pair_df', pair_df)

    last_query = f"""
        SELECT CAST(category_id_1 AS BIGINT) AS category_id_1,
            CAST(category_id_2 AS BIGINT) AS category_id_2,
            COUNT(category_id_2) AS cnt
        FROM pair_df
        GROUP BY category_id_1, category_id_2
        ORDER BY cnt desc
        LIMIT {q30_limit}
    """
    result = bc.sql(last_query)
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
