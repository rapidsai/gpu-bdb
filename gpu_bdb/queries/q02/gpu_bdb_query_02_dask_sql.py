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

import sys
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader
from bdb_tools.sessionization import get_distinct_sessions
from dask.distributed import wait


# -------- Q2 -----------
q02_item_sk = 10001
q02_limit = 30
q02_session_timeout_inSec = 3600


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )
    wcs_cols = ["wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"]
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    c.create_table("web_clickstreams", wcs_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query_1 = """
        SELECT
            CAST(wcs_user_sk AS INTEGER) AS wcs_user_sk,
            CAST(wcs_item_sk AS INTEGER) AS wcs_item_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec
        FROM web_clickstreams
        WHERE wcs_item_sk IS NOT NULL
        AND   wcs_user_sk IS NOT NULL
        DISTRIBUTE BY wcs_user_sk
    """
    wcs_result = c.sql(query_1)

    session_df = wcs_result.map_partitions(
        get_distinct_sessions,
        keep_cols=["wcs_user_sk", "wcs_item_sk"],
        time_out=q02_session_timeout_inSec,
    )
    del wcs_result

    c.create_table('session_df', session_df, persist=False)

    last_query = f"""
        WITH item_df AS (
            SELECT wcs_user_sk, session_id
            FROM session_df
            WHERE wcs_item_sk = {q02_item_sk}
        )
        SELECT sd.wcs_item_sk as item_sk_1,
            count(sd.wcs_item_sk) as cnt
        FROM session_df sd
        INNER JOIN item_df id
        ON sd.wcs_user_sk = id.wcs_user_sk
        AND sd.session_id = id.session_id
        AND sd.wcs_item_sk <> {q02_item_sk}
        GROUP BY sd.wcs_item_sk
        ORDER BY cnt desc
        LIMIT {q02_limit}
    """
    result = c.sql(last_query)
    result["item_sk_2"] = q02_item_sk
    result_order = ["item_sk_1", "item_sk_2", "cnt"]
    result = result[result_order]

    del session_df
    c.drop_table("session_df")
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
