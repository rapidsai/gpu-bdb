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

from xbb_tools.sessionization import get_distinct_sessions
from dask.distributed import wait

# -------- Q2 -----------
q02_item_sk = 10001
q02_limit = 30
q02_session_timeout_inSec = 3600


def read_tables(data_dir, bc):
    bc.create_table("web_clickstreams",
                    data_dir + "web_clickstreams/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query_1 = """
        SELECT
            CAST(wcs_user_sk AS INTEGER) AS wcs_user_sk,
            CAST(wcs_item_sk AS INTEGER) AS wcs_item_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec
        FROM web_clickstreams
        WHERE wcs_item_sk IS NOT NULL
        AND   wcs_user_sk IS NOT NULL
        ORDER BY wcs_user_sk
    """
    wcs_result = bc.sql(query_1)

    session_df = wcs_result.map_partitions(
        get_distinct_sessions,
        keep_cols=["wcs_user_sk", "wcs_item_sk"],
        time_out=q02_session_timeout_inSec,
    )
    del wcs_result

    session_df = session_df.persist()
    wait(session_df)
    bc.create_table('session_df', session_df)

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
    result = bc.sql(last_query)
    result["item_sk_2"] = q02_item_sk
    result_order = ["item_sk_1", "item_sk_2", "cnt"]
    result = result[result_order]

    del session_df
    bc.drop_table("session_df")
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
