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

from xbb_tools.sessionization import get_distinct_sessions

cli_args = tpcxbb_argparser()


# -------- Q2 -----------
q02_item_sk = 10001
q02_limit = 30
q02_session_timeout_inSec = 3600


def get_relevant_item_series(df, q02_item_sk):
    """
        Returns relevant items directly
    """
    item_df = df[df["wcs_item_sk"] == q02_item_sk].reset_index(drop=True)
    pair_df = item_df.merge(
        df, on=["wcs_user_sk", "session_id"], suffixes=["_t1", "_t2"], how="inner"
    )
    return pair_df[pair_df["wcs_item_sk_t2"] != q02_item_sk][
        "wcs_item_sk_t2"
    ].reset_index(drop=True)


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("web_clickstreams",
                    data_dir + "web_clickstreams/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = """
        SELECT
            wcs_user_sk,
            wcs_item_sk,
            (wcs_click_date_sk * 86400 + wcs_click_time_sk) AS tstamp_inSec
        FROM web_clickstreams
        WHERE wcs_item_sk IS NOT NULL
        AND   wcs_user_sk IS NOT NULL
        ORDER BY wcs_user_sk, tstamp_inSec
    """
    wcs_result = bc.sql(query)

    session_df = wcs_result.map_partitions(
        get_distinct_sessions,
        keep_cols=["wcs_user_sk", "wcs_item_sk"],
        time_out=q02_session_timeout_inSec,
    )

    item_series = session_df.map_partitions(get_relevant_item_series,
                                            q02_item_sk)

    # bringing unique items viewed with query item at cudf level
    items_value_counts = item_series.value_counts().head(q02_limit)
    items_value_counts.columns = ["item_sk_1", "cnt"]
    items_value_counts = items_value_counts.to_frame()

    result = items_value_counts.reset_index(drop=False)
    result.columns = ["item_sk_1", "cnt"]
    result["item_sk_2"] = q02_item_sk
    result = result[["item_sk_1", "item_sk_2", "cnt"]]

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
