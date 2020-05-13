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
import traceback


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
import os
import dask_cudf


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
    get_query_number,
)

from xbb_tools.sessionization import get_distinct_sessions
from tpcx_bb_query_02 import get_relevant_item_series

cli_args = tpcxbb_argparser()

# -------- Q2-----------
q02_item_sk = 10001
q02_limit = 30
q02_session_timeout_inSec = 3600


@benchmark()
def read_tables(data_dir):
    bc.create_table("web_clickstreams", data_dir + "web_clickstreams/*.parquet")


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(data_dir):
    read_tables(data_dir)

    # first step -- BlazingSQL
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

    result = bc.sql(query)

    # second step -- DataFrame manipulation
    session_df = result.map_partitions(
        get_distinct_sessions,
        keep_cols=["wcs_user_sk", "wcs_item_sk"],
        time_out=q02_session_timeout_inSec,
    )

    # item_series = filtered_pair_df.map_partitions(get_item_series)
    item_series = session_df.map_partitions(get_relevant_item_series)

    # bringing unique items viewed with query item at cudf level
    items_value_counts = item_series.value_counts().head(q02_limit)
    items_value_counts = items_value_counts.rename("item_sk_1")
    items_value_counts = items_value_counts.to_frame()

    result_df = items_value_counts.reset_index(drop=False).rename(
        {"index": "item_sk_1", "item_sk_1": "cnt"}
    )
    result_df["item_sk_2"] = q02_item_sk
    result_order = ["item_sk_1", "item_sk_2", "cnt"]
    result_df = result_df[result_order]

    return result_df


if __name__ == "__main__":
    try:
        client = attach_to_cluster(cli_args)

        bc = BlazingContext(
            allocator="existing",
            dask_client=client,
            network_interface=os.environ.get("INTERFACE", "eth0"),
        )

        result_df = main(cli_args["data_dir"])
        write_result(
            result_df, output_directory=cli_args["output_dir"],
        )

    except:

        print(traceback.format_exc())
