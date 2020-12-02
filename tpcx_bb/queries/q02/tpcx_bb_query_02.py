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

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader
from xbb_tools.sessionization import get_distinct_sessions

### Implementation Notes:

### Future Notes:
# The bottleneck of current implimenation is `set-index`, once ucx is working correctly
# it should go away

# -------- Q2 -----------
q02_item_sk = 10001
q02_MAX_ITEMS_PER_BASKET = 5000000
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


def reduction_function(df, q02_session_timeout_inSec):

    ### get session_df
    df = get_distinct_sessions(
        df, keep_cols=["wcs_user_sk", "wcs_item_sk"], time_out=q02_session_timeout_inSec
    )

    item_series = get_relevant_item_series(df, q02_item_sk)
    # bringing unique items viewed with query item at cudf level
    del df

    grouped_df = item_series.value_counts().reset_index(drop=False)
    del item_series
    grouped_df.columns = ["i_item_sk", "cnt"]
    return grouped_df


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )
    wcs_cols = ["wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"]
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)
    return wcs_df


def pre_repartition_task(wcs_df):

    f_wcs_df = wcs_df[
        wcs_df["wcs_item_sk"].notnull() & wcs_df["wcs_user_sk"].notnull()
    ].reset_index(drop=True)
    f_wcs_df["tstamp_inSec"] = (
        f_wcs_df["wcs_click_date_sk"] * 24 * 60 * 60 + f_wcs_df["wcs_click_time_sk"]
    )

    cols_2_keep = ["wcs_user_sk", "wcs_item_sk", "tstamp_inSec"]
    f_wcs_df = f_wcs_df[cols_2_keep]

    ### for map reduce task, we set index to ensure
    ### that the click for each user ends up at the same partition
    f_wcs_df["wcs_user_sk"] = f_wcs_df["wcs_user_sk"].astype("int32")
    f_wcs_df["wcs_item_sk"] = f_wcs_df["wcs_item_sk"].astype("int32")

    return f_wcs_df


def main(client, config):

    wcs_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    ### filter nulls
    # SELECT
    #  wcs_user_sk,
    #  wcs_item_sk,
    #  (wcs_click_date_sk * 24 * 60 * 60 + wcs_click_time_sk) AS tstamp_inSec
    # FROM web_clickstreams
    # WHERE wcs_item_sk IS NOT NULL
    # AND   wcs_user_sk IS NOT NULL

    f_wcs_df = wcs_df.map_partitions(pre_repartition_task)
    f_wcs_df = f_wcs_df.shuffle(on=["wcs_user_sk"])

    ### Main Query
    # SELECT
    #  item_sk_1,${hiveconf:q02_item_sk} AS item_sk_2, COUNT (*) AS cnt
    # FROM
    # (
    # )
    # GROUP BY item_sk_1
    # ORDER BY
    #  cnt DESC,
    #  item_sk_1
    # LIMIT ${hiveconf:q02_limit};

    # q02_limit=30
    grouped_df = f_wcs_df.map_partitions(reduction_function, q02_session_timeout_inSec)
    items_value_counts = grouped_df.groupby(["i_item_sk"]).cnt.sum()

    items_value_counts = items_value_counts.map_partitions(
        lambda ser: ser.sort_values(ascending=False)
    )

    ### final calculation on 30 values
    result_df = items_value_counts.reset_index(drop=False)
    result_df.columns = ["item_sk_1", "cnt"]
    result_df = result_df.head(q02_limit)
    result_df["item_sk_2"] = q02_item_sk
    result_order = ["item_sk_1", "item_sk_2", "cnt"]
    result_df = result_df[result_order]
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
