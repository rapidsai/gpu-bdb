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

# Original code /tpcx-bb/tpcx-bb1.3.1/rapids-queries/q02/tpcx-bb-query-02.py
import sys

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader
from xbb_tools.sessionization import get_session_id, get_distinct_sessions, get_pairs

from dask import delayed

### needed for set index
import os
import numpy as np
import glob

### Implementation Notes:

### Future Notes:
# The bottleneck of current implementation is `set-index`, once ucx is working correctly
# it should go away


### session timeout in secs
q30_session_timeout_inSec = 3600
### query output limit
q30_limit = 40


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    item_df = table_reader.read("item", relevant_cols=item_cols)
    return item_df


def pre_repartition_task(wcs_fn, f_item_df):
    """
        Runs the pre-repartition task
    """
    import cudf

    wcs_cols = ["wcs_user_sk", "wcs_item_sk", "wcs_click_date_sk", "wcs_click_time_sk"]
    wcs_df = cudf.read_parquet(wcs_fn, columns=wcs_cols)

    f_wcs_df = wcs_df[wcs_df["wcs_user_sk"].notnull()].reset_index(drop=True)
    merged_df = f_wcs_df.merge(
        f_item_df, left_on=["wcs_item_sk"], right_on=["i_item_sk"]
    )
    del wcs_df

    merged_df["tstamp_inSec"] = (
        merged_df["wcs_click_date_sk"] * 24 * 60 * 60 + merged_df["wcs_click_time_sk"]
    )
    cols_keep = ["wcs_user_sk", "tstamp_inSec", "i_category_id"]
    merged_df = merged_df[cols_keep]
    merged_df["i_category_id"] = merged_df["i_category_id"].astype(np.int8)

    return merged_df


def main(client, config):
    import dask_cudf
    import cudf

    item_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    """
    Filter and Join web_clickstreams and item table.
    SELECT wcs_user_sk,
      (wcs_click_date_sk*24L*60L*60L + wcs_click_time_sk) AS tstamp_inSec,
      i_category_id
    FROM web_clickstreams wcs, item i
    WHERE wcs.wcs_item_sk = i.i_item_sk
    AND i.i_category_id IS NOT NULL
    AND wcs.wcs_user_sk IS NOT NULL
    """
    f_item_df = item_df[item_df["i_category_id"].notnull()].reset_index(drop=True)

    # The main idea is that we don't fuse a filtration task with reading task yet
    # this  causes more memory pressures as we try to read the whole thing ( and spill that)
    # at once and then do filtration .

    ### Below Pr has the dashboard snapshot which makes the problem clear
    ### https://github.com/rapidsai/tpcx-bb-internal/pull/496#issue-399946141

    web_clickstream_flist = glob.glob(config["data_dir"] + "web_clickstreams/*.parquet")
    task_ls = [
        delayed(pre_repartition_task)(fn, f_item_df.to_delayed()[0])
        for fn in web_clickstream_flist
    ]

    meta_d = {
        "wcs_user_sk": np.ones(1, dtype=np.int64),
        "tstamp_inSec": np.ones(1, dtype=np.int64),
        "i_category_id": np.ones(1, dtype=np.int8),
    }
    meta_df = cudf.DataFrame(meta_d)

    merged_df = dask_cudf.from_delayed(task_ls, meta=meta_df)

    ### that the click for each user ends up at the same partition
    merged_df = merged_df.shuffle(on=["wcs_user_sk"])

    ### Main Query
    ### sessionize logic.
    distinct_session_df = merged_df.map_partitions(
        get_distinct_sessions,
        keep_cols=["wcs_user_sk", "i_category_id"],
        time_out=q30_session_timeout_inSec,
    )

    del merged_df
    ### create pairs out of item category id's.
    pair_df = distinct_session_df.map_partitions(
        get_pairs,
        pair_col="i_category_id",
        output_col_1="category_id_1",
        output_col_2="category_id_2",
    )

    del distinct_session_df
    ### apply groupby on "category_id_1", "category_id_2"
    grouped_df = (
        pair_df.groupby(["category_id_1", "category_id_2"])
        .size(split_every=2)
        .reset_index()
    )

    grouped_df.columns = ["category_id_1", "category_id_2", "cnt"]

    result_df = grouped_df.repartition(npartitions=1).persist()
    ### sort records in desc order and reset index.
    ### below only has 40 rows so leaving as cudf frame should be fine
    result_df = result_df.map_partitions(
        lambda x: x.sort_values("cnt", ascending=False)
    )
    result_df = result_df.reset_index(drop=True).head(q30_limit)
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
