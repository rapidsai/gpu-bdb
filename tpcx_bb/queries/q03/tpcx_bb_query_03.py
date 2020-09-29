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
import os


from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)
from xbb_tools.readers import build_reader

from distributed import wait
import numpy as np

from numba import cuda
import glob
from dask import delayed


q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001
q03_purchased_item_category_IN = [2, 3]
q03_limit = 100


def get_wcs_minima(config):
    import dask_cudf

    wcs_df = dask_cudf.read_parquet(
        os.path.join(config["data_dir"], "web_clickstreams/*.parquet"),
        columns=["wcs_click_date_sk", "wcs_click_time_sk"],
    )

    wcs_df["tstamp"] = wcs_df["wcs_click_date_sk"] * 86400 + wcs_df["wcs_click_time_sk"]

    wcs_tstamp_min = wcs_df["tstamp"].min().compute()
    return wcs_tstamp_min


def pre_repartition_task(wcs_fn, item_df, wcs_tstamp_min):
    import cudf

    wcs_cols = [
        "wcs_user_sk",
        "wcs_sales_sk",
        "wcs_item_sk",
        "wcs_click_date_sk",
        "wcs_click_time_sk",
    ]
    wcs_df = cudf.read_parquet(wcs_fn, columns=wcs_cols)
    wcs_df = wcs_df._drop_na_rows(subset=["wcs_user_sk", "wcs_item_sk"])
    wcs_df["tstamp"] = wcs_df["wcs_click_date_sk"] * 86400 + wcs_df["wcs_click_time_sk"]
    wcs_df["tstamp"] = wcs_df["tstamp"] - wcs_tstamp_min

    wcs_df["tstamp"] = wcs_df["tstamp"].astype("int32")
    wcs_df["wcs_user_sk"] = wcs_df["wcs_user_sk"].astype("int32")
    wcs_df["wcs_sales_sk"] = wcs_df["wcs_sales_sk"].astype("int32")
    wcs_df["wcs_item_sk"] = wcs_df["wcs_item_sk"].astype("int32")

    merged_df = wcs_df.merge(
        item_df, left_on=["wcs_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    del wcs_df
    del item_df

    cols_keep = [
        "wcs_user_sk",
        "tstamp",
        "wcs_item_sk",
        "wcs_sales_sk",
        "i_category_id",
    ]
    merged_df = merged_df[cols_keep]

    merged_df["wcs_user_sk"] = merged_df["wcs_user_sk"].astype("int32")
    merged_df["wcs_sales_sk"] = merged_df["wcs_sales_sk"].astype("int32")
    merged_df["wcs_item_sk"] = merged_df["wcs_item_sk"].astype("int32")
    merged_df["wcs_sales_sk"] = merged_df.wcs_sales_sk.fillna(0)
    return merged_df


def reduction_function(df, item_df_filtered):
    """
         Combines all the reduction ops into a single frame
    """
    product_view_results = apply_find_items_viewed(df, item_mappings=item_df_filtered)

    grouped_df = product_view_results.groupby(["i_item_sk"]).size().reset_index()
    grouped_df.columns = ["i_item_sk", "cnt"]
    return grouped_df


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    item_df = table_reader.read("item", relevant_cols=item_cols)
    return item_df


@cuda.jit
def find_items_viewed_before_purchase_kernel(
    relevant_idx_col, user_col, timestamp_col, item_col, out_col, N
):
    """
    Find the past N items viewed after a relevant purchase was made,
    as defined by the configuration of this query.
    """
    i = cuda.grid(1)
    relevant_item = q03_purchased_item_IN

    if i < (relevant_idx_col.size):  # boundary guard
        # every relevant row gets N rows in the output, so we need to map the indexes
        # back into their position in the original array
        orig_idx = relevant_idx_col[i]
        current_user = user_col[orig_idx]

        # look at the previous N clicks (assume sorted descending)
        rows_to_check = N
        remaining_rows = user_col.size - orig_idx

        if remaining_rows <= rows_to_check:
            rows_to_check = remaining_rows - 1

        for k in range(1, rows_to_check + 1):
            if current_user != user_col[orig_idx + k]:
                out_col[i * N + k - 1] = 0

            # only checking relevant purchases via the relevant_idx_col
            elif (timestamp_col[orig_idx + k] <= timestamp_col[orig_idx]) & (
                timestamp_col[orig_idx + k]
                >= (timestamp_col[orig_idx] - q03_days_in_sec_before_purchase)
            ):
                out_col[i * N + k - 1] = item_col[orig_idx + k]
            else:
                out_col[i * N + k - 1] = 0


def apply_find_items_viewed(df, item_mappings):
    import cudf

    # need to sort descending to ensure that the
    # next N rows are the previous N clicks
    df = df.sort_values(
        by=["wcs_user_sk", "tstamp", "wcs_sales_sk", "wcs_item_sk"],
        ascending=[False, False, False, False],
    )
    df.reset_index(drop=True, inplace=True)
    df["relevant_flag"] = (df.wcs_sales_sk != 0) & (
        df.wcs_item_sk == q03_purchased_item_IN
    )
    df["relevant_idx_pos"] = df.index.to_series()
    df.reset_index(drop=True, inplace=True)
    # only allocate output for the relevant rows
    sample = df.loc[df.relevant_flag == True]
    sample.reset_index(drop=True, inplace=True)

    N = q03_views_before_purchase
    size = len(sample)

    # we know this can be int32, since it's going to contain item_sks
    out_arr = cuda.device_array(size * N, dtype=df["wcs_item_sk"].dtype)

    find_items_viewed_before_purchase_kernel.forall(size)(
        sample["relevant_idx_pos"],
        df["wcs_user_sk"],
        df["tstamp"],
        df["wcs_item_sk"],
        out_arr,
        N,
    )

    result = cudf.DataFrame({"prior_item_viewed": out_arr})

    del out_arr
    del df
    del sample

    filtered = result.merge(
        item_mappings,
        how="inner",
        left_on=["prior_item_viewed"],
        right_on=["i_item_sk"],
    )
    return filtered


def main(client, config):
    import dask_cudf
    import cudf

    item_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    wcs_tstamp_min = get_wcs_minima(config)

    item_df["i_item_sk"] = item_df["i_item_sk"].astype("int32")
    item_df["i_category_id"] = item_df["i_category_id"].astype("int8")

    # we eventually will only care about these categories, so we can filter now
    item_df_filtered = item_df.loc[
        item_df.i_category_id.isin(q03_purchased_item_category_IN)
    ].reset_index(drop=True)

    # The main idea is that we don't fuse a filtration task with reading task yet
    # this causes more memory pressures as we try to read the whole thing ( and spill that)
    # at once and then do filtration .

    ### Below Pr has the dashboard snapshot which makes the problem clear
    ### https://github.com/rapidsai/tpcx-bb-internal/pull/496#issue-399946141

    web_clickstream_flist = glob.glob(os.path.join(config["data_dir"], "web_clickstreams/*.parquet"))
    task_ls = [
        delayed(pre_repartition_task)(fn, item_df.to_delayed()[0], wcs_tstamp_min)
        for fn in web_clickstream_flist
    ]

    meta_d = {
        "wcs_user_sk": np.ones(1, dtype=np.int32),
        "tstamp": np.ones(1, dtype=np.int32),
        "wcs_item_sk": np.ones(1, dtype=np.int32),
        "wcs_sales_sk": np.ones(1, dtype=np.int32),
        "i_category_id": np.ones(1, dtype=np.int8),
    }
    meta_df = cudf.DataFrame(meta_d)

    merged_df = dask_cudf.from_delayed(task_ls, meta=meta_df)

    merged_df = merged_df.repartition(columns="wcs_user_sk")

    meta_d = {
        "i_item_sk": np.ones(1, dtype=merged_df["wcs_item_sk"].dtype),
        "cnt": np.ones(1, dtype=merged_df["wcs_item_sk"].dtype),
    }
    meta_df = cudf.DataFrame(meta_d)

    grouped_df = merged_df.map_partitions(
        reduction_function, item_df_filtered.to_delayed()[0], meta=meta_df
    )

    ### todo: check if this has any impact on stability
    grouped_df = grouped_df.persist(priority=10000)
    ### todo: remove this later after more testing
    wait(grouped_df)
    print("---" * 20)
    print("grouping complete ={}".format(len(grouped_df)))
    grouped_df = grouped_df.groupby(["i_item_sk"]).sum(split_every=2).reset_index()
    grouped_df.columns = ["i_item_sk", "cnt"]
    result_df = grouped_df.map_partitions(
        lambda df: df.sort_values(by=["cnt"], ascending=False)
    )

    result_df.columns = ["lastviewed_item", "cnt"]
    result_df["purchased_item"] = q03_purchased_item_IN
    cols_order = ["purchased_item", "lastviewed_item", "cnt"]
    result_df = result_df[cols_order]
    result_df = result_df.persist()
    ### todo: remove this later after more testing
    wait(result_df)
    print(len(result_df))
    result_df = result_df.head(q03_limit)
    print("result complete")
    print("---" * 20)
    return result_df


if __name__ == "__main__":
    from xbb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
