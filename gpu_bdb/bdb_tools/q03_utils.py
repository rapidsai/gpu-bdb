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

import cudf

from numba import cuda, jit

from bdb_tools.readers import build_reader

q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001
q03_purchased_item_category_IN = 2, 3
q03_limit = 100

def read_tables(config, c=None):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
        backend=config["backend"],
    )

    item_cols = ["i_category_id", "i_item_sk"]
    wcs_cols = [
        "wcs_user_sk",
        "wcs_click_time_sk",
        "wcs_click_date_sk",
        "wcs_item_sk",
        "wcs_sales_sk",
    ]

    item_df = table_reader.read("item", relevant_cols=item_cols)
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)

    if c:
        c.create_table("web_clickstreams", wcs_df, persist=False)
        c.create_table("item", item_df, persist=False)

    return item_df


@cuda.jit
def find_items_viewed_before_purchase_kernel_gpu(
    relevant_idx_col, user_col, timestamp_col, item_col, out_col, N
):
    """
    Find the past N items viewed after a relevant purchase was made,
    as defined by the configuration of this query.
    """
    i = cuda.grid(1)

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

@jit(nopython=True)
def find_items_viewed_before_purchase_kernel_cpu(
    relevant_idx_col, user_col, timestamp_col, item_col, out_col, N
):
    """
    Find the past N items viewed after a relevant purchase was made,
    as defined by the configuration of this query.
    """
#     i = cuda.grid(1)
    relevant_item = q03_purchased_item_IN

    for i in range(relevant_idx_col.size):  # boundary guard
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
    # need to sort descending to ensure that the
    # next N rows are the previous N clicks
    import pandas as pd
    import numpy as np
    
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
    if isinstance(df, cudf.DataFrame):
        out_arr = cuda.device_array(size * N, dtype=df["wcs_item_sk"].dtype)
        find_items_viewed_before_purchase_kernel_gpu.forall(size)(
            sample["relevant_idx_pos"],
            df["wcs_user_sk"],
            df["tstamp"],
            df["wcs_item_sk"],
            out_arr,
            N,
        ) 
        result = cudf.DataFrame({"prior_item_viewed": out_arr})
    else: 
        out_arr = np.zeros(size * N, dtype=df["wcs_item_sk"].dtype, like=df["wcs_item_sk"].values)
        find_items_viewed_before_purchase_kernel_cpu(
            sample["relevant_idx_pos"].to_numpy(),
            df["wcs_user_sk"].to_numpy(),
            df["tstamp"].to_numpy(),
            df["wcs_item_sk"].to_numpy(),
            out_arr,
            N,
        )
        result = pd.DataFrame({"prior_item_viewed": out_arr})
        
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

