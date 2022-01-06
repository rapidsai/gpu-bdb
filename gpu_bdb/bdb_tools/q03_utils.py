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

import os
import sys

import cudf

from numba import cuda

# -------- Q03 -----------
q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001

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

    load_q03()
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


def load_q03():
    import importlib, types

    fn = os.path.join(os.getcwd(), "gpu_bdb_query_03_dask_sql.py")
    if not os.path.isfile(fn):
        fn = os.path.join(os.getcwd(), "queries/q03/gpu_bdb_query_03_dask_sql.py")

    loader = importlib.machinery.SourceFileLoader("03", fn)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    sys.modules[loader.name] = mod
    return mod.main

