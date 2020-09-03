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
from numba import cuda

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

from dask.distributed import wait

# -------- Q03 -----------
q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001
# --see q1 for categories
q03_purchased_item_category_IN = "2,3"
q03_limit = 100


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


def read_tables(data_dir, bc):
    bc.create_table("web_clickstreams",
                    data_dir + "web_clickstreams/*.parquet")
    bc.create_table("item", data_dir + "item/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query_1 = """
        SELECT i_item_sk,
            CAST(i_category_id AS TINYINT) AS i_category_id
        FROM item
    """
    item_df = bc.sql(query_1)

    item_df = item_df.persist()
    wait(item_df)
    bc.create_table("item_df", item_df)

    query_2 = """
        SELECT CAST(w.wcs_user_sk AS INTEGER) as wcs_user_sk,
            wcs_click_date_sk * 86400 + wcs_click_time_sk AS tstamp,
            CAST(w.wcs_item_sk AS INTEGER) as wcs_item_sk,
            CAST(COALESCE(w.wcs_sales_sk, 0) AS INTEGER) as wcs_sales_sk
        FROM web_clickstreams AS w
        INNER JOIN item_df AS i ON w.wcs_item_sk = i.i_item_sk
        WHERE w.wcs_user_sk IS NOT NULL
        AND w.wcs_item_sk IS NOT NULL
        ORDER BY w.wcs_user_sk
    """
    merged_df = bc.sql(query_2)

    query_3 = f"""
        SELECT i_item_sk, i_category_id
        FROM item_df
        WHERE i_category_id IN ({q03_purchased_item_category_IN})
    """
    item_df_filtered = bc.sql(query_3)

    product_view_results = merged_df.map_partitions(
        apply_find_items_viewed, item_mappings=item_df_filtered
    )
    
    product_view_results = product_view_results.persist()
    wait(product_view_results)

    bc.drop_table("item_df")
    del item_df
    del merged_df
    del item_df_filtered

    bc.create_table('product_result', product_view_results)

    last_query = f"""
        SELECT CAST({q03_purchased_item_IN} AS BIGINT) AS purchased_item,
            i_item_sk AS lastviewed_item,
            COUNT(i_item_sk) AS cnt
        FROM product_result
        GROUP BY i_item_sk
        ORDER BY purchased_item, cnt desc, lastviewed_item
        LIMIT {q03_limit}
    """
    result = bc.sql(last_query)

    bc.drop_table("product_result")
    del product_view_results
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
