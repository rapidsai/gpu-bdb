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
import glob

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)
from bdb_tools.readers import build_reader

from distributed import wait
import numpy as np
from dask import delayed

### Current Implementation Assumption
# The filtered item table will fit in GPU memory :
# At scale `100` ->non filtered-rows `178,200` -> filtered rows `60,059`
# Extrapolation to scale `1_000_000` ->non filtered-rows `17,820,000` -> filtered rows `6,005,900` (So should scale up)


### These parameters are not used
# q12_startDate='2001-09-02'
# q12_endDate1='2001-10-02'
# q12_endDate2='2001-12-02'
q12_i_category_IN = ["Books", "Electronics"]

### below was hard coded in the orignal query
q12_store_sale_sk_start_date = 37134

item_cols = ["i_item_sk", "i_category"]
store_sales_cols = ["ss_item_sk", "ss_sold_date_sk", "ss_customer_sk"]

### Util Functions
def string_filter(df, col_name, col_values):
    """
        Filters strings based on values
    """
    bool_flag = None

    for val in col_values:
        if bool_flag is None:
            bool_flag = df[col_name] == val
        else:
            bool_flag = (df[col_name] == val) | (bool_flag)

    return df[bool_flag].reset_index(drop=True)


def read_tables(config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_df = table_reader.read("item", relevant_cols=item_cols)
    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)

    return item_df, store_sales_df


def filter_wcs_table(web_clickstreams_fn, filtered_item_df):
    """
        Filter web clickstreams table
        
        ###  SELECT
        ###    wcs_user_sk,
        ###    wcs_click_date_sk
        ###  FROM web_clickstreams, item
        ###  WHERE wcs_click_date_sk BETWEEN 37134 AND (37134 + 30) -- in a given month and year
        ###  -- AND i_category IN ({q12_i_category_IN}) -- filter given category
        ###   AND wcs_item_sk = i_item_sk
        ##    AND wcs_user_sk IS NOT NULL
        ###   AND wcs_sales_sk IS NULL --only views, not purchases
    """
    import cudf

    web_clickstreams_cols = [
        "wcs_user_sk",
        "wcs_click_date_sk",
        "wcs_item_sk",
        "wcs_sales_sk",
    ]
    web_clickstreams_df = cudf.read_parquet(
        web_clickstreams_fn, columns=web_clickstreams_cols
    )

    filter_wcs_df = web_clickstreams_df[
        web_clickstreams_df["wcs_user_sk"].notnull()
        & web_clickstreams_df["wcs_sales_sk"].isnull()
    ].reset_index(drop=True)

    filter_wcs_df = filter_wcs_df.loc[
        (filter_wcs_df["wcs_click_date_sk"] >= q12_store_sale_sk_start_date)
        & (filter_wcs_df["wcs_click_date_sk"] <= (q12_store_sale_sk_start_date + 30))
    ].reset_index(drop=True)

    filter_wcs_df = filter_wcs_df.merge(
        filtered_item_df, left_on=["wcs_item_sk"], right_on=["i_item_sk"], how="inner"
    )

    return filter_wcs_df[["wcs_user_sk", "wcs_click_date_sk"]]


def filter_ss_table(store_sales_df, filtered_item_df):
    """
        Filter store sales table
        
        ###  SELECT
        ###    ss_customer_sk,
        ###    ss_sold_date_sk
        ###  FROM store_sales, item
        ###  WHERE ss_sold_date_sk BETWEEN 37134 AND (37134 + 90) -- in the three consecutive months.
        ###  AND i_category IN ({q12_i_category_IN}) -- filter given category
        ###  AND ss_item_sk = i_item_sk
        ###  AND ss_customer_sk IS NOT NULL

    """

    filtered_ss_df = store_sales_df[
        store_sales_df["ss_customer_sk"].notnull()
    ].reset_index(drop=True)

    filtered_ss_df = filtered_ss_df.loc[
        (filtered_ss_df["ss_sold_date_sk"] >= q12_store_sale_sk_start_date)
        & (filtered_ss_df["ss_sold_date_sk"] <= (q12_store_sale_sk_start_date + 90))
    ].reset_index(drop=True)

    filtered_ss_df = filtered_ss_df.merge(
        filtered_item_df, left_on=["ss_item_sk"], right_on=["i_item_sk"], how="inner"
    )
    return filtered_ss_df[["ss_customer_sk", "ss_sold_date_sk"]]


def main(client, config):
    import cudf, dask_cudf

    item_df, store_sales_df = benchmark(
        read_tables,
        config=config,
        compute_result=config["get_read_time"],
        dask_profile=config["dask_profile"],
    )

    ### Query 0. Filtering item table
    filtered_item_df = string_filter(item_df, "i_category", q12_i_category_IN)
    ### filtered_item_df is a single partition to allow a nx1 merge using map partitions
    filtered_item_df = filtered_item_df.repartition(npartitions=1)
    filtered_item_df = filtered_item_df.persist()
    wait(filtered_item_df)
    ###  Query 1

    # The main idea is that we don't fuse a filtration task with reading task yet
    # this  causes more memory pressures as we try to read the whole thing ( and spill that)
    # at once and then do filtration .

    meta_d = {
        "wcs_user_sk": np.ones(1, dtype=np.int64),
        "wcs_click_date_sk": np.ones(1, dtype=np.int64),
    }
    meta_df = cudf.DataFrame(meta_d)
    web_clickstream_flist = glob.glob(os.path.join(config["data_dir"], "web_clickstreams/*.parquet"))
    task_ls = [
        delayed(filter_wcs_table)(fn, filtered_item_df.to_delayed()[0])
        for fn in web_clickstream_flist
    ]
    filter_wcs_df = dask_cudf.from_delayed(task_ls, meta=meta_df)
    ###  Query 2

    # The main idea is that we don't fuse a filtration task with reading task yet
    # this  causes more memory pressures as we try to read the whole thing ( and spill that)
    # at once and then do filtration .

    meta_d = {
        "ss_customer_sk": np.ones(1, dtype=store_sales_df["ss_customer_sk"].dtype),
        "ss_sold_date_sk": np.ones(1, dtype=np.int64),
    }
    meta_df = cudf.DataFrame(meta_d)

    filtered_ss_df = store_sales_df.map_partitions(
        filter_ss_table, filtered_item_df.to_delayed()[0], meta=meta_df
    )

    ### Result Query
    ### SELECT DISTINCT wcs_user_sk
    ### ....
    ### webInRange
    ### storeInRange
    ### WHERE wcs_user_sk = ss_customer_sk
    ### AND wcs_click_date_sk < ss_sold_date_sk -- buy AFTER viewed on website
    ### ORDER BY wcs_user_sk

    ### Note: Below brings it down to a single partition
    filter_wcs_df_d = filter_wcs_df.drop_duplicates()
    filter_wcs_df_d = filter_wcs_df_d.persist()
    wait(filter_wcs_df_d)

    filtered_ss_df_d = filtered_ss_df.drop_duplicates()
    filtered_ss_df_d = filtered_ss_df_d.persist()
    wait(filtered_ss_df_d)

    ss_wcs_join = filter_wcs_df_d.merge(
        filtered_ss_df_d, left_on="wcs_user_sk", right_on="ss_customer_sk", how="inner"
    )

    ss_wcs_join = ss_wcs_join[
        ss_wcs_join["wcs_click_date_sk"] < ss_wcs_join["ss_sold_date_sk"]
    ]
    ss_wcs_join = ss_wcs_join["wcs_user_sk"]

    ### todo: check performence by replacing with 1 single drop_duplicates call

    ### below decreases memory usage on the single gpu to help with subsequent compute
    ss_wcs_join = ss_wcs_join.map_partitions(lambda sr: sr.drop_duplicates())
    ss_wcs_join = ss_wcs_join.repartition(npartitions=1).persist()
    ss_wcs_join = ss_wcs_join.drop_duplicates().reset_index(drop=True)
    ss_wcs_join = ss_wcs_join.map_partitions(lambda ser: ser.sort_values())

    # todo:check if repartition helps for writing efficiency
    # context: 0.1 seconds on sf-1k
    ss_wcs_join = ss_wcs_join.persist()
    wait(ss_wcs_join)
    return ss_wcs_join.to_frame()


if __name__ == "__main__":
    from bdb_tools.cluster_startup import attach_to_cluster
    import cudf
    import dask_cudf

    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main)
