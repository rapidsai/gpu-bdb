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
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os

import dask
import numpy as np
import sys
import os
from numba import cuda
import rmm

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
    get_query_number,
)
from tpcx_bb_query_03 import apply_find_items_viewed

cli_args = tpcxbb_argparser()

q03_days_in_sec_before_purchase = 864000
q03_views_before_purchase = 5
q03_purchased_item_IN = 10001
q03_purchased_item_category_IN = [2, 3]
q03_limit = 100


q03_purchased_item_category_IN_sql_str = str(q03_purchased_item_category_IN)
q03_purchased_item_category_IN_sql_str = q03_purchased_item_category_IN_sql_str.replace(
    "[", "("
)
q03_purchased_item_category_IN_sql_str = q03_purchased_item_category_IN_sql_str.replace(
    "]", ")"
)


@benchmark()
def read_tables(data_dir):
    bc.create_table("web_clickstreams", data_dir + "web_clickstreams/*.parquet")
    bc.create_table("item", data_dir + "item/*.parquet")


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(data_dir):
    read_tables(data_dir)

    query = "SELECT i_item_sk, i_category_id from item"
    item_df = bc.sql(query)
    item_df["i_category_id"] = item_df["i_category_id"].astype("int8")
    bc.create_table("item_df", item_df)

    query = """ 
		SELECT  CAST(w.wcs_user_sk AS INTEGER) as wcs_user_sk, 
				wcs_click_date_sk * 86400 + wcs_click_time_sk AS tstamp, 
				CAST(w.wcs_item_sk AS INTEGER) as wcs_item_sk, 
				COALESCE(w.wcs_sales_sk, 0) as wcs_sales_sk, 
				i.i_category_id
		FROM web_clickstreams AS w
		INNER JOIN item_df AS i ON w.wcs_item_sk = i.i_item_sk
		WHERE w.wcs_user_sk IS NOT NULL 
		AND w.wcs_item_sk IS NOT NULL
		ORDER BY w.wcs_user_sk
	"""
    merged_df = bc.sql(query)

    # OR using partition below
    # query = """
    # 	SELECT  CAST(w.wcs_user_sk AS INTEGER) as wcs_user_sk,
    # 			wcs_click_date_sk * 86400 + wcs_click_time_sk AS tstamp,
    # 			CAST(w.wcs_item_sk AS INTEGER) as wcs_item_sk,
    # 			COALESCE(w.wcs_sales_sk, 0) as wcs_sales_sk,
    # 			i.i_category_id
    # 	FROM web_clickstreams AS w
    # 	INNER JOIN item_df AS i ON w.wcs_item_sk = i.i_item_sk
    # 	WHERE w.wcs_user_sk IS NOT NULL
    # 	AND w.wcs_item_sk IS NOT NULL
    # """
    # result_df = bc.sql(query)
    # result_df = bc.partition(result_df, by=['wcs_user_sk'])

    query = f"SELECT i_item_sk, i_category_id from item_df WHERE i_category_id IN {q03_purchased_item_category_IN_sql_str}"
    item_df_filtered = bc.sql(query)

    meta = cudf.DataFrame(
        {
            "prior_item_viewed": np.array([], dtype=merged_df["wcs_item_sk"].dtype),
            "i_category_id": np.array([], dtype=merged_df["i_category_id"].dtype),
            "i_item_sk": np.array([], dtype=merged_df["wcs_item_sk"].dtype),
        }
    )

    # product_view_results = merged_df.map_partitions(
    #     apply_find_items_viewed, item_mappings=item_df_filtered, meta=meta
    # )
    product_view_results = merged_df.map_partitions(
        apply_find_items_viewed, item_mappings=item_df_filtered
    )

    grouped_df = product_view_results.groupby(["i_item_sk"]).size().reset_index()
    grouped_df.columns = ["i_item_sk", "cnt"]

    result_df = grouped_df.map_partitions(
        lambda df: df.sort_values(by=["cnt"], ascending=False)
    )

    result_df.columns = ["lastviewed_item", "cnt"]
    result_df["purchased_item"] = q03_purchased_item_IN
    cols_order = ["purchased_item", "lastviewed_item", "cnt"]
    result_df = result_df[cols_order]
    result_df = result_df.head(q03_limit)

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
        cli_args["query_status"] = "Failed"
        print(traceback.format_exc())
