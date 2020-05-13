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


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os

import numpy as np
from distributed import wait
import cupy as cp

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)
from tpcx_bb_query_08 import (
    get_session_id_from_session_boundary,
    get_session_id,
    get_sessions,
    get_unique_sales_keys_from_sessions,
    prep_for_sessionization,
)

cli_args = tpcxbb_argparser()


q08_SECONDS_BEFORE_PURCHASE = 259200
REVIEW_CAT_CODE = 6
NA_FLAG = 0


@benchmark()
def read_tables(data_dir):
    bc.create_table("web_clickstreams", data_dir + "/web_clickstreams/*.parquet")
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")
    bc.create_table("web_page", data_dir + "/web_page/*.parquet")
    bc.create_table("web_returns", data_dir + "/web_returns/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")
    bc.create_table("warehouse", data_dir + "/warehouse/*.parquet")


@benchmark(dask_profile=cli_args.get("dask_profile"))
def main(data_dir):
    read_tables(data_dir)

    dates = bc.sql(
        f"""
	select d_date_sk from date_dim
	where CAST(d_date as date) in (date '2001-09-02', date '2002-09-02')
	order by cast(d_date as date) asc
	"""
    )
    dates_cpu = dates.compute()
    dates_cpu.index = list(range(0, dates_cpu.shape[0]))
    start = dates_cpu["d_date_sk"][0]
    end = dates_cpu["d_date_sk"][1]

    merged_df = bc.sql(
        f"""
	select 
		CAST(wcs_user_sk AS INTEGER) as wcs_user_sk,
		(wcs_click_date_sk * 86400 + wcs_click_time_sk) as tstamp_inSec,
		wcs_sales_sk,
		wp_type,
		case when wp_type = 'review' then 1 else 0 end as review_flag
	from web_clickstreams
	inner join web_page on wcs_web_page_sk = wp_web_page_sk
	where wcs_user_sk IS NOT NULL and wcs_click_date_sk between {start} and {end}
	-- order by wcs_user_sk, tstamp_inSec, wcs_sales_sk, wp_type
	"""
    )

    temp = merged_df
    merged_df["wp_type"] = merged_df["wp_type"].astype("category")

    merged_df["wp_type_codes"] = merged_df["wp_type"].cat.codes
    merged_df["wp_type_codes"] = merged_df["wp_type_codes"].astype("int8")

    col_keep = [
        "wcs_user_sk",
        "tstamp_inSec",
        "wcs_sales_sk",
        "wp_type_codes",
        "review_flag",
    ]
    merged_df = merged_df[col_keep]

    merged_df = merged_df.repartition(columns=["wcs_user_sk"])

    prepped = merged_df.map_partitions(
        prep_for_sessionization, review_cat_code=REVIEW_CAT_CODE
    )

    sessionized = prepped.map_partitions(get_sessions)

    unique_review_sales = sessionized.map_partitions(
        get_unique_sales_keys_from_sessions, review_cat_code=REVIEW_CAT_CODE
    )

    bc.create_table("reviews", unique_review_sales.to_frame().persist())

    result = bc.sql(
        f"""
	select
		CAST(review_total AS BIGINT)as q08_review_sales_amount,
		CAST(total - review_total AS BIGINT) as no_q08_review_sales_amount
	from (
		select
		sum(ws_net_paid) as total,
		sum(case when wcs_sales_sk is null then 0 else 1 end * ws_net_paid) as review_total
		from web_sales
		left outer join reviews on ws_order_number = wcs_sales_sk
		where ws_sold_date_sk between {start} and {end}
	)
	"""
    )

    return result


if __name__ == "__main__":

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
