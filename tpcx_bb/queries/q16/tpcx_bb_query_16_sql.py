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


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os

import datetime
from datetime import timedelta
from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

cli_args = tpcxbb_argparser()


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir, bc):
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")
    bc.create_table("web_returns", data_dir + "/web_returns/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")
    bc.create_table("warehouse", data_dir + "/warehouse/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    date = datetime.datetime(2001, 3, 16)
    start = (date + timedelta(days=-30)).strftime("%Y-%m-%d")
    end = (date + timedelta(days=30)).strftime("%Y-%m-%d")
    mid = date.strftime("%Y-%m-%d")

    date_query = f"""
        SELECT d_date_sk 
        FROM date_dim 
        WHERE CAST(d_date as DATE) IN (DATE '{start}', DATE '{mid}', DATE '{end}') 
        ORDER BY CAST(d_date as date) ASC
    """

    dates = bc.sql(date_query)

    cpu_dates = dates["d_date_sk"].compute().to_pandas()
    cpu_dates.index = list(range(0, cpu_dates.shape[0]))

    last_query = f"""
        SELECT w_state, i_item_id,
        SUM
        (
            CASE WHEN ws_sold_date_sk < {str(cpu_dates[1])}
            THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
            ELSE 0.0 END
        ) AS sales_before,
        SUM
        (
            CASE WHEN ws_sold_date_sk >= {str(cpu_dates[1])}
            THEN ws_sales_price - COALESCE(wr_refunded_cash,0)
            ELSE 0.0 END
        ) AS sales_after
        FROM 
        (
            SELECT ws_item_sk, 
                ws_warehouse_sk, 
                ws_sold_date_sk, 
                ws_sales_price, 
                wr_refunded_cash
            FROM web_sales ws
            LEFT OUTER JOIN web_returns wr ON 
            (
                ws.ws_order_number = wr.wr_order_number
                AND ws.ws_item_sk = wr.wr_item_sk
            )
            WHERE ws_sold_date_sk BETWEEN {str(cpu_dates[0])}
            AND {str(cpu_dates[2])}
        ) a1
        JOIN item i ON a1.ws_item_sk = i.i_item_sk
        JOIN warehouse w ON a1.ws_warehouse_sk = w.w_warehouse_sk
        GROUP BY w_state,i_item_id 
        ORDER BY w_state,i_item_id
        LIMIT 100
    """

    result = bc.sql(last_query)
    return result


if __name__ == "__main__":
    client, bc = attach_to_cluster(cli_args, create_blazing_context=True)
    run_query(cli_args=cli_args, client=client, query_func=main, blazing_context=bc)