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

from bdb_tools.cluster_startup import attach_to_cluster

import cudf
import datetime
from datetime import timedelta
from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q16_utils import read_tables

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

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

    dates = c.sql(date_query)
 
    cpu_dates = dates["d_date_sk"].compute()
    
    if isinstance(cpu_dates, cudf.Series):
        cpu_dates = cpu_dates.to_pandas()
        
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

    result = c.sql(last_query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
