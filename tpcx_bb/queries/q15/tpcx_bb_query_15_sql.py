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
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)

cli_args = tpcxbb_argparser()


# -------- Q15 -----------
# --store_sales date range
q15_startDate = "2001-09-02"
# --+1year
q15_endDate = "2002-09-02"
q15_store_sk = 10


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir, bc):
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = f"""
        SELECT *
        FROM
        (
            SELECT
                cat,
                ( (count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x) * SUM(x)) )  AS slope,
                (SUM(y) - ((count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x)*SUM(x)) ) * SUM(x)) / count(x) AS intercept
            FROM
            (
                SELECT
                    i.i_category_id AS cat,
                    s.ss_sold_date_sk AS x,
                    CAST(SUM(s.ss_net_paid) AS DOUBLE) AS y,
                    CAST(s.ss_sold_date_sk * SUM(s.ss_net_paid) AS DOUBLE) AS xy,
                    CAST(s.ss_sold_date_sk * s.ss_sold_date_sk AS DOUBLE) AS xx
                FROM store_sales s
                INNER JOIN item i ON s.ss_item_sk = i.i_item_sk
                INNER JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
                WHERE s.ss_store_sk = {q15_store_sk}
                AND i.i_category_id IS NOT NULL
                AND CAST(d.d_date AS DATE) >= DATE '{q15_startDate}'
                AND   CAST(d.d_date AS DATE) <= DATE '{q15_endDate}'
                GROUP BY i.i_category_id, s.ss_sold_date_sk
            ) temp
            GROUP BY cat
        ) regression
        WHERE slope <= 0.0
        ORDER BY cat
    """
    result = bc.sql(query)
    return result


if __name__ == "__main__":
    client, bc = attach_to_cluster(cli_args, create_blazing_context=True)
    run_query(cli_args=cli_args, client=client, query_func=main, blazing_context=bc)