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
    bc.create_table(
        "household_demographics", data_dir + "/household_demographics/*.parquet"
    )
    bc.create_table("web_page", data_dir + "/web_page/*.parquet")
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")
    bc.create_table("time_dim", data_dir + "/time_dim/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client, bc):
    read_tables(data_dir, bc)

    query = """ 
		SELECT CASE WHEN pmc > 0.0 THEN CAST (amc AS DOUBLE) / CAST (pmc AS DOUBLE) ELSE -1.0 END AS am_pm_ratio
		FROM 
		(
			SELECT SUM(amc1) AS amc, SUM(pmc1) AS pmc
			FROM
			(
				SELECT
					CASE WHEN t_hour BETWEEN 7 AND 8 THEN COUNT(1) ELSE 0 END AS amc1,
					CASE WHEN t_hour BETWEEN 19 AND 20 THEN COUNT(1) ELSE 0 END AS pmc1
				FROM web_sales ws
				JOIN household_demographics hd ON (hd.hd_demo_sk = ws.ws_ship_hdemo_sk and hd.hd_dep_count = 5)
				JOIN web_page wp ON (wp.wp_web_page_sk = ws.ws_web_page_sk and wp.wp_char_count BETWEEN 5000 AND 6000)
				JOIN time_dim td ON (td.t_time_sk = ws.ws_sold_time_sk and td.t_hour IN (7,8,19,20))
				GROUP BY t_hour
			) cnt_am_pm
		) sum_am_pm
	"""

    result = bc.sql(query)
    return result


if __name__ == "__main__":
    client, bc = attach_to_cluster(cli_args, create_blazing_context=True)
    run_query(cli_args=cli_args, client=client, query_func=main, blazing_context=bc)
