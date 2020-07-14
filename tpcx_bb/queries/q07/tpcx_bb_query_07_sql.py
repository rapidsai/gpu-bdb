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
    bc.create_table("item", data_dir + "/item/*.parquet")
    bc.create_table("customer", data_dir + "/customer/*.parquet")
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("customer_address", data_dir + "/customer_address/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client, bc):
    read_tables(data_dir, bc)

    query = """
		WITH temp_table as 
		(
			SELECT k.i_item_sk
			FROM item k,
			(
				SELECT i_category, 
					SUM(j.i_current_price) / COUNT(j.i_current_price) * 1.2 AS avg_price
				FROM item j
				GROUP BY j.i_category
			) avgCategoryPrice
			WHERE avgCategoryPrice.i_category = k.i_category
			AND k.i_current_price > avgCategoryPrice.avg_price 
		)
		SELECT ca_state, COUNT(*) AS cnt
		FROM
			customer_address a,
			customer c,
			store_sales s,
			temp_table highPriceItems
		WHERE a.ca_address_sk = c.c_current_addr_sk
		AND c.c_customer_sk = s.ss_customer_sk
		AND ca_state IS NOT NULL
		AND ss_item_sk = highPriceItems.i_item_sk
		AND s.ss_sold_date_sk IN
		( 
			SELECT d_date_sk
			FROM date_dim
			WHERE d_year = 2004
			AND d_moy = 7
		)
		GROUP BY ca_state
		HAVING COUNT(*) >= 10
		ORDER BY cnt DESC, ca_state
		LIMIT 10
	"""

    result = bc.sql(query)
    return result


if __name__ == "__main__":
    client, bc = attach_to_cluster(cli_args, create_blazing_context=True)
    run_query(cli_args=cli_args, client=client, query_func=main, blazing_context=bc)
