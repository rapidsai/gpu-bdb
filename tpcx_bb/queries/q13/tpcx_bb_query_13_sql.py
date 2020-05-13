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

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)

cli_args = tpcxbb_argparser()


@benchmark(dask_profile=cli_args["dask_profile"])
def read_tables(data_dir):
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("customer", data_dir + "/customer/*.parquet")
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
		WITH temp_table1 AS 
		(
			SELECT
				ss.ss_customer_sk AS customer_sk,
				sum(case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) first_year_total,
				sum(case when (d_year = 2002) THEN ss_net_paid ELSE 0.0 END) second_year_total
			FROM store_sales ss
			JOIN 
			(
				SELECT d_date_sk, d_year
				FROM date_dim d
				WHERE d.d_year in (2001, 2002)
			) dd ON (ss.ss_sold_date_sk = dd.d_date_sk)
			GROUP BY ss.ss_customer_sk 
			HAVING sum(case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) > 0.0
		),
		temp_table2 AS 
		(
			SELECT
				ws.ws_bill_customer_sk AS customer_sk,
				sum(case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) first_year_total,
				sum(case when (d_year = 2002) THEN ws_net_paid ELSE 0.0 END) second_year_total
			FROM web_sales ws
			JOIN 
			(
				SELECT d_date_sk, d_year
				FROM date_dim d
				WHERE d.d_year in (2001, 2002)
			) dd ON (ws.ws_sold_date_sk = dd.d_date_sk)
			GROUP BY ws.ws_bill_customer_sk 
			HAVING sum(case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) > 0.0
		)
		SELECT
			c_customer_sk,
			c_first_name,
			c_last_name,
			(store.second_year_total / store.first_year_total) AS storeSalesIncreaseRatio,
			(web.second_year_total / web.first_year_total) AS webSalesIncreaseRatio 
		FROM temp_table1 store,
			temp_table2 web,
			customer c
		WHERE store.customer_sk = web.customer_sk
		AND web.customer_sk = c_customer_sk
		AND (web.second_year_total / web.first_year_total) > (store.second_year_total / store.first_year_total) 
		ORDER BY webSalesIncreaseRatio DESC,
			c_customer_sk,
			c_first_name,
			c_last_name
		LIMIT 100
    """

    result = bc.sql(query)
    return result


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        dask_client=client,
        pool=True,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    result_df = main(cli_args["data_dir"])
    write_result(
        result_df, output_directory=cli_args["output_dir"],
    )
