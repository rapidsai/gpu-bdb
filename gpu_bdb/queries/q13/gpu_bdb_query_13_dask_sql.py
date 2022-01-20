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

import sys

from bdb_tools.cluster_startup import attach_to_cluster
from dask.distributed import Client
import os

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q13_utils import read_tables

from dask.distributed import wait

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

    query_1 = """
		SELECT
			ss.ss_customer_sk AS customer_sk,
			sum( case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) first_year_total,
			sum( case when (d_year = 2002) THEN ss_net_paid ELSE 0.0 END) second_year_total
		FROM store_sales ss
		JOIN 
		(
			SELECT d_date_sk, d_year
			FROM date_dim d
			WHERE d.d_year in (2001, 2002)
		) dd on ( ss.ss_sold_date_sk = dd.d_date_sk )
		GROUP BY ss.ss_customer_sk 
		HAVING sum( case when (d_year = 2001) THEN ss_net_paid ELSE 0.0 END) > 0.0
	"""
    temp_table1 = c.sql(query_1)

    temp_table1 = temp_table1.persist()
    wait(temp_table1)
    c.create_table("temp_table1", temp_table1, persist=False)
    query_2 = """
		SELECT
			ws.ws_bill_customer_sk AS customer_sk,
			sum( case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) first_year_total,
			sum( case when (d_year = 2002) THEN ws_net_paid ELSE 0.0 END) second_year_total
		FROM web_sales ws
		JOIN 
		(
			SELECT d_date_sk, d_year
			FROM date_dim d
			WHERE d.d_year in (2001, 2002)
		) dd ON ( ws.ws_sold_date_sk = dd.d_date_sk )
		GROUP BY ws.ws_bill_customer_sk 
		HAVING sum( case when (d_year = 2001) THEN ws_net_paid ELSE 0.0 END) > 0.0
	"""
    temp_table2 = c.sql(query_2)

    temp_table2 = temp_table2.persist()
    wait(temp_table2)
    c.create_table("temp_table2", temp_table2, persist=False)
    query = """
		SELECT
			CAST(c_customer_sk AS BIGINT) as c_customer_sk,
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
    result = c.sql(query)

    c.drop_table("temp_table1")
    c.drop_table("temp_table2")
    return result


if __name__ == "__main__":
	config = gpubdb_argparser()
	client, c = attach_to_cluster(config, create_sql_context=True)
	run_query(config=config, client=client, query_func=main, sql_context=c)
