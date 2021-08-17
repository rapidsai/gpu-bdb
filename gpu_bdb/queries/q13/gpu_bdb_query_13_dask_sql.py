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

from bdb_tools.cluster_startup import attach_to_cluster
from dask.distributed import Client
import os

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask.distributed import wait

from dask_sql import Context

def read_tables(data_dir, bc):
	table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
	)

	date_cols = ["d_date_sk", "d_year"]
	date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)

	customer_cols = ["c_customer_sk", "c_customer_id", "c_first_name", "c_last_name"]
	customer_df = table_reader.read("customer", relevant_cols=customer_cols)

	s_sales_cols = ["ss_sold_date_sk", "ss_customer_sk", "ss_net_paid"]
	s_sales_df = table_reader.read("store_sales", relevant_cols=s_sales_cols)

	w_sales_cols = ["ws_sold_date_sk", "ws_bill_customer_sk", "ws_net_paid"]
	web_sales_df = table_reader.read("web_sales", relevant_cols=w_sales_cols)

	bc.create_table("date_dim", date_dim_df, persist=False)
	bc.create_table("customer", customer_df, persist=False)
	bc.create_table("store_sales", s_sales_df, persist=False)
	bc.create_table("web_sales", web_sales_df, persist=False)

    # bc.create_table("date_dim", os.path.join(data_dir, "date_dim/*.parquet"))
    # bc.create_table("customer", os.path.join(data_dir, "customer/*.parquet"))
    # bc.create_table("store_sales", os.path.join(data_dir, "store_sales/*.parquet"))
    # bc.create_table("web_sales", os.path.join(data_dir, "web_sales/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

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
    temp_table1 = bc.sql(query_1)

    temp_table1 = temp_table1.persist()
    wait(temp_table1)
    bc.create_table("temp_table1", temp_table1, persist=False)
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
    temp_table2 = bc.sql(query_2)

    temp_table2 = temp_table2.persist()
    wait(temp_table2)
    bc.create_table("temp_table2", temp_table2, persist=False)
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
    result = bc.sql(query)

    bc.drop_table("temp_table1")
    bc.drop_table("temp_table2")
    return result


if __name__ == "__main__":
	config = gpubdb_argparser()
	client, _ = attach_to_cluster(config)
	c = Context()
	run_query(config=config, client=client, query_func=main, blazing_context=c)
