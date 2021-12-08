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


def read_tables(data_dir, c, config):
	table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )
	
	item_cols = ["i_item_sk", "i_current_price", "i_category"]
	store_sales_cols = ["ss_item_sk", "ss_customer_sk", "ss_sold_date_sk"]
	store_cols = ["s_store_sk"]
	date_cols = ["d_date_sk", "d_year", "d_moy"]
	customer_cols = ["c_customer_sk", "c_current_addr_sk"]
	customer_address_cols = ["ca_address_sk", "ca_state"]

	item_df = table_reader.read("item", relevant_cols=item_cols)
	store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
	# store_df = table_reader.read("store", relevant_cols=store_cols)
	date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
	customer_df = table_reader.read("customer", relevant_cols=customer_cols)
	customer_address_df = table_reader.read(
		"customer_address", relevant_cols=customer_address_cols
	)

	c.create_table("item", item_df, persist=False)
	c.create_table("customer", customer_df, persist=False)
	c.create_table("store_sales", store_sales_df, persist=False)
	c.create_table("date_dim", date_dim_df, persist=False)
	c.create_table("customer_address", customer_address_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

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

    result = c.sql(query)
    return result


if __name__ == "__main__":
	config = gpubdb_argparser()
	client, c = attach_to_cluster(config)
	run_query(config=config, client=client, query_func=main, sql_context=c)
