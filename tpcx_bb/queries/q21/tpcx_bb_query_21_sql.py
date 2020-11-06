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

from xbb_tools.cluster_startup import attach_to_cluster

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)


def read_tables(data_dir, bc):
    bc.create_table("store_sales", os.path.join(data_dir, "store_sales/*.parquet"))
    bc.create_table("date_dim", os.path.join(data_dir, "date_dim/*.parquet"))
    bc.create_table("item", os.path.join(data_dir, "item/*.parquet"))
    bc.create_table("web_sales", os.path.join(data_dir, "web_sales/*.parquet"))
    bc.create_table("store_returns", os.path.join(data_dir, "store_returns/*.parquet"))
    bc.create_table("store", os.path.join(data_dir, "store/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query = """
		SELECT
			part_i.i_item_id AS i_item_id,
            part_i.i_item_desc AS i_item_desc,
            part_s.s_store_id AS s_store_id,
            part_s.s_store_name AS s_store_name,
            CAST(SUM(part_ss.ss_quantity) AS BIGINT) AS store_sales_quantity,
            CAST(SUM(part_sr.sr_return_quantity) AS BIGINT) AS store_returns_quantity,
            CAST(SUM(part_ws.ws_quantity) AS BIGINT) AS web_sales_quantity
		FROM 
		(
			SELECT
				sr_item_sk,
				sr_customer_sk,
				sr_ticket_number,
				sr_return_quantity
			FROM
				store_returns sr,
				date_dim d2
			WHERE d2.d_year = 2003
			AND d2.d_moy BETWEEN 1 AND 7 --which were returned in the next six months
			AND sr.sr_returned_date_sk = d2.d_date_sk
		) part_sr
		INNER JOIN 
		(
			SELECT
				ws_item_sk,
				ws_bill_customer_sk,
				ws_quantity
			FROM
				web_sales ws,
				date_dim d3
			-- in the following three years (re-purchased by the returning customer afterwards through the web sales channel)
			WHERE d3.d_year BETWEEN 2003 AND 2005 
			AND ws.ws_sold_date_sk = d3.d_date_sk
		) part_ws ON 
		(
			part_sr.sr_item_sk = part_ws.ws_item_sk
			AND part_sr.sr_customer_sk = part_ws.ws_bill_customer_sk
		) INNER JOIN 
		(
			SELECT
				ss_item_sk,
				ss_store_sk,
				ss_customer_sk,
				ss_ticket_number,
				ss_quantity
			FROM
				store_sales ss,
				date_dim d1
			WHERE d1.d_year = 2003
			AND d1.d_moy = 1
			AND ss.ss_sold_date_sk = d1.d_date_sk
		) part_ss ON 
		(
			part_ss.ss_ticket_number = part_sr.sr_ticket_number
			AND part_ss.ss_item_sk = part_sr.sr_item_sk
			AND part_ss.ss_customer_sk = part_sr.sr_customer_sk
		)
		INNER JOIN store part_s ON 
		(
			part_s.s_store_sk = part_ss.ss_store_sk
		)
		INNER JOIN item part_i ON 
		(
			part_i.i_item_sk = part_ss.ss_item_sk
		)
		GROUP BY
			part_i.i_item_id,
			part_i.i_item_desc,
			part_s.s_store_id,
			part_s.s_store_name
		ORDER BY
			part_i.i_item_id,
			part_i.i_item_desc,
			part_s.s_store_id,
			part_s.s_store_name
		LIMIT 100
	"""
    result = bc.sql(query)
    result['i_item_desc'] = result['i_item_desc'].str.strip()
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
