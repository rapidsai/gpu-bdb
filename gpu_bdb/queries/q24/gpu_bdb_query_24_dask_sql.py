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
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q24_utils import read_tables

def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

    query = """
		WITH temp_table as 
		(
			SELECT
				i_item_sk, 
				imp_sk,
				(imp_competitor_price - i_current_price) / i_current_price AS price_change,
				imp_start_date, 
				(imp_end_date - imp_start_date) AS no_days_comp_price
			FROM item i ,item_marketprices imp 
			WHERE i.i_item_sk = imp.imp_item_sk
			AND i.i_item_sk = 10000
			ORDER BY i_item_sk, imp_sk, imp_start_date
		)
		SELECT ws_item_sk,
		-- avg ( (current_ss_quant + current_ws_quant - prev_ss_quant - prev_ws_quant) / ((prev_ss_quant + prev_ws_quant) * ws.price_change) ) -- single node
			sum( (current_ss_quant+current_ws_quant-prev_ss_quant-prev_ws_quant) / (prev_ss_quant*ws.price_change+prev_ws_quant*ws.price_change) ) 
			/ count( (current_ss_quant + current_ws_quant - prev_ss_quant - prev_ws_quant) / ((prev_ss_quant + prev_ws_quant) * ws.price_change) ) AS cross_price_elasticity
		FROM
		( 
			SELECT
				ws_item_sk,
				imp_sk,
				price_change,
				SUM( CASE WHEN ( (ws_sold_date_sk >= c.imp_start_date) AND (ws_sold_date_sk < (c.imp_start_date + c.no_days_comp_price))) THEN ws_quantity ELSE 0 END ) AS current_ws_quant,
				SUM( CASE WHEN ( (ws_sold_date_sk >= (c.imp_start_date - c.no_days_comp_price)) AND (ws_sold_date_sk < c.imp_start_date)) THEN ws_quantity ELSE 0 END ) AS prev_ws_quant
			FROM web_sales ws
			JOIN temp_table c ON ws.ws_item_sk = c.i_item_sk
			GROUP BY ws_item_sk, imp_sk, price_change
		) ws JOIN
		( 
			SELECT
				ss_item_sk,
				imp_sk,
				price_change,
				SUM( CASE WHEN ((ss_sold_date_sk >= c.imp_start_date) AND (ss_sold_date_sk < (c.imp_start_date + c.no_days_comp_price))) THEN ss_quantity ELSE 0 END) AS current_ss_quant,
				SUM( CASE WHEN ((ss_sold_date_sk >= (c.imp_start_date - c.no_days_comp_price)) AND (ss_sold_date_sk < c.imp_start_date)) THEN ss_quantity ELSE 0 END) AS prev_ss_quant
			FROM store_sales ss
			JOIN temp_table c ON c.i_item_sk = ss.ss_item_sk
			GROUP BY ss_item_sk, imp_sk, price_change
		) ss
		ON (ws.ws_item_sk = ss.ss_item_sk and ws.imp_sk = ss.imp_sk)
		GROUP BY  ws.ws_item_sk
  	"""

    result = c.sql(query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
