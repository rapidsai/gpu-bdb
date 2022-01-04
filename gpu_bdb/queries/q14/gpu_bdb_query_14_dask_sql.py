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

	ws_columns = ["ws_ship_hdemo_sk", "ws_web_page_sk", "ws_sold_time_sk"]
	web_sales = table_reader.read("web_sales", relevant_cols=ws_columns)

	hd_columns = ["hd_demo_sk", "hd_dep_count"]
	household_demographics = table_reader.read(
		"household_demographics", relevant_cols=hd_columns
	)

	wp_columns = ["wp_web_page_sk", "wp_char_count"]
	web_page = table_reader.read("web_page", relevant_cols=wp_columns)

	td_columns = ["t_time_sk", "t_hour"]
	time_dim = table_reader.read("time_dim", relevant_cols=td_columns)

	c.create_table("household_demographics", household_demographics, persist=False)
	c.create_table("web_page", web_page, persist=False)
	c.create_table("web_sales", web_sales, persist=False)
	c.create_table("time_dim", time_dim, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

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

    result = c.sql(query)
    return result


if __name__ == "__main__":
	config = gpubdb_argparser()
	client, c = attach_to_cluster(config)
	run_query(config=config, client=client, query_func=main, sql_context=c)
