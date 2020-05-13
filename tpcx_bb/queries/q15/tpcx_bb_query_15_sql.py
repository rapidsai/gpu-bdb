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
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
		SELECT *
		FROM 
		(
			SELECT
				cat,
				( (count(x)*SUM(xy) - SUM(x)*SUM(y)) / (count(x)*SUM(xx) - SUM(x)*SUM(x)) )  AS slope,
				(SUM(y) - ((count(x)*SUM(xy) - SUM(x)*SUM(y)) / (count(x)*SUM(xx) - SUM(x)*SUM(x)) ) * SUM(x)) / count(x) AS intercept
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
				WHERE EXISTS
				(
					SELECT * 
					FROM 
					(
						SELECT d_date_sk
						FROM date_dim d
						WHERE d.d_date >= DATE '2001-09-02'
						AND   d.d_date <= DATE '2002-09-02'
					) dd
					WHERE s.ss_sold_date_sk = dd.d_date_sk 
				)
				AND i.i_category_id IS NOT NULL
				AND s.ss_store_sk = 10 
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
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        allocator="existing",
        dask_client=client,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    result_df = main(cli_args["data_dir"])
    write_result(
        result_df, output_directory=cli_args["output_dir"],
    )
