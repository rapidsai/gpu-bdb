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
    bc.create_table("inventory", data_dir + "/inventory/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")
    bc.create_table("warehouse", data_dir + "/warehouse/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
		SELECT
			w_warehouse_name,
			i_item_id,
			SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_before,
			SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_after
		FROM 
			inventory inv,
			item i,
			warehouse w,
			date_dim d
		WHERE i_current_price BETWEEN 0.98 AND 1.5
		AND i_item_sk        = inv_item_sk
		AND inv_warehouse_sk = w_warehouse_sk
		AND inv_date_sk      = d_date_sk
		AND timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 >= -30
		AND timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 <= 30
		GROUP BY w_warehouse_name, i_item_id
		HAVING SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) > 0
		AND
		(
			CAST(
			SUM (CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS DOUBLE)
			/ CAST( SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
			AS DOUBLE) >= CAST(0.6666667 AS DOUBLE)
		)
		AND 
		(
			CAST(
			SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS DOUBLE)
		 	/ CAST ( SUM(CASE WHEN timestampdiff(DAY, timestamp '2001-05-08 00:00:00', CAST(d_date  || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
		 AS DOUBLE) <= CAST(1.50 AS DOUBLE)
		)
		ORDER BY w_warehouse_name, i_item_id
		LIMIT 100
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
