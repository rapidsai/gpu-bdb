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
    bc.create_table("web_clickstreams", data_dir + "/web_clickstreams/*.parquet")
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table("item", data_dir + "/item/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
		SELECT DISTINCT wcs_user_sk 
		FROM
		( 
			SELECT
				wcs_user_sk,
				wcs_click_date_sk
			FROM web_clickstreams, item
			WHERE wcs_click_date_sk BETWEEN 37134 AND 37164
			AND i_category IN ('Books', 'Electronics') 
			AND wcs_item_sk = i_item_sk
			AND wcs_user_sk IS NOT NULL
			AND wcs_sales_sk IS NULL 
		) webInRange,
		( 
			SELECT
				ss_customer_sk,
				ss_sold_date_sk
			FROM store_sales, item
			WHERE ss_sold_date_sk BETWEEN 37134 AND 37224
			AND i_category IN ('Books', 'Electronics') -- filter given category 
			AND ss_item_sk = i_item_sk
			AND ss_customer_sk IS NOT NULL
		) storeInRange
		WHERE wcs_user_sk = ss_customer_sk
		AND wcs_click_date_sk < ss_sold_date_sk 
		ORDER BY wcs_user_sk
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
