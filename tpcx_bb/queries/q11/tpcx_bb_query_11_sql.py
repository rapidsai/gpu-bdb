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
import cudf

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    write_result,
)

cli_args = tpcxbb_argparser()


@benchmark(dask_profile=cli_args["dask_profile"])
def read_tables(data_dir):
    bc.create_table("web_sales", data_dir + "/web_sales/*.parquet")
    bc.create_table("product_reviews", data_dir + "/product_reviews/*.parquet")
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir):
    read_tables(data_dir)

    query = """
        WITH p AS
        (
            SELECT
                pr_item_sk,
                count(pr_item_sk) AS r_count,
                AVG( CAST(pr_review_rating AS DOUBLE) ) avg_rating  
            FROM product_reviews
            WHERE pr_item_sk IS NOT NULL
            GROUP BY pr_item_sk
        ), s AS
        (
            SELECT
                ws_item_sk
            FROM web_sales ws
            WHERE EXISTS 
            (
                SELECT d_date_sk FROM 
                (
                    SELECT d_date_sk 
                    FROM date_dim d 
                    WHERE CAST(d.d_date AS DATE) >= DATE '2003-01-02'
                    AND  CAST(d.d_date AS DATE) <= DATE '2003-02-02'
                ) dd 
                WHERE ws.ws_sold_date_sk = dd.d_date_sk
            )
            AND ws_item_sk IS NOT null
            GROUP BY ws_item_sk
        )
        SELECT p.r_count    AS x,
            p.avg_rating AS y
        FROM s INNER JOIN p ON p.pr_item_sk = s.ws_item_sk
    """

    result = bc.sql(query)
    sales_corr = result["x"].corr(result["y"]).compute()
    result = cudf.DataFrame(
        [sales_corr], columns=["corr(CAST(reviews_count AS DOUBLE), avg_rating)"]
    )

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
