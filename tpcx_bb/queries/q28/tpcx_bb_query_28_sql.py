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
import os


from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
)
from tpcx_bb_query_28 import (
    post_etl_processing,
    write_result,
    register_serialization,
)

cli_args = tpcxbb_argparser()


@benchmark(dask_profile=cli_args["dask_profile"])
def read_tables(data_dir):
    bc.create_table("product_reviews", data_dir + "product_reviews/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(client, data_dir):
    read_tables(data_dir)

    # 10 % of data
    query1 = """
        SELECT 
            pr_review_sk,
            pr_review_rating,
            pr_review_content
        FROM product_reviews 
        WHERE mod(pr_review_sk, 10) IN (0) 
        AND pr_review_content IS NOT NULL
        ORDER BY pr_review_sk
    """

    test_data = bc.sql(query1)

    # 90 % of data
    query2 = """
        SELECT 
            pr_review_sk,
            pr_review_rating,
            pr_review_content
        FROM product_reviews 
        WHERE mod(pr_review_sk, 10) IN (1,2,3,4,5,6,7,8,9)
        AND pr_review_content IS NOT NULL
        ORDER BY pr_review_sk
    """

    train_data = bc.sql(query2)

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    result_df, acc, prec, cmat = post_etl_processing(
        client=client, train_data=train_data, test_data=test_data
    )
    return result_df, acc, prec, cmat


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        allocator="existing",
        dask_client=client,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    register_serialization()
    client.run(register_serialization)

    result_df, acc, prec, cmat = main(client=client, data_dir=cli_args["data_dir"])
    write_result(
        result_df, acc, prec, cmat, output_directory=cli_args["output_dir"],
    )

    if cli_args["verify_results"]:
        result_verified = verify_results(cli_args["verify_dir"])
    cli_args["result_verified"] = result_verified
