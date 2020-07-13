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

from blazingsql import BlazingContext
from xbb_tools.cluster_startup import attach_to_cluster
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
)

cli_args = tpcxbb_argparser()


@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("store_sales", data_dir + "/store_sales/*.parquet")
    bc.create_table(
        "customer_address", data_dir + "/customer_address/*.parquet"
    )
    bc.create_table(
        "customer_demographics", data_dir + "/customer_demographics/*.parquet"
    )
    bc.create_table("date_dim", data_dir + "/date_dim/*.parquet")
    bc.create_table("store", data_dir + "/store/*.parquet")


@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    query = """
        SELECT SUM(ss1.ss_quantity)
        FROM store_sales ss1,
            date_dim dd,customer_address ca1,
            store s,
            customer_demographics cd
        -- select date range
        WHERE ss1.ss_sold_date_sk = dd.d_date_sk
        AND dd.d_year = 2001
        AND ss1.ss_addr_sk = ca1.ca_address_sk
        AND s.s_store_sk = ss1.ss_store_sk
        AND cd.cd_demo_sk = ss1.ss_cdemo_sk
        AND
        (
            (
                cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 100 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 150
            )
            OR
            (
                cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 50 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 200
            )
            OR
            (
                cd.cd_marital_status = 'M'
                AND cd.cd_education_status = '4 yr Degree'
                AND 150 <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= 200
            )
        )
        AND
        (
            (
                ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('KY', 'GA', 'NM')
                AND 0 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 2000
            )
            OR
            (
                ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('MT', 'OR', 'IN')
                AND 150 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 3000
            )
            OR
            (
                ca1.ca_country = 'United States'
                AND ca1.ca_state IN ('WI', 'MO', 'WV')
                AND 50 <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= 25000
            )
        )
    """
    result = bc.sql(query)
    result.columns = ["sum(ss_quantity)"]
    return result


if __name__ == "__main__":
    client = attach_to_cluster(cli_args)

    bc = BlazingContext(
        dask_client=client,
        pool=True,
        network_interface=os.environ.get("INTERFACE", "eth0"),
    )

    run_bsql_query(
        cli_args=cli_args, client=client, query_func=main
    )
