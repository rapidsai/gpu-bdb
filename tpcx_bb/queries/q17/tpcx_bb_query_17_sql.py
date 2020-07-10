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
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import os

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_bsql_query,
)

cli_args = tpcxbb_argparser()

# ------- Q17 ------
q17_year = 2001
q17_month = 12
q17_i_category_IN = "('Books', 'Music')"
date_string = str(q17_year) + "-" + str(q17_month) + "-"

@benchmark(
    compute_result=cli_args["get_read_time"], dask_profile=cli_args["dask_profile"]
)
def read_tables(data_dir):
    bc.create_table("store_sales", data_dir + "store_sales/*.parquet")
    bc.create_table("item", data_dir + "item/*.parquet")
    bc.create_table("customer", data_dir + "customer/*.parquet")
    bc.create_table("store", data_dir + "store/*.parquet")
    bc.create_table("date_dim", data_dir + "date_dim/*.parquet")
    bc.create_table("customer_address", data_dir + "customer_address/*.parquet")
    bc.create_table("promotion", data_dir + "promotion/*.parquet")

@benchmark(dask_profile=cli_args["dask_profile"])
def main(data_dir, client):
    read_tables(data_dir)

    dates_pandas = (
        bc.sql(
            """select min(d_date_sk) as start_date, max(d_date_sk) as end_date
    from date_dim
    where d_year = """
            + str(q17_year)
            + """ and d_moy = """
            + str(q17_month)
            + """
    """
        )
        .compute()
        .to_pandas()
    )
    date_start_sk = dates_pandas["start_date"][0]
    date_end_sk = dates_pandas["end_date"][0]

    query = (
        """
        SELECT CAST(sum(promotional) AS DOUBLE) as promotional, sum(total) as total,
        CASE WHEN sum(total) > 0 THEN CAST (100 * sum(promotional) AS DOUBLE) /
            CAST(sum(total) AS DOUBLE) ELSE 0.0 END as promo_percent
        FROM
        (
            SELECT p_channel_email,
                p_channel_dmail,
                p_channel_tv,
                CASE WHEN (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
                    THEN SUM(ss_ext_sales_price) ELSE 0 END as promotional,
                SUM(ss_ext_sales_price) total
            FROM store_sales ss
            INNER JOIN promotion p ON ss.ss_promo_sk = p.p_promo_sk
            inner join item i on ss.ss_item_sk = i.i_item_sk
            inner join store s on ss.ss_store_sk = s.s_store_sk
            inner join customer c on c.c_customer_sk = ss.ss_customer_sk
            inner join customer_address ca on c.c_current_addr_sk = ca.ca_address_sk
            WHERE i.i_category IN """
        + q17_i_category_IN
        + """ and s.s_gmt_offset = -5.0 and ca.ca_gmt_offset = -5.0 and
            ss.ss_sold_date_sk >= """
        + str(date_start_sk)
        + """ and ss.ss_sold_date_sk <= """
        + str(date_end_sk)
        + """
            
            GROUP BY p_channel_email, p_channel_dmail, p_channel_tv
        ) sum_promotional
        -- we don't need a 'ON' join condition. result is just two numbers.
        ORDER by promotional, total
        LIMIT 100
    """
    )

    result = bc.sql(query)
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
