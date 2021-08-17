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

from bdb_tools.cluster_startup import attach_to_cluster
import os

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask_sql import Context

# -------- Q9 -----------
q09_year = 2001

q09_part1_ca_country = "United States"
q09_part1_ca_state_IN = "'KY', 'GA', 'NM'"
q09_part1_net_profit_min = 0
q09_part1_net_profit_max = 2000
q09_part1_education_status = "4 yr Degree"
q09_part1_marital_status = "M"
q09_part1_sales_price_min = 100
q09_part1_sales_price_max = 150

q09_part2_ca_country = "United States"
q09_part2_ca_state_IN = "'MT', 'OR', 'IN'"
q09_part2_net_profit_min = 150
q09_part2_net_profit_max = 3000
q09_part2_education_status = "4 yr Degree"
q09_part2_marital_status = "M"
q09_part2_sales_price_min = 50
q09_part2_sales_price_max = 200

q09_part3_ca_country = "United States"
q09_part3_ca_state_IN = "'WI', 'MO', 'WV'"
q09_part3_net_profit_min = 50
q09_part3_net_profit_max = 25000
q09_part3_education_status = "4 yr Degree"
q09_part3_marital_status = "M"
q09_part3_sales_price_min = 150
q09_part3_sales_price_max = 200


def read_tables(data_dir, bc):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    ss_columns = [
        "ss_quantity",
        "ss_sold_date_sk",
        "ss_addr_sk",
        "ss_store_sk",
        "ss_cdemo_sk",
        "ss_sales_price",
        "ss_net_profit",
    ]

    store_sales = table_reader.read("store_sales", relevant_cols=ss_columns)

    ca_columns = ["ca_address_sk", "ca_country", "ca_state"]
    customer_address = table_reader.read("customer_address", relevant_cols=ca_columns)

    cd_columns = ["cd_demo_sk", "cd_marital_status", "cd_education_status"]
    customer_demographics = table_reader.read(
        "customer_demographics", relevant_cols=cd_columns
    )

    dd_columns = ["d_year", "d_date_sk"]
    date_dim = table_reader.read("date_dim", relevant_cols=dd_columns)

    s_columns = ["s_store_sk"]
    store = table_reader.read("store", relevant_cols=s_columns)

    bc.create_table("store_sales", store_sales, persist=False)
    bc.create_table("customer_address", customer_address, persist=False)
    bc.create_table("customer_demographics", customer_demographics, persist=False)
    bc.create_table("date_dim", date_dim, persist=False)
    bc.create_table("store", store, persist=False)

    # bc.create_table("store_sales", os.path.join(data_dir, "store_sales/*.parquet"))
    # bc.create_table("customer_address", os.path.join(data_dir, "customer_address/*.parquet"))
    # bc.create_table(
    #     "customer_demographics", os.path.join(data_dir, "customer_demographics/*.parquet"
    # ))
    # bc.create_table("date_dim", os.path.join(data_dir, "date_dim/*.parquet"))
    # bc.create_table("store", os.path.join(data_dir, "store/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query = f"""
        SELECT SUM(ss1.ss_quantity)
        FROM store_sales ss1,
            date_dim dd,customer_address ca1,
            store s,
            customer_demographics cd
        -- select date range
        WHERE ss1.ss_sold_date_sk = dd.d_date_sk
        AND dd.d_year = {q09_year}
        AND ss1.ss_addr_sk = ca1.ca_address_sk
        AND s.s_store_sk = ss1.ss_store_sk
        AND cd.cd_demo_sk = ss1.ss_cdemo_sk
        AND
        (
            (
                cd.cd_marital_status = '{q09_part1_marital_status}'
                AND cd.cd_education_status = '{q09_part1_education_status}'
                AND {q09_part1_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part1_sales_price_max}
            )
            OR
            (
                cd.cd_marital_status = '{q09_part2_marital_status}'
                AND cd.cd_education_status = '{q09_part2_education_status}'
                AND {q09_part2_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part2_sales_price_max}
            )
            OR
            (
                cd.cd_marital_status = '{q09_part3_marital_status}'
                AND cd.cd_education_status = '{q09_part3_education_status}'
                AND {q09_part3_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part3_sales_price_max}
            )
        )
        AND
        (
            (
                ca1.ca_country = '{q09_part1_ca_country}'
                AND ca1.ca_state IN ({q09_part1_ca_state_IN})
                AND {q09_part1_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part1_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part2_ca_country}'
                AND ca1.ca_state IN ({q09_part2_ca_state_IN})
                AND {q09_part2_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part2_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part3_ca_country}'
                AND ca1.ca_state IN ({q09_part3_ca_state_IN})
                AND {q09_part3_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part3_net_profit_max}
            )
        )
    """
    result = bc.sql(query)
    result.columns = ["sum(ss_quantity)"]
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, _ = attach_to_cluster(config)
    c = Context()
    run_query(config=config, client=client, query_func=main, blazing_context=c)
