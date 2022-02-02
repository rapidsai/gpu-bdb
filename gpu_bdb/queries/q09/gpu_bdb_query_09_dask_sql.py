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

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.q09_utils import (
    q09_year,
    q09_part1_ca_country,
    q09_part1_ca_state_IN,
    q09_part1_net_profit_min,
    q09_part1_net_profit_max,
    q09_part1_education_status,
    q09_part1_marital_status,
    q09_part1_sales_price_min,
    q09_part1_sales_price_max,
    q09_part2_ca_country,
    q09_part2_ca_state_IN,
    q09_part2_net_profit_min,
    q09_part2_net_profit_max,
    q09_part2_education_status,
    q09_part2_marital_status,
    q09_part2_sales_price_min,
    q09_part2_sales_price_max,
    q09_part3_ca_country,
    q09_part3_ca_state_IN,
    q09_part3_net_profit_min,
    q09_part3_net_profit_max,
    q09_part3_education_status,
    q09_part3_marital_status,
    q09_part3_sales_price_min,
    q09_part3_sales_price_max,
    read_tables
)


def main(data_dir, client, c, config):
    benchmark(read_tables, config, c, dask_profile=config["dask_profile"])

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
                AND ca1.ca_state IN {q09_part1_ca_state_IN}
                AND {q09_part1_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part1_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part2_ca_country}'
                AND ca1.ca_state IN {q09_part2_ca_state_IN}
                AND {q09_part2_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part2_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part3_ca_country}'
                AND ca1.ca_state IN {q09_part3_ca_state_IN}
                AND {q09_part3_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part3_net_profit_max}
            )
        )
    """
    result = c.sql(query)
    result.columns = ["sum(ss_quantity)"]
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
