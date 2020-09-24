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

from xbb_tools.cluster_startup import attach_to_cluster

from xbb_tools.utils import (
    benchmark,
    tpcxbb_argparser,
    run_query,
)


# -------- Q6 -----------
q06_LIMIT = 100
# --web_sales and store_sales date
q06_YEAR = 2001


def read_tables(data_dir, bc):
    bc.create_table('web_sales', data_dir + "web_sales/*.parquet")
    bc.create_table('store_sales', data_dir + "store_sales/*.parquet")
    bc.create_table('date_dim', data_dir + "date_dim/*.parquet")
    bc.create_table('customer', data_dir + "customer/*.parquet")


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query = f"""
        WITH temp_table_1 as
        (
            SELECT ss_customer_sk AS customer_sk,
                sum( case when (d_year = {q06_YEAR}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS first_year_total,
                sum( case when (d_year = {q06_YEAR + 1}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS second_year_total
            FROM store_sales,
                date_dim
            WHERE ss_sold_date_sk = d_date_sk
            AND   d_year BETWEEN {q06_YEAR} AND {q06_YEAR + 1}
            GROUP BY ss_customer_sk
            -- first_year_total is an aggregation, rewrite all sum () statement
            HAVING sum( case when (d_year = {q06_YEAR}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END) > 0.0
        ),
        temp_table_2 AS
        (
            SELECT ws_bill_customer_sk AS customer_sk ,
                sum( case when (d_year = {q06_YEAR}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS first_year_total,
                sum( case when (d_year = {q06_YEAR + 1}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS second_year_total
            FROM web_sales,
                 date_dim
            WHERE ws_sold_date_sk = d_date_sk
            AND   d_year BETWEEN {q06_YEAR} AND {q06_YEAR + 1}
            GROUP BY ws_bill_customer_sk
            -- required to avoid division by 0, because later we will divide by this value
            HAVING sum( case when (d_year = {q06_YEAR}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0)ELSE 0.0 END) > 0.0
        )
        -- MAIN QUERY
        SELECT
            CAST( (web.second_year_total / web.first_year_total) AS DOUBLE) AS web_sales_increase_ratio,
            c_customer_sk,
            c_first_name,
            c_last_name,
            c_preferred_cust_flag,
            c_birth_country,
            c_login,
            c_email_address
        FROM temp_table_1 store,
            temp_table_2 web,
            customer c
        WHERE store.customer_sk = web.customer_sk
        AND  web.customer_sk = c_customer_sk
        -- if customer has sales in first year for both store and websales,
        -- select him only if web second_year_total/first_year_total
        -- ratio is bigger then his store second_year_total/first_year_total ratio.
        AND (web.second_year_total / web.first_year_total) >
            (store.second_year_total / store.first_year_total)
        ORDER BY
            web_sales_increase_ratio DESC,
            c_customer_sk,
            c_first_name,
            c_last_name,
            c_preferred_cust_flag,
            c_birth_country,
            c_login
        LIMIT {q06_LIMIT}
    """
    result = bc.sql(query)
    return result


if __name__ == "__main__":
    config = tpcxbb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)

