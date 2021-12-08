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
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader


# ------- Q17 ------
q17_gmt_offset = -5.0
# --store_sales date
q17_year = 2001
q17_month = 12
q17_i_category_IN = "'Books', 'Music'"

store_sales_cols = [
    "ss_ext_sales_price",
    "ss_sold_date_sk",
    "ss_store_sk",
    "ss_customer_sk",
    "ss_promo_sk",
    "ss_item_sk",
]
item_cols = ["i_category", "i_item_sk"]
customer_cols = ["c_customer_sk", "c_current_addr_sk"]
store_cols = ["s_gmt_offset", "s_store_sk"]
date_cols = ["d_date_sk", "d_year", "d_moy"]
customer_address_cols = ["ca_address_sk", "ca_gmt_offset"]
promotion_cols = ["p_channel_email", "p_channel_dmail", "p_channel_tv", "p_promo_sk"]

def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    item_df = table_reader.read("item", relevant_cols=item_cols)
    customer_df = table_reader.read("customer", relevant_cols=customer_cols)
    store_df = table_reader.read("store", relevant_cols=store_cols)
    date_dim_df = table_reader.read("date_dim", relevant_cols=date_cols)
    customer_address_df = table_reader.read(
        "customer_address", relevant_cols=customer_address_cols
    )
    promotion_df = table_reader.read("promotion", relevant_cols=promotion_cols)

    c.create_table("store_sales", store_sales_df, persist=False)
    c.create_table("item", item_df, persist=False)
    c.create_table("customer", customer_df, persist=False)
    c.create_table("store", store_df, persist=False)
    c.create_table("date_dim", date_dim_df, persist=False)
    c.create_table("customer_address", customer_address_df, persist=False)
    c.create_table("promotion", promotion_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query_date = f"""
        select min(d_date_sk) as min_d_date_sk,
            max(d_date_sk) as max_d_date_sk
        from date_dim
        where d_year = {q17_year}
        and d_moy = {q17_month}
    """
    dates_result = c.sql(query_date).compute()

    min_date_sk_val = dates_result["min_d_date_sk"][0]
    max_date_sk_val = dates_result["max_d_date_sk"][0]

    query = f"""
        SELECT sum(promotional) as promotional,
            sum(total) as total,
            CASE WHEN sum(total) > 0.0 THEN (100.0 * sum(promotional)) / sum(total)
                ELSE 0.0 END as promo_percent
        FROM
        (
            SELECT p_channel_email,
                p_channel_dmail,
                p_channel_tv,
                SUM( CAST(ss_ext_sales_price AS DOUBLE) ) total,
                CASE WHEN (p_channel_dmail = 'Y' OR p_channel_email = 'Y' OR p_channel_tv = 'Y')
                    THEN SUM(CAST(ss_ext_sales_price AS DOUBLE)) ELSE 0 END as promotional
            FROM store_sales ss
            INNER JOIN promotion p ON ss.ss_promo_sk = p.p_promo_sk
            inner join item i on ss.ss_item_sk = i.i_item_sk
            inner join store s on ss.ss_store_sk = s.s_store_sk
            inner join customer c on c.c_customer_sk = ss.ss_customer_sk
            inner join customer_address ca
            on c.c_current_addr_sk = ca.ca_address_sk
            WHERE i.i_category IN ({q17_i_category_IN})
            AND s.s_gmt_offset = {q17_gmt_offset}
            AND ca.ca_gmt_offset = {q17_gmt_offset}
            AND ss.ss_sold_date_sk >= {min_date_sk_val}
            AND ss.ss_sold_date_sk <= {max_date_sk_val}
            GROUP BY p_channel_email, p_channel_dmail, p_channel_tv
        ) sum_promotional
        -- we don't need a 'ON' join condition. result is just two numbers.
    """
    result = c.sql(query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config)
    run_query(config=config, client=client, query_func=main, sql_context=c)
