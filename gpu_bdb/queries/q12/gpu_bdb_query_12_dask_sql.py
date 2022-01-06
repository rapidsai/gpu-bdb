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

import sys
import os

from bdb_tools.cluster_startup import attach_to_cluster

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader


# -------- Q12 -----------
q12_i_category_IN = "'Books', 'Electronics'"

item_cols = ["i_item_sk", "i_category"]
store_sales_cols = ["ss_item_sk", "ss_sold_date_sk", "ss_customer_sk"]
wcs_cols = ["wcs_user_sk", "wcs_click_date_sk", "wcs_item_sk", "wcs_sales_sk"]


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    item_df = table_reader.read("item", relevant_cols=item_cols)
    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    wcs_df = table_reader.read("web_clickstreams", relevant_cols=wcs_cols)
    
    c.create_table("web_clickstreams", wcs_df, persist=False)
    c.create_table("store_sales", store_sales_df, persist=False)
    c.create_table("item", item_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query = f"""
        SELECT DISTINCT wcs_user_sk
        FROM
        (
            SELECT DISTINCT
                wcs_user_sk,
                wcs_click_date_sk
            FROM web_clickstreams, item
            WHERE wcs_click_date_sk BETWEEN 37134 AND 37164
            AND i_category IN ({q12_i_category_IN})
            AND wcs_item_sk = i_item_sk
            AND wcs_user_sk IS NOT NULL
            AND wcs_sales_sk IS NULL
        ) webInRange,
        (
            SELECT DISTINCT
                ss_customer_sk,
                ss_sold_date_sk
            FROM store_sales, item
            WHERE ss_sold_date_sk BETWEEN 37134 AND 37224
            AND i_category IN ({q12_i_category_IN}) -- filter given category
            AND ss_item_sk = i_item_sk
            AND ss_customer_sk IS NOT NULL
        ) storeInRange
        WHERE wcs_user_sk = ss_customer_sk
        AND wcs_click_date_sk < ss_sold_date_sk
        ORDER BY wcs_user_sk
    """
    result = c.sql(query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
