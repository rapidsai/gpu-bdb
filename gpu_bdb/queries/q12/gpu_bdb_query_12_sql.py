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


# -------- Q12 -----------
q12_i_category_IN = "'Books', 'Electronics'"


def read_tables(data_dir, bc):
    bc.create_table("web_clickstreams",
                    os.path.join(data_dir, "web_clickstreams/*.parquet"))
    bc.create_table("store_sales",
                    os.path.join(data_dir, "store_sales/*.parquet"))
    bc.create_table("item", os.path.join(data_dir, "item/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

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
    result = bc.sql(query)
    return result


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, bc = attach_to_cluster(config, create_blazing_context=True)
    run_query(config=config, client=client, query_func=main, blazing_context=bc)
