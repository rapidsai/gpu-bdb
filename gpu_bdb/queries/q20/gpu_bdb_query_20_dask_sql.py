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
from dask import delayed
from dask.distributed import wait
import numpy as np

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from bdb_tools.q20_utils import get_clusters

from dask_sql import Context


def read_tables(data_dir, c, config):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    store_sales_cols = [
        "ss_customer_sk",
        "ss_ticket_number",
        "ss_item_sk",
        "ss_net_paid",
    ]
    store_returns_cols = [
        "sr_item_sk",
        "sr_customer_sk",
        "sr_ticket_number",
        "sr_return_amt",
    ]

    store_sales_df = table_reader.read("store_sales", relevant_cols=store_sales_cols)
    store_returns_df = table_reader.read(
        "store_returns", relevant_cols=store_returns_cols
    )

    c.create_table("store_sales", store_sales_df, persist=False)
    c.create_table("store_returns", store_returns_df, persist=False)


def main(data_dir, client, c, config):
    benchmark(read_tables, data_dir, c, config, dask_profile=config["dask_profile"])

    query = """
        SELECT
            ss_customer_sk AS user_sk,
            round(CASE WHEN ((returns_count IS NULL) OR (orders_count IS NULL)
                OR ((returns_count / orders_count) IS NULL) ) THEN 0.0
                ELSE (returns_count / orders_count) END, 7) AS orderRatio,
            round(CASE WHEN ((returns_items IS NULL) OR (orders_items IS NULL)
                OR ((returns_items / orders_items) IS NULL) ) THEN 0.0
                ELSE (returns_items / orders_items) END, 7) AS itemsRatio,
            round(CASE WHEN ((returns_money IS NULL) OR (orders_money IS NULL)
                OR ((returns_money / orders_money) IS NULL) ) THEN 0.0
                ELSE (returns_money / orders_money) END, 7) AS monetaryRatio,
            round(CASE WHEN ( returns_count IS NULL) THEN 0.0
                ELSE returns_count END, 0) AS frequency
        FROM
        (
            SELECT
                ss_customer_sk,
                -- return order ratio
                CAST (COUNT(distinct(ss_ticket_number)) AS DOUBLE)
                    AS orders_count,
                -- return ss_item_sk ratio
                CAST (COUNT(ss_item_sk) AS DOUBLE) AS orders_items,
                -- return monetary amount ratio
                CAST(SUM( ss_net_paid ) AS DOUBLE) AS orders_money
            FROM store_sales s
            GROUP BY ss_customer_sk
        ) orders
        LEFT OUTER JOIN
        (
            SELECT
                sr_customer_sk,
                -- return order ratio
                CAST(count(distinct(sr_ticket_number)) AS DOUBLE)
                    AS returns_count,
                -- return ss_item_sk ratio
                CAST (COUNT(sr_item_sk) AS DOUBLE) AS returns_items,
                -- return monetary amount ratio
                CAST( SUM( sr_return_amt ) AS DOUBLE) AS returns_money
            FROM store_returns
            GROUP BY sr_customer_sk
        ) returned ON ss_customer_sk=sr_customer_sk
    """
    final_df = c.sql(query)

    final_df = final_df.fillna(0)
    final_df = final_df.repartition(npartitions=1).persist()
    wait(final_df)

    final_df = final_df.sort_values(["user_sk"]).reset_index(drop=True)
    final_df = final_df.persist()
    wait(final_df)

    feature_cols = ["orderRatio", "itemsRatio", "monetaryRatio", "frequency"]

    results_dict = get_clusters(
        client=client, ml_input_df=final_df, feature_cols=feature_cols
    )

    return results_dict


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, c = attach_to_cluster(config, create_sql_context=True)
    run_query(config=config, client=client, query_func=main, sql_context=c)
