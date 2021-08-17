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
import cudf

from bdb_tools.utils import (
    benchmark,
    gpubdb_argparser,
    run_query,
)

from bdb_tools.readers import build_reader

from dask_sql import Context

def read_tables(data_dir, bc):
    table_reader = build_reader(
        data_format=config["file_format"],
        basepath=config["data_dir"],
        split_row_groups=config["split_row_groups"],
    )

    product_review_cols = [
        "pr_review_rating",
        "pr_item_sk",
    ]
    web_sales_cols = [
        "ws_sold_date_sk",
        "ws_net_paid",
        "ws_item_sk",
    ]
    date_cols = ["d_date_sk", "d_date"]

    pr_df = table_reader.read("product_reviews", relevant_cols=product_review_cols)
    # we only read int columns here so it should scale up to sf-10k as just 26M rows
    pr_df = pr_df.repartition(npartitions=1)

    ws_df = table_reader.read("web_sales", relevant_cols=web_sales_cols)
    date_df = table_reader.read("date_dim", relevant_cols=date_cols)

    bc.create_table("web_sales", ws_df, persist=False)
    bc.create_table("product_reviews", pr_df, persist=False)
    bc.create_table("date_dim", date_df, persist=False)

    # bc.create_table("web_sales", os.path.join(data_dir, "web_sales/*.parquet"))
    # bc.create_table("product_reviews", os.path.join(data_dir, "product_reviews/*.parquet"))
    # bc.create_table("date_dim", os.path.join(data_dir, "date_dim/*.parquet"))


def main(data_dir, client, bc, config):
    benchmark(read_tables, data_dir, bc, dask_profile=config["dask_profile"])

    query = """
        WITH p AS
        (
            SELECT
                pr_item_sk,
                count(pr_item_sk) AS r_count,
                AVG( CAST(pr_review_rating AS DOUBLE) ) avg_rating  
            FROM product_reviews
            WHERE pr_item_sk IS NOT NULL
            GROUP BY pr_item_sk
        ), s AS
        (
            SELECT
                ws_item_sk
            FROM web_sales ws
            INNER JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
            WHERE ws_item_sk IS NOT null
            AND CAST(d.d_date AS DATE) >= DATE '2003-01-02'
            AND CAST(d.d_date AS DATE) <= DATE '2003-02-02'
            GROUP BY ws_item_sk
        )
        SELECT p.r_count    AS x,
            p.avg_rating AS y
        FROM s INNER JOIN p ON p.pr_item_sk = s.ws_item_sk
    """

    result = bc.sql(query)
    sales_corr = result["x"].corr(result["y"]).compute()
    result_df = cudf.DataFrame([sales_corr])
    result_df.columns = ["corr(CAST(reviews_count AS DOUBLE), avg_rating)"]
    return result_df


if __name__ == "__main__":
    config = gpubdb_argparser()
    client, _ = attach_to_cluster(config)
    c = Context()
    run_query(config=config, client=client, query_func=main, blazing_context=c)
